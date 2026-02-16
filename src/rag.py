"""
RAG (Retrieval-Augmented Generation) module
Handles chunking, embeddings, FAISS vector store, and retrieval
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class Chunk:
    """A text chunk with metadata"""
    text: str
    section_id: str
    chunk_id: str
    start_char: int
    end_char: int


class EmbeddingModel:
    """Wrapper for sentence transformer embedding model"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model name for sentence-transformers
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode([text])[0]


class VectorStore:
    """FAISS-based vector store for semantic search"""

    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize vector store

        Args:
            embedding_model: Embedding model to use
        """
        if not HAS_FAISS:
            raise ImportError("faiss is not installed. Install with: pip install faiss-cpu")

        self.embedding_model = embedding_model
        self.index = None
        self.chunks: List[Chunk] = []

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Build FAISS index from chunks

        Args:
            chunks: List of text chunks
        """
        self.chunks = chunks

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for most similar chunks

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode_single(query)

        # Search
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )

        # Convert L2 distances to similarity scores (1 / (1 + distance))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1.0 / (1.0 + dist)
            results.append((self.chunks[idx], similarity))

        return results


def chunk_text(text: str, section_id: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to chunk
        section_id: ID of the section this text belongs to
        chunk_size: Target size of each chunk in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of Chunk objects
    """
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunk = Chunk(
            text=chunk_text,
            section_id=section_id,
            chunk_id=f"{section_id}_chunk_{chunk_idx}",
            start_char=start,
            end_char=min(end, len(text))
        )
        chunks.append(chunk)

        chunk_idx += 1
        start = end - overlap

    return chunks


def compute_similarity(text1: str, text2: str, embedding_model: EmbeddingModel) -> float:
    """
    Compute cosine similarity between two texts

    Args:
        text1: First text
        text2: Second text
        embedding_model: Embedding model to use

    Returns:
        Similarity score between 0 and 1
    """
    embeddings = embedding_model.encode([text1, text2])

    # Cosine similarity
    emb1 = embeddings[0]
    emb2 = embeddings[1]

    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)

    # Normalize to 0-1 range
    return float((similarity + 1) / 2)


class RAGSystem:
    """Complete RAG system for policy documents"""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize RAG system with embedding model"""
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.vector_store = VectorStore(self.embedding_model)

    def index_sections(self, sections: List, chunk_size: int = 500) -> None:
        """
        Index sections into vector store

        Args:
            sections: List of Section objects
            chunk_size: Size of text chunks
        """
        all_chunks = []

        for section in sections:
            chunks = chunk_text(section.text, section.section_id, chunk_size=chunk_size)
            all_chunks.extend(chunks)

        self.vector_store.build_index(all_chunks)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of (chunk, score) tuples
        """
        return self.vector_store.search(query, k=k)

    def compute_section_similarity(self, section1, section2) -> float:
        """
        Compute similarity between two sections

        Args:
            section1: First Section object
            section2: Second Section object

        Returns:
            Similarity score
        """
        # Use title + first 200 chars for comparison
        text1 = section1.title + " " + section1.text[:200]
        text2 = section2.title + " " + section2.text[:200]

        return compute_similarity(text1, text2, self.embedding_model)
