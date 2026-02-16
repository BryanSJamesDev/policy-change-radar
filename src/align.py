"""
Section alignment module
Aligns sections between two policy versions using embedding similarity
"""
from typing import List, Tuple, Set
from src.schemas import Section, SectionAlignment
from src.rag import RAGSystem


def greedy_bipartite_matching(
    sections_a: List[Section],
    sections_b: List[Section],
    similarity_matrix: List[List[float]],
    threshold: float = 0.5
) -> List[Tuple[int, int, float]]:
    """
    Greedy bipartite matching algorithm for section alignment

    Args:
        sections_a: Sections from version A
        sections_b: Sections from version B
        similarity_matrix: Matrix of similarity scores [i][j] for section_a[i] and section_b[j]
        threshold: Minimum similarity to consider a match

    Returns:
        List of (index_a, index_b, score) tuples for matched sections
    """
    matches = []
    used_a: Set[int] = set()
    used_b: Set[int] = set()

    # Create list of all possible matches above threshold
    candidates = []
    for i in range(len(sections_a)):
        for j in range(len(sections_b)):
            score = similarity_matrix[i][j]
            if score >= threshold:
                candidates.append((i, j, score))

    # Sort by score descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedily select best matches
    for idx_a, idx_b, score in candidates:
        if idx_a not in used_a and idx_b not in used_b:
            matches.append((idx_a, idx_b, score))
            used_a.add(idx_a)
            used_b.add(idx_b)

    return matches


def align_sections(
    sections_a: List[Section],
    sections_b: List[Section],
    rag_system: RAGSystem,
    threshold: float = 0.5
) -> List[SectionAlignment]:
    """
    Align sections between two policy versions

    Args:
        sections_a: Sections from policy version A
        sections_b: Sections from policy version B
        rag_system: RAG system with embedding model
        threshold: Minimum similarity threshold for matching

    Returns:
        List of SectionAlignment objects
    """
    # Build similarity matrix
    similarity_matrix = []

    for section_a in sections_a:
        row = []
        for section_b in sections_b:
            score = rag_system.compute_section_similarity(section_a, section_b)
            row.append(score)
        similarity_matrix.append(row)

    # Perform greedy matching
    matches = greedy_bipartite_matching(sections_a, sections_b, similarity_matrix, threshold)

    # Create alignment objects
    alignments = []

    # Add matched pairs
    matched_a_indices = set()
    matched_b_indices = set()

    for idx_a, idx_b, score in matches:
        alignment = SectionAlignment(
            section_a_id=sections_a[idx_a].section_id,
            section_b_id=sections_b[idx_b].section_id,
            similarity_score=score,
            is_matched=True
        )
        alignments.append(alignment)
        matched_a_indices.add(idx_a)
        matched_b_indices.add(idx_b)

    # Add unmatched sections from A (removed sections)
    for idx, section_a in enumerate(sections_a):
        if idx not in matched_a_indices:
            alignment = SectionAlignment(
                section_a_id=section_a.section_id,
                section_b_id=None,
                similarity_score=0.0,
                is_matched=False
            )
            alignments.append(alignment)

    # Add unmatched sections from B (added sections)
    for idx, section_b in enumerate(sections_b):
        if idx not in matched_b_indices:
            alignment = SectionAlignment(
                section_a_id=None,
                section_b_id=section_b.section_id,
                similarity_score=0.0,
                is_matched=False
            )
            alignments.append(alignment)

    return alignments


def get_alignment_stats(alignments: List[SectionAlignment]) -> dict:
    """
    Get statistics about section alignments

    Args:
        alignments: List of alignments

    Returns:
        Dictionary with statistics
    """
    matched = [a for a in alignments if a.is_matched]
    removed = [a for a in alignments if a.section_a_id and not a.section_b_id]
    added = [a for a in alignments if a.section_b_id and not a.section_a_id]

    avg_similarity = sum(a.similarity_score for a in matched) / len(matched) if matched else 0.0

    return {
        "total_alignments": len(alignments),
        "matched_pairs": len(matched),
        "removed_sections": len(removed),
        "added_sections": len(added),
        "avg_similarity": avg_similarity
    }


def find_section_by_id(sections: List[Section], section_id: str) -> Section:
    """
    Find a section by its ID

    Args:
        sections: List of sections
        section_id: Section ID to find

    Returns:
        Section object or None
    """
    for section in sections:
        if section.section_id == section_id:
            return section
    return None
