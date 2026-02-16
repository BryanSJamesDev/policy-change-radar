"""
Unit tests for section alignment
"""
import pytest
from src.segment import segment_document
from src.align import greedy_bipartite_matching, align_sections, find_section_by_id
from src.rag import RAGSystem
from src.schemas import Section


def test_greedy_bipartite_matching():
    """Test greedy bipartite matching algorithm"""
    # Create dummy sections
    sections_a = [
        Section(section_id="a1", title="Coverage", text="Coverage text", start_char=0, end_char=10),
        Section(section_id="a2", title="Exclusions", text="Exclusions text", start_char=10, end_char=20)
    ]

    sections_b = [
        Section(section_id="b1", title="Coverage", text="Coverage text updated", start_char=0, end_char=15),
        Section(section_id="b2", title="Exclusions", text="Exclusions text modified", start_char=15, end_char=30)
    ]

    # Create similarity matrix (high similarity for matching sections)
    similarity_matrix = [
        [0.95, 0.2],  # a1 matches b1
        [0.2, 0.92]   # a2 matches b2
    ]

    matches = greedy_bipartite_matching(sections_a, sections_b, similarity_matrix, threshold=0.5)

    # Should find 2 matches
    assert len(matches) == 2

    # Check matches are correct
    assert (0, 0, 0.95) in matches  # a1 -> b1
    assert (1, 1, 0.92) in matches  # a2 -> b2


def test_greedy_bipartite_matching_with_threshold():
    """Test matching with threshold filtering"""
    sections_a = [
        Section(section_id="a1", title="Section 1", text="Text 1", start_char=0, end_char=10)
    ]

    sections_b = [
        Section(section_id="b1", title="Section 1", text="Text 1", start_char=0, end_char=10)
    ]

    # Low similarity - below threshold
    similarity_matrix = [[0.3]]

    matches = greedy_bipartite_matching(sections_a, sections_b, similarity_matrix, threshold=0.5)

    # Should find no matches
    assert len(matches) == 0


def test_align_sections_basic():
    """Test basic section alignment"""
    # Create test documents
    text_a = """
SECTION 1: COVERAGE

Coverage details for version A.

SECTION 2: EXCLUSIONS

Exclusions for version A.
"""

    text_b = """
SECTION 1: COVERAGE

Coverage details for version B - updated.

SECTION 2: EXCLUSIONS

Exclusions for version B - modified.
"""

    sections_a = segment_document(text_a, "doc_a")
    sections_b = segment_document(text_b, "doc_b")

    # Initialize RAG system
    rag_system = RAGSystem()

    # Align sections
    alignments = align_sections(sections_a, sections_b, rag_system, threshold=0.3)

    # Should have alignments
    assert len(alignments) > 0

    # Count matched pairs
    matched = [a for a in alignments if a.is_matched]
    assert len(matched) >= 2  # Should match both sections


def test_align_sections_with_additions():
    """Test alignment when sections are added"""
    text_a = """
SECTION 1: COVERAGE

Coverage text.
"""

    text_b = """
SECTION 1: COVERAGE

Coverage text.

SECTION 2: NEW SECTION

This is a new section.
"""

    sections_a = segment_document(text_a, "doc_a")
    sections_b = segment_document(text_b, "doc_b")

    rag_system = RAGSystem()
    alignments = align_sections(sections_a, sections_b, rag_system, threshold=0.3)

    # Should have alignments for both matched and unmatched
    assert len(alignments) >= 2

    # Should have at least one unmatched (added) section
    unmatched_b = [a for a in alignments if a.section_b_id and not a.section_a_id]
    assert len(unmatched_b) >= 1


def test_align_sections_with_removals():
    """Test alignment when sections are removed"""
    text_a = """
SECTION 1: COVERAGE

Coverage text.

SECTION 2: REMOVED SECTION

This will be removed.
"""

    text_b = """
SECTION 1: COVERAGE

Coverage text.
"""

    sections_a = segment_document(text_a, "doc_a")
    sections_b = segment_document(text_b, "doc_b")

    rag_system = RAGSystem()
    alignments = align_sections(sections_a, sections_b, rag_system, threshold=0.3)

    # Should have alignments
    assert len(alignments) >= 2

    # Should have at least one unmatched (removed) section
    unmatched_a = [a for a in alignments if a.section_a_id and not a.section_b_id]
    assert len(unmatched_a) >= 1


def test_find_section_by_id():
    """Test finding section by ID"""
    sections = [
        Section(section_id="s1", title="Section 1", text="Text 1", start_char=0, end_char=10),
        Section(section_id="s2", title="Section 2", text="Text 2", start_char=10, end_char=20)
    ]

    # Should find section
    found = find_section_by_id(sections, "s1")
    assert found is not None
    assert found.section_id == "s1"

    # Should not find non-existent section
    not_found = find_section_by_id(sections, "s99")
    assert not_found is None
