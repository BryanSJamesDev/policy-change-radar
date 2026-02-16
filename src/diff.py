"""
Diff detection and classification module
Detects changes between aligned sections and classifies severity
"""
import re
from typing import List
from difflib import SequenceMatcher
from src.schemas import Section, SectionAlignment, ChangeSummary, ChangeType


# Keywords that indicate high-severity changes
HIGH_SEVERITY_KEYWORDS = [
    'coverage', 'exclusion', 'premium', 'deductible', 'limit', 'liability',
    'claim', 'payment', 'benefit', 'penalty', 'cancellation', 'termination',
    'requirement', 'obligation', 'prohibited', 'must', 'shall', 'mandatory'
]

MEDIUM_SEVERITY_KEYWORDS = [
    'notice', 'notification', 'timeframe', 'deadline', 'process', 'procedure',
    'documentation', 'evidence', 'proof', 'submit', 'file', 'report'
]


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity ratio between two texts using sequence matching

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity ratio between 0 and 1
    """
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def count_keyword_matches(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in text"""
    text_lower = text.lower()
    count = 0
    for keyword in keywords:
        if keyword in text_lower:
            count += 1
    return count


def calculate_severity_score(section_a: Section, section_b: Section, similarity: float) -> int:
    """
    Calculate severity score (1-5) based on changes

    Args:
        section_a: Section from version A
        section_b: Section from version B
        similarity: Similarity score between sections

    Returns:
        Severity score from 1 (minor) to 5 (critical)
    """
    # Base score on similarity
    if similarity > 0.95:
        base_score = 1
    elif similarity > 0.85:
        base_score = 2
    elif similarity > 0.7:
        base_score = 3
    elif similarity > 0.5:
        base_score = 4
    else:
        base_score = 5

    # Check for high-severity keywords in either version
    combined_text = section_a.text + " " + section_b.text
    high_severity_count = count_keyword_matches(combined_text, HIGH_SEVERITY_KEYWORDS)
    medium_severity_count = count_keyword_matches(combined_text, MEDIUM_SEVERITY_KEYWORDS)

    # Boost score if high-severity keywords present
    if high_severity_count >= 3:
        base_score = min(5, base_score + 1)
    elif medium_severity_count >= 3:
        base_score = min(5, base_score + 0)

    return base_score


def classify_change(
    section_a: Section,
    section_b: Section,
    similarity: float
) -> ChangeType:
    """
    Classify the type of change between two sections

    Args:
        section_a: Section from version A (can be None for added)
        section_b: Section from version B (can be None for removed)
        similarity: Similarity score

    Returns:
        ChangeType enum value
    """
    if section_a is None:
        return ChangeType.ADDED
    if section_b is None:
        return ChangeType.REMOVED

    # Check similarity
    text_similarity = compute_text_similarity(section_a.text, section_b.text)

    if text_similarity > 0.98:
        return ChangeType.UNCHANGED
    elif text_similarity > 0.85:
        return ChangeType.MODIFIED_MINOR
    else:
        return ChangeType.MODIFIED_MAJOR


def detect_changes(
    sections_a: List[Section],
    sections_b: List[Section],
    alignments: List[SectionAlignment]
) -> List[ChangeSummary]:
    """
    Detect and classify changes between aligned sections

    Args:
        sections_a: Sections from policy A
        sections_b: Sections from policy B
        alignments: Section alignments

    Returns:
        List of ChangeSummary objects (without LLM-generated content)
    """
    # Create lookup dictionaries
    sections_a_dict = {s.section_id: s for s in sections_a}
    sections_b_dict = {s.section_id: s for s in sections_b}

    changes = []

    for alignment in alignments:
        section_a = sections_a_dict.get(alignment.section_a_id) if alignment.section_a_id else None
        section_b = sections_b_dict.get(alignment.section_b_id) if alignment.section_b_id else None

        # Classify change type
        change_type = classify_change(section_a, section_b, alignment.similarity_score)

        # Calculate severity
        if section_a and section_b:
            severity = calculate_severity_score(section_a, section_b, alignment.similarity_score)
        elif section_a or section_b:
            # Removed or added sections are at least medium severity
            severity = 3
        else:
            severity = 1

        # Create change summary (LLM will fill in details later)
        change = ChangeSummary(
            section_a_id=alignment.section_a_id,
            section_b_id=alignment.section_b_id,
            change_type=change_type,
            severity_score=severity,
            what_changed=[],
            why_it_matters=[],
            evidence=[]
        )

        changes.append(change)

    return changes


def extract_diff_snippets(text1: str, text2: str, max_snippets: int = 3) -> List[str]:
    """
    Extract snippets showing differences between two texts

    Args:
        text1: First text
        text2: Second text
        max_snippets: Maximum number of snippets to extract

    Returns:
        List of text snippets highlighting differences
    """
    # Use SequenceMatcher to find differences
    matcher = SequenceMatcher(None, text1, text2)
    snippets = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ['replace', 'delete', 'insert']:
            # Extract context around the change
            if tag == 'delete' or tag == 'replace':
                start = max(0, i1 - 20)
                end = min(len(text1), i2 + 20)
                snippet = text1[start:end].strip()
                if snippet and len(snippet.split()) <= 25:
                    snippets.append(snippet)

            if len(snippets) >= max_snippets:
                break

    return snippets[:max_snippets]


def get_change_statistics(changes: List[ChangeSummary]) -> dict:
    """
    Get statistics about detected changes

    Args:
        changes: List of change summaries

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_changes": len(changes),
        "unchanged": 0,
        "modified_minor": 0,
        "modified_major": 0,
        "removed": 0,
        "added": 0,
        "avg_severity": 0.0,
        "high_severity_count": 0  # severity >= 4
    }

    for change in changes:
        stats[change.change_type.value] += 1
        stats["avg_severity"] += change.severity_score
        if change.severity_score >= 4:
            stats["high_severity_count"] += 1

    if changes:
        stats["avg_severity"] /= len(changes)

    return stats
