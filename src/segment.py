"""
Section segmentation module
Detects headings using heuristics and splits document into sections
"""
import re
from typing import List, Tuple
from src.schemas import Section


def is_all_caps_heading(line: str) -> bool:
    """Check if line is an ALL CAPS heading"""
    line = line.strip()
    if len(line) < 3 or len(line) > 100:
        return False

    # Remove common punctuation
    cleaned = re.sub(r'[:\-\.]', '', line)

    # Check if alphabetic chars are all uppercase
    alpha_chars = [c for c in cleaned if c.isalpha()]
    if len(alpha_chars) < 3:
        return False

    return all(c.isupper() for c in alpha_chars)


def is_numbered_heading(line: str) -> bool:
    """
    Check if line is a numbered heading like:
    - "1.2 Coverage Details"
    - "SECTION 3: Definitions"
    - "Article IV - Claims"
    """
    line = line.strip()
    if len(line) < 3 or len(line) > 100:
        return False

    patterns = [
        r'^\d+\.\d+\s+\w+',  # 1.2 Heading
        r'^\d+\.\s+\w+',      # 1. Heading
        r'^SECTION\s+\d+',    # SECTION 3
        r'^Article\s+[IVX]+', # Article IV
        r'^ARTICLE\s+\d+',    # ARTICLE 3
        r'^Part\s+\d+',       # Part 2
        r'^Chapter\s+\d+',    # Chapter 5
    ]

    for pattern in patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True

    return False


def detect_headings(text: str) -> List[Tuple[int, str]]:
    """
    Detect headings in text using heuristics

    Returns:
        List of (char_position, heading_text) tuples
    """
    lines = text.split('\n')
    headings = []
    char_pos = 0

    for line in lines:
        stripped = line.strip()

        if stripped and (is_all_caps_heading(stripped) or is_numbered_heading(stripped)):
            headings.append((char_pos, stripped))

        char_pos += len(line) + 1  # +1 for newline

    return headings


def segment_document(text: str, doc_name: str = "document") -> List[Section]:
    """
    Segment document into sections based on detected headings

    Args:
        text: Full document text
        doc_name: Name/identifier for the document

    Returns:
        List of Section objects
    """
    headings = detect_headings(text)

    if not headings:
        # No headings found, treat entire document as one section
        return [Section(
            section_id=f"{doc_name}_0",
            title="Full Document",
            text=text,
            start_char=0,
            end_char=len(text)
        )]

    sections = []

    for i, (start_pos, heading) in enumerate(headings):
        # Determine end position
        if i < len(headings) - 1:
            end_pos = headings[i + 1][0]
        else:
            end_pos = len(text)

        # Extract section text (excluding the heading line itself for cleaner text)
        section_text = text[start_pos:end_pos].strip()

        # Create section ID
        section_id = f"{doc_name}_{i}"

        sections.append(Section(
            section_id=section_id,
            title=heading,
            text=section_text,
            start_char=start_pos,
            end_char=end_pos
        ))

    # Check for preamble (text before first heading)
    if headings and headings[0][0] > 0:
        preamble_text = text[0:headings[0][0]].strip()
        if preamble_text and len(preamble_text) > 50:  # Only add if substantial
            preamble = Section(
                section_id=f"{doc_name}_preamble",
                title="Preamble",
                text=preamble_text,
                start_char=0,
                end_char=headings[0][0]
            )
            sections.insert(0, preamble)

    return sections


def get_section_preview(section: Section, max_chars: int = 200) -> str:
    """
    Get a preview of section text for display

    Args:
        section: Section object
        max_chars: Maximum characters to return

    Returns:
        Preview text with ellipsis if truncated
    """
    text = section.text.strip()
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def merge_small_sections(sections: List[Section], min_length: int = 100) -> List[Section]:
    """
    Merge sections that are too small with the previous section

    Args:
        sections: List of sections
        min_length: Minimum section length in characters

    Returns:
        Merged sections list
    """
    if not sections:
        return []

    merged = [sections[0]]

    for section in sections[1:]:
        if len(section.text) < min_length and merged:
            # Merge with previous section
            prev = merged[-1]
            prev.text = prev.text + "\n\n" + section.text
            prev.end_char = section.end_char
        else:
            merged.append(section)

    return merged
