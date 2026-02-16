"""
Unit tests for section segmentation
"""
import pytest
from src.segment import (
    is_all_caps_heading,
    is_numbered_heading,
    detect_headings,
    segment_document
)


def test_is_all_caps_heading():
    """Test ALL CAPS heading detection"""
    assert is_all_caps_heading("SECTION 1: COVERAGE") is True
    assert is_all_caps_heading("COVERAGE LIMITS") is True
    assert is_all_caps_heading("DEFINITIONS") is True

    # Should not match
    assert is_all_caps_heading("This is normal text") is False
    assert is_all_caps_heading("Section 1") is False
    assert is_all_caps_heading("A") is False  # Too short
    assert is_all_caps_heading("123456") is False  # No alpha chars


def test_is_numbered_heading():
    """Test numbered heading detection"""
    assert is_numbered_heading("1. Introduction") is True
    assert is_numbered_heading("1.2 Coverage Details") is True
    assert is_numbered_heading("SECTION 3: Terms") is True
    assert is_numbered_heading("Article IV - Claims") is True
    assert is_numbered_heading("ARTICLE 5") is True
    assert is_numbered_heading("Part 2") is True
    assert is_numbered_heading("Chapter 3") is True

    # Should not match
    assert is_numbered_heading("This is normal text") is False
    assert is_numbered_heading("The number 1 appears here") is False


def test_detect_headings():
    """Test heading detection in text"""
    text = """
PREAMBLE

This is the preamble text.

SECTION 1: COVERAGE

Coverage details go here.

1.2 Specific Coverage

More details.
"""

    headings = detect_headings(text)

    # Should detect PREAMBLE, SECTION 1, and 1.2
    assert len(headings) >= 3

    # Check that headings are detected
    heading_texts = [h[1] for h in headings]
    assert "PREAMBLE" in heading_texts
    assert any("SECTION 1" in h for h in heading_texts)


def test_segment_document_basic():
    """Test basic document segmentation"""
    text = """
SECTION 1: COVERAGE

This section covers coverage details.
Coverage limits apply.

SECTION 2: EXCLUSIONS

This section lists exclusions.
"""

    sections = segment_document(text, "test_doc")

    # Should have at least 2 sections
    assert len(sections) >= 2

    # Check section structure
    for section in sections:
        assert section.section_id.startswith("test_doc")
        assert len(section.title) > 0
        assert len(section.text) > 0
        assert section.start_char >= 0
        assert section.end_char > section.start_char


def test_segment_document_no_headings():
    """Test document with no headings"""
    text = "This is just plain text with no headings at all."

    sections = segment_document(text, "plain_doc")

    # Should create one section for full document
    assert len(sections) == 1
    assert sections[0].title == "Full Document"
    assert sections[0].text == text


def test_segment_document_with_preamble():
    """Test document with text before first heading"""
    text = """This is preamble text before any heading.
It should be captured.

SECTION 1: FIRST SECTION

First section content.
"""

    sections = segment_document(text, "doc_with_preamble")

    # Should have preamble + section 1
    assert len(sections) >= 2

    # Check if preamble is captured
    has_preamble = any("preamble" in s.title.lower() for s in sections)
    assert has_preamble is True


def test_segment_document_mixed_headings():
    """Test document with mixed heading styles"""
    text = """
ALL CAPS HEADING

Some text.

1.2 Numbered Heading

More text.

SECTION 3: Mixed Style

Final text.
"""

    sections = segment_document(text, "mixed_doc")

    # Should detect all three heading styles
    assert len(sections) >= 3

    section_titles = [s.title for s in sections]
    assert "ALL CAPS HEADING" in section_titles
    assert "1.2 Numbered Heading" in section_titles
