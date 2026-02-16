"""
PDF and text extraction module
Supports both PDF files (via PyMuPDF) and plain text files
"""
import io
from pathlib import Path
from typing import Union

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def extract_text_from_pdf(pdf_path: Union[str, Path, bytes]) -> str:
    """
    Extract text from PDF using PyMuPDF

    Args:
        pdf_path: Path to PDF file or bytes content

    Returns:
        Extracted text as string
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF (fitz) is not installed. Install with: pip install pymupdf")

    text_parts = []

    if isinstance(pdf_path, bytes):
        # Handle uploaded file bytes
        doc = fitz.open(stream=pdf_path, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(text)
    finally:
        doc.close()

    return "\n".join(text_parts)


def extract_text_from_file(file_path: Union[str, Path]) -> str:
    """
    Extract text from a text file

    Args:
        file_path: Path to text file

    Returns:
        File contents as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_text(source: Union[str, Path, bytes], file_type: str = None) -> str:
    """
    Main extraction function that handles both PDF and text files

    Args:
        source: File path or bytes content
        file_type: 'pdf' or 'txt', auto-detected from path if None

    Returns:
        Extracted text
    """
    if isinstance(source, bytes):
        # Must be PDF bytes from upload
        if file_type is None or file_type == 'pdf':
            return extract_text_from_pdf(source)
        else:
            # Decode bytes as text
            return source.decode('utf-8')

    source_path = Path(source)

    # Auto-detect file type
    if file_type is None:
        if source_path.suffix.lower() == '.pdf':
            file_type = 'pdf'
        else:
            file_type = 'txt'

    if file_type == 'pdf':
        return extract_text_from_pdf(source_path)
    else:
        return extract_text_from_file(source_path)


def clean_text(text: str) -> str:
    """
    Clean extracted text
    - Remove excessive whitespace
    - Normalize line breaks

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    import re
    text = re.sub(r' +', ' ', text)

    # Replace multiple newlines with max 2
    text = re.sub(r'\n\n+', '\n\n', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text
