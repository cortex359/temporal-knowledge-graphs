"""Document loading for various file formats."""

import mimetypes
from pathlib import Path
from typing import Dict, Optional, Tuple

from bs4 import BeautifulSoup
import PyPDF2
import markdown

from temporal_kg_rag.models.document import Document
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Load documents from various file formats."""

    SUPPORTED_FORMATS = {
        ".pdf": "pdf",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".html": "html",
        ".htm": "html",
    }

    def __init__(self):
        """Initialize document loader."""
        pass

    def load(
        self,
        file_path: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Tuple[Document, str]:
        """
        Load a document from a file.

        Args:
            file_path: Path to the file
            title: Optional document title (defaults to filename)
            metadata: Optional metadata dictionary

        Returns:
            Tuple of (Document object, extracted text content)

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        content_type = self.SUPPORTED_FORMATS[suffix]
        logger.info(f"Loading document: {path.name} (type: {content_type})")

        # Extract text based on file type
        if content_type == "pdf":
            text = self._load_pdf(path)
        elif content_type == "markdown":
            text = self._load_markdown(path)
        elif content_type == "html":
            text = self._load_html(path)
        else:  # text
            text = self._load_text(path)

        # Create Document object
        doc_title = title or path.stem
        doc_metadata = metadata or {}
        doc_metadata.update({
            "filename": path.name,
            "file_size": path.stat().st_size,
            "file_extension": suffix,
        })

        document = Document(
            title=doc_title,
            source=str(path.absolute()),
            content_type=content_type,
            file_path=str(path.absolute()),
            metadata=doc_metadata,
        )

        logger.info(
            f"Document loaded: {document.title} "
            f"({len(text)} characters, {len(text.split())} words)"
        )

        return document, text

    def _load_pdf(self, path: Path) -> str:
        """
        Load text from PDF file.

        Args:
            path: Path to PDF file

        Returns:
            Extracted text content
        """
        text_parts = []

        try:
            with open(path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                logger.debug(f"PDF has {num_pages} pages")

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")

            text = "\n\n".join(text_parts)

            if not text.strip():
                logger.warning(f"No text extracted from PDF: {path}")

            return text

        except Exception as e:
            logger.error(f"Error loading PDF {path}: {e}")
            raise

    def _load_markdown(self, path: Path) -> str:
        """
        Load text from Markdown file.

        Args:
            path: Path to Markdown file

        Returns:
            Plain text content (HTML converted to text)
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                md_content = file.read()

            # Convert Markdown to HTML
            html_content = markdown.markdown(md_content)

            # Convert HTML to plain text
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            return text

        except Exception as e:
            logger.error(f"Error loading Markdown {path}: {e}")
            raise

    def _load_html(self, path: Path) -> str:
        """
        Load text from HTML file.

        Args:
            path: Path to HTML file

        Returns:
            Plain text content
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                html_content = file.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            return text

        except Exception as e:
            logger.error(f"Error loading HTML {path}: {e}")
            raise

    def _load_text(self, path: Path) -> str:
        """
        Load text from plain text file.

        Args:
            path: Path to text file

        Returns:
            Text content
        """
        try:
            # Try UTF-8 first
            try:
                with open(path, "r", encoding="utf-8") as file:
                    text = file.read()
            except UnicodeDecodeError:
                # Fall back to latin-1
                logger.warning(f"UTF-8 decode failed for {path}, trying latin-1")
                with open(path, "r", encoding="latin-1") as file:
                    text = file.read()

            return text

        except Exception as e:
            logger.error(f"Error loading text file {path}: {e}")
            raise

    def is_supported(self, file_path: str) -> bool:
        """
        Check if a file format is supported.

        Args:
            file_path: Path to the file

        Returns:
            True if format is supported, False otherwise
        """
        suffix = Path(file_path).suffix.lower()
        return suffix in self.SUPPORTED_FORMATS

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """Get list of supported file formats."""
        return list(cls.SUPPORTED_FORMATS.keys())


def load_document(
    file_path: str,
    title: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[Document, str]:
    """
    Convenience function to load a document.

    Args:
        file_path: Path to the file
        title: Optional document title
        metadata: Optional metadata dictionary

    Returns:
        Tuple of (Document object, extracted text content)
    """
    loader = DocumentLoader()
    return loader.load(file_path, title, metadata)
