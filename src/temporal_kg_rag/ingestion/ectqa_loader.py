"""
ECT-QA Dataset Loader.

Loads documents from the ECT-QA (Earnings Call Transcripts Q&A) dataset format.

Each line in the JSONL file has the structure:
{
    "company_name": "Crocs, Inc.",
    "stock_code": "CROX",
    "sector": "consumer_discretionary",
    "year": "2020",
    "quarter": "q1",
    "URL": "https://...",
    "raw_content": "...",
    "cleaned_content": "...",
    "token_count": 4110
}

The `cleaned_content` field is used as the document text.
All other fields are stored as metadata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from temporal_kg_rag.models.document import Document
from temporal_kg_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ECTQALoader:
    """Load documents from ECT-QA dataset (JSONL format)."""

    def __init__(self):
        """Initialize ECT-QA loader."""
        pass

    def load_jsonl(
        self,
        file_path: str,
        use_cleaned_content: bool = True,
        limit: Optional[int] = None,
        filter_sector: Optional[str] = None,
        filter_year: Optional[str] = None,
        filter_quarter: Optional[str] = None,
        filter_stock_code: Optional[str] = None,
    ) -> Generator[Tuple[Document, str], None, None]:
        """
        Load documents from ECT-QA JSONL file.

        Args:
            file_path: Path to the JSONL file
            use_cleaned_content: Use cleaned_content (True) or raw_content (False)
            limit: Maximum number of documents to load (None for all)
            filter_sector: Only load documents from this sector
            filter_year: Only load documents from this year
            filter_quarter: Only load documents from this quarter
            filter_stock_code: Only load documents for this stock code

        Yields:
            Tuples of (Document, text_content)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"ECT-QA file not found: {file_path}")

        if path.suffix.lower() not in [".jsonl", ".json"]:
            raise ValueError(f"Expected JSONL file, got: {path.suffix}")

        logger.info(f"Loading ECT-QA documents from: {path}")

        count = 0
        skipped = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if limit and count >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    skipped += 1
                    continue

                # Apply filters
                if filter_sector and record.get("sector") != filter_sector:
                    continue
                if filter_year and record.get("year") != filter_year:
                    continue
                if filter_quarter and record.get("quarter") != filter_quarter:
                    continue
                if filter_stock_code:
                    record_code = (record.get("stock_code") or "").upper()
                    if record_code != filter_stock_code.upper():
                        continue

                # Extract document
                try:
                    document, text = self._parse_record(record, use_cleaned_content)
                    count += 1
                    yield document, text
                except Exception as e:
                    logger.warning(f"Failed to parse record at line {line_num}: {e}")
                    skipped += 1
                    continue

        logger.info(
            f"ECT-QA loading complete: {count} documents loaded, {skipped} skipped"
        )

    def load_all(
        self,
        file_path: str,
        use_cleaned_content: bool = True,
        limit: Optional[int] = None,
        **filters,
    ) -> List[Tuple[Document, str]]:
        """
        Load all documents from ECT-QA JSONL file into a list.

        Args:
            file_path: Path to the JSONL file
            use_cleaned_content: Use cleaned_content (True) or raw_content (False)
            limit: Maximum number of documents to load
            **filters: Filter options (filter_sector, filter_year, filter_quarter, filter_stock_code)

        Returns:
            List of (Document, text_content) tuples
        """
        return list(self.load_jsonl(file_path, use_cleaned_content, limit, **filters))

    def _parse_record(
        self,
        record: Dict,
        use_cleaned_content: bool = True,
    ) -> Tuple[Document, str]:
        """
        Parse a single ECT-QA record into a Document.

        Args:
            record: The JSON record from the JSONL file
            use_cleaned_content: Use cleaned_content or raw_content

        Returns:
            Tuple of (Document, text_content)
        """
        # Extract text content
        if use_cleaned_content:
            text = record.get("cleaned_content", "")
        else:
            text = record.get("raw_content", "")

        if not text:
            raise ValueError("No content found in record")

        # Build document title
        company_name = record.get("company_name", "Unknown Company")
        stock_code = record.get("stock_code", "")
        year = record.get("year", "")
        quarter = record.get("quarter", "").upper()

        title = f"{company_name} ({stock_code}) - {quarter} {year} Earnings Call"

        # Build metadata from all other fields
        metadata = {
            "company_name": company_name,
            "stock_code": stock_code,
            "sector": record.get("sector", ""),
            "year": year,
            "quarter": quarter,
            "url": record.get("URL", ""),
            "token_count": record.get("token_count", len(text.split())),
            "dataset": "ECT-QA",
            "content_type_used": "cleaned" if use_cleaned_content else "raw",
        }

        # Parse year/quarter into a timestamp for temporal tracking
        document_date = self._parse_quarter_date(year, quarter)

        # Create Document
        document = Document(
            title=title,
            source=record.get("URL", f"ectqa://{stock_code}/{year}/{quarter}"),
            content_type="earnings_call_transcript",
            file_path=record.get("URL", ""),
            metadata=metadata,
            created_at=document_date,
        )

        return document, text

    @staticmethod
    def _parse_quarter_date(year: str, quarter: str) -> datetime:
        """
        Parse year and quarter into a datetime.

        Args:
            year: Year string (e.g., "2020")
            quarter: Quarter string (e.g., "q1", "Q2")

        Returns:
            datetime representing the end of that quarter
        """
        try:
            year_int = int(year)
        except (ValueError, TypeError):
            return datetime.now()

        quarter_lower = quarter.lower() if quarter else ""

        # Map quarters to end-of-quarter months
        quarter_months = {
            "q1": 3,   # End of Q1: March
            "q2": 6,   # End of Q2: June
            "q3": 9,   # End of Q3: September
            "q4": 12,  # End of Q4: December
        }

        month = quarter_months.get(quarter_lower, 1)

        # Return last day of the quarter month
        if month in [1, 3, 5, 7, 8, 10, 12]:
            day = 31
        elif month in [4, 6, 9, 11]:
            day = 30
        else:  # February
            day = 28

        try:
            return datetime(year_int, month, day)
        except ValueError:
            return datetime.now()

    def get_dataset_stats(self, file_path: str) -> Dict:
        """
        Get statistics about the ECT-QA dataset.

        Args:
            file_path: Path to the JSONL file

        Returns:
            Dictionary with dataset statistics
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"ECT-QA file not found: {file_path}")

        stats = {
            "total_records": 0,
            "sectors": {},
            "years": {},
            "quarters": {},
            "companies": set(),
            "total_tokens": 0,
        }

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    stats["total_records"] += 1

                    # Count by sector
                    sector = record.get("sector", "unknown")
                    stats["sectors"][sector] = stats["sectors"].get(sector, 0) + 1

                    # Count by year
                    year = record.get("year", "unknown")
                    stats["years"][year] = stats["years"].get(year, 0) + 1

                    # Count by quarter
                    quarter = record.get("quarter", "unknown")
                    stats["quarters"][quarter] = stats["quarters"].get(quarter, 0) + 1

                    # Track companies
                    company = record.get("company_name", "")
                    if company:
                        stats["companies"].add(company)

                    # Sum tokens
                    stats["total_tokens"] += record.get("token_count", 0)

                except json.JSONDecodeError:
                    continue

        # Convert set to count
        stats["unique_companies"] = len(stats["companies"])
        del stats["companies"]

        return stats


# Global loader instance
_ectqa_loader: Optional[ECTQALoader] = None


def get_ectqa_loader() -> ECTQALoader:
    """Get the global ECT-QA loader instance."""
    global _ectqa_loader
    if _ectqa_loader is None:
        _ectqa_loader = ECTQALoader()
    return _ectqa_loader


def load_ectqa_documents(
    file_path: str,
    limit: Optional[int] = None,
    **filters,
) -> Generator[Tuple[Document, str], None, None]:
    """
    Convenience function to load ECT-QA documents.

    Args:
        file_path: Path to the JSONL file
        limit: Maximum number of documents
        **filters: Filter options

    Yields:
        Tuples of (Document, text_content)
    """
    loader = get_ectqa_loader()
    yield from loader.load_jsonl(file_path, limit=limit, **filters)
