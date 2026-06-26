"""
processing/metadata/extractor.py

What problem does this solve?
- Documents contain structured information (dates, emails, categories) buried
  in unstructured text. Vector stores can filter on structured fields but
  cannot filter on text buried in a string. Extraction makes that data queryable.

Why regex + rule-based instead of an NLP model (spaCy NER, etc.)?
- Regex is deterministic, zero latency, and zero GPU cost.
- Dates, emails, URLs, and phones follow predictable patterns that regex
  handles with >99% recall.
- Document category is reliably predicted by keyword frequency — no model needed.
- NLP-based entity extraction (people, orgs) is Phase 2, after the pipeline
  is stable. Adding it here would add ~500MB model weight and seconds of latency.

Why langdetect for language detection?
- Already installed (pulled in by unstructured). Lightweight. Supports 55 languages.
- Deterministic with seed=0. Sufficient for routing to the right embedding model.
- Alternative (fasttext langdetect) is more accurate but adds a 900MB model binary.
"""

import re
from typing import TYPE_CHECKING

from agent.processing.models import ExtractedMetadata

if TYPE_CHECKING:
    from agent.ingestion.models import Document

# ── Compiled regex patterns ────────────────────────────────────────────────────

# Dates: ISO (2024-01-15), US (01/15/2024), Written (January 15, 2024)
_DATE_PATTERNS = [
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
    re.compile(
        r'\b(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\.?\s+\d{1,2},?\s+\d{4}\b',
        re.IGNORECASE,
    ),
]

# Emails: standard RFC 5322 simplified pattern
_EMAIL_RE = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
)

# Phone numbers: US (+1-xxx-xxx-xxxx) and international (+xx xxxx xxxx)
_PHONE_PATTERNS = [
    re.compile(r'\+?1?\s*[-.]?\s*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    re.compile(r'\+\d{1,3}[\s.-]\d{4,14}\b'),
]

# URLs: http and https only (no bare domains to avoid false positives)
_URL_RE = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

# Section headers: numbered (1.2 Title), markdown (## Title), ALL CAPS lines
_HEADER_PATTERNS = [
    re.compile(r'(?m)^#{1,6}\s+(.+)$'),                 # ## Markdown header
    re.compile(r'(?m)^\d+(?:\.\d+)*\s{1,4}([A-Z].+)$'), # 1.2 Section Title
    re.compile(r'(?m)^([A-Z][A-Z\s]{4,})$'),             # ALL CAPS HEADER
]

# Document category keyword mapping
# Scored by keyword frequency — highest score wins.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "invoice":       ["invoice", "bill", "payment due", "total amount", "tax invoice", "purchase order"],
    "contract":      ["contract", "agreement", "parties", "hereinafter", "obligations", "whereas", "indemnify"],
    "policy":        ["policy", "procedure", "guideline", "shall", "compliance", "must not", "is required to"],
    "report":        ["report", "analysis", "findings", "conclusion", "executive summary", "recommendations"],
    "email":         ["from:", "to:", "subject:", "sent:", "dear", "regards", "sincerely"],
    "presentation":  ["slide", "agenda", "presentation", "overview", "key takeaways"],
    "specification": ["specification", "requirement", "functional", "technical spec", "system design"],
    "resume":        ["experience", "education", "skills", "objective", "work history", "references"],
    "manual":        ["chapter", "installation", "configuration", "troubleshoot", "step by step", "how to"],
}


class MetadataExtractor:
    """
    What problem does this solve?
    - Structured data buried in document text (dates, emails, category) is
      extracted into typed fields so the vector store can filter on them
      at query time without scanning document text.

    Why does this class exist?
    - Centralises all extraction logic. Adding a new field (e.g. currency amounts)
      only touches this file and the ExtractedMetadata model.
    - Stateless — safe to share one instance across all documents in a pipeline.
    """

    def extract(self, text: str, document: "Document") -> ExtractedMetadata:
        """
        What problem does this solve?
        - Converts unstructured text into a typed ExtractedMetadata object
          with queryable structured fields.

        Why are both text and document required?
        - text:     The cleaned/normalised text — this is what gets analysed.
                    Using document.text (raw) would extract from noisy text,
                    reducing regex accuracy.
        - document: Provides source_type, title, and existing metadata as
                    hints for category detection and language detection.

        Why use only the first 2000 chars for language detection?
        - langdetect is accurate on 200+ chars. Using the full document
          is wasteful — the language is set by the first few paragraphs.
        - Caps CPU time at a fixed cost regardless of document length.

        Why deduplicate extracted values with set()?
        - Dates and emails repeat across pages in headers/footers.
          Returning 50 copies of the same date is noise, not signal.

        Returns ExtractedMetadata with all fields populated.
        """
        dates = self._extract_dates(text)
        emails = self._extract_emails(text)
        phones = self._extract_phones(text)
        urls = self._extract_urls(text)
        headers = self._extract_section_headers(text)
        language = self._detect_language(text)
        category = self._detect_category(text, document.source_type, document.title)

        word_count = len(text.split())
        char_count = len(text)
        # Average adult reading speed: 200 words per minute
        reading_time = round(word_count / 200, 1)

        return ExtractedMetadata(
            language=language,
            category=category,
            dates=dates,
            emails=emails,
            phone_numbers=phones,
            urls=urls,
            section_headers=headers[:20],   # cap at 20 to avoid huge metadata payloads
            word_count=word_count,
            char_count=char_count,
            reading_time_minutes=reading_time,
        )

    # ── Private extraction methods ─────────────────────────────────────────────

    def _extract_dates(self, text: str) -> list[str]:
        """Extracts all date patterns, deduplicates, and returns sorted list."""
        found: set[str] = set()
        for pattern in _DATE_PATTERNS:
            found.update(pattern.findall(text))
        return sorted(found)

    def _extract_emails(self, text: str) -> list[str]:
        """Extracts email addresses, lowercases for deduplication."""
        found = {m.lower() for m in _EMAIL_RE.findall(text)}
        return sorted(found)

    def _extract_phones(self, text: str) -> list[str]:
        """Extracts phone numbers across US and international formats."""
        found: set[str] = set()
        for pattern in _PHONE_PATTERNS:
            found.update(pattern.findall(text))
        # Strip surrounding whitespace from each match
        return sorted(p.strip() for p in found)

    def _extract_urls(self, text: str) -> list[str]:
        """Extracts http/https URLs, deduplicates."""
        found = set(_URL_RE.findall(text))
        return sorted(found)

    def _extract_section_headers(self, text: str) -> list[str]:
        """Extracts section headers using structural patterns."""
        found: list[str] = []
        for pattern in _HEADER_PATTERNS:
            for match in pattern.finditer(text):
                header = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if header and len(header) > 3:  # skip very short false positives
                    found.append(header)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = []
        for h in found:
            if h not in seen:
                seen.add(h)
                unique.append(h)
        return unique

    def _detect_language(self, text: str) -> str:
        """
        Detects document language using langdetect.

        Why seed=0?
        - langdetect uses random sampling internally. seed=0 makes it
          deterministic across runs — same text always returns same language.

        Why first 2000 chars?
        - Language is stable across a document. Analysing the full text
          wastes CPU. First 2000 chars is sufficient for accurate detection.
        """
        try:
            from langdetect import detect, DetectorFactory
            from langdetect.lang_detect_exception import LangDetectException
            DetectorFactory.seed = 0  # deterministic
            return detect(text[:2000])
        except Exception:
            # langdetect fails on very short text or non-linguistic content
            return "unknown"

    def _detect_category(self, text: str, source_type: str, title: str) -> str:
        """
        Scores the document against keyword sets and returns the best match.

        Why keyword scoring instead of a classifier model?
        - Zero latency, zero model size, fully auditable.
        - Enterprise document categories are well-defined and keyword-rich.
        - A classifier would need labelled training data and adds complexity
          without meaningfully better accuracy for these category definitions.

        Why combine text + title + source_type in scoring?
        - Title is often the most discriminative signal ("Q3 Invoice.pdf").
        - source_type provides a strong prior: .eml → email, .pptx → presentation.
        """
        # source_type gives a strong prior — use it directly if unambiguous
        _SOURCE_TYPE_MAP = {
            "eml": "email",
            "msg": "email",
            "pptx": "presentation",
        }
        if source_type in _SOURCE_TYPE_MAP:
            return _SOURCE_TYPE_MAP[source_type]

        # Score by keyword frequency across lowercased text + title
        combined = (text + " " + title).lower()
        scores: dict[str, int] = {}

        for category, keywords in _CATEGORY_KEYWORDS.items():
            score = sum(combined.count(kw.lower()) for kw in keywords)
            if score > 0:
                scores[category] = score

        if not scores:
            return "unknown"

        # Return the category with the highest keyword frequency score
        return max(scores, key=lambda k: scores[k])
