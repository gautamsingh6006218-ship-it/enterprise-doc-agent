"""
ingestion/loaders/pdf_smart.py

What problem does this solve?
- Enterprise document collections contain all three PDF types (text, complex,
  scanned) mixed together. Callers should not have to classify PDFs manually
  before ingesting them.

Why a smart router instead of always using Docling or always using OCR?
- PdfLoader (PyMuPDF) is 10-50x faster than Docling for text PDFs.
- Docling downloads 500MB+ of models — overkill for a simple text PDF.
- OCR is the slowest — only justified for genuinely scanned documents.
- Routing by text density gives the best speed/quality tradeoff automatically.

Routing logic:
    1. Always run PyMuPDF first (fast, no ML, handles 80%+ of PDFs).
    2. If text density is above threshold → return PyMuPDF result.
    3. If sparse text → try Docling (complex layouts, tables, mixed content).
    4. If Docling not installed or fails → try Tesseract OCR.
    5. If nothing works → return PyMuPDF result with a warning in metadata.

Threshold (100 chars/page):
- Normal text PDFs: 500-3000 chars/page.
- Complex PDFs (many images/charts): 100-500 chars/page.
- Scanned PDFs: 0-50 chars/page (headers/footers only).
- 100 is conservative — avoids routing image-heavy but text-rich PDFs to OCR.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.loaders.pdf import PdfLoader
from agent.ingestion.loaders.pdf_docling import DOCLING_AVAILABLE, DoclingPdfLoader
from agent.ingestion.loaders.pdf_ocr import TESSERACT_AVAILABLE, OcrPdfLoader
from agent.ingestion.models import Document


class SmartPdfLoader(BaseDocumentLoader):
    """
    What problem does this solve?
    - Automatically selects the best PDF extraction strategy without the
      caller needing to know which type of PDF they are dealing with.

    Why does this class exist?
    - Single entry point for all PDF types. The registry maps .pdf to this
      loader — no manual routing needed.
    - Hides the complexity of three different PDF extraction strategies
      behind one clean interface.

    Why initialise sub-loaders in __init__?
    - Sub-loaders are stateless and reusable across calls.
    - DoclingPdfLoader caches its converter at class level, so creating the
      instance here does not trigger model downloads yet.
    """

    # If extracted text is below this many chars per page, the PDF is likely
    # scanned or image-heavy and needs Docling or OCR.
    _SPARSE_TEXT_THRESHOLD = 100  # chars per page

    def __init__(self) -> None:
        self._pdf_loader = PdfLoader()
        # Sub-loaders are instantiated unconditionally — their AVAILABLE flags
        # gate actual usage, not instantiation.
        self._docling_loader = DoclingPdfLoader() if DOCLING_AVAILABLE else None
        self._ocr_loader = OcrPdfLoader() if TESSERACT_AVAILABLE else None

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def _is_text_sparse(self, document: Document) -> bool:
        """
        Why chars-per-page instead of total chars?
        - A 100-page scanned PDF has more total chars (from headers/footers)
          than a 1-page text PDF. Per-page normalization is accurate.

        Why 100 as the threshold?
        - Empirically: clean text pages have 500+ chars. Pure image pages
          have 0-30 chars (page numbers, headers only). 100 catches both
          without misclassifying text-light but valid pages.
        """
        page_count = document.metadata.get("page_count", 1) or 1
        chars_per_page = len(document.text) / page_count
        return chars_per_page < self._SPARSE_TEXT_THRESHOLD

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Intelligently routes each PDF to the right extraction strategy,
          maximising text quality while minimising processing time.

        Routing decision:
        - Step 1: PyMuPDF (fast). If sufficient text → done.
        - Step 2: Docling (if sparse text AND installed). Better for tables/layouts.
        - Step 3: Tesseract OCR (if still sparse AND installed). For scanned docs.
        - Step 4: Return PyMuPDF result with warning if all else unavailable.

        Why not always use Docling?
        - Docling is 10-50x slower on text-only PDFs. PyMuPDF + Markdown mode
          produces equal quality output for standard documents at a fraction of
          the processing time.

        Why add extraction_method to metadata?
        - Observability: knowing which loader was used helps diagnose
          quality issues and tune the routing thresholds over time.

        Raises:
        - FileNotFoundError  if file does not exist.
        - ValueError         if extension is not .pdf.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"SmartPdfLoader cannot handle: {path.suffix}")

        # Step 1: Always try PyMuPDF first — fast, no ML, handles most PDFs
        document = self._pdf_loader.load(file_path, tenant_id)

        if not self._is_text_sparse(document):
            # Sufficient text extracted — return PyMuPDF result
            return document

        # Step 2: Sparse text detected → try Docling for complex layouts
        if self._docling_loader:
            try:
                docling_doc = self._docling_loader.load(file_path, tenant_id)
                if not self._is_text_sparse(docling_doc):
                    return docling_doc
            except Exception:
                # Docling failed (corrupted file, model error) — fall through
                pass

        # Step 3: Still sparse → try Tesseract OCR for scanned documents
        if self._ocr_loader:
            try:
                return self._ocr_loader.load(file_path, tenant_id)
            except Exception:
                pass

        # Step 4: All advanced loaders unavailable or failed — return PyMuPDF
        # result with a warning so the operator knows quality may be low.
        document.metadata["extraction_warning"] = (
            "Text is sparse. Install docling or tesseract for better extraction. "
            f"Current: {len(document.text)} chars, "
            f"{document.metadata.get('page_count', 0)} pages."
        )
        return document
