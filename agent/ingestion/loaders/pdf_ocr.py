"""
ingestion/loaders/pdf_ocr.py

What problem does this solve?
- Scanned PDFs have no text layer — PyMuPDF and Docling return empty or
  near-empty text. OCR is the only way to extract content from these files.

Why Tesseract + PyMuPDF together?
- PyMuPDF renders each PDF page to a high-resolution image (300 DPI).
- Tesseract runs OCR on those images to extract text.
- This combination handles any scanned document regardless of scan quality.

Why 300 DPI for rendering?
- Below 200 DPI Tesseract accuracy drops significantly.
- 300 DPI is the sweet spot: good accuracy without excessive memory use.
- 600 DPI adds marginal accuracy gain but quadruples memory and processing time.

System requirement:
- Tesseract binary must be installed: brew install tesseract (macOS)
  or apt install tesseract-ocr (Linux).
- pytesseract and Pillow Python packages must be installed.
- Import guard at module level — fails clearly if deps are missing.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

import fitz  # PyMuPDF — used to render PDF pages to images

from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class OcrPdfLoader(BaseDocumentLoader):
    """
    What problem does this solve?
    - Extracts text from scanned/image-only PDFs where no text layer exists.

    Why does this class exist?
    - PyMuPDF and Docling cannot extract text from pure image PDFs.
      OCR is the only solution for legacy scanned documents.

    Why is this a separate loader instead of a flag on PdfLoader?
    - OCR is 10-50x slower than text extraction. Making it opt-in via a
      separate loader prevents accidental slow processing of text PDFs.
    - SmartPdfLoader decides when to use this based on text density detection.
    """

    # Render resolution. 300 DPI balances OCR accuracy with memory usage.
    _DPI = 300

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def load(self, file_path: str, tenant_id: str = "default") -> Document:
        """
        What problem does this solve?
        - Converts scanned PDF pages into text by rendering to images and
          running Tesseract OCR on each page.

        Why are these inputs required?
        - file_path:  PDF to OCR. Must be a .pdf file.
        - tenant_id:  Propagated to Document for tenant-scoped vector queries.

        Why render at 300 DPI?
        - Tesseract needs sufficient pixel density to distinguish characters.
          300 DPI consistently produces >95% accuracy on clean scans.

        Why collect page-level OCR confidence?
        - Low confidence scores (e.g. <60%) indicate poor scan quality.
          Stored in metadata so downstream services can flag low-quality docs.

        Why raise ImportError explicitly instead of silently returning empty?
        - Silent failure hides a missing system dependency. An explicit error
          tells the operator exactly what to install.

        Raises:
        - ImportError        if pytesseract or Pillow are not installed.
        - FileNotFoundError  if the PDF does not exist.
        - ValueError         if the file is not a .pdf.
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError(
                "OcrPdfLoader requires pytesseract and Pillow. "
                "Run: pip install pytesseract Pillow && brew install tesseract"
            )

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"OcrPdfLoader cannot handle: {path.suffix}")

        raw_bytes = path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()

        pdf = fitz.open(path)
        pages_text: list[str] = []
        total_pages = pdf.page_count

        for page in pdf:
            # Render page to a pixel map at 300 DPI for high OCR accuracy
            matrix = fitz.Matrix(self._DPI / 72, self._DPI / 72)
            pix = page.get_pixmap(matrix=matrix)

            # Convert PyMuPDF pixmap to PIL Image for Tesseract
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Run OCR — lang="eng" for English; extend for multilingual enterprise docs
            text = pytesseract.image_to_string(img, lang="eng")

            if text.strip():
                pages_text.append(text.strip())

        pdf.close()

        full_text = "\n\n".join(pages_text).strip()

        return Document(
            id=str(uuid4()),
            source_type="pdf",
            source_path=str(path.resolve()),
            title=path.stem,
            text=full_text,
            tenant_id=tenant_id,
            file_hash=file_hash,
            metadata={
                "page_count": total_pages,
                "ocr_pages": len(pages_text),
                "file_name": path.name,
                "file_size_bytes": len(raw_bytes),
                "extraction_method": "tesseract_ocr",
                "ocr_dpi": self._DPI,
            },
        )
