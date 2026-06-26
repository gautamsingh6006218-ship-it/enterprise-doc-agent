import textwrap
from pathlib import Path

import pytest

from agent.ingestion.loader_registry import LoaderRegistry
from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document
from agent.services.ingestion_service import IngestionResult, IngestionService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def service() -> IngestionService:
    return IngestionService()


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """Minimal valid PDF with one text page."""
    content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj
4 0 obj<</Length 44>>
stream
BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
trailer<</Size 5/Root 1 0 R>>
startxref
368
%%EOF"""
    p = tmp_path / "sample.pdf"
    p.write_bytes(content)
    return p


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "notes.txt"
    p.write_text("Enterprise document content.\nSecond line.", encoding="utf-8")
    return p


@pytest.fixture()
def sample_md(tmp_path: Path) -> Path:
    p = tmp_path / "readme.md"
    p.write_text("# Title\n\nSome markdown content.", encoding="utf-8")
    return p


@pytest.fixture()
def sample_html(tmp_path: Path) -> Path:
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><title>My Page</title></head>"
        "<body><p>Hello HTML</p><script>alert(1)</script></body></html>",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def sample_docx(tmp_path: Path) -> Path:
    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_paragraph("Enterprise DOCX content.")
    doc.add_paragraph("Second paragraph.")
    path = tmp_path / "report.docx"
    doc.save(str(path))
    return path


# ── LoaderRegistry tests ───────────────────────────────────────────────────────

class TestLoaderRegistry:

    def test_supported_extensions_includes_core_formats(self):
        registry = LoaderRegistry()
        assert ".pdf" in registry.supported_extensions
        assert ".docx" in registry.supported_extensions
        assert ".txt" in registry.supported_extensions
        assert ".md" in registry.supported_extensions
        assert ".html" in registry.supported_extensions
        assert ".htm" in registry.supported_extensions

    def test_get_loader_returns_correct_type_for_pdf(self):
        from agent.ingestion.loaders.pdf import PdfLoader
        registry = LoaderRegistry()
        loader = registry.get_loader("file.pdf")
        assert isinstance(loader, PdfLoader)

    def test_get_loader_raises_for_unsupported_extension(self):
        registry = LoaderRegistry()
        with pytest.raises(ValueError, match="No loader registered for '.xyz'"):
            registry.get_loader("file.xyz")

    def test_register_custom_loader_overrides_extension(self):
        class FakeLoader(BaseDocumentLoader):
            @property
            def supported_extensions(self) -> list[str]:
                return [".pdf"]

            def load(self, file_path: str, tenant_id: str = "default") -> Document:
                raise NotImplementedError

        registry = LoaderRegistry()
        registry.register(FakeLoader())
        assert isinstance(registry.get_loader("doc.pdf"), FakeLoader)


# ── IngestionService tests ─────────────────────────────────────────────────────

class TestIngestionService:

    def test_ingest_txt_returns_success(self, service: IngestionService, sample_txt: Path):
        result = service.ingest(str(sample_txt))
        assert result.success is True
        assert result.document is not None
        assert result.error is None

    def test_ingest_txt_document_fields(self, service: IngestionService, sample_txt: Path):
        doc = service.ingest(str(sample_txt)).document
        assert doc.source_type == "txt"
        assert doc.title == "notes"
        assert "Enterprise document content" in doc.text
        assert len(doc.file_hash) == 32  # MD5 hex
        assert doc.created_at is not None

    def test_ingest_markdown_sets_source_type(self, service: IngestionService, sample_md: Path):
        doc = service.ingest(str(sample_md)).document
        assert doc.source_type == "markdown"

    def test_ingest_html_strips_script_tags(self, service: IngestionService, sample_html: Path):
        doc = service.ingest(str(sample_html)).document
        assert "alert" not in doc.text
        assert "Hello HTML" in doc.text

    def test_ingest_html_uses_title_tag(self, service: IngestionService, sample_html: Path):
        doc = service.ingest(str(sample_html)).document
        assert doc.title == "My Page"

    def test_ingest_docx_extracts_paragraphs(self, service: IngestionService, sample_docx: Path):
        doc = service.ingest(str(sample_docx)).document
        assert doc.source_type == "docx"
        assert "Enterprise DOCX content" in doc.text

    def test_ingest_sets_tenant_id(self, service: IngestionService, sample_txt: Path):
        doc = service.ingest(str(sample_txt), tenant_id="acme-corp").document
        assert doc.tenant_id == "acme-corp"

    def test_ingest_default_tenant_id(self, service: IngestionService, sample_txt: Path):
        doc = service.ingest(str(sample_txt)).document
        assert doc.tenant_id == "default"

    def test_ingest_file_not_found_returns_failure(self, service: IngestionService):
        result = service.ingest("/non/existent/file.txt")
        assert result.success is False
        assert result.document is None
        assert "not found" in result.error.lower()

    def test_ingest_unsupported_format_returns_failure(self, service: IngestionService, tmp_path: Path):
        f = tmp_path / "data.xyz"
        f.write_text("data")
        result = service.ingest(str(f))
        assert result.success is False
        assert ".xyz" in result.error

    def test_supported_formats_property(self, service: IngestionService):
        formats = service.supported_formats
        assert isinstance(formats, list)
        assert len(formats) > 0

    def test_ingest_generates_unique_ids(self, service: IngestionService, sample_txt: Path):
        doc1 = service.ingest(str(sample_txt)).document
        doc2 = service.ingest(str(sample_txt)).document
        assert doc1.id != doc2.id

    def test_ingest_same_file_same_hash(self, service: IngestionService, sample_txt: Path):
        doc1 = service.ingest(str(sample_txt)).document
        doc2 = service.ingest(str(sample_txt)).document
        assert doc1.file_hash == doc2.file_hash
