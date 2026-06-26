"""
tests/ingestion/test_ingestion_service.py

What does this test file cover?
- LoaderRegistry: extension routing, custom loader registration, error on unknown format.
- IngestionService: all supported formats, RBAC field attachment, error handling.
- Models: field defaults, deduplication via file_hash, tenant isolation.

Why are tests structured as classes?
- Groups related assertions. Running a single class (e.g. TestRBAC) is faster
  during development without running the full suite.
"""

from pathlib import Path

import pytest

from agent.ingestion.loader_registry import LoaderRegistry
from agent.ingestion.loaders.base import BaseDocumentLoader
from agent.ingestion.models import Document
from agent.services.ingestion_service import IngestionResult, IngestionService


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def service() -> IngestionService:
    """Default service with all built-in loaders."""
    return IngestionService()


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """Minimal valid single-page PDF with extractable text."""
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


# ── LoaderRegistry ─────────────────────────────────────────────────────────────

class TestLoaderRegistry:
    """Verifies extension-to-loader routing and extensibility."""

    def test_supported_extensions_includes_core_formats(self):
        registry = LoaderRegistry()
        for ext in [".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm"]:
            assert ext in registry.supported_extensions

    def test_get_loader_returns_correct_type_for_pdf(self):
        from agent.ingestion.loaders.pdf import PdfLoader
        assert isinstance(LoaderRegistry().get_loader("file.pdf"), PdfLoader)

    def test_get_loader_raises_for_unsupported_extension(self):
        with pytest.raises(ValueError, match="No loader registered for '.xyz'"):
            LoaderRegistry().get_loader("file.xyz")

    def test_register_custom_loader_overrides_existing(self):
        """Confirms open/closed principle — new loader replaces old without code changes."""
        class FakeLoader(BaseDocumentLoader):
            @property
            def supported_extensions(self) -> list[str]:
                return [".pdf"]

            def load(self, file_path: str, tenant_id: str = "default") -> Document:
                raise NotImplementedError

        registry = LoaderRegistry()
        registry.register(FakeLoader())
        assert isinstance(registry.get_loader("doc.pdf"), FakeLoader)


# ── IngestionService — format loading ─────────────────────────────────────────

class TestIngestionServiceFormats:
    """Verifies each file format loads correctly and sets expected fields."""

    def test_ingest_txt_success(self, service: IngestionService, sample_txt: Path):
        result = service.ingest(str(sample_txt))
        assert result.success is True
        assert result.document is not None
        assert result.error is None

    def test_ingest_txt_fields(self, service: IngestionService, sample_txt: Path):
        doc = service.ingest(str(sample_txt)).document
        assert doc.source_type == "txt"
        assert doc.title == "notes"
        assert "Enterprise document content" in doc.text
        assert len(doc.file_hash) == 32      # MD5 is always 32 hex chars
        assert doc.created_at is not None

    def test_ingest_markdown_source_type(self, service: IngestionService, sample_md: Path):
        doc = service.ingest(str(sample_md)).document
        assert doc.source_type == "markdown"

    def test_ingest_html_strips_scripts(self, service: IngestionService, sample_html: Path):
        doc = service.ingest(str(sample_html)).document
        assert "alert" not in doc.text       # <script> content must be stripped
        assert "Hello HTML" in doc.text

    def test_ingest_html_title_from_tag(self, service: IngestionService, sample_html: Path):
        doc = service.ingest(str(sample_html)).document
        assert doc.title == "My Page"        # reads <title> not filename

    def test_ingest_docx_extracts_paragraphs(self, service: IngestionService, sample_docx: Path):
        doc = service.ingest(str(sample_docx)).document
        assert doc.source_type == "docx"
        assert "Enterprise DOCX content" in doc.text

    def test_supported_formats_lists_all_loaders(self, service: IngestionService):
        formats = service.supported_formats
        assert isinstance(formats, list)
        assert len(formats) >= 7            # pdf, docx, txt, md, markdown, html, htm


# ── IngestionService — RBAC ────────────────────────────────────────────────────

class TestIngestionServiceRBAC:
    """
    Verifies RBAC fields are attached by the service (not the loader).
    Loaders are format-extraction only — RBAC is a service-layer concern.
    """

    def test_default_rbac_fields(self, service: IngestionService, sample_txt: Path):
        doc = service.ingest(str(sample_txt)).document
        assert doc.owner_id == "system"
        assert doc.access_roles == []
        assert doc.visibility == "public"

    def test_custom_owner_id(self, service: IngestionService, sample_txt: Path):
        doc = service.ingest(str(sample_txt), owner_id="user-42").document
        assert doc.owner_id == "user-42"

    def test_access_roles_attached(self, service: IngestionService, sample_txt: Path):
        doc = service.ingest(
            str(sample_txt), access_roles=["hr", "legal"]
        ).document
        assert "hr" in doc.access_roles
        assert "legal" in doc.access_roles

    def test_visibility_restricted(self, service: IngestionService, sample_txt: Path):
        doc = service.ingest(str(sample_txt), visibility="restricted").document
        assert doc.visibility == "restricted"

    def test_visibility_private(self, service: IngestionService, sample_txt: Path):
        doc = service.ingest(str(sample_txt), visibility="private").document
        assert doc.visibility == "private"

    def test_tenant_id_isolation(self, service: IngestionService, sample_txt: Path):
        doc_a = service.ingest(str(sample_txt), tenant_id="tenant-a").document
        doc_b = service.ingest(str(sample_txt), tenant_id="tenant-b").document
        assert doc_a.tenant_id != doc_b.tenant_id


# ── IngestionService — error handling ─────────────────────────────────────────

class TestIngestionServiceErrors:
    """Verifies the service never raises — always returns IngestionResult."""

    def test_file_not_found_returns_failure(self, service: IngestionService):
        result = service.ingest("/non/existent/file.txt")
        assert result.success is False
        assert result.document is None
        assert "not found" in result.error.lower()

    def test_unsupported_format_returns_failure(self, service: IngestionService, tmp_path: Path):
        f = tmp_path / "data.xyz"
        f.write_text("data")
        result = service.ingest(str(f))
        assert result.success is False
        assert ".xyz" in result.error


# ── IngestionService — deduplication ──────────────────────────────────────────

class TestIngestionServiceDeduplication:
    """
    Verifies file_hash behaviour.
    The hash is used by the vector store to skip re-ingestion of unchanged files.
    """

    def test_same_file_produces_same_hash(self, service: IngestionService, sample_txt: Path):
        doc1 = service.ingest(str(sample_txt)).document
        doc2 = service.ingest(str(sample_txt)).document
        assert doc1.file_hash == doc2.file_hash

    def test_each_ingest_produces_unique_id(self, service: IngestionService, sample_txt: Path):
        # id is a new UUID every load — file_hash is the dedup key, not id.
        doc1 = service.ingest(str(sample_txt)).document
        doc2 = service.ingest(str(sample_txt)).document
        assert doc1.id != doc2.id

    def test_different_files_produce_different_hashes(
        self, service: IngestionService, tmp_path: Path
    ):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A", encoding="utf-8")
        f2.write_text("content B", encoding="utf-8")
        hash1 = service.ingest(str(f1)).document.file_hash
        hash2 = service.ingest(str(f2)).document.file_hash
        assert hash1 != hash2
