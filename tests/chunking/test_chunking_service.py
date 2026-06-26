"""
tests/chunking/test_chunking_service.py

What does this cover?
- TokenCounter:         count, count_batch, exceeds_limit, encoding_name.
- SentenceWindowStrategy: prose chunking, empty text, single sentence,
                           chunk overlap, multi-paragraph.
- TokenStrategy:        tabular text chunking, empty input.
- StructureAwareStrategy: markdown header chunking, breadcrumb prepended,
                          HTML fallback, empty input.
- ChunkingRouter:       correct strategy routed per source_type, fallback.
- ChunkingPipeline:     RBAC propagation, sentence window prev/next links,
                        stats keys, empty text handling, chunk IDs.
- ChunkingService:      success path, error wrapping via DI, stats returned,
                        chunks have required metadata keys.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent.chunking.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_TARGET_TOKENS,
    TIKTOKEN_ENCODING,
)
from agent.chunking.pipeline import ChunkingPipeline
from agent.chunking.router import ChunkingRouter
from agent.chunking.strategies.base import BaseChunkingStrategy, ChunkResult
from agent.chunking.strategies.sentence_window import SentenceWindowStrategy
from agent.chunking.strategies.structure_aware import StructureAwareStrategy
from agent.chunking.strategies.token_chunker import TokenStrategy
from agent.chunking.token_counter import TokenCounter
from agent.ingestion.models import Document, DocumentChunk
from agent.processing.metadata.extractor import MetadataExtractor
from agent.processing.models import ExtractedMetadata, ProcessedDocument
from agent.services.chunking_service import ChunkingResult, ChunkingService


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_document(
    text: str = "Sample document text.",
    source_type: str = "txt",
    tenant_id: str = "acme",
    owner_id: str = "user-1",
    access_roles: list | None = None,
    visibility: str = "public",
) -> Document:
    return Document(
        id="doc-001",
        source_type=source_type,
        source_path="/test/path.txt",
        title="Test Document",
        text=text,
        tenant_id=tenant_id,
        owner_id=owner_id,
        access_roles=access_roles or ["hr"],
        visibility=visibility,
        file_hash="abc123",
        created_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def make_processed(
    text: str = "Sample document text.",
    source_type: str = "txt",
    cleaned_text: str | None = None,
    **doc_kwargs,
) -> ProcessedDocument:
    doc = make_document(text=text, source_type=source_type, **doc_kwargs)
    meta = ExtractedMetadata(
        language="en",
        category="report",
        word_count=len(text.split()),
        char_count=len(text),
    )
    return ProcessedDocument(
        document=doc,
        cleaned_text=cleaned_text if cleaned_text is not None else text,
        extracted_metadata=meta,
        processing_stats={"original": len(text), "final": len(cleaned_text or text)},
    )


PROSE = (
    "Artificial intelligence is transforming enterprise operations. "
    "Companies are leveraging machine learning models to automate repetitive tasks. "
    "Natural language processing enables intelligent document search. "
    "Vector databases store high-dimensional embeddings for fast similarity retrieval. "
    "Production systems require careful chunking strategies to preserve semantic context. "
    "Sentence-level chunking respects natural language boundaries better than character splitting. "
    "Token overlap ensures that context at chunk boundaries is not silently lost. "
    "Enterprise deployments must handle multiple document formats uniformly. "
    "PDF, DOCX, XLSX, markdown, and HTML all require different extraction strategies. "
    "A unified ingestion pipeline simplifies deployment and monitoring."
)


# ── TokenCounter ───────────────────────────────────────────────────────────────

class TestTokenCounter:

    def setup_method(self):
        self.counter = TokenCounter()

    def test_count_returns_integer(self):
        result = self.counter.count("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_count_empty_string(self):
        assert self.counter.count("") == 0

    def test_count_increases_with_more_text(self):
        short = self.counter.count("hello")
        long = self.counter.count("hello world this is a longer sentence")
        assert long > short

    def test_count_batch_returns_correct_length(self):
        texts = ["first chunk", "second chunk", "third chunk"]
        counts = self.counter.count_batch(texts)
        assert len(counts) == 3

    def test_count_batch_matches_individual_counts(self):
        texts = ["the quick brown fox", "jumps over the lazy dog"]
        batch = self.counter.count_batch(texts)
        individual = [self.counter.count(t) for t in texts]
        assert batch == individual

    def test_count_batch_empty_list(self):
        assert self.counter.count_batch([]) == []

    def test_exceeds_limit_true(self):
        text = "word " * 100
        assert self.counter.exceeds_limit(text, limit=10) is True

    def test_exceeds_limit_false(self):
        text = "short"
        assert self.counter.exceeds_limit(text, limit=100) is False

    def test_exceeds_limit_exactly_at_boundary(self):
        # Exactly at limit = NOT exceeded (limit is exclusive upper bound)
        text = "hello world"
        count = self.counter.count(text)
        # count tokens == limit → should NOT exceed
        assert self.counter.exceeds_limit(text, limit=count) is False
        # count tokens < limit → should NOT exceed
        assert self.counter.exceeds_limit(text, limit=count + 1) is False
        # count tokens > limit → should exceed
        assert self.counter.exceeds_limit(text, limit=count - 1) is True

    def test_encoding_name_property(self):
        assert self.counter.encoding_name == TIKTOKEN_ENCODING

    def test_disallowed_special_tokens_handled(self):
        # Should not raise on text containing angle-bracket patterns
        text = "Document contains <|endoftext|> somewhere."
        result = self.counter.count(text)
        assert isinstance(result, int)


# ── SentenceWindowStrategy ─────────────────────────────────────────────────────

class TestSentenceWindowStrategy:

    def setup_method(self):
        # Use small chunk_size to produce multiple chunks from short test strings
        self.strategy = SentenceWindowStrategy(chunk_size=50, chunk_overlap=5)

    def test_strategy_name(self):
        assert self.strategy.strategy_name == "SentenceWindowStrategy"

    def test_returns_chunk_result(self):
        result = self.strategy.chunk(PROSE)
        assert isinstance(result.texts, list)
        assert isinstance(result.token_counts, list)
        assert result.strategy_name == "SentenceWindowStrategy"

    def test_empty_text_returns_empty_result(self):
        result = self.strategy.chunk("")
        assert result.texts == []
        assert result.token_counts == []

    def test_whitespace_only_returns_empty_result(self):
        result = self.strategy.chunk("   \n\n  ")
        assert result.texts == []

    def test_texts_and_token_counts_same_length(self):
        result = self.strategy.chunk(PROSE)
        assert len(result.texts) == len(result.token_counts)

    def test_produces_multiple_chunks_from_long_text(self):
        result = self.strategy.chunk(PROSE)
        assert len(result.texts) > 1

    def test_all_chunks_are_non_empty(self):
        result = self.strategy.chunk(PROSE)
        for text in result.texts:
            assert text.strip() != ""

    def test_token_counts_are_positive(self):
        result = self.strategy.chunk(PROSE)
        for count in result.token_counts:
            assert count > 0

    def test_chunks_together_cover_original_content(self):
        """All content should appear across chunks (no silent data loss)."""
        sentences = [
            "First important sentence about contracts.",
            "Second important sentence about invoices.",
            "Third important sentence about policies.",
        ]
        text = " ".join(sentences)
        result = self.strategy.chunk(text)
        combined = " ".join(result.texts)
        for sentence in sentences:
            # Key words from each sentence should appear somewhere
            first_word = sentence.split()[0]
            assert first_word in combined

    def test_single_short_sentence_produces_one_chunk(self):
        result = self.strategy.chunk("Hello.")
        assert len(result.texts) == 1
        assert "Hello" in result.texts[0]


# ── TokenStrategy ──────────────────────────────────────────────────────────────

class TestTokenStrategy:

    def setup_method(self):
        self.strategy = TokenStrategy(chunk_size=20, chunk_overlap=2)

    def test_strategy_name(self):
        assert self.strategy.strategy_name == "TokenStrategy"

    def test_empty_text_returns_empty_result(self):
        result = self.strategy.chunk("")
        assert result.texts == []
        assert result.token_counts == []

    def test_returns_chunk_result(self):
        tabular = "\n".join([f"SKU{i},19.99,{i*10}" for i in range(30)])
        result = self.strategy.chunk(tabular)
        assert isinstance(result.texts, list)
        assert len(result.texts) > 0

    def test_texts_and_token_counts_same_length(self):
        tabular = "\n".join([f"row{i},val{i}" for i in range(30)])
        result = self.strategy.chunk(tabular)
        assert len(result.texts) == len(result.token_counts)

    def test_produces_multiple_chunks_from_long_table(self):
        tabular = "\n".join([f"row{i},value{i},extra{i}" for i in range(50)])
        result = self.strategy.chunk(tabular)
        assert len(result.texts) > 1


# ── StructureAwareStrategy ─────────────────────────────────────────────────────

class TestStructureAwareStrategy:

    def setup_method(self):
        self.strategy = StructureAwareStrategy(chunk_size=100, chunk_overlap=10)

    def test_strategy_name(self):
        assert self.strategy.strategy_name == "StructureAwareStrategy"

    def test_empty_text_returns_empty_result(self):
        result = self.strategy.chunk("", source_type="markdown")
        assert result.texts == []
        assert result.token_counts == []

    def test_markdown_splits_on_headers(self):
        md = (
            "# Introduction\n\n"
            "This section introduces the system.\n\n"
            "## Background\n\n"
            "Historical context about this project.\n\n"
            "## Objectives\n\n"
            "The goals of this document are listed here."
        )
        result = self.strategy.chunk(md, source_type="markdown")
        assert len(result.texts) > 0

    def test_markdown_breadcrumb_prepended(self):
        md = (
            "# Installation\n\n"
            "Follow these steps to install.\n\n"
            "## Troubleshooting\n\n"
            "Common problems and solutions are described here."
        )
        result = self.strategy.chunk(md, source_type="markdown")
        # At least one chunk should contain the breadcrumb
        combined = " ".join(result.texts)
        assert "Installation" in combined or "Troubleshooting" in combined

    def test_html_produces_chunks(self):
        html_text = "This is HTML content extracted from a webpage. " * 30
        result = self.strategy.chunk(html_text, source_type="html")
        assert len(result.texts) > 0

    def test_texts_and_token_counts_same_length(self):
        md = "# Section\n\nContent about something important.\n\n## Sub\n\nMore details."
        result = self.strategy.chunk(md, source_type="markdown")
        assert len(result.texts) == len(result.token_counts)

    def test_token_counts_are_positive(self):
        md = "# Header\n\nSome content here for testing."
        result = self.strategy.chunk(md, source_type="markdown")
        for count in result.token_counts:
            assert count > 0

    def test_markdown_no_headers_falls_back(self):
        """Markdown text without any headers should still produce chunks."""
        plain = "Plain text document with no headers. " * 20
        result = self.strategy.chunk(plain, source_type="markdown")
        assert len(result.texts) > 0


# ── ChunkingRouter ─────────────────────────────────────────────────────────────

class TestChunkingRouter:

    def setup_method(self):
        self.router = ChunkingRouter()

    def test_pdf_routes_to_sentence_window(self):
        strategy = self.router.route("pdf")
        assert strategy.strategy_name == "SentenceWindowStrategy"

    def test_docx_routes_to_sentence_window(self):
        strategy = self.router.route("docx")
        assert strategy.strategy_name == "SentenceWindowStrategy"

    def test_txt_routes_to_sentence_window(self):
        strategy = self.router.route("txt")
        assert strategy.strategy_name == "SentenceWindowStrategy"

    def test_eml_routes_to_sentence_window(self):
        strategy = self.router.route("eml")
        assert strategy.strategy_name == "SentenceWindowStrategy"

    def test_markdown_routes_to_structure_aware(self):
        strategy = self.router.route("markdown")
        assert strategy.strategy_name == "StructureAwareStrategy"

    def test_html_routes_to_structure_aware(self):
        strategy = self.router.route("html")
        assert strategy.strategy_name == "StructureAwareStrategy"

    def test_xlsx_routes_to_token_strategy(self):
        strategy = self.router.route("xlsx")
        assert strategy.strategy_name == "TokenStrategy"

    def test_csv_routes_to_token_strategy(self):
        strategy = self.router.route("csv")
        assert strategy.strategy_name == "TokenStrategy"

    def test_unknown_source_type_falls_back_to_sentence_window(self):
        strategy = self.router.route("xyz_unknown_format")
        assert strategy.strategy_name == "SentenceWindowStrategy"

    def test_uppercase_source_type_normalised(self):
        strategy = self.router.route("PDF")
        assert strategy.strategy_name == "SentenceWindowStrategy"

    def test_injected_strategy_used(self):
        """DI: injected strategy instance is returned for its source types."""
        stub = MagicMock(spec=BaseChunkingStrategy)
        stub.strategy_name = "StubStrategy"
        router = ChunkingRouter(sentence_strategy=stub)
        result = router.route("pdf")
        assert result is stub


# ── ChunkingPipeline ───────────────────────────────────────────────────────────

class TestChunkingPipeline:

    def setup_method(self):
        self.pipeline = ChunkingPipeline()

    def test_returns_chunks_and_stats(self):
        pd = make_processed(text=PROSE)
        chunks, stats = self.pipeline.run(pd)
        assert isinstance(chunks, list)
        assert isinstance(stats, dict)

    def test_chunks_are_document_chunk_instances(self):
        pd = make_processed(text=PROSE)
        chunks, _ = self.pipeline.run(pd)
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_empty_cleaned_text_returns_no_chunks(self):
        pd = make_processed(text="placeholder", cleaned_text="")
        chunks, stats = self.pipeline.run(pd)
        assert chunks == []
        assert stats["total_chunks"] == 0

    def test_rbac_fields_propagated_to_every_chunk(self):
        pd = make_processed(
            text=PROSE,
            tenant_id="tenant-xyz",
            owner_id="owner-abc",
            access_roles=["legal", "finance"],
            visibility="restricted",
        )
        chunks, _ = self.pipeline.run(pd)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["tenant_id"] == "tenant-xyz"
            assert chunk.metadata["owner_id"] == "owner-abc"
            assert chunk.metadata["access_roles"] == ["legal", "finance"]
            assert chunk.metadata["visibility"] == "restricted"

    def test_chunk_ids_are_deterministic(self):
        pd = make_processed(text=PROSE)
        chunks, _ = self.pipeline.run(pd)
        for i, chunk in enumerate(chunks):
            assert chunk.id == f"doc-001_chunk_{i}"

    def test_chunk_index_sequential(self):
        pd = make_processed(text=PROSE)
        chunks, _ = self.pipeline.run(pd)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_first_chunk_has_no_prev_link(self):
        pd = make_processed(text=PROSE)
        chunks, _ = self.pipeline.run(pd)
        assert chunks[0].metadata["prev_chunk_id"] is None

    def test_last_chunk_has_no_next_link(self):
        pd = make_processed(text=PROSE)
        chunks, _ = self.pipeline.run(pd)
        assert chunks[-1].metadata["next_chunk_id"] is None

    def test_middle_chunks_have_both_links(self):
        pd = make_processed(text=PROSE)
        chunks, _ = self.pipeline.run(pd)
        if len(chunks) > 2:
            mid = chunks[1]
            assert mid.metadata["prev_chunk_id"] == f"doc-001_chunk_0"
            assert mid.metadata["next_chunk_id"] == f"doc-001_chunk_2"

    def test_stats_has_required_keys(self):
        pd = make_processed(text=PROSE)
        _, stats = self.pipeline.run(pd)
        required = {
            "total_input_tokens",
            "total_chunks",
            "strategy",
            "avg_tokens_per_chunk",
            "max_chunk_tokens",
            "min_chunk_tokens",
        }
        assert required.issubset(stats.keys())

    def test_stats_total_chunks_matches_chunk_list(self):
        pd = make_processed(text=PROSE)
        chunks, stats = self.pipeline.run(pd)
        assert stats["total_chunks"] == len(chunks)

    def test_metadata_includes_source_info(self):
        pd = make_processed(text=PROSE)
        chunks, _ = self.pipeline.run(pd)
        for chunk in chunks:
            assert "source_type" in chunk.metadata
            assert "source_path" in chunk.metadata
            assert "title" in chunk.metadata
            assert "document_id" in chunk.metadata

    def test_metadata_includes_token_count(self):
        pd = make_processed(text=PROSE)
        chunks, _ = self.pipeline.run(pd)
        for chunk in chunks:
            assert "token_count" in chunk.metadata
            assert chunk.metadata["token_count"] > 0

    def test_metadata_includes_language_and_category(self):
        pd = make_processed(text=PROSE)
        chunks, _ = self.pipeline.run(pd)
        for chunk in chunks:
            assert "language" in chunk.metadata
            assert "category" in chunk.metadata

    def test_markdown_routes_to_structure_aware(self):
        md = "# Introduction\n\nSome intro text.\n\n## Methods\n\nMethod details here."
        pd = make_processed(text=md, source_type="markdown")
        chunks, stats = self.pipeline.run(pd)
        assert stats["strategy"] == "StructureAwareStrategy"

    def test_xlsx_routes_to_token_strategy(self):
        csv_text = "\n".join([f"col1,col2,col3\nval{i},num{i},extra{i}" for i in range(10)])
        pd = make_processed(text=csv_text, source_type="xlsx")
        chunks, stats = self.pipeline.run(pd)
        assert stats["strategy"] == "TokenStrategy"

    def test_injected_router_used(self):
        """DI: injected router controls which strategy is selected."""
        stub_result = ChunkResult(
            texts=["stub chunk one", "stub chunk two"],
            token_counts=[3, 3],
            strategy_name="StubStrategy",
        )
        stub_strategy = MagicMock(spec=BaseChunkingStrategy)
        stub_strategy.strategy_name = "StubStrategy"
        stub_strategy.chunk.return_value = stub_result

        stub_router = MagicMock(spec=ChunkingRouter)
        stub_router.route.return_value = stub_strategy

        pipeline = ChunkingPipeline(router=stub_router)
        pd = make_processed(text="any text")
        chunks, stats = pipeline.run(pd)

        assert len(chunks) == 2
        assert stats["strategy"] == "StubStrategy"


# ── ChunkingService ────────────────────────────────────────────────────────────

class TestChunkingService:

    def setup_method(self):
        self.service = ChunkingService()

    def test_success_result(self):
        pd = make_processed(text=PROSE)
        result = self.service.chunk(pd)
        assert result.success is True
        assert result.chunks is not None
        assert result.error is None

    def test_result_has_stats(self):
        pd = make_processed(text=PROSE)
        result = self.service.chunk(pd)
        assert result.stats is not None
        assert "total_chunks" in result.stats

    def test_chunks_list_not_empty_for_prose(self):
        pd = make_processed(text=PROSE)
        result = self.service.chunk(pd)
        assert len(result.chunks) > 0

    def test_empty_text_produces_success_with_zero_chunks(self):
        pd = make_processed(text="placeholder", cleaned_text="")
        result = self.service.chunk(pd)
        assert result.success is True
        assert result.chunks == []
        assert result.stats["total_chunks"] == 0

    def test_error_on_pipeline_failure(self):
        """Inject a broken pipeline to verify error wrapping."""
        class BrokenPipeline:
            def run(self, _):
                raise RuntimeError("simulated chunking error")

        service = ChunkingService(pipeline=BrokenPipeline())
        pd = make_processed(text=PROSE)
        result = service.chunk(pd)
        assert result.success is False
        assert result.chunks is None
        assert "simulated chunking error" in result.error

    def test_rbac_propagated_end_to_end(self):
        pd = make_processed(
            text=PROSE,
            tenant_id="corp-tenantA",
            owner_id="emp-42",
            access_roles=["hr"],
            visibility="restricted",
        )
        result = self.service.chunk(pd)
        assert result.success is True
        for chunk in result.chunks:
            assert chunk.metadata["tenant_id"] == "corp-tenantA"
            assert chunk.metadata["owner_id"] == "emp-42"
            assert chunk.metadata["visibility"] == "restricted"

    def test_chunks_are_document_chunk_type(self):
        pd = make_processed(text=PROSE)
        result = self.service.chunk(pd)
        for chunk in result.chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_all_required_metadata_keys_present(self):
        """Every chunk must have all keys for vector store filtering."""
        pd = make_processed(text=PROSE)
        result = self.service.chunk(pd)
        required_keys = {
            "tenant_id", "owner_id", "access_roles", "visibility",
            "source_type", "source_path", "title", "document_id",
            "chunk_index", "total_chunks", "token_count", "strategy",
            "prev_chunk_id", "next_chunk_id", "language", "category",
        }
        for chunk in result.chunks:
            missing = required_keys - chunk.metadata.keys()
            assert not missing, f"Chunk {chunk.id} missing metadata keys: {missing}"

    def test_injected_pipeline_used(self):
        """DI: injected pipeline controls chunking behaviour."""
        stub_chunk = DocumentChunk(
            id="stub_chunk_0",
            document_id="doc-001",
            text="stub text",
            chunk_index=0,
            metadata={"test": True},
        )

        class StubPipeline:
            def run(self, _):
                return [stub_chunk], {"total_chunks": 1, "strategy": "stub"}

        service = ChunkingService(pipeline=StubPipeline())
        pd = make_processed(text="any text")
        result = service.chunk(pd)
        assert result.success is True
        assert len(result.chunks) == 1
        assert result.chunks[0].id == "stub_chunk_0"
