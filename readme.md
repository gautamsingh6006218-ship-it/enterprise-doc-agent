# Enterprise Document AI Agent

Production-grade RAG (Retrieval-Augmented Generation) pipeline for 500,000+ enterprise documents. Microservices architecture — every service has a clean boundary, returns Result objects, and never raises.

---

## Pipeline Overview

```
Raw File
   │
   ▼
[1] IngestionService        →  Document (text + RBAC metadata)
   │
   ▼
[2] PreprocessingService    →  ProcessedDocument (cleaned text + extracted metadata)
   │
   ▼
[3] ChunkingService         →  list[DocumentChunk] (512-token chunks + window links)
   │
   ▼
[4] EmbeddingService        →  vectors stored in PostgreSQL + pgvector
   │
   ▼
[5] RetrievalService        →  RetrievedContext (reranked chunks + window context)
```

---

## Technology Stack by Pipeline Stage

### Stage 1 — Document Loading (Ingestion)

| Tool | Version | Why Used |
|---|---|---|
| **PyMuPDF4LLM** | 1.27.x | Primary PDF loader — preserves markdown structure (headers, tables, lists) from text-based PDFs |
| **Docling** | 2.x | Fallback for complex PDFs — handles multi-column layouts, scanned tables, figures |
| **Tesseract OCR** + pytesseract | 0.3.x | Final fallback for scanned/image-only PDFs — renders page at 300 DPI, extracts text via OCR |
| **python-docx** | 1.2.x | DOCX loader — extracts paragraphs, tables, headings from Word documents |
| **openpyxl** | 3.1.x | XLSX loader — reads spreadsheet rows/cells |
| **Unstructured** | 0.23.x | Universal loader for PPTX, RTF, EPUB, EML, MSG, ODT, images — one library handles 14+ formats |
| **BeautifulSoup4** | 4.15.x | HTML loader — strips tags, extracts clean text |
| **lxml** | 6.x | HTML/XML parser used by BeautifulSoup4 |
| **Pillow** | 12.x | Image rendering for OCR pipeline (PDF page → PIL image → Tesseract) |

**Smart routing:** `SmartPdfLoader` auto-detects PDF type by chars/page density. Text PDF → PyMuPDF4LLM → Docling (if sparse) → OCR (if still sparse). Other formats → LoaderRegistry routes to the correct loader.

---

### Stage 2 — Data Cleaning & Normalization (Preprocessing)

All implemented as pure Python — no external library dependencies for the core cleaning logic.

| Component | What It Does |
|---|---|
| **WhitespaceCleaner** | Removes control characters (`\x00–\x1f`), normalises `\r\n` → `\n`, collapses 3+ blank lines to 2, strips trailing spaces |
| **NoiseCleaner** | Source-type-aware: removes PDF page numbers, watermarks (standalone lines only), TOC dot leaders, email headers (only for `.eml`/`.msg`) |
| **UnicodeNormalizer** | Maps 35+ Unicode variants to ASCII equivalents — curly quotes → straight, em-dash → hyphen, bullets → `-`, zero-width chars removed. Uses `str.translate()` for O(n) single-pass |
| **HyphenNormalizer** | Repairs PDF line-break hyphenation (`infor-\nmation` → `information`). Preserves compound prefixes (`self-`, `multi-`, `cross-`, etc.) |
| **langdetect** | 1.0.9 | Language detection on first 2000 chars. Seeded (`seed=0`) for deterministic output |

---

### Stage 3 — Metadata Extraction (Preprocessing)

| Component | What It Extracts |
|---|---|
| **Regex patterns** | Dates (ISO, US, written), email addresses, phone numbers, URLs |
| **Section headers** | Markdown `##`, numbered `1.2.3`, ALL CAPS standalone lines |
| **Category detection** | Keyword scoring against known categories: invoice, contract, policy, report, email, presentation, specification, resume, manual |
| **langdetect** | Document language code (`en`, `de`, `fr`, etc.) |
| **Stats** | word_count, char_count, reading_time_minutes |

---

### Stage 4 — Chunking

| Tool | Version | Why Used |
|---|---|---|
| **chonkie** | 1.6.8 | `SentenceChunker` (primary — respects sentence boundaries for prose), `TokenChunker` (hard token budget for tables/CSV) |
| **tiktoken** | 0.13.0 | Token counting with `cl100k_base` encoding. Pre/post chunk token counting, 85-90% accurate for nomic-embed-text |
| **langchain-text-splitters** | 1.1.x | `MarkdownHeaderTextSplitter` (splits on `#/##/###` headers) + `RecursiveCharacterTextSplitter.from_tiktoken_encoder` for overflow sections |

**Chunking strategies by document type:**

| Source Type | Strategy | Why |
|---|---|---|
| PDF, DOCX, TXT, email, PPTX | `SentenceWindowStrategy` (chonkie) | Prose — sentence boundaries matter |
| Markdown, HTML | `StructureAwareStrategy` (LangChain) | Headers define semantic sections |
| XLSX, CSV | `TokenStrategy` (chonkie) | Tables have no sentence boundaries |

**Parameters:** 512 token target, 1024 token hard max, 51 token overlap (~10%). Every chunk stores `prev_chunk_id` / `next_chunk_id` for sentence window retrieval (±2 neighbours at query time).

---

### Stage 5 — Embedding

| Tool | Version | Why Used |
|---|---|---|
| **FlagEmbedding** (BGE-M3) | 1.4.x | `BAAI/bge-m3` — single model that produces **both** dense (1024-dim) AND sparse (learned lexical) vectors in one forward pass. Eliminates need for a separate BM25 index |
| **psycopg2-binary** | 2.9.x | PostgreSQL adapter — connection, batch upsert via `execute_batch` |
| **pgvector** | 0.4.x | Registers `vector(1024)` type in psycopg2; enables `<=>` cosine distance operator |
| **numpy** | 2.4.x | Converts float lists to `float32` arrays for pgvector type compatibility |
| **PostgreSQL + pgvector** | pg16 | Stores dense vectors (HNSW index), sparse weights (JSONB), metadata (JSONB with GIN index) |

**Vector storage schema:**
- `dense_vector vector(1024)` — HNSW index, cosine distance
- `sparse_weights JSONB` — BGE-M3 lexical weights stored for future sparse search
- `metadata JSONB` — RBAC fields + provenance (tenant_id, owner_id, access_roles, visibility, source_type, chunk_index, prev/next chunk IDs, language, category)

---

### Stage 6 — Retrieval

| Tool | Version | Why Used |
|---|---|---|
| **FlagEmbedding** (BGE-M3) | 1.4.x | `encode_query()` — encodes user query to 1024-dim vector for similarity search |
| **PostgreSQL FTS** | pg16 built-in | `ts_rank_cd` + `plainto_tsquery` for keyword/BM25-style search. GIN index on `tsvector(text)` |
| **Custom RRF** | — | Reciprocal Rank Fusion (Cormack et al. 2009, k=60). Merges dense + keyword result lists without LangChain dependency (12 lines of Python) |
| **FlagEmbedding** (bge-reranker-v2-m3) | 1.4.x | Local cross-encoder reranker — jointly encodes (query, chunk) to score relevance. Fully on-premises, no external API |

**Retrieval flow:**
```
Query → BGE-M3 encode → Dense search (top 50) ─┐
                       → Keyword search (top 50) ─┤
                                                   ├─ RRF → top 20 → bge-reranker → top 5
                                                              └─ Sentence window expand (±2 chunks)
```

---

## RBAC (Role-Based Access Control)

Every `Document` carries: `tenant_id`, `owner_id`, `access_roles`, `visibility`. These propagate to every `DocumentChunk.metadata` at chunking time. All vector store queries enforce RBAC in SQL:

```
visibility = "public"     → any user in the tenant
visibility = "restricted" → users whose roles ∩ access_roles ≠ ∅
visibility = "private"    → owner_id only
```

Tenant isolation is always enforced first (`metadata->>'tenant_id' = $1`).

---

## Project Structure

```
agent/
├── ingestion/
│   ├── models.py              # Document, DocumentChunk dataclasses
│   ├── loader_registry.py     # LoaderRegistry singleton (open/closed)
│   └── loaders/
│       ├── base.py            # BaseDocumentLoader ABC
│       ├── pdf.py             # PyMuPDF4LLM
│       ├── pdf_smart.py       # SmartPdfLoader (auto-route)
│       ├── pdf_docling.py     # Docling fallback
│       ├── pdf_ocr.py         # Tesseract OCR fallback
│       ├── unstructured_loader.py  # PPTX/email/images
│       ├── txt.py             # Plain text
│       └── html.py            # HTML
├── processing/
│   ├── models.py              # ProcessedDocument, ExtractedMetadata
│   ├── pipeline.py            # PreprocessingPipeline
│   ├── cleaning/
│   │   ├── whitespace_cleaner.py
│   │   └── noise_cleaner.py
│   ├── normalization/
│   │   ├── unicode_normalizer.py
│   │   └── hyphen_normalizer.py
│   └── metadata/
│       └── extractor.py
├── chunking/
│   ├── config.py              # CHUNK_TARGET_TOKENS=512, STRATEGY_MAP
│   ├── token_counter.py       # tiktoken wrapper
│   ├── router.py              # source_type → strategy
│   ├── pipeline.py            # ChunkingPipeline
│   └── strategies/
│       ├── base.py            # BaseChunkingStrategy ABC, ChunkResult
│       ├── sentence_window.py # chonkie SentenceChunker
│       ├── structure_aware.py # LangChain header + recursive splitter
│       └── token_chunker.py   # chonkie TokenChunker
├── embeddings/
│   ├── config.py              # BGE_M3_MODEL, DENSE_VECTOR_DIM=1024
│   ├── bge_encoder.py         # BGEEncoder (dense + sparse)
│   └── store.py               # PgVectorStore (upsert, search_dense, search_keyword)
├── retrieval/
│   ├── models.py              # RBACContext, SearchResult, RetrievedContext
│   ├── rrf.py                 # reciprocal_rank_fusion (custom, no LangChain)
│   ├── reranker.py            # BGEReranker (bge-reranker-v2-m3, local)
│   └── pipeline.py            # RetrievalPipeline (encode→search→RRF→rerank→expand)
└── services/
    ├── ingestion_service.py   # IngestionResult
    ├── preprocessing_service.py # PreprocessingResult
    ├── chunking_service.py    # ChunkingResult
    ├── embedding_service.py   # EmbeddingResult
    └── retrieval_service.py   # RetrievalResult

scripts/
└── setup_db.sql               # PostgreSQL schema (HNSW + GIN + metadata indexes)

docker-compose.yml             # PostgreSQL 16 + pgvector (pgvector/pgvector:pg16)

tests/
├── ingestion/                 # 27 tests
├── processing/                # 49 tests
├── chunking/                  # 71 tests
├── embedding/                 # 29 tests
└── retrieval/                 # 22 tests
```

---

## Quick Start

```bash
# 1. Start PostgreSQL + pgvector
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
python3 -m pytest tests/ -v

# 4. Set database URL
export PGVECTOR_URL="postgresql://raguser:ragpassword@localhost:5432/ragdb"
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| PostgreSQL + pgvector (not Qdrant/Pinecone) | Avoids dual-write problem — vectors and RBAC metadata in one DB, one transaction |
| BGE-M3 (not separate dense + BM25) | One model produces both dense and sparse vectors — no second service to manage |
| bge-reranker-v2-m3 (not Cohere Rerank API) | Enterprise docs (HR/legal/finance) must not leave the organisation's infrastructure |
| Custom RRF (not LangChain EnsembleRetriever) | RRF is 12 lines; EnsembleRetriever requires LangChain's Document/Retriever abstractions which don't match our DocumentChunk model |
| Result pattern everywhere (never raises) | All services return `XxxResult(success, data, error)` — safe across Celery workers and async handlers |
| RBAC enforced in SQL (not post-filter) | Post-filtering would silently reduce result count below top_k; SQL-level filtering guarantees exactly top_k RBAC-valid results |
