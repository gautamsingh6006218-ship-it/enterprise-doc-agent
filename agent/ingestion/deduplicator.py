"""
ingestion/deduplicator.py

What problem does this solve?
- Exact file deduplication (MD5 hash) misses near-duplicates: the same policy
  document with a date changed on the footer, or a contract with a clause
  amended. Without near-dedup, 50 versions of the same HR policy all get
  chunked and embedded — retrieval returns near-identical chunks that waste
  context window and confuse the LLM.

Why MinHash LSH and not simpler alternatives?
- Exact hash (current): misses any byte difference — fails for near-duplicates.
- Edit distance (Levenshtein): O(n²) per pair — unusable at 500K documents.
- Cosine similarity on embeddings: requires embedding first — expensive,
  and we haven't embedded the document yet at ingestion time.
- MinHash LSH: O(1) lookup, O(document length) to compute, probabilistic
  but accurate enough for enterprise dedup (Jaccard ≥ threshold → duplicate).
  Industry-standard for large-scale document dedup (used by Common Crawl, etc.)

How MinHash LSH works (brief):
- Represent a document as a set of k-shingles (k consecutive words).
- MinHash reduces this set to a compact signature (num_perm hash functions).
- LSH groups similar signatures into the same bucket — query in O(1).
- Two documents are near-duplicates if their Jaccard similarity ≥ threshold.
  Jaccard = |A ∩ B| / |A ∪ B| — fraction of shared shingles.

Why k=5 word shingles?
- k=1 (single words) has too many false positives (shared vocabulary, not content).
- k=5 strikes the right balance: captures phrase similarity without being so
  specific that reordered sentences are missed.
- k=8+ starts missing paraphrased near-duplicates.

Why num_perm=128?
- Error bound ≈ 1/sqrt(num_perm) ≈ 0.088 at 128 permutations.
- At 256, error ≈ 0.0625 but compute doubles. 128 is the industry default.

Why in-memory index?
- For production, the index should persist in Redis (MinHashLSH supports it).
  In-memory is correct for single-process batch ingestion and testing.
  The deduplicator is injectable — swap the backend without changing callers.
"""

from dataclasses import dataclass

try:
    from datasketch import MinHash, MinHashLSH
    _DATASKETCH_AVAILABLE = True
except ImportError:
    _DATASKETCH_AVAILABLE = False

_DEFAULT_THRESHOLD = 0.85   # Jaccard similarity threshold for near-duplicate
_DEFAULT_NUM_PERM  = 128    # MinHash permutations — controls accuracy/speed
_DEFAULT_SHINGLE_K = 5      # k-word shingles — controls granularity


@dataclass
class DeduplicationResult:
    """
    What problem does this solve?
    - check() callers need to know: is this a duplicate, and if so, of what?
      A dataclass keeps these paired and named instead of a bare tuple.

    Fields:
    - is_duplicate:     True if a near-duplicate already exists in the index.
    - duplicate_of:     document_id of the existing near-duplicate. None if not duplicate.
    - similarity_score: Estimated Jaccard similarity (0–1). 0.0 if not duplicate.
    - was_indexed:      True if this document was added to the index (only when not duplicate).
    """

    is_duplicate: bool
    duplicate_of: str | None = None
    similarity_score: float = 0.0
    was_indexed: bool = False


class MinHashDeduplicator:
    """
    What problem does this solve?
    - Detects near-duplicate documents before they enter the chunking and
      embedding pipeline, saving compute and keeping retrieval results clean.

    Why injectable (threshold, num_perm as constructor args)?
    - Different document types need different thresholds:
      Legal contracts: 0.95 (only skip near-identical versions)
      News articles:   0.80 (skip paraphrased duplicates)
    - Tests use small num_perm for speed without changing production config.

    Why store document_id → MinHash separately from the LSH index?
    - LSH.query() returns document IDs but not the similarity score.
      Storing the MinHash lets us compute the exact estimated similarity
      for the duplicate_of document — useful for monitoring and reporting.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        num_perm: int = _DEFAULT_NUM_PERM,
        shingle_k: int = _DEFAULT_SHINGLE_K,
        disabled: bool = False,
    ) -> None:
        """
        Args:
        - threshold:  Jaccard similarity above which documents are near-duplicates.
        - num_perm:   MinHash accuracy (more = slower + more accurate).
        - shingle_k:  Size of word shingles for document representation.
        - disabled:   If True, check() always returns not-duplicate. For environments
                      where datasketch is unavailable or dedup should be bypassed.
        """
        self._threshold = threshold
        self._num_perm = num_perm
        self._shingle_k = shingle_k
        self._disabled = disabled or not _DATASKETCH_AVAILABLE

        if not self._disabled:
            self._lsh: "MinHashLSH" = MinHashLSH(
                threshold=threshold,
                num_perm=num_perm,
            )
            self._hashes: dict[str, "MinHash"] = {}

    def check(self, document_id: str, text: str) -> DeduplicationResult:
        """
        What problem does this solve?
        - Determines if a document is a near-duplicate of one already ingested,
          and if not, adds it to the index for future comparisons.

        Why index on first check (not in a separate add() call)?
        - The common pattern is: check → skip if duplicate → process if not.
          Combining check+index into one call means callers never forget to
          index new documents. Simpler API, fewer bugs.

        Why query BEFORE inserting?
        - LSH.query() includes the document being inserted if it's already in
          the index. Querying first prevents self-matching.

        Why use cleaned/lowercased text for shingling?
        - Case differences and extra whitespace should not prevent dedup.
          "CONFIDENTIAL POLICY" and "confidential policy" are the same document.

        Args:
        - document_id: Unique identifier for this document (Document.id).
        - text:        The document's extracted text (from loader, before cleaning).

        Returns DeduplicationResult with is_duplicate, duplicate_of, similarity_score.
        """
        if self._disabled:
            return DeduplicationResult(is_duplicate=False, was_indexed=False)

        # Skip very short texts — not enough shingles for reliable comparison
        words = text.lower().split()
        if len(words) < self._shingle_k * 3:
            return DeduplicationResult(is_duplicate=False, was_indexed=False)

        m = self._compute_minhash(words)

        # Query BEFORE inserting (avoid self-match)
        matches = self._lsh.query(m)
        if matches:
            best_match = matches[0]
            # Estimate Jaccard similarity between this doc and the best match
            similarity = m.jaccard(self._hashes[best_match])
            return DeduplicationResult(
                is_duplicate=True,
                duplicate_of=best_match,
                similarity_score=round(similarity, 4),
                was_indexed=False,
            )

        # Not a duplicate — add to index
        self._lsh.insert(document_id, m)
        self._hashes[document_id] = m
        return DeduplicationResult(
            is_duplicate=False,
            was_indexed=True,
        )

    def remove(self, document_id: str) -> None:
        """
        What problem does this solve?
        - When a document is deleted or re-ingested (replacing an old version),
          its MinHash must be removed so the next version isn't flagged as
          a duplicate of the old one.

        Why silently ignore unknown document_ids?
        - The deduplicator may not have indexed the document (e.g. it was too
          short, or was ingested before the deduplicator was added). Raising
          would break deletion flows that don't track whether dedup was applied.
        """
        if self._disabled:
            return
        if document_id in self._hashes:
            try:
                self._lsh.remove(document_id)
            except ValueError:
                pass
            del self._hashes[document_id]

    def _compute_minhash(self, words: list[str]) -> "MinHash":
        """
        Builds a MinHash from k-word shingles of the document's word list.

        Why iterate shingles and not the full text?
        - MinHash is a set-similarity algorithm. Shingling converts a sequence
          (ordered) into a set of overlapping n-grams (unordered).
          Two near-identical documents with sentences in different order will
          still share most of their 5-word shingles.
        """
        m = MinHash(num_perm=self._num_perm)
        k = self._shingle_k
        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i : i + k])
            m.update(shingle.encode("utf-8"))
        return m

    @property
    def indexed_count(self) -> int:
        """Number of documents currently in the index."""
        return len(self._hashes) if not self._disabled else 0
