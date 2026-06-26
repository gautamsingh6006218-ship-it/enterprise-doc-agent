"""
chunking/token_counter.py

What problem does this solve?
- Before chunking we need to know how many tokens a document contains so
  the ChunkingPipeline can select the right strategy and log meaningful stats.
- After chunking we need to verify each chunk is within the embedding model's
  token budget — preventing silent truncation at embedding time.

Why tiktoken and not the model's native tokenizer?
- nomic-embed-text uses a BERT WordPiece tokenizer. Loading it requires
  downloading the full model weights (~270MB) just to count tokens.
- tiktoken cl100k_base is 85-90% accurate for nomic-embed-text and loads
  instantly from a small file. The 10-15% error is acceptable for chunking
  targets — we budget conservatively (512 target vs 1024 hard max).
- When switching to a model whose tokenizer is critical (e.g. LLaMA), swap
  the encoding name here — one line change, nothing else moves.

Why a class instead of module-level functions?
- Caches the tiktoken encoding object on the instance. tiktoken.get_encoding()
  reads from disk on first call — caching avoids repeated disk reads when
  counting tokens across thousands of chunks.
"""

import tiktoken

from agent.chunking.config import TIKTOKEN_ENCODING


class TokenCounter:
    """
    What problem does this solve?
    - Centralises all token counting so the rest of the pipeline never
      calls tiktoken directly. If we swap the tokenizer later, only this
      file changes.

    Why does this class exist?
    - Caches the encoding object (disk I/O on first call).
    - Provides a clean interface: count() for single texts, count_batch()
      for lists of chunk texts.
    - Stateless after initialisation — safe to share one instance.
    """

    def __init__(self, encoding_name: str = TIKTOKEN_ENCODING) -> None:
        """
        Why cache the encoding?
        - tiktoken.get_encoding() reads from disk on first call.
          Caching avoids repeated disk reads when processing thousands of
          chunks across a single document ingestion batch.
        """
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._encoding_name = encoding_name

    def count(self, text: str) -> int:
        """
        What problem does this solve?
        - Returns the token count of a single text string.

        Why encode without allowed_special?
        - Strict mode (default) raises on special tokens like <|endoftext|>.
          disallowed_special=() silently ignores them — appropriate for
          user-supplied document text that may contain angle-bracket patterns.

        Args:
        - text: Any string to count tokens for.

        Returns integer token count.
        """
        return len(self._encoding.encode(text, disallowed_special=()))

    def count_batch(self, texts: list[str]) -> list[int]:
        """
        What problem does this solve?
        - Counts tokens for a list of chunk texts efficiently.
          Used to validate all chunks after splitting.

        Why not just call count() in a loop?
        - encode_batch() processes all strings in a single C-level pass —
          faster than N individual encode() calls for large chunk lists.

        Returns list of token counts in the same order as input texts.
        """
        encoded = self._encoding.encode_batch(
            texts, disallowed_special=()
        )
        return [len(e) for e in encoded]

    def exceeds_limit(self, text: str, limit: int) -> bool:
        """
        What problem does this solve?
        - Quick check whether a text exceeds a token limit without storing
          the full token count. Used to validate chunks against hard ceiling.

        Why not just count() > limit?
        - Identical logic — this is a named convenience method so call
          sites read as intent: exceeds_limit(text, MAX_CHUNK_TOKENS)
          instead of count(text) > MAX_CHUNK_TOKENS.
        """
        return self.count(text) > limit

    @property
    def encoding_name(self) -> str:
        """Returns the tiktoken encoding being used — for logging and stats."""
        return self._encoding_name
