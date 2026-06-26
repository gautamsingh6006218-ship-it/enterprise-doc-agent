import os

from agent.llm.base import BaseLLMProvider, LLMResult
from agent.retrieval.models import SearchResult

_SYSTEM_PROMPT = """You are an enterprise document assistant. Your job is to extract and present information from the provided document excerpts.

Instructions:
- Read ALL provided source excerpts carefully
- Extract and present ALL relevant information that answers the question
- If a source contains a list (skills, bullet points, items), reproduce that list completely
- Be direct — answer the question immediately without preamble
- Cite sources inline (e.g. "According to Source 1...")
- Only say you cannot answer if the topic is genuinely absent from ALL sources
- Never ignore data that is clearly present in the excerpts"""


def _source_name(chunk: SearchResult) -> str:
    meta = chunk.metadata

    # title is overridden with original_filename at ingest time
    title = meta.get("title", "")
    if title and not title.startswith(("ingest_", "watch_")):
        return title

    # title is a temp name — try source_path basename
    source_path = meta.get("source_path", "")
    if source_path:
        name = os.path.basename(source_path)
        if not name.startswith(("ingest_", "watch_")):
            return name

    # Last resort: category + document_id prefix
    category = meta.get("category", "")
    doc_id = chunk.document_id[:8]
    return f"{category} ({doc_id})" if category else doc_id


class LLMService:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    def answer(self, query: str, chunks: list[SearchResult]) -> LLMResult:
        if not chunks:
            return LLMResult(
                success=True,
                answer="No relevant documents found to answer this question.",
            )

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = _source_name(chunk)
            context_parts.append(f"[Source {i}: {source}]\n{chunk.text}")

        context = "\n\n---\n\n".join(context_parts)
        user_message = f"Document excerpts:\n\n{context}\n\nQuestion: {query}"

        return self._provider.generate(_SYSTEM_PROMPT, user_message)
