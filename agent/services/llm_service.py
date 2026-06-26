from agent.llm.base import BaseLLMProvider, LLMResult
from agent.retrieval.models import SearchResult

_SYSTEM_PROMPT = """You are an enterprise document assistant. Answer questions based ONLY on the provided document excerpts.

Rules:
- Answer directly and concisely using only the information in the provided context
- If the answer is not clearly present in the context, say "I don't have enough information in the provided documents to answer this."
- Never make up, infer, or hallucinate facts beyond what is explicitly stated
- Cite which source excerpt your answer comes from (e.g. "According to Source 1...")
- If quoting directly, use quotation marks"""


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
            source = (
                chunk.metadata.get("file_name")
                or chunk.metadata.get("source")
                or chunk.document_id
            )
            context_parts.append(f"[Source {i}: {source}]\n{chunk.text}")

        context = "\n\n---\n\n".join(context_parts)
        user_message = f"Document excerpts:\n\n{context}\n\nQuestion: {query}"

        return self._provider.generate(_SYSTEM_PROMPT, user_message)
