from app.schemas import RetrievedChunk


class ContextBuilder:
    def __init__(self, max_chunks: int = 5) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self.max_chunks = max_chunks

    def build(self, chunks: list[RetrievedChunk]) -> str:
        unique_chunks = self._deduplicate(chunks)[: self.max_chunks]
        blocks = [
            (
                f"[SOURCE {index}]\n"
                f"Title: {chunk.title}\n"
                f"URL: {chunk.url}\n"
                "Content:\n"
                f"{chunk.text.strip()}"
            )
            for index, chunk in enumerate(unique_chunks, start=1)
            if chunk.text.strip()
        ]
        return "\n\n".join(blocks)

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique_chunks.append(chunk)
        return unique_chunks
