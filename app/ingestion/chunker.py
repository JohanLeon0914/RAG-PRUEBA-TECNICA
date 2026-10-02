import hashlib
import re

from app.schemas import DocumentChunk, SourceDocument


class DocumentChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be lower than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[SourceDocument]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for document in documents:
            text_chunks = self._chunk_text(document.content)
            for index, text in enumerate(text_chunks):
                chunks.append(
                    DocumentChunk(
                        chunk_id=self._chunk_id(document.document_id, index, text),
                        document_id=document.document_id,
                        text=text,
                        url=document.url,
                        title=document.title,
                        section=document.section,
                        chunk_index=index,
                        scraped_at=document.scraped_at,
                        metadata={
                            **document.metadata,
                            "chunk_size_unit": "characters",
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                        },
                    )
                )

        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        units = self._split_into_units(text)
        chunks: list[str] = []
        current = ""

        for unit in units:
            if len(unit) > self.chunk_size:
                chunks.extend(self._flush_with_long_unit(current, unit))
                current = ""
                continue

            candidate = self._join(current, unit)
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
            current = self._join(self._overlap_suffix(current), unit)

            if len(current) > self.chunk_size:
                chunks.append(unit)
                current = ""

        if current:
            chunks.append(current)

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_into_units(self, text: str) -> list[str]:
        paragraphs = [
            paragraph.strip() for paragraph in re.split(r"\n{1,}", text) if paragraph.strip()
        ]
        units: list[str] = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                units.append(paragraph)
            else:
                units.extend(self._split_long_paragraph(paragraph))

        return units

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?¿¡])\s+", paragraph)
            if sentence.strip()
        ]
        units: list[str] = []

        for sentence in sentences:
            if len(sentence) <= self.chunk_size:
                units.append(sentence)
            else:
                units.extend(self._split_by_words(sentence))

        return units

    def _split_by_words(self, text: str) -> list[str]:
        words = text.split()
        units: list[str] = []
        current = ""

        for word in words:
            candidate = self._join(current, word, separator=" ")
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                units.append(current)
            current = word

        if current:
            units.append(current)

        return units

    def _flush_with_long_unit(self, current: str, unit: str) -> list[str]:
        chunks: list[str] = []
        if current:
            chunks.append(current)

        for part in self._split_long_paragraph(unit):
            if not chunks or self.chunk_overlap == 0:
                chunks.append(part)
                continue
            candidate = self._join(self._overlap_suffix(chunks[-1]), part)
            chunks.append(candidate if len(candidate) <= self.chunk_size else part)

        return chunks

    def _overlap_suffix(self, text: str) -> str:
        if self.chunk_overlap == 0 or not text:
            return ""

        suffix = text[-self.chunk_overlap :]
        match = re.search(r"\s", suffix)
        if match:
            suffix = suffix[match.end() :]
        return suffix.strip()

    def _join(self, left: str, right: str, separator: str = "\n") -> str:
        if not left:
            return right.strip()
        if not right:
            return left.strip()
        return f"{left.strip()}{separator}{right.strip()}"

    def _chunk_id(self, document_id: str, chunk_index: int, text: str) -> str:
        digest = hashlib.sha256(f"{document_id}:{chunk_index}:{text}".encode()).hexdigest()
        return f"{document_id}:{chunk_index}:{digest[:16]}"
