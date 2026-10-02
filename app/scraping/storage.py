import json
from pathlib import Path

from app.schemas import SourceDocument


def save_documents_jsonl(documents: list[SourceDocument], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(document.model_dump_json() + "\n")


def load_documents_jsonl(path: str) -> list[SourceDocument]:
    source = Path(path)
    documents: list[SourceDocument] = []

    with source.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(SourceDocument.model_validate(json.loads(line)))

    return documents
