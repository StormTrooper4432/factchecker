import argparse
import uuid
from pathlib import Path
from typing import List, Iterable

from chromadb.errors import InvalidArgumentError

from . import config
from .tagging import generate_tags
from .text_utils import clean_text_regex, chunk_by_sentences
from .storage import get_client, get_collection, recreate_collection

SUPPORTED_EXTENSIONS = {".txt", ".xml", ".nxml"}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _iter_data_files(data_dir: str) -> Iterable[Path]:
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])


def ingest_directory(data_dir: str = config.DATA_DIR):
    client = get_client()
    collection = get_collection(client)

    paths = list(_iter_data_files(data_dir))
    if not paths:
        print(f"No supported files found under {data_dir}")
        return

    ingested = 0
    total_chunks = 0

    for p in paths:
        text = read_text(p)
        if not text.strip():
            continue

        cleaned = clean_text_regex(text)
        tags = generate_tags(cleaned)
        chunks = chunk_by_sentences(cleaned, config.SENTENCES_PER_CHUNK)
        if not chunks:
            continue

        source_id = f"{p.stem}-{uuid.uuid4().hex[:8]}"
        ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source_id": source_id, "file": str(p), "tags": tags, "chunk_index": i}
            for i in range(len(chunks))
        ]

        try:
            collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        except InvalidArgumentError as exc:
            if "expecting embedding with dimension" in str(exc).lower():
                collection = recreate_collection(client)
                collection.add(ids=ids, documents=chunks, metadatas=metadatas)
            else:
                raise

        ingested += 1
        total_chunks += len(chunks)
        print(f"Ingested {len(chunks)} chunks from {p}")

    print(f"Done. Files: {ingested}, Chunks: {total_chunks}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data files into Chroma.")
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="Root data directory to ingest.")
    args = parser.parse_args()
    ingest_directory(args.data_dir)


if __name__ == "__main__":
    main()
