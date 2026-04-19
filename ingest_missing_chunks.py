#!/usr/bin/env python3
"""
Ingest any missing crawl chunk files into the existing ChromaDB collection.

This script scans `data/crawl_chunks/**/*.chunk.txt`, generates deterministic
chunk IDs from the file paths, checks which IDs are already present in Chroma,
and adds only the missing chunks.

Usage:
  python -B ingest_missing_chunks.py
"""

from pathlib import Path
from typing import List

from foodrag.storage import get_client, get_collection

CHUNK_ROOT = Path("data/crawl_chunks")
BATCH_SIZE = 128


def chunk_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return f"crawl_chunks::{relative.as_posix()}"


def collect_chunk_paths(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.chunk.txt"))


def read_chunk(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def existing_ids(collection, ids: List[str]) -> List[str]:
    if not ids:
        return []
    response = collection.get(ids=ids)
    return response.get("ids", [])


def main() -> None:
    chunk_paths = collect_chunk_paths(CHUNK_ROOT)
    print(f"Found {len(chunk_paths)} chunk files in {CHUNK_ROOT}")
    if not chunk_paths:
        return

    client = get_client()
    collection = get_collection(client)

    all_ids = [chunk_id(path, CHUNK_ROOT) for path in chunk_paths]
    missing_ids = []
    missing_paths = []

    for i in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[i : i + BATCH_SIZE]
        present = set(existing_ids(collection, batch_ids))
        for path, cid in zip(chunk_paths[i : i + BATCH_SIZE], batch_ids):
            if cid not in present:
                missing_ids.append(cid)
                missing_paths.append(path)

    print(f"Found {len(missing_paths)} missing chunk files to ingest")
    if not missing_paths:
        return

    added = 0
    for i in range(0, len(missing_paths), BATCH_SIZE):
        batch_paths = missing_paths[i : i + BATCH_SIZE]
        ids = [chunk_id(path, CHUNK_ROOT) for path in batch_paths]
        docs = [read_chunk(path) for path in batch_paths]
        metadatas = []
        for path in batch_paths:
            suffix = path.stem.rsplit("_chunk_", 1)[-1]
            if suffix.endswith(".chunk"):
                suffix = suffix[: -len(".chunk")]
            try:
                chunk_index = int(suffix)
            except ValueError:
                chunk_index = -1
            metadatas.append(
                {
                    "file": str(path),
                    "source": "crawl_chunks",
                    "chunk_index": chunk_index,
                }
            )
        collection.add(ids=ids, documents=docs, metadatas=metadatas)
        added += len(batch_paths)
        print(f"Added {added}/{len(missing_paths)} chunks")

    print(f"Done. Added {added} missing chunk(s) to ChromaDB.")


if __name__ == "__main__":
    main()
