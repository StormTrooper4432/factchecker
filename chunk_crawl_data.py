#!/usr/bin/env python3
"""
Create pre-chunked crawl evidence for faster, more reliable retrieval.

This reads raw text files from the crawled corpus in `data/crawl`, splits them into
quality chunks using the same logic as the retrieval pipeline, and writes the
chunk files to `data/crawl_chunks`.

Usage:
  python -B chunk_crawl_data.py --source data/crawl --out data/crawl_chunks --max-files 1000
"""

import argparse
from pathlib import Path

from foodrag.retrieval import write_chunked_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-chunk crawled data for retrieval.")
    parser.add_argument("--source", default="data/crawl", help="Raw crawl corpus directory.")
    parser.add_argument("--out", default="data/crawl_chunks", help="Output directory for chunk files.")
    parser.add_argument("--max-files", type=int, default=1000, help="Max raw text files to process.")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    written = write_chunked_corpus(str(source_dir), str(Path(args.out)), max_files=args.max_files)
    print(f"Wrote {written} chunk files to {args.out}")


if __name__ == "__main__":
    main()
