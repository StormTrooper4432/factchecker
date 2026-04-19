"""
Direct file-based fact checking (no Chroma).
Steps:
 1) LLM (or heuristic) extracts key terms from the claim.
 2) Filter local text/XML files containing those terms.
 3) Chunk into 2-sentence blocks.
 4) Embed with lightweight food embedding and score similarity.
 5) Keep top 5 chunks; send to LLM (or heuristic) for verdict.
"""

import argparse
from pathlib import Path

from foodrag import config
from foodrag.retrieval import retrieve_top_chunks, extract_terms, find_files_with_terms
from foodrag.factcheck import run_fact_check


def main():
    parser = argparse.ArgumentParser(description="Direct fact-check from local files (no vector DB).")
    parser.add_argument("claim", help="Claim to evaluate.")
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="Root data directory.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="Top-k evidence chunks.")
    parser.add_argument("--max-files", type=int, default=config.MAX_FILES, help="Max files to scan for terms.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        return
    print(f"Using data dir: {data_dir}")
    print(f"Max files to scan: {args.max_files}, Top-k: {args.top_k}")

    # Debug: show terms and file prefilter count
    terms = extract_terms(args.claim)
    print(f"Extracted terms ({len(terms)}): {terms}")
    matched_files = find_files_with_terms(str(data_dir), terms, max_files=args.max_files)
    print(f"Files matched after term filter: {len(matched_files)}")
    if matched_files:
        print("Sample matched files:")
        for p in matched_files[:5]:
            print("  ", p)

    evidence = retrieve_top_chunks(args.claim, str(data_dir), k=args.top_k, max_files=args.max_files)
    print(f"Retrieved evidence chunks: {len(evidence)}")
    if not evidence:
        print("No evidence found.")
        return
    verdict, reasoning, supporting = run_fact_check(args.claim, evidence)
    print(f"Verdict: {verdict}")
    print(f"Reasoning: {reasoning}")
    print("Evidence:")
    for i, ev in enumerate(evidence, 1):
        mark = "*" if i in supporting else " "
        meta = ev.get("metadata", {})
        print(f"{mark}[{i}] {ev['text']} (file: {meta.get('file')}, sim: {meta.get('similarity')})\n")


if __name__ == "__main__":
    main()
