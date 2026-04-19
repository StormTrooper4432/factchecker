"""
Pipeline:
1) LLM extracts important terms from the claim.
2) Filter files containing those terms.
3) Chunk with LangChain (fallback splitter).
4) Embed and score; keep top 5 chunks.
5) Fact-check via LLM (fallback heuristic).
"""

import argparse

from foodrag import config
from foodrag.retrieval import retrieve_top_chunks
from foodrag.factcheck import run_fact_check


def main():
    parser = argparse.ArgumentParser(description="Fact-check a claim using top-5 chunks from local data.")
    parser.add_argument("claim", help="Claim to evaluate.")
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="Root directory of text files.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="Number of evidence chunks.")
    args = parser.parse_args()

    evidence = retrieve_top_chunks(args.claim, args.data_dir, k=args.top_k)
    if not evidence:
        print("No evidence found.")
        return
    verdict, reasoning, supporting = run_fact_check(args.claim, evidence)
    print(f"Verdict: {verdict}")
    print(f"Reasoning: {reasoning}")
    print("Evidence used:")
    for idx, ev in enumerate(evidence, 1):
        mark = "*" if idx in supporting else " "
        meta = ev.get("metadata", {})
        print(f"{mark}[{idx}] {ev['text']} (file: {meta.get('file')}, sim: {meta.get('similarity')})")


if __name__ == "__main__":
    main()
