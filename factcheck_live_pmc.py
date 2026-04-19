"""
End-to-end fact checking that:
 1) Extracts keywords from the claim.
 2) Uses PMC E-utilities to fetch OA articles matching the keywords.
 3) Runs on-the-fly chunking/embedding retrieval over those fresh articles.
 4) Calls the LLM for a verdict.

Prints status at each step so you see progress.

Usage:
  GEMINI_API_KEY=... python -B factcheck_live_pmc.py "Omega-3 supplements lower triglyceride levels by around 20–30% in adults with high fasting triglycerides." \
    --max-api 40 --max-files 200 --top-k 8
"""

import argparse
import tempfile
from pathlib import Path

from foodrag.retrieval import extract_terms, merge_top_chunks, retrieve_top_chunks
from foodrag.factcheck import run_fact_check
from foodrag import config
from pmc_pull_api import search_and_download


def main():
    parser = argparse.ArgumentParser(description="Live PMC pull + fact check.")
    parser.add_argument("claim", help="Claim to fact-check.")
    parser.add_argument("--max-api", type=int, default=40, help="Max articles to pull from PMC for this claim.")
    parser.add_argument("--max-files", type=int, default=200, help="Max files to scan for retrieval.")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k evidence chunks to send to LLM.")
    parser.add_argument(
        "--fallback-data-dir",
        default=config.DATA_DIR,
        help="Final fallback corpus (default: existing baseline corpus).",
    )
    parser.add_argument(
        "--crawl-dir",
        default="data/crawl",
        help="Optional intermediate fallback: crawled open-web text folder.",
    )
    args = parser.parse_args()

    print(f"[1/4] Extracting keywords from claim...")
    terms = extract_terms(args.claim)
    # Build an AND query with anchor-first to avoid over-strict noise
    if terms:
        anchor_terms = [t for t in terms if "gluten" in t or "celiac" in t or "weight" in t]
        anchor = anchor_terms[0] if anchor_terms else terms[0]
        extras = [t for t in terms if t != anchor][:2]
        key_terms = [anchor] + extras
    else:
        key_terms = [args.claim]
    query = " AND ".join(key_terms)
    print(f"    Terms: {terms}")
    print(f"    Using key terms: {key_terms}")
    print(f"    PMC query: {query!r}")

    # Use a temp dir for the fetched articles
    with tempfile.TemporaryDirectory(prefix="pmc_live_") as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"[2/4] Fetching up to {args.max_api} PMC OA articles into {tmp_path} ...")
        try:
            saved = search_and_download(query, args.max_api, tmp_path)
        except Exception as exc:
            print(f"    PMC fetch failed ({type(exc).__name__}: {exc}); continuing with local corpora.")
            saved = 0
        print(f"    Downloaded {saved} articles.")
        if saved == 0:
            print("No articles downloaded from PMC; will rely on local corpora if available.")

        print(f"[3/4] Retrieving top chunks from downloaded articles...")
        pmc_evidence = retrieve_top_chunks(
            args.claim, data_dir=str(tmp_path), k=args.top_k, max_files=args.max_files
        )
        print(f"    Retrieved {len(pmc_evidence)} evidence chunks from PMC.")

        crawl_evidence = []
        crawl_chunks = Path("data/crawl_chunks")
        if crawl_chunks.exists():
            crawl_evidence = retrieve_top_chunks(
                args.claim,
                data_dir=str(crawl_chunks),
                k=args.top_k,
                max_files=args.max_files,
            )
            print(f"    Retrieved {len(crawl_evidence)} evidence chunks from pre-chunked crawl corpus.")
        elif Path(args.crawl_dir).exists():
            crawl_evidence = retrieve_top_chunks(
                args.claim,
                data_dir=args.crawl_dir,
                k=args.top_k,
                max_files=args.max_files,
            )
            print(f"    Retrieved {len(crawl_evidence)} evidence chunks from raw crawled corpus.")

        if pmc_evidence and crawl_evidence:
            evidence = merge_top_chunks([pmc_evidence, crawl_evidence], k=args.top_k)
        elif pmc_evidence:
            evidence = pmc_evidence
        else:
            evidence = crawl_evidence

        print(f"    Using {len(evidence)} evidence chunks after merging sources.")

        if not evidence:
            # Baseline fallback
            print(f"No evidence from PMC or crawl; falling back to baseline corpus {args.fallback_data_dir} ...")
            evidence = retrieve_top_chunks(
                args.claim,
                data_dir=args.fallback_data_dir,
                k=args.top_k,
                max_files=args.max_files,
            )
            print(f"    Retrieved {len(evidence)} evidence chunks from baseline corpus.")

            if not evidence:
                # Final broad PMC retry with OR query to force some content
                broad_terms = terms or [args.claim]
                broad_query = " OR ".join(broad_terms[:4])
                print(f"No evidence found; broadening PMC query to {broad_query!r} ...")
                saved2 = search_and_download(broad_query, args.max_api, tmp_path)
                print(f"    Downloaded {saved2} articles on broad retry.")
                evidence = retrieve_top_chunks(
                    args.claim, data_dir=str(tmp_path), k=args.top_k, max_files=args.max_files
                )
                print(f"    Retrieved {len(evidence)} evidence chunks after broad retry.")
                if not evidence:
                    print("No evidence found anywhere; aborting.")
                    return

        print(f"[4/4] Asking LLM for verdict...")
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
