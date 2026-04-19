"""
Download Open Access PMC articles by keyword using NCBI E-utilities.

Examples:
  python -B pmc_pull_api.py --query "triglyceride" --max 50 --outdir data/oa_api
  python -B pmc_pull_api.py --query "vitamin D fracture" --max 30

Notes:
- Uses esearch on db=pmc with the Open Access filter.
- Downloads individual articles via efetch (rettype=full, retmode=xml) and saves
  each as PMC<id>.nxml under the chosen output directory.
- Saved XML is directly readable by the existing ingestion pipeline.

Can also be imported; see `search_and_download`.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import requests


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def esearch(query: str, retmax: int) -> List[str]:
    """Return a list of PMC IDs matching the query within the Open Access subset."""
    term = f"{query} open access[filter]"
    params = {
        "db": "pmc",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
    }
    try:
        resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except requests.RequestException as exc:  # network/DNS/API hiccups
        print(f"[pmc] esearch failed: {exc}", file=sys.stderr)
        return []


def fetch_one(pmcid: str, outdir: Path) -> bool:
    """Download a single PMC article as NXML. Returns True if saved."""
    params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": "full",
        "retmode": "xml",
    }
    try:
        resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
        resp.raise_for_status()
        if not resp.text.strip():
            return False
        fname = outdir / f"PMC{pmcid}.nxml"
        fname.write_text(resp.text, encoding="utf-8")
        return True
    except requests.RequestException as exc:
        print(f"[pmc] fetch failed for PMC{pmcid}: {exc}", file=sys.stderr)
        return False


def search_and_download(query: str, retmax: int, outdir: Path, delay: float = 0.35) -> int:
    """Search PMC OA and download up to retmax articles. Returns count saved."""
    outdir.mkdir(parents=True, exist_ok=True)
    ids = esearch(query, retmax)
    saved = 0
    for pmcid in ids:
        ok = fetch_one(pmcid, outdir)
        if ok:
            saved += 1
        time.sleep(delay)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download OA PMC articles by keyword.")
    parser.add_argument("--query", required=True, help="Keyword phrase to search PMC OA subset.")
    parser.add_argument("--max", type=int, default=50, help="Maximum number of articles to fetch.")
    parser.add_argument(
        "--outdir",
        default="data/oa_api",
        help="Directory to store downloaded NXML files.",
    )
    parser.add_argument("--delay", type=float, default=0.35, help="Seconds between requests (NCBI courtesy).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Searching PMC OA for: {args.query!r} (max {args.max})")
    ids = esearch(args.query, args.max)
    if not ids:
        print("No PMC IDs found.")
        return

    print(f"Found {len(ids)} IDs. Downloading to {outdir} ...")
    saved = 0
    for pmcid in ids:
        ok = fetch_one(pmcid, outdir)
        if ok:
            saved += 1
        time.sleep(args.delay)
    print(f"Done. Saved {saved} articles to {outdir}")


if __name__ == "__main__":
    main()
