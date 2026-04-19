"""
Lightweight crawl of open, license-friendly health/nutrition domains.

Purpose: Grab a small batch of authoritative pages (HTML → plain text) for the
fact-checker corpus without hard-coding topics.

Defaults (tuned to finish quickly):
- Allowlist domains: NIH, CDC, AHA, USPSTF, Nutrition.gov, Mayo Clinic, ACE, ACSM, NSCA, NHS
- max_depth = 1 (follow only direct links from the seed page)
- max_pages_per_domain = 10
- max_total_pages = 200

Output:
- Saves cleaned text files under data/crawl/<domain>/page_<n>.txt
- Skips binary/PDF to stay fast.

Usage:
  python -B crawl_open_domains.py
  python -B crawl_open_domains.py --max-total 20 --max-domain 5
"""

import argparse
import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

ALLOW_DOMAINS = [
    "nih.gov",
    "cdc.gov",
    "heart.org",
    "uspreventiveservicestaskforce.org",
    "nutrition.gov",
    "mayoclinic.org",
    "acefitness.org",
    "acsm.org",
    "nsca.com",
    "nhs.uk",
]

DEFAULT_SEEDS = [
    "https://www.nhlbi.nih.gov/health/heart-healthy-living",
    "https://www.cdc.gov/nutrition/index.html",
    "https://www.cdc.gov/physicalactivity/index.html",
    "https://www.heart.org/en/healthy-living",
    "https://www.acefitness.org/education-and-resources/lifestyle",
    "https://www.acsm.org/read-research",
    "https://www.nsca.com/education/articles/",
    "https://www.mayoclinic.org/healthy-lifestyle",
    "https://www.nutrition.gov/topics/whats-food",
    "https://www.nhs.uk/live-well/",
]

USER_AGENT = "NutritionFactChecker/0.1 (+https://example.com)"


def same_domain(url: str, allowed: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host.endswith(d) for d in allowed)


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    query = "&".join(sorted([q for q in parsed.query.split("&") if q]))
    return urlunparse((scheme, netloc, path, "", query, ""))


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles/nav/footer
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def crawl(seeds, outdir: Path, max_depth: int, max_domain: int, max_total: int, timeout: float = 10.0):
    outdir.mkdir(parents=True, exist_ok=True)
    seen = set()
    per_domain = {d: 0 for d in ALLOW_DOMAINS}
    total = 0
    queue = deque([(url, 0) for url in seeds])

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    while queue and total < max_total:
        url, depth = queue.popleft()
        normalized = normalize_url(url)
        if normalized in seen or depth > max_depth:
            continue
        if not same_domain(url, ALLOW_DOMAINS):
            continue
        domain = next(d for d in ALLOW_DOMAINS if urlparse(url).netloc.lower().endswith(d))
        if per_domain.get(domain, 0) >= max_domain:
            continue

        try:
            resp = session.get(url, timeout=timeout)
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            text = clean_text(resp.text)
            if len(text) < 300:
                continue
            domain_dir = outdir / domain.replace(".", "_")
            domain_dir.mkdir(parents=True, exist_ok=True)
            page_num = per_domain[domain] + 1
            fname = domain_dir / f"page_{page_num}.txt"
            meta_fname = domain_dir / f"page_{page_num}.meta.json"
            fname.write_text(text, encoding="utf-8")
            meta_fname.write_text(json.dumps({"url": url, "domain": domain}, ensure_ascii=False), encoding="utf-8")
            per_domain[domain] += 1
            total += 1
            print(f"Saved {fname} (len={len(text)})")
        except Exception as e:
            print(f"Skip {url}: {e}", file=sys.stderr)
            continue

        seen.add(normalized)

        # enqueue children
        if depth < max_depth:
            try:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("#") or href.startswith("mailto:"):
                        continue
                    child = urljoin(url, href)
                    child_norm = normalize_url(child)
                    if same_domain(child, ALLOW_DOMAINS) and child_norm not in seen:
                        queue.append((child, depth + 1))
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Quick crawl of open health/nutrition domains.")
    parser.add_argument("--outdir", default="data/crawl", help="Where to store text outputs.")
    parser.add_argument("--max-depth", type=int, default=15, help="Max link depth to follow.")
    parser.add_argument("--max-domain", type=int, default=20, help="Max pages per domain.")
    parser.add_argument("--max-total", type=int, default=200, help="Max total pages.")
    args = parser.parse_args()

    print(f"Starting crawl (max_total={args.max_total}, max_domain={args.max_domain}, max_depth={args.max_depth})")
    crawl(DEFAULT_SEEDS, Path(args.outdir), args.max_depth, args.max_domain, args.max_total)
    print("Done.")


if __name__ == "__main__":
    main()
