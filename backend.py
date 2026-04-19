import tempfile
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, jsonify, request

from foodrag import config
from foodrag.factcheck import run_fact_check, run_fact_check_llm_only
from foodrag.hardcoded_prompts import get_available_claims

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/claims", methods=["GET"])
def list_claims():
    return jsonify({"claims": get_available_claims()})


@app.route("/api/evaluate", methods=["POST", "OPTIONS"])
def evaluate_claim():
    if request.method == "OPTIONS":
        return jsonify({})

    payload = request.get_json(silent=True) or {}
    claim = str(payload.get("claim", "")).strip()
    mode = str(payload.get("mode", "llm_only")).strip().lower()
    if not claim:
        return jsonify({"error": "Missing claim text."}), 400

    # Fast path: LLM-only mode (no retrieval)
    if mode != "hybrid":
        verdict, reasoning, sources = run_fact_check_llm_only(claim)
        evidence = []
        for idx, src in enumerate(sources, 1):
            title = src.get("title") or f"Source {idx}"
            url = src.get("url", "")
            quote = src.get("quote", "")
            source_label = ""
            if url:
                try:
                    parsed = urlparse(url)
                    if parsed.netloc:
                        source_label = parsed.netloc
                except Exception:
                    source_label = ""
            evidence.append(
                {
                    "id": idx,
                    "text": quote or title,
                    "source": source_label or title,
                    "source_url": url,
                    "similarity": 1.0,
                }
            )
        return jsonify(
            {
                "claim": claim,
                "verdict": verdict,
                "reasoning": reasoning,
                "evidence": evidence,
                "supporting_ids": [],
                "query": None,
                "pmc_error": None,
                "mode": "llm_only",
            }
        )

    try:
        from foodrag.retrieval import (
            extract_terms,
            merge_top_chunks,
            retrieve_top_chunks,
            retrieve_top_chunks_chroma,
        )
        from pmc_pull_api import search_and_download
    except Exception as exc:
        return jsonify(
            {
                "error": "Hybrid retrieval mode is unavailable in this deployment.",
                "details": f"{type(exc).__name__}: {exc}",
                "mode": "hybrid",
            }
        ), 501

    terms = extract_terms(claim)
    query_terms = []
    generic_filters = {
        "levels",
        "adults",
        "adult",
        "people",
        "high",
        "lower",
        "around",
        "with",
        "in",
        "by",
        "and",
        "the",
        "for",
        "study",
        "studies",
        "results",
        "may",
        "was",
        "were",
        "associated",
        "observed",
        "significant",
        "significantly",
    }

    for term in terms:
        tokens = set(term.split())
        if not tokens or tokens <= generic_filters:
            continue
        query_terms.append(term)

    if not query_terms:
        query_terms = terms

    def term_score(term: str) -> int:
        score = len(term)
        for keyword in [
            "omega",
            "triglyceride",
            "cholesterol",
            "egg",
            "eggs",
            "heart disease",
            "fasting",
            "supplement",
            "diet",
            "gluten",
            "celiac",
            "protein",
            "carbohydrate",
            "fatty",
            "acid",
            "bmi",
        ]:
            if keyword in term:
                score += 10
        if any(ch.isdigit() for ch in term):
            score += 5
        return score

    query_terms = sorted(query_terms, key=term_score, reverse=True)
    query = " AND ".join(query_terms[:3]) if query_terms else claim

    with tempfile.TemporaryDirectory(prefix="pmc_live_") as tmpdir:
        tmp_path = Path(tmpdir)
        saved = 0
        pmc_error = None
        try:
            saved = search_and_download(query, 20, tmp_path)
        except Exception as exc:  # network/DNS/API hiccups shouldn't crash the API
            pmc_error = f"PMC search failed: {type(exc).__name__}: {exc}"
            app.logger.warning(pmc_error)
        evidence_objs = []
        source_label = "PMC OA"

        pmc_objs = []
        if saved > 0:
            pmc_objs = retrieve_top_chunks(claim, str(tmp_path), k=5, max_files=100)

        crawl_objs = []
        crawl_chunks_dir = Path("data/crawl_chunks")
        if crawl_chunks_dir.exists():
            crawl_objs = retrieve_top_chunks(claim, str(crawl_chunks_dir), k=5, max_files=100)
            source_label = "curated crawl chunks"
        else:
            crawl_dir = Path("data/crawl")
            if crawl_dir.exists():
                crawl_objs = retrieve_top_chunks(claim, str(crawl_dir), k=5, max_files=100)
                source_label = "curated crawl"

        chroma_objs = retrieve_top_chunks_chroma(claim, k=5)

        sources = [pmc_objs, crawl_objs, chroma_objs]
        available_sources = [s for s in sources if s]
        if available_sources:
            evidence_objs = merge_top_chunks(available_sources, k=5)
        else:
            evidence_objs = []

        if not evidence_objs:
            baseline_dir = Path(config.DATA_DIR)
            if baseline_dir.exists():
                evidence_objs = retrieve_top_chunks(claim, str(baseline_dir), k=5, max_files=100)
                source_label = "baseline corpus"

        if not evidence_objs:
            return jsonify(
                {
                    "claim": claim,
                    "verdict": "unknown",
                    "reasoning": (
                        "No relevant evidence was found from PMC OA, curated crawl data, or baseline local corpus."
                    ),
                    "evidence": [],
                }
            ), 404

        verdict, reasoning, supporting_ids = run_fact_check(claim, evidence_objs)
        evidence = []

        def _friendly_source_name(file_path: str, source_url: str) -> str:
            if source_url:
                try:
                    parsed = urlparse(source_url)
                    if parsed.netloc:
                        return parsed.netloc
                except Exception:
                    pass
                return source_url

            path_obj = Path(file_path)
            file_name = path_obj.name
            if file_name.startswith("PMC") and file_name.endswith(".nxml"):
                return f"PMC {file_name[:-5]}"
            if file_name.startswith("PMC") and file_name.endswith(".txt"):
                return f"PMC {file_name[:-4]}"

            parts = path_obj.parts
            if "crawl_chunks" in parts:
                try:
                    idx = parts.index("crawl_chunks")
                    if idx + 1 < len(parts):
                        return parts[idx + 1].replace("_", " ")
                except ValueError:
                    pass

            return file_name

        for idx, ev in enumerate(evidence_objs, 1):
            source_path = ev["metadata"].get("file", "")
            source_url = ev["metadata"].get("source_url", "")
            if not source_url and source_path:
                source_file = Path(source_path)
                stem = source_file.stem
                if stem.startswith("PMC") and source_file.suffix.lower() in {".txt", ".nxml", ".xml"}:
                    source_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{stem}/"

            source = _friendly_source_name(source_path, source_url)
            text = ev["text"].strip()
            if len(text) > 260:
                text = text[:260].rsplit(" ", 1)[0] + "..."
            evidence.append(
                {
                    "id": idx,
                    "text": text,
                    "source": source,
                    "source_url": source_url,
                    "similarity": ev["metadata"]["similarity"],
                }
            )

    return jsonify(
        {
            "claim": claim,
            "verdict": verdict,
            "reasoning": reasoning,
            "evidence": evidence,
            "supporting_ids": supporting_ids,
            "query": query,
            "pmc_error": pmc_error,
            "mode": "hybrid",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
