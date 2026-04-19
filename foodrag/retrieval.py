import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup

# Gemini import with fallback
try:
    from google import genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    try:
        import google.genai as genai  # type: ignore
        _HAS_GENAI = True
    except Exception:
        genai = None  # type: ignore
        _HAS_GENAI = False

from . import config
from .storage import get_client, get_collection
from .text_utils import split_sentences

GENERIC_TERM_STOPWORDS = {
    "for",
    "everyone",
    "every",
    "day",
    "and",
    "the",
    "a",
    "in",
    "of",
    "to",
    "with",
    "on",
    "by",
    "is",
    "are",
    "be",
    "was",
    "were",
    "may",
    "this",
    "that",
    "study",
    "studies",
    "risk",
    "risks",
    "results",
    "associated",
    "observed",
    "significant",
    "significantly",
    "increases",
    "raises",
}
# Short domain-relevant tokens we never want to drop even though length <= 3
SHORT_TERM_WHITELIST = {"egg", "eggs", "bmi", "fat"}


def _has_gemini() -> bool:
    return _HAS_GENAI and bool(os.getenv("GEMINI_API_KEY"))


def _extract_text(resp) -> str:
    if getattr(resp, "text", None):
        return resp.text
    try:
        cand = resp.candidates[0]
        content = getattr(cand, "content", cand)
        parts = getattr(content, "parts", []) or []
        return "\n".join([getattr(p, "text", "") for p in parts if getattr(p, "text", "")])
    except Exception:
        return ""


def extract_terms(claim: str) -> List[str]:
    # Capture hyphenated, numeric, and space-separated key phrases (e.g., "omega-3 supplements").
    words_raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", claim)
    phrases = []
    for i in range(len(words_raw) - 1):
        phrases.append(f"{words_raw[i]} {words_raw[i+1]}")
    STOP = {
        "for everyone",
        "everyone",
        "healthier and",
        "and promotes",
        "promotes",
        "loss for",
        "weight",
        "loss",
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
    base_terms = {
        w.lower()
        for w in words_raw
        if (len(w) > 3 or w.lower() in SHORT_TERM_WHITELIST) and w.lower() not in STOP
    }
    base_terms |= {
        p.lower()
        for p in phrases
        if len(p) > 5 and not all(tok.lower() in STOP for tok in p.split())
    }
    if _has_gemini():
        try:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            resp = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", config.GEMINI_MODEL),
                contents=(
                    "Extract 6-12 salient nutrition keywords/phrases for retrieval. "
                    "Return JSON array of lowercase strings. Claim: " + claim
                ),
                config=genai.types.GenerateContentConfig(max_output_tokens=80, temperature=0.0),
            )
            raw = _extract_text(resp)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                llm_terms = [str(t).lower() for t in parsed if str(t).strip()]
                return llm_terms + list(base_terms - set(llm_terms))
        except Exception:
            pass
    # heuristic fallback
    seen = []
    for t in base_terms:
        if t not in seen:
            seen.append(t)
    return seen[:12]


def _normalize(text: str) -> str:
    # Normalize various hyphens/dashes to ASCII hyphen and lowercase.
    return (
        text.replace("‐", "-")
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
        .lower()
    )


def _clean_latex_and_math(text: str) -> str:
    # Strip LaTeX/math markup and common PMC XML artifacts.
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", text)
    text = re.sub(r"\$\$.*?\$\$", "", text)
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_text(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".xml", ".nxml"}:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                soup = BeautifulSoup(txt, "xml")
        except Exception:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                soup = BeautifulSoup(txt, "html.parser")
        for tag in soup.find_all(["formula", "inline-formula", "tex-math", "disp-formula", "graphic", "xref", "table-wrap"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        txt = " ".join([p for p in paragraphs if p])
        txt = _clean_latex_and_math(txt)
    return txt


CHUNK_FILE_SUFFIX = ".chunk.txt"


def _is_quality_chunk(text: str) -> bool:
    if len(text.split()) < 8:
        return False
    if len(re.findall(r"\\\\|documentclass|usepackage|begin\{|end\{|tex-math|\\\[|\\\]|\\\(|\\\)", text)) > 0:
        return False
    if len(re.findall(r"[^a-zA-Z0-9 .,;:'\"()\-]", text)) / max(len(text), 1) > 0.15:
        return False
    if text.count("figure") > 3 or text.count("table") > 3:
        return False
    return True


def _is_chunk_file(path: Path) -> bool:
    return path.name.endswith(CHUNK_FILE_SUFFIX)


def chunk_text(text: str) -> List[str]:
    sentences = split_sentences(text)
    chunks = []
    for i in range(0, len(sentences), config.SENTENCES_PER_CHUNK):
        chunk = " ".join(sentences[i : i + config.SENTENCES_PER_CHUNK]).strip()
        if chunk and _is_quality_chunk(chunk):
            chunks.append(chunk)
    return chunks


def write_chunked_corpus(source_dir: str, target_dir: str, max_files: int = 2000) -> int:
    root = Path(source_dir)
    out_root = Path(target_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    files = list(root.rglob("*.txt"))
    written = 0
    for idx, source_path in enumerate(files):
        if idx >= max_files:
            break
        if source_path.name.endswith(CHUNK_FILE_SUFFIX):
            continue
        try:
            text = read_text(source_path)
        except Exception:
            continue
        if not text or len(text.split()) < 30:
            continue
        chunks = chunk_text(_normalize(text))
        if not chunks:
            continue
        relative = source_path.relative_to(root)
        domain_dir = out_root / relative.parent
        domain_dir.mkdir(parents=True, exist_ok=True)
        metadata = {}
        meta_source = source_path.with_suffix(".meta.json")
        if meta_source.exists():
            try:
                metadata = json.loads(meta_source.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                metadata = {}
        for chunk_index, chunk in enumerate(chunks, start=1):
            out_path = domain_dir / f"{source_path.stem}_chunk_{chunk_index}{CHUNK_FILE_SUFFIX}"
            out_path.write_text(chunk, encoding="utf-8")
            chunk_meta = {
                "source_url": metadata.get("url", ""),
                "domain": metadata.get("domain", ""),
                "source_file": str(source_path),
                "chunk_index": chunk_index,
            }
            out_path.with_suffix(".meta.json").write_text(json.dumps(chunk_meta, ensure_ascii=False), encoding="utf-8")
            written += 1
    return written


def _token_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9\-']*", text.lower())
    return {t for t in tokens if len(t) > 2}


def _is_generic_term(term: str) -> bool:
    tokens = [t for t in term.split() if t]
    if not tokens:
        return True
    content_tokens = [t for t in tokens if t not in GENERIC_TERM_STOPWORDS]
    if any(t in SHORT_TERM_WHITELIST for t in tokens):
        return False
    if not content_tokens:
        return True
    return all(len(token) <= 3 for token in content_tokens)


def _bigram_set(text: str) -> set[str]:
    tokens = [t for t in re.findall(r"[a-z0-9][a-z0-9\-']*", text.lower()) if len(t) > 2]
    return {" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)}


def _phrase_overlap(chunk_str: str, terms: List[str]) -> float:
    if not terms:
        return 0.0
    weighted_matches = 0
    total_weight = 0
    for term in terms:
        weight = len(term.split())
        total_weight += weight
        if term in chunk_str:
            weighted_matches += weight
    return weighted_matches / max(total_weight, 1)


DOMAIN_URL_OVERRIDES = {
    "nih_gov": "https://www.nih.gov",
    "cdc_gov": "https://www.cdc.gov",
    "heart_org": "https://www.heart.org",
    "uspreventiveservicestaskforce_org": "https://www.uspreventiveservicestaskforce.org",
    "nutrition_gov": "https://www.nutrition.gov",
    "mayoclinic_org": "https://www.mayoclinic.org",
    "acefitness_org": "https://www.acefitness.org",
    "acsm_org": "https://www.acsm.org",
    "nsca_com": "https://www.nsca.com",
    "nhs_uk": "https://www.nhs.uk",
}


def _load_chunk_metadata(path: Path) -> Dict[str, str]:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _infer_source_url(path: Path) -> str:
    parts = path.parts
    if "data" in parts and "crawl_chunks" in parts:
        try:
            idx = parts.index("crawl_chunks")
            domain_key = parts[idx + 1]
            return DOMAIN_URL_OVERRIDES.get(domain_key, "")
        except Exception:
            return ""
    return ""


def _normalize_terms_for_matching(text: str) -> str:
    return (
        text.replace("‐", "-")
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
        .lower()
    )


def find_files_with_terms(data_dir: str, terms: List[str], max_files: int) -> List[Path]:
    root = Path(data_dir)
    files = [f for f in root.rglob("*.txt") if not _is_chunk_file(f)]
    files += list(root.rglob("*.chunk.txt"))
    files += list(root.rglob("*.xml")) + list(root.rglob("*.nxml"))
    if not terms:
        return files[:max_files]
    normalized_terms = [_normalize_terms_for_matching(t) for t in terms]
    anchor_keywords = [
        "egg",
        "eggs",
        "cholesterol",
        "diet",
        "heart disease",
        "blood pressure",
        "diabetes",
        "gluten",
        "celiac",
        "triglyceride",
        "saturated",
        "fatty",
        "protein",
        "carbohydrate",
        "obesity",
        "bmi",
        "stroke",
        "cardiovascular",
        "sodium",
        "salt",
    ]
    anchor_terms = [t for t in normalized_terms if any(a in t for a in anchor_keywords)]
    other_terms = [t for t in normalized_terms if t not in anchor_terms]
    key_terms = (anchor_terms + other_terms)[:5]
    matches = []
    for f in files:
        try:
            # Read only first 1MB for term check to speed up filtering
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                snippet = fh.read(1_000_000)
        except Exception:
            continue
        snippet = _normalize_terms_for_matching(snippet)
        if anchor_terms and not any(a in snippet for a in anchor_terms):
            continue
        if key_terms and not any(t in snippet for t in key_terms):
            continue
        matches.append(f)
        if len(matches) >= max_files:
            break
    return matches


def retrieve_top_chunks(claim: str, data_dir: str, k: int = config.TOP_K, max_files: int = config.MAX_FILES) -> List[Dict]:
    terms = extract_terms(claim)
    filtered_terms = [
        t for t in terms if not _is_generic_term(t) and (len(t) > 3 or t in SHORT_TERM_WHITELIST)
    ]
    if filtered_terms:
        terms_for_matching = filtered_terms
    else:
        terms_for_matching = terms

    files = find_files_with_terms(data_dir, terms_for_matching, max_files=max_files)
    if not files:
        return []

    claim_tokens = _token_set(_normalize(claim))
    claim_bigrams = _bigram_set(_normalize(claim))
    results = []
    for f in files:
        try:
            text = read_text(f)
        except Exception:
            continue
        normed_text = _normalize(text)
        if _is_chunk_file(f):
            chunks = [normed_text]
        else:
            chunks = chunk_text(normed_text)
        if not chunks:
            continue
        metadata = _load_chunk_metadata(f)
        source_url = metadata.get("source_url", metadata.get("url", ""))
        if not source_url:
            source_url = _infer_source_url(f)
        for idx, chunk_str in enumerate(chunks):
            chunk_tokens = _token_set(chunk_str)
            chunk_bigrams = _bigram_set(chunk_str)
            if not claim_tokens or not chunk_tokens:
                score = 0.0
            else:
                token_overlap = len(claim_tokens & chunk_tokens) / (len(claim_tokens) + 1e-8)
                bigram_overlap = len(claim_bigrams & chunk_bigrams) / (len(claim_bigrams) + 1e-8)
                phrase_overlap = _phrase_overlap(chunk_str, terms_for_matching)
                score = 0.15 * token_overlap + 0.1 * bigram_overlap + 0.75 * phrase_overlap
            results.append(
                {
                    "id": f"{f.name}_chunk_{idx}",
                    "text": chunk_str,
                    "metadata": {
                        "file": str(f),
                        "chunk_index": idx,
                        "similarity": float(score),
                        "terms": terms,
                        "source_url": source_url,
                    },
                    "score": float(score),
                }
            )
    results.sort(key=lambda x: x["score"], reverse=True)
    filtered = [r for r in results if r["score"] >= config.SIM_THRESHOLD]
    if not filtered:
        return []
    return filtered[:k]


def retrieve_top_chunks_chroma(claim: str, k: int = config.TOP_K) -> List[Dict]:
    try:
        client = get_client()
        collection = get_collection(client)
        response = collection.query(query_texts=[claim], n_results=k)
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
    except Exception:
        return []

    results = []
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        source_file = meta.get("file", "")
        source_url = meta.get("source_url", meta.get("url", ""))
        if not source_url and source_file:
            source_url = _infer_source_url(Path(source_file))
        similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
        results.append(
            {
                "id": f"chroma_{idx}",
                "text": doc,
                "metadata": {
                    "file": source_file,
                    "chunk_index": idx,
                    "similarity": similarity,
                    "source_url": source_url,
                },
                "score": float(similarity),
            }
        )
    return results


def merge_top_chunks(results_list: List[List[Dict]], k: int = config.TOP_K) -> List[Dict]:
    seen = set()
    merged = []
    for results in results_list:
        for r in results:
            key = (r["metadata"].get("file"), r["metadata"].get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:k]
