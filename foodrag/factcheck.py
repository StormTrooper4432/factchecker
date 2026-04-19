import os
from typing import Dict, List, Tuple

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


def _extract_text(response) -> str:
    if getattr(response, "text", None):
        return response.text
    try:
        cand = response.candidates[0]
        content = getattr(cand, "content", cand)
        parts = getattr(content, "parts", []) or []
        texts = []
        for p in parts:
            if getattr(p, "text", None):
                texts.append(p.text)
            elif getattr(p, "inline_data", None):
                data = getattr(p.inline_data, "data", b"")
                try:
                    texts.append(data.decode("utf-8"))
                except Exception:
                    try:
                        texts.append(str(data))
                    except Exception:
                        pass
        return "\n".join([t for t in texts if t])
    except Exception:
        return ""


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences and leading 'json' labels."""
    t = text.strip()
    if t.startswith("```"):
        # Drop surrounding backticks and optional language line.
        t = t.strip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1]
    if t.lower().startswith("json"):
        t = t[4:].strip()
    return t.strip()


def _parse_llm_json(raw: str):
    """Best-effort JSON parsing that tolerates fences and extra text."""
    import json

    cleaned = _strip_code_fences(raw)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    # Try to find the first balanced JSON object.
    snippet = cleaned[start:]
    open_braces = 0
    end_idx = None
    for i, ch in enumerate(snippet):
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces -= 1
            if open_braces == 0:
                end_idx = i
                break
    if end_idx is not None:
        snippet = snippet[: end_idx + 1]
    else:
        # If unbalanced, pessimistically close remaining braces.
        snippet = snippet + ("}" * max(open_braces, 0))

    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError as e:
        # Last attempt: strip after last brace and retry.
        last = snippet.rfind("}")
        if last != -1:
            try:
                parsed = json.loads(snippet[: last + 1])
            except Exception:
                # Soft fallback: regex extraction of key fields from partial JSON
                import re
                verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', snippet, re.IGNORECASE)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)', snippet, re.IGNORECASE)
                supporting_match = re.search(r'"supporting_ids"\s*:\s*\[([^\]]*)\]', snippet, re.IGNORECASE)
                verdict = (verdict_match.group(1).lower() if verdict_match else "unknown")
                reasoning = reasoning_match.group(1) if reasoning_match else ""
                supporting_ids = []
                if supporting_match:
                    for token in supporting_match.group(1).split(","):
                        try:
                            supporting_ids.append(int(token))
                        except Exception:
                            continue
                return verdict, reasoning, supporting_ids
        raise ValueError(f"JSON decode failed: {e}") from e

    verdict = str(parsed.get("verdict", "unknown")).lower()
    if verdict not in {"true", "false", "unknown"}:
        verdict = "unknown"
    reasoning = str(parsed.get("reasoning", ""))
    supporting = parsed.get("supporting_ids", parsed.get("supporting", []))
    if isinstance(supporting, str):
        try:
            supporting = json.loads(supporting)
        except Exception:
            supporting = [s.strip() for s in supporting.split(",") if s.strip()]
    if not isinstance(supporting, list):
        supporting = []
    supporting_ids = []
    for s in supporting:
        try:
            supporting_ids.append(int(s))
        except Exception:
            continue
    return verdict, reasoning, supporting_ids


def _parse_llm_sources(raw: str) -> List[Dict[str, str]]:
    """Parse a sources array from an LLM JSON response."""
    import json

    cleaned = _strip_code_fences(raw)
    start = cleaned.find("{")
    if start != -1:
        snippet = cleaned[start:]
    else:
        snippet = cleaned
    try:
        parsed = json.loads(snippet)
    except Exception:
        return []
    sources = parsed.get("sources", []) if isinstance(parsed, dict) else []
    results = []
    if isinstance(sources, list):
        for src in sources:
            if not isinstance(src, dict):
                continue
            title = str(src.get("title", "")).strip() or str(src.get("source", "")).strip()
            url = str(src.get("url", "")).strip()
            quote = str(src.get("quote", "")).strip()
            if not title and url:
                title = url
            if title or url:
                results.append({"title": title, "url": url, "quote": quote})
    return results


def _has_gemini() -> bool:
    return _HAS_GENAI and bool(os.getenv("GEMINI_API_KEY"))


def build_factcheck_prompt(claim: str, evidence: List[Dict]) -> str:
    context_blocks = []
    for idx, ev in enumerate(evidence, 1):
        context_blocks.append(f"[{idx}] {ev['text']}")
    context = "\n".join(context_blocks)
    prompt = (
        "You are a nutrition and fitness fact-checker. Use ONLY the provided evidence. "
        "Return ONLY a single JSON object (no prose, no code fences) with keys: "
        "verdict (true/false/unknown), reasoning (2-4 sentences), supporting_ids (list of evidence numbers). "
        'If unsure, return {"verdict":"unknown","reasoning":"not enough info","supporting_ids":[]}. '
        f"\nClaim: {claim}\n\nEvidence:\n{context}\n\nJSON:"
    )
    return prompt


def run_fact_check(claim: str, evidence: List[Dict]) -> Tuple[str, str, List[int]]:
    if not evidence:
        return "unknown", "No evidence found.", []
    prompt = build_factcheck_prompt(claim, evidence)
    raw = ""
    attempt = 0
    debug = bool(os.getenv("GEMINI_DEBUG"))
    debug_path = os.getenv("GEMINI_DEBUG_PATH", "gemini_last_response.txt")

    def _llm_call(content: str):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=content,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
            ),
        )

    if _has_gemini():
        try:
            response = _llm_call(prompt)
            if debug:
                try:
                    with open(debug_path, "w", encoding="utf-8") as dbg:
                        dbg.write(repr(response))
                except Exception:
                    pass
            raw = _extract_text(response)
            verdict, reasoning, supporting = _parse_llm_json(raw)
            return verdict, reasoning, supporting
        except Exception:
            # One retry with a minimal prompt in case the model added prose
            attempt = 1
            try:
                minimal_prompt = (
                    "Return ONLY JSON like "
                    '{"verdict":"true|false|unknown","reasoning":"...","supporting_ids":[1,2]}. '
                    f"Claim: {claim}. Evidence:\n{prompt.split('Evidence:',1)[1]}"
                )
                response = _llm_call(minimal_prompt)
                if debug:
                    try:
                        with open(debug_path, "w", encoding="utf-8") as dbg:
                            dbg.write(repr(response))
                    except Exception:
                        pass
                raw = _extract_text(response)
                verdict, reasoning, supporting = _parse_llm_json(raw)
                return verdict, reasoning, supporting
            except Exception as e2:
                return (
                    "unknown",
                    f"LLM call failed ({type(e2).__name__}: {e2}); falling back to heuristic. Raw: {raw!r}",
                    [],
                )
    # Heuristic fallback: lexical overlap
    lowered_claim = claim.lower()
    matches = [ev for ev in evidence if overlap_ratio(lowered_claim, ev["text"].lower()) > 0.3]
    verdict = "true" if matches else "unknown"
    if _has_gemini():
        reasoning = "Heuristic verdict based on lexical overlap because the API response could not be parsed."
    else:
        reasoning = "Heuristic verdict based on lexical overlap; API key is not available in the backend environment."
    supporting = list(range(1, len(matches) + 1))
    return verdict, reasoning, supporting


def run_fact_check_llm_only(claim: str) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    LLM-only fact check that asks Gemini for verdict + reasoning + reputable sources.
    Returns verdict, reasoning, and a list of sources [{title, url, quote}].
    """
    if not _has_gemini():
        return (
            "unknown",
            "Gemini API key is not configured; cannot run LLM-only mode.",
            [],
        )

    prompt = (
        "You are a nutrition fact-checker. Using your internal knowledge ONLY (no browsing), "
        "evaluate the claim and respond with strict JSON (no code fences): "
        '{\"verdict\":\"true|false|unknown\",\"reasoning\":\"3-5 sentences\",'
        '"sources\":[{"title":"source title","url":"https://...",'
        '"quote":"short supporting quote"}]}. '
        "Use only reputable sources (NIH, CDC, WHO, Mayo Clinic, Harvard, PubMed/PMC, "
        "UpToDate, major medical journals). Do not invent URLs. "
        "If unsure, set verdict to unknown and return an empty sources list. "
        f"Claim: {claim}"
    )

    def _llm_call(content: str):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", config.GEMINI_MODEL),
            contents=content,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
            ),
        )

    raw = ""
    try:
        response = _llm_call(prompt)
        raw = _extract_text(response)
        verdict, reasoning, _supporting = _parse_llm_json(raw)
        sources = _parse_llm_sources(raw)
        return verdict, reasoning, sources
    except Exception as exc:
        return (
            "unknown",
            f"LLM-only mode failed ({type(exc).__name__}: {exc}); raw: {raw!r}",
            [],
        )


def overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
