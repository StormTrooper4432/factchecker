import re
from typing import List


SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_END_RE.split(text) if s.strip()]


def chunk_by_sentences(text: str, sentences_per_chunk: int) -> List[str]:
    sentences = split_sentences(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def clean_text_regex(text: str) -> str:
    # Basic cleanup: collapse whitespace, drop control chars.
    text = re.sub(r"\s+", " ", text)
    return text.strip()
