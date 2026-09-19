"""Minimal RAG over the local travel guides in guides/*.md.

Pure-Python TF-IDF + cosine similarity over character n-grams — no heavy deps,
and character n-grams work for Chinese text without a word segmenter.
Implementing it by hand (rather than importing sklearn) keeps the demo light
and makes the retrieval mechanism explicit.
"""
import glob
import math
import os
import re
from collections import Counter

_GUIDE_DIR = os.path.join(os.path.dirname(__file__), "guides")


def _ngrams(text: str, lo: int = 2, hi: int = 3) -> list:
    text = re.sub(r"\s+", "", text)
    grams = []
    for n in range(lo, hi + 1):
        grams.extend(text[i:i + n] for i in range(len(text) - n + 1))
    return grams


def _load_chunks():
    """Split each guide into paragraph-sized chunks tagged with their source file."""
    chunks, sources = [], []
    for path in sorted(glob.glob(os.path.join(_GUIDE_DIR, "*.md"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for para in (p.strip() for p in text.split("\n\n")):
            if len(para) >= 8:
                chunks.append(para)
                sources.append(name)
    return chunks, sources


_CHUNKS, _SOURCES = _load_chunks()

# Build IDF over all chunks, then a tf-idf vector (dict) + norm per chunk.
_N = len(_CHUNKS)
_df = Counter()
_chunk_grams = []
for _c in _CHUNKS:
    grams = _ngrams(_c)
    _chunk_grams.append(grams)
    for g in set(grams):
        _df[g] += 1
_idf = {g: math.log((_N + 1) / (df + 1)) + 1 for g, df in _df.items()}


def _vectorize(grams: list) -> tuple:
    vec = {}
    for g, tf in Counter(grams).items():
        if g in _idf:
            vec[g] = tf * _idf[g]
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return vec, norm


_CHUNK_VECS = [_vectorize(g) for g in _chunk_grams]


def retrieve(query: str, k: int = 3) -> str:
    """Return the top-k most relevant guide passages, joined for the system prompt."""
    if not _CHUNKS:
        return ""
    qvec, qnorm = _vectorize(_ngrams(query))
    scored = []
    for i, (cvec, cnorm) in enumerate(_CHUNK_VECS):
        # dot product over the smaller dict
        small, big = (qvec, cvec) if len(qvec) < len(cvec) else (cvec, qvec)
        dot = sum(w * big.get(g, 0.0) for g, w in small.items())
        scored.append((dot / (qnorm * cnorm), i))
    scored.sort(reverse=True)
    picked = [f"【{_SOURCES[i]}】{_CHUNKS[i]}" for score, i in scored[:k] if score > 0]
    return "\n".join(picked)
