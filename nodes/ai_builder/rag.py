# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
Retrieval over the local reference docs (docs/ai_builder/reference/*.md).

Why: Blender's Python API drifts between versions and small local models hallucinate
socket names, operator ids and deprecated calls. Grounding every generation in a curated,
version-pinned reference is the cheapest reliability win available.

How: markdown files are split on headings into chunks. Ranking is BM25 (pure Python, no
dependencies). If an Ollama embedding model is available the scores are blended with cosine
similarity; embeddings are cached in `.embeddings_cache.json` next to the docs.
"""

import glob
import hashlib
import json
import math
import os
import re
from collections import Counter

from . import config

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{1,}|\d+")
_STOP = set("the a an and or of to in for with on at by from is are be this that it as use using into".split())


def tokenize(text):
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOP]


def chunk_markdown(text, source, max_chars=1400):
    """Split on '## ' / '### ' headings, then hard-split long sections."""
    chunks = []
    current_title, buf = source, []

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        while len(body) > max_chars:
            cut = body.rfind("\n\n", 0, max_chars)
            cut = cut if cut > max_chars // 3 else max_chars
            chunks.append({"source": source, "title": current_title, "text": body[:cut].strip()})
            body = body[cut:].strip()
        chunks.append({"source": source, "title": current_title, "text": body})

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+", line):
            flush()
            buf = [line]
            current_title = line.lstrip("# ").strip()
        else:
            buf.append(line)
    flush()
    return chunks


class ReferenceIndex:
    def __init__(self, docs_dirs=None, extra_text=""):
        dirs = list(docs_dirs or [config.REFERENCE_DOCS_DIR])
        self.chunks = []
        for d in dirs:
            for path in sorted(glob.glob(os.path.join(d, "**", "*.md"), recursive=True)):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    self.chunks.extend(chunk_markdown(f.read(), os.path.relpath(path, d)))
        if extra_text and extra_text.strip():
            self.chunks.extend(chunk_markdown(extra_text, "user_notes.md"))
        self._build_bm25()
        self.cache_path = os.path.join(dirs[0], ".embeddings_cache.json") if dirs else None
        self._embeddings = None

    # ------------------------------------------------------------------ BM25
    def _build_bm25(self):
        self.doc_tokens = [tokenize(c["title"] + " " + c["text"]) for c in self.chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        df = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n = max(len(self.chunks), 1)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(t) for t in self.doc_tokens]

    def _bm25_scores(self, query, k1=1.5, b=0.75):
        q = tokenize(query)
        scores = []
        for i in range(len(self.chunks)):
            s = 0.0
            for t in q:
                if t not in self.tf[i]:
                    continue
                f = self.tf[i][t]
                s += self.idf.get(t, 0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * self.doc_len[i] / self.avg_len))
            scores.append(s)
        return scores

    # ------------------------------------------------------------------ embeddings
    def _chunk_hash(self, chunk, model):
        return hashlib.sha1((model + "|" + chunk["text"]).encode("utf-8")).hexdigest()

    def ensure_embeddings(self, client):
        if self._embeddings is not None or client is None or not self.cache_path:
            return self._embeddings
        model = client.cfg.get("embed_model") or ""
        cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        missing = [(i, c) for i, c in enumerate(self.chunks) if self._chunk_hash(c, model) not in cache]
        if missing:
            vectors = client.embed([c["text"] for _, c in missing])
            if not vectors:
                self._embeddings = False
                return None
            for (_, c), v in zip(missing, vectors):
                cache[self._chunk_hash(c, model)] = v
            try:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
            except Exception:
                pass
        self._embeddings = [cache.get(self._chunk_hash(c, model)) for c in self.chunks]
        return self._embeddings

    @staticmethod
    def _cos(a, b):
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query, k=6, client=None, always_include=("00_", "06_")):
        """Top-k chunks. Files whose name starts with `always_include` prefixes get a boost so the
        core rules and the toolbox protocol are never left out."""
        if not self.chunks:
            return []
        bm25 = self._bm25_scores(query)
        mx = max(bm25) or 1.0
        combined = [s / mx for s in bm25]
        embs = self.ensure_embeddings(client) if client is not None else None
        if embs:
            qv = client.embed([query])
            if qv:
                sims = [self._cos(qv[0], e) for e in embs]
                smax = max(sims) or 1.0
                combined = [0.5 * c + 0.5 * (s / smax) for c, s in zip(combined, sims)]
        for i, c in enumerate(self.chunks):
            if any(c["source"].startswith(p) for p in always_include):
                combined[i] += 0.15
        order = sorted(range(len(self.chunks)), key=lambda i: combined[i], reverse=True)
        return [dict(self.chunks[i], score=round(combined[i], 3)) for i in order[:k]]

    @staticmethod
    def format_context(chunks, max_chars=9000):
        out, total = [], 0
        for c in chunks:
            block = f"### [{c['source']}] {c['title']}\n{c['text']}\n"
            if total + len(block) > max_chars:
                break
            out.append(block)
            total += len(block)
        return "\n".join(out)
