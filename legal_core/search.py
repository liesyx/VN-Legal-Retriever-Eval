"""Thuật toán tìm — cấu hình thắng benchmark: hybrid_weighted_a0.7 + rerank20_title_mix0.5.
Dùng chung cho production (Qdrant) và luồng build (đọc file, để kiểm tra).

  1. Dense (AITeamVN) + BM25 (âm tiết + bigram) trên CÂU HỎI ĐẦY ĐỦ, mỗi nhánh 1.000 chunk.
  2. 0,7·dense + 0,3·BM25 (min-max từng nhánh về [0,05; 1]).
  3. Reranker chấm 20 chunk đầu bằng CÂU NGẮN (n8n viết bằng LLM; trống -> lấy câu hỏi chính trong đoạn).
  4. Điểm 20 chunk đó = 0,5·reranker + 0,5·hybrid (min-max trên 20 chunk); chunk sau hạng 20 giữ thứ tự hybrid.
  5. Gộp chunk -> Điều, trả top-k Điều kèm toàn văn.
"""
from __future__ import annotations

import gzip
import json
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .fusion import _minmax, weighted_fusion
from .text import syllable_bigrams


def qdrant_client(url: str, api_key: str | None = None, timeout: int = 30):
    """QdrantClient mặc định cổng 6333 khi URL không ghi cổng; Qdrant sau reverse proxy (Coolify, Caddy…) chạy 443/80."""
    from urllib.parse import urlparse
    from qdrant_client import QdrantClient
    u = urlparse(url)
    port = u.port or (443 if u.scheme == "https" else 80 if u.scheme == "http" and u.hostname not in ("localhost", "127.0.0.1") else 6333)
    return QdrantClient(url=url, port=port, api_key=api_key, timeout=timeout)


@dataclass
class Config:
    emb_model: str = "AITeamVN/Vietnamese_Embedding"
    rerank_model: str = "AITeamVN/Vietnamese_Reranker"
    device: str | None = None                    # None = cuda nếu có
    # Tham số đã chốt trên tập dev của benchmark — không dò lại ở đây
    topn: int = 1000
    alpha: float = 0.7
    floor: float = 0.05
    k1: float = 2.0
    b: float = 0.75
    k3: float = 8.0
    depth: int = 20                              # 0 = không rerank (chỉ hybrid, không nạp model rerank)
    beta: float = 0.5


class SparseQuery:
    """Biến câu hỏi thành sparse vector BM25 (phía câu hỏi). Bảng từ do luồng build xuất ra (bm25_vocab_vN.json.gz).
    Làm đúng như CountVectorizer(analyzer=syllable_bigrams).transform + bão hoà k3 của BM25 benchmark."""

    def __init__(self, vocab: dict, k3: float, collection: str):
        self.vocab, self.k3, self.collection = vocab, k3, collection

    @classmethod
    def load(cls, path: str) -> "SparseQuery":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], d["k3"], d["collection"])

    def save(self, path: str, **meta):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"collection": self.collection, "k3": self.k3, **meta, "vocab": self.vocab}, f, ensure_ascii=False)

    def encode(self, question: str):
        tf = Counter(t for t in syllable_bigrams(question) if t in self.vocab)
        idx = np.array([self.vocab[t] for t in tf], dtype=np.int64)
        c = np.array(list(tf.values()), dtype=np.float32)
        return idx, ((self.k3 + 1) * c / (self.k3 + c)).astype(np.float32)


def fuse(dense_ids, dense_sc, lex_ids, lex_sc, cfg: Config):
    """0,7·dense + 0,3·BM25 (min-max có sàn). Chunk BM25 = 0 không được cộng điểm; khi ít hơn topn chunk có từ chung,
    mốc thấp nhất của min-max vẫn là 0 như benchmark (benchmark lấp danh sách bằng chunk 0 điểm)."""
    lex_ids, lex_sc = np.asarray(lex_ids, int), np.asarray(lex_sc, float)
    keep = lex_sc > 0
    lex_ids, lex_sc = lex_ids[keep], lex_sc[keep]
    if 0 < len(lex_ids) < cfg.topn:
        lex_ids, lex_sc = np.append(lex_ids, -1), np.append(lex_sc, 0.0)
    ix, sc = weighted_fusion(([np.asarray(dense_ids, int)], [np.asarray(dense_sc, float)]), ([lex_ids], [lex_sc]),
                             cfg.alpha, cfg.topn + 1, floor=cfg.floor)
    ix, sc = np.asarray(ix[0]), np.asarray(sc[0])
    ok = ix != -1
    return ix[ok][: cfg.topn], sc[ok][: cfg.topn]


class Searcher:
    """Bước 3–5 (rerank, trộn, gộp Điều). Lớp con cài _hybrid() và _payload()."""
    cfg: Config
    reranker = None

    def _hybrid(self, question: str):            # -> (ix, sc) đã trộn, sắp giảm dần
        raise NotImplementedError

    def _payload(self, ids: list) -> dict:        # id chunk -> {embed_text, body, key, article: {citation, title, effective_date, text}}
        raise NotImplementedError

    def search(self, question: str, rerank_query: str | None = None, k: int = 8, max_chars: int = 2500) -> dict:
        cfg = self.cfg
        rq = (rerank_query or "").strip() or main_question(question)
        t0 = time.perf_counter()
        ix, sc = self._hybrid(question)
        t1 = time.perf_counter()

        d = min(cfg.depth, len(ix)) if self.reranker is not None else 0
        head = [int(i) for i in ix[:d]]
        pl = self._payload(head)
        if d:
            ce = np.asarray(self.reranker.predict([(rq, pl[i]["embed_text"]) for i in head], show_progress_bar=False),
                            dtype=np.float64).ravel()
            f = cfg.beta * _minmax(ce) + (1 - cfg.beta) * _minmax(np.asarray(sc[:d], np.float64))
            head = [head[j] for j in np.argsort(-f, kind="stable")]
        order = head + [int(i) for i in ix[d:]]
        t2 = time.perf_counter()

        hits, seen, pos = [], set(), 0
        while len(hits) < k and pos < len(order):
            batch = order[pos: pos + 64]; pos += 64
            pl.update(self._payload([i for i in batch if i not in pl]))
            for i in batch:
                p = pl[i]
                if p["key"] in seen:
                    continue
                seen.add(p["key"])
                a = p["article"]
                hits.append(dict(n=len(hits) + 1, key=p["key"], citation=a["citation"], title=a["title"],
                                 effective_date=a["effective_date"], text=fit_text(a["text"], p["body"], max_chars, a.get("len"))))
                if len(hits) == k:
                    break
        return dict(rerank_query=rq, sources=hits,
                    timings_ms=dict(hybrid=round((t1 - t0) * 1000, 1), rerank=round((t2 - t1) * 1000, 1)))


MAX_CHARS = 20_000      # trần max_chars của API; Điều dài hơn không cần lưu toàn văn ở Qdrant


def fit_text(text: str, best_chunk: str, limit: int, full_len: int | None = None) -> str:
    """Điều ngắn: nguyên văn. Điều dài: tiêu đề + đúng chunk khớp câu hỏi nhất.
    full_len: độ dài thật khi `text` chỉ là dòng tiêu đề (Điều > MAX_CHARS, Qdrant không lưu toàn văn)."""
    if (full_len or len(text)) <= limit:
        return text
    head = text.split("\n", 1)[0]
    return f"{head}\n…\n{best_chunk[: max(0, limit - len(head) - 10)]}\n…"


_CUE = re.compile(r"(xin hỏi|cho (tôi|em|cháu|mình) hỏi|hỏi là|có được|có phải|có quyền|thì sao|như thế nào|"
                  r"làm (thế|như) nào|bao nhiêu|không\s*\?)", re.I)


def main_question(question: str) -> str:
    """Không có câu ngắn từ LLM: lấy câu chứa ý hỏi (thường ở cuối) thay cho cả câu chuyện."""
    sents = [s.strip() for s in re.split(r"(?<=[.?!\n])\s+", question or "") if s.strip()]
    asked = [s for s in sents if "?" in s or _CUE.search(s)]
    return ((asked or sents or [question or ""])[-1])[:300]


def encode_query(embedder, question: str) -> np.ndarray:
    return np.asarray(embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=False), dtype=np.float32)[0]


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_embedder(name: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer
    device = device or _auto_device()
    m = SentenceTransformer(name, device=device)
    if device.startswith("cuda"):
        m.half()
    m.max_seq_length = 514
    return m


def load_reranker(name: str, device: str | None = None):
    from sentence_transformers import CrossEncoder
    device = device or _auto_device()
    ce = CrossEncoder(name, device=device, max_length=1024)
    if device.startswith("cuda"):
        ce.model.half()
    return ce
