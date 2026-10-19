"""Đọc output notebook embedding và dựng dữ liệu cho tìm kiếm (chỉ dùng ở luồng build)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from legal_core.search import Config
from legal_core.text import article_key, dieu_number, syllable_bigrams
from pipeline.bm25 import BM25


def load_index(index_dir: str):
    """-> chunks (DataFrame), E (float32, đã chuẩn hoá) — kiểm tra hai file khớp nhau."""
    chunks = pd.read_parquet(Path(index_dir) / "chunks.parquet")
    E = np.load(Path(index_dir) / "embeddings.f16.npy").astype(np.float32)
    if len(E) != len(chunks):
        raise ValueError(f"embeddings ({len(E)}) ≠ chunks ({len(chunks)}) — index hỏng hoặc lệch phiên bản")
    return chunks, E


def article_keys(chunks: pd.DataFrame) -> np.ndarray:
    """Khoá Điều 'số hiệu|Điều' cho từng chunk — giống benchmark."""
    dieu = chunks["origin_article"].map(dieu_number)
    key = {r: article_key(s, d) or f"rec:{r}" for r, s, d in zip(chunks.record_id, chunks.origin_so_hieu, dieu)}
    return chunks.record_id.map(key).to_numpy(dtype=object)


def build_articles(chunks: pd.DataFrame, keys: np.ndarray) -> dict:
    """Ghép toàn văn mỗi Điều: các chunk cùng record_id, theo chunk_index. Câu dẫn ngắn của Điều
    chỉ nằm trong header (embed_text = header + "\\n" + body) nên lấy từ header của chunk đầu."""
    c = chunks.assign(_key=keys).sort_values(["record_id", "chunk_index"])
    out = {}
    for k, g in c.groupby("_key", sort=False):
        r = g.iloc[0]
        subject = r.subject_title_vi if isinstance(r.subject_title_vi, str) else ""
        head = r.embed_text[: len(r.embed_text) - len(r.body)].rstrip("\n").split("\n")
        if subject and head and head[0].strip() == subject.strip():
            head = head[1:]
        cite = " ".join(x for x in (r.origin_article, r.origin_doc_type, r.origin_so_hieu) if isinstance(x, str) and x)
        eff = r.get("origin_effective_date")
        out[k] = dict(citation=cite, title=r.article_title or "", subject=subject,
                      text="\n".join([*head, join_dedup(g["body"].tolist())]).strip(),
                      effective_date=eff if isinstance(eff, str) else None)
    return out


def join_dedup(parts: list, max_overlap: int = 600) -> str:
    """Nối các chunk liên tiếp, bỏ phần chồng lấn (chunker cắt câu quá dài thành cửa sổ có overlap)."""
    out = ""
    for p in parts:
        p = p or ""
        k = next((n for n in range(min(len(out), len(p), max_overlap), 19, -1) if out.endswith(p[:n])), 0)
        out += ("" if k else ("\n" if out else "")) + p[k:]
    return out


def build_bm25(texts: list, cfg: Config) -> BM25:
    return BM25(syllable_bigrams, k1=cfg.k1, b=cfg.b, max_df=0.5, qtf="k3", k3=cfg.k3).fit(texts)
