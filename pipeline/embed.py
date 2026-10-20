"""Dựng index từ Bộ Pháp điển: chunk theo cấu trúc + embedding (GPU, đa GPU, chạy tiếp được khi bị ngắt).

Ra thư mục gồm chunks.parquet + embeddings.f16.npy (hàng i ↔ hàng i) + manifest.json — đúng định dạng
publish_qdrant / FileRetriever / mlops.labels đang đọc. Chạy trên Kaggle (2×T4 ~ 30–40 phút cho ~87k chunk).
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import chunker as lc


@dataclass
class IndexConfig:
    dataset_id: str = "tmquan/phapdien-moj-gov-vn"
    dataset_cfg: str = "documents"
    model_id: str = "AITeamVN/Vietnamese_Embedding"
    max_tokens: int = 512         # ngân sách token / chunk (gồm header)
    overlap: int = 48             # chồng lấn, chỉ dùng khi phải cắt cửa sổ
    batch_size: int = 64          # / GPU, tự giảm khi hết bộ nhớ
    shard_size: int = 8192        # số văn bản mỗi shard lưu đĩa (để chạy tiếp)
    fp16: bool = True


def load_model(cfg: IndexConfig, revision: str | None, device: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(cfg.model_id, revision=revision, device=device)
    m.max_seq_length = cfg.max_tokens + 2          # + <s>, </s>  (= 514, giống load_embedder phía production)
    if cfg.fp16 and device.startswith("cuda"):
        m.half()
    return m


def make_chunks(df: pd.DataFrame, cfg: IndexConfig, tokenizer, progress=None) -> pd.DataFrame:
    """df: split train của dataset Pháp điển (config documents)."""
    df = df[df["content_char_len"].fillna(0) > 0].reset_index(drop=True)      # bỏ dòng phân cách rỗng
    assert df["record_id"].is_unique, "record_id phải duy nhất"
    recs = lc.build_chunk_records(df.to_dict("records"), lc.TokenCounter(tokenizer),
                                  max_tokens=cfg.max_tokens, overlap=cfg.overlap, progress=progress)
    chunks = pd.DataFrame(recs)
    assert (chunks["tokens"] <= cfg.max_tokens).all(), "có chunk vượt ngân sách token"
    assert chunks["chunk_id"].is_unique
    return chunks


def embed(texts: list[str], models: list, cfg: IndexConfig, shard_dir: Path, progress=None) -> np.ndarray:
    """Mỗi GPU một model, chia shard xen kẽ; shard nào đã có trên đĩa thì bỏ qua (chạy tiếp sau khi bị ngắt)."""
    import torch
    shard_dir.mkdir(parents=True, exist_ok=True)
    n = math.ceil(len(texts) / cfg.shard_size)

    def encode(m, part, bs):
        while True:
            try:
                return m.encode(part, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                if bs == 1:
                    raise
                bs = max(1, bs // 2); print(f"  hết bộ nhớ GPU -> batch {bs}")

    def worker(w, ids, m):
        for k in (progress(ids, desc=f"GPU {w}", position=w) if progress else ids):
            f = shard_dir / f"shard_{k:05d}.npy"
            if f.exists():
                continue
            e = encode(m, texts[k * cfg.shard_size:(k + 1) * cfg.shard_size], cfg.batch_size)
            np.save(f.with_suffix(".tmp.npy"), e.astype(np.float16))
            os.replace(f.with_suffix(".tmp.npy"), f)                 # ghi nguyên tử

    try:
        with ThreadPoolExecutor(len(models)) as ex:
            list(ex.map(worker, range(len(models)), [list(range(i, n, len(models))) for i in range(len(models))], models))
    except Exception as e:                                           # đa GPU lỗi -> làm nốt bằng GPU đầu
        print("đa GPU lỗi:", repr(e), "-> chạy tiếp bằng 1 GPU")
        worker(0, list(range(n)), models[0])
    return np.concatenate([np.load(shard_dir / f"shard_{k:05d}.npy") for k in range(n)])


def build_index(df: pd.DataFrame, out_dir: str, cfg: IndexConfig = IndexConfig(), *, dataset_rev: str | None = None,
                model_rev: str | None = None, devices: list[str] | None = None, progress=None) -> tuple[pd.DataFrame, np.ndarray]:
    import torch
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    devices = devices or [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    t0 = time.time()
    models = [load_model(cfg, model_rev, d) for d in devices]
    chunks = make_chunks(df, cfg, models[0].tokenizer, progress)
    print(f"{chunks.record_id.nunique():,} Điều -> {len(chunks):,} chunk ({time.time()-t0:.0f}s)")

    # embed mỗi nội dung đúng 1 lần, dài trước (lỗi hết bộ nhớ lộ ra sớm, batch đều độ dài)
    uniq = chunks.drop_duplicates("text_hash").sort_values("tokens", ascending=False).reset_index(drop=True)
    fp = hashlib.sha1("|".join([str(model_rev), str(cfg.max_tokens), str(cfg.shard_size), *uniq.text_hash]).encode()).hexdigest()[:12]
    t1 = time.time()
    U = embed(uniq["embed_text"].tolist(), models, cfg, out / "shards" / fp, progress)
    assert U.shape[0] == len(uniq) and np.isfinite(U.astype(np.float32)).all()
    pos = {h: i for i, h in enumerate(uniq.text_hash)}
    E = U[chunks.text_hash.map(pos).to_numpy()]
    print(f"embedding {E.shape} trong {(time.time()-t1)/60:.1f} phút ({len(devices)} thiết bị)")

    chunks.to_parquet(out / "chunks.parquet", index=False)
    np.save(out / "embeddings.f16.npy", E.astype(np.float16))
    dim = E.shape[1]
    (out / "manifest.json").write_text(json.dumps(dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dataset=dict(id=cfg.dataset_id, config=cfg.dataset_cfg, revision=dataset_rev, rows=int(chunks.record_id.nunique())),
        model=dict(id=cfg.model_id, revision=model_rev, dim=dim, normalize=True, similarity="dot", max_seq_length=cfg.max_tokens + 2),
        chunker=dict(version=lc.CHUNKER_VERSION, max_tokens=cfg.max_tokens, overlap=cfg.overlap),
        config=asdict(cfg), counts=dict(chunks=len(chunks), unique_texts=len(uniq), articles=int(chunks.record_id.nunique())),
    ), ensure_ascii=False, indent=2))
    import shutil
    shutil.rmtree(out / "shards", ignore_errors=True)
    return chunks, E
