"""Luồng build: đẩy MỘT phiên bản dữ liệu lên Qdrant (collection phapdien_vN) + xuất bảng từ BM25.

  python -m pipeline.publish_qdrant --index-dir phapdien_index --version 2   # -> phapdien_v2 + bm25_vocab_v2.json.gz
  python -m pipeline.publish_qdrant --list
  python -m pipeline.publish_qdrant --delete phapdien_v1

Biến môi trường: QDRANT_URL, QDRANT_API_KEY (quyền ghi). Bị ngắt -> chạy lại đúng lệnh, tự đẩy tiếp.
Lên bản: chép bảng từ vào MinIO data/phapdien_vN/bm25_vocab.json.gz rồi đổi data.version trong values.yaml.
"""
from __future__ import annotations

import argparse
import os
import time

from legal_core.search import MAX_CHARS, Config, SparseQuery, qdrant_client
from pipeline.index_data import article_keys, build_articles, build_bm25, load_index

PREFIX = "phapdien"
BATCH, BATCH_BYTES = 256, 12_000_000      # Qdrant từ chối request > 32 MB -> chia lô theo số point và dung lượng
# Nhãn để lọc / tô màu trên Dashboard Qdrant (không ảnh hưởng tìm kiếm)
META = {"chu_de": "topic_title_vi", "de_muc": "subject_title_vi", "loai_vb": "origin_doc_type"}


def meta_columns(chunks) -> dict:
    out = {f: [v if isinstance(v, str) and v.strip() else "Khác" for v in chunks[c]] for f, c in META.items() if c in chunks}
    if "origin_effective_date" in chunks:
        out["nam"] = [int(v[:4]) if isinstance(v, str) and v[:4].isdigit() else 0 for v in chunks["origin_effective_date"]]
    return out


def completeness(client, name: str) -> tuple[int, int]:
    """(số point đang có, số point cần có) — mỗi point lưu n_total."""
    have = client.count(name, exact=True).count
    p = client.retrieve(name, [0], with_payload=["n_total"]) if have else []
    return have, int(p[0].payload.get("n_total", -1)) if p else -1


def publish(client, index_dir: str, version: int, vocab_out: str | None = None, cfg: Config = Config()) -> str:
    from qdrant_client import models
    name, t0 = f"{PREFIX}_v{version}", time.time()
    chunks, E = load_index(index_dir)
    texts, keys = chunks["embed_text"].tolist(), article_keys(chunks)
    articles = build_articles(chunks, keys)
    bm = build_bm25(texts, cfg)
    W = bm.WT.T.tocsr()                   # hàng i = trọng số BM25 của chunk i (idf × tf bão hoà × chuẩn hoá độ dài)
    vocab = {t: int(i) for t, i in bm.vec.vocabulary_.items()}
    total, meta = len(chunks), meta_columns(chunks)
    print(f"{total:,} chunk · {len(articles):,} Điều · bảng từ {len(vocab):,} · {time.time()-t0:.0f}s")

    vocab_out = vocab_out or f"bm25_vocab_v{version}.json.gz"
    SparseQuery(vocab, cfg.k3, name).save(vocab_out, k1=cfg.k1, b=cfg.b, n_chunks=total)
    print(f"bảng từ -> {vocab_out} ({os.path.getsize(vocab_out)/1e6:.1f} MB)")

    if not client.collection_exists(name):
        client.create_collection(name, vectors_config={"dense": models.VectorParams(size=E.shape[1], distance=models.Distance.DOT)},
                                 sparse_vectors_config={"bm25": models.SparseVectorParams()}, on_disk_payload=True)
    have, old = completeness(client, name)
    if have and old not in (-1, total):
        raise SystemExit(f"{name} chứa dữ liệu khác ({old:,} chunk ≠ {total:,}). Dùng --version khác hoặc --delete {name}")
    start = max(0, have - 2 * BATCH) if have < total else total   # đẩy tiếp: lùi 1 đoạn, upsert ghi đè nên không trùng

    def point(i):
        a, c, row = articles[keys[i]], chunks.iloc[i], W[i]
        text = a["text"] if len(a["text"]) <= MAX_CHARS else a["text"].split("\n", 1)[0]   # Điều quá dài: chỉ lưu tiêu đề
        return models.PointStruct(
            id=i,
            vector={"dense": E[i].tolist(), "bm25": models.SparseVector(indices=row.indices.tolist(), values=row.data.tolist())},
            payload=dict(key=keys[i], record_id=c.record_id, chunk_index=int(c.chunk_index), embed_text=texts[i], body=c.body,
                         citation=a["citation"], title=a["title"], effective_date=a["effective_date"],
                         article_text=text, article_len=len(a["text"]), n_total=total, **{f: v[i] for f, v in meta.items()}))

    def size(p):                          # ước lượng JSON: chữ ~2 byte, số thực ~20 byte
        pl = p.payload
        return 2 * sum(len(pl[f]) for f in ("embed_text", "body", "article_text")) + 20 * (E.shape[1] + 2 * len(p.vector["bm25"].indices)) + 2000

    pts, nbytes, last = [], 0, time.time()
    for i in range(start, total):
        p = point(i)
        b = size(p)
        if pts and (len(pts) >= BATCH or nbytes + b > BATCH_BYTES):
            client.upsert(name, pts, wait=True)
            pts, nbytes = [], 0
            if time.time() - last > 30:
                print(f"  {i:,}/{total:,}", flush=True); last = time.time()
        pts.append(p); nbytes += b
    if pts:
        client.upsert(name, pts, wait=True)
    n = client.count(name, exact=True).count
    if n != total:
        raise SystemExit(f"Qdrant có {n} point, cần {total}")
    for f in meta:
        client.create_payload_index(name, f, models.PayloadSchemaType.INTEGER if f == "nam" else models.PayloadSchemaType.KEYWORD)
    print(f"✓ {name}: {n:,} point · {time.time()-t0:.0f}s")
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir")
    ap.add_argument("--version", type=int)
    ap.add_argument("--vocab-out")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--delete")
    a = ap.parse_args()
    client = qdrant_client(os.environ["QDRANT_URL"], os.environ.get("QDRANT_API_KEY"), timeout=120)
    if a.list:
        for c in sorted(x.name for x in client.get_collections().collections if x.name.startswith(PREFIX)):
            have, need = completeness(client, c)
            print(f"{c:16} {have:>8,} point  {'đủ' if have == need else f'CHƯA XONG (cần {need:,})'}")
    elif a.delete:
        client.delete_collection(a.delete)
        print(f"đã xoá {a.delete}")
    elif a.index_dir and a.version:
        publish(client, a.index_dir, a.version, a.vocab_out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
