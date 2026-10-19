"""Index giả nhỏ (90 Điều, 120 chunk) cùng định dạng output notebook embedding — để CI chạy test không cần dữ liệu thật.
Từ vựng rộng (3.000 từ) để BM25 không hoà điểm ở mốc cắt top-n (hoà ở mốc cắt -> file và Qdrant chọn chunk khác nhau)."""
from pathlib import Path

import numpy as np
import pandas as pd

DOCS = [("Bộ luật", "45/2019/QH14", "Lao động", "2021-01-01"), ("Nghị định", "145/2020/NĐ-CP", "Việc làm", "2021-02-01"),
        ("Bộ luật", "91/2015/QH13", "Dân sự", "2017-01-01")]


def make_index(out: str, dim: int = 16, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    words = lambda n: " ".join(f"từ{int(x)}" for x in rng.integers(0, 3000, n))
    rows = []
    for d, (dtype, so_hieu, subject, eff) in enumerate(DOCS):
        for a in range(1, 31):
            rid = f"r{d}-{a}"
            title = f"Điều X.{a}. {subject} {a}"
            parts = [f"1. {words(35)} quy định."] + ([f"2. {words(35)} áp dụng."] if a % 3 == 0 else [])
            for j, body in enumerate(parts):
                rows.append(dict(chunk_id=f"{rid}-{j}", record_id=rid, chunk_index=j, origin_article=f"Điều {a}",
                                 origin_so_hieu=so_hieu, origin_doc_type=dtype, origin_effective_date=eff,
                                 source_note_text=f"(Điều {a} {dtype} số {so_hieu}, có hiệu lực thi hành kể từ ngày {eff})",
                                 subject_title_vi=subject, article_title=title, body=body,
                                 embed_text=f"{subject}\n{title}\n{body}"))
    chunks = pd.DataFrame(rows)
    E = rng.normal(size=(len(chunks), dim))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(f"{out}/chunks.parquet", index=False)
    np.save(f"{out}/embeddings.f16.npy", E.astype(np.float16))
    return out
