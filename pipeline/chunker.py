"""Chunker theo cấu trúc văn bản pháp luật Việt Nam (Điều -> Khoản -> Điểm -> câu -> cửa sổ token).

Thiết kế:
- Đơn vị gốc là 1 Điều (record trong Bộ Pháp điển).
- Điều ngắn (vừa ngân sách token) -> 1 chunk.
- Điều dài -> tách theo Khoản (1. 2. 3. ...), Khoản dài -> tách theo Điểm (a) b) c) ...),
  Điểm dài -> tách theo câu, câu dài -> cửa sổ token có chồng lấn (chỉ dùng ở bước cuối).
- Các mảnh nhỏ liên tiếp được ghép lại (packing) cho tới khi gần chạm ngân sách.
- Mỗi chunk mang một "header ngữ cảnh" (đề mục, chương, tiêu đề Điều, câu dẫn) để embedding
  hiểu chunk thuộc về đâu; header được tính vào ngân sách token.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

CHUNKER_VERSION = "legal-structural-v3"

_PUNCT_BEFORE = r'[.;:!?…"”)\]]'
# Ranh giới hợp lệ: đầu chuỗi, đầu dòng mới, hoặc sau dấu câu + khoảng trắng.
_BOUNDARY = r'(?:^|(?<=\n)|(?<=' + _PUNCT_BEFORE + r'\s))'
KHOAN_RE = re.compile(_BOUNDARY + r'(\d{1,3})\.\s+(?=\S)')
DIEM_RE = re.compile(_BOUNDARY + r'([a-zđ])\)\s+(?=\S)')
SENT_SPLIT_RE = re.compile(r'(?<=[.;!?…])\s+')
DIEM_LETTERS = "a b c d đ e g h i k l m n o p q r s t u v x y".split()


# --------------------------------------------------------------------------------------
# Tách cấu trúc
# --------------------------------------------------------------------------------------
def _sequential_spans(text: str, regex: re.Pattern, sequence: list[str], min_items: int = 2):
    """Tìm các vị trí bắt đầu Khoản/Điểm, CHỈ nhận khi đánh số liên tiếp (1,2,3 / a,b,c).

    Trả về (preamble, [(label, body), ...]) hoặc None nếu không đủ cấu trúc.
    Việc bắt buộc số liên tiếp giúp tránh tách nhầm ở "Điều 8. ..." hay "khoản 2. của Điều ...".
    """
    starts = []
    expect = 0
    for m in regex.finditer(text):
        if expect < len(sequence) and m.group(1) == sequence[expect]:
            starts.append((m.start(), m.end(), m.group(1)))
            expect += 1
    if len(starts) < min_items:
        return None
    preamble = text[: starts[0][0]].strip()
    items = []
    for i, (s, e, label) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        items.append((label, text[s:end].strip()))
    return preamble, items


def split_khoan(text: str):
    seq = [str(i) for i in range(1, 300)]
    return _sequential_spans(text, KHOAN_RE, seq)


def split_diem(text: str):
    return _sequential_spans(text, DIEM_RE, DIEM_LETTERS)


_LABEL_ONLY_RE = re.compile(r"^(?:\d{1,3}\.|[a-zđ]\))$")


def split_sentences(text: str) -> list[str]:
    """Tách câu, nhưng không để nhãn "2." hay "a)" bị tách rời khỏi nội dung của nó."""
    out: list[str] = []
    carry = ""
    for s in SENT_SPLIT_RE.split(text):
        s = s.strip()
        if not s:
            continue
        if _LABEL_ONLY_RE.match(s):
            carry = f"{carry} {s}".strip()
            continue
        out.append(f"{carry} {s}".strip() if carry else s)
        carry = ""
    if carry:
        out.append(carry)
    return out


# --------------------------------------------------------------------------------------
# Token counting (tiêm từ ngoài vào để test được không cần model)
# --------------------------------------------------------------------------------------
class TokenCounter:
    """Bọc tokenizer HuggingFace. Có thể thay bằng bản giả khi test."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def count(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        out = self.tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
        return [len(x) for x in out]

    def windows(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        enc = self.tok(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        if len(offs) <= max_tokens:
            return [text]
        step = max(1, max_tokens - overlap)
        pieces = []
        for s in range(0, len(offs), step):
            e = min(s + max_tokens, len(offs))
            pieces.append(text[offs[s][0]: offs[e - 1][1]].strip())
            if e == len(offs):
                break
        return [p for p in pieces if p]


class WhitespaceCounter(TokenCounter):
    """Bộ đếm giả: 1 từ = 1 token. Chỉ dùng để test logic."""

    def __init__(self):
        pass

    def count(self, texts):
        return [len(t.split()) for t in texts]

    def windows(self, text, max_tokens, overlap):
        w = text.split()
        if len(w) <= max_tokens:
            return [text]
        step = max(1, max_tokens - overlap)
        out = []
        for s in range(0, len(w), step):
            out.append(" ".join(w[s: s + max_tokens]))
            if s + max_tokens >= len(w):
                break
        return out


# --------------------------------------------------------------------------------------
# Chunk
# --------------------------------------------------------------------------------------
@dataclass
class Piece:
    text: str
    khoan: Optional[str] = None   # số Khoản chứa mảnh này
    diem: Optional[str] = None    # chữ Điểm (nếu có)
    tokens: int = 0


@dataclass
class Chunk:
    body: str
    embed_text: str
    khoan_from: Optional[str]
    khoan_to: Optional[str]
    diem: list[str] = field(default_factory=list)
    tokens: int = 0


def _refine(piece: Piece, budget: int, counter: TokenCounter, overlap: int) -> list[Piece]:
    """Chia nhỏ một mảnh cho tới khi mọi mảnh <= budget token."""
    if piece.tokens <= budget:
        return [piece]
    # 1) thử tách Điểm
    if piece.diem is None:
        sp = split_diem(piece.text)
        if sp:
            pre, items = sp
            subs = []
            (pre_tok,) = counter.count([pre]) if pre else (0,)
            if pre and pre_tok <= 48:
                # câu dẫn của Khoản ngắn ("2. Người lao động có quyền sau đây:") -> gắn vào từng Điểm
                subs += [Piece(f"{pre} {body}", piece.khoan, label) for label, body in items]
            else:
                if pre:
                    subs.append(Piece(pre, piece.khoan, None))
                subs += [Piece(body, piece.khoan, label) for label, body in items]
            for p, n in zip(subs, counter.count([p.text for p in subs])):
                p.tokens = n
            out = []
            for p in subs:
                p.diem = p.diem or "-"   # đánh dấu đã thử tách điểm
                out += _refine(p, budget, counter, overlap)
            for p in out:
                if p.diem == "-":
                    p.diem = None
            return out
    # 2) tách câu
    sents = split_sentences(piece.text)
    if len(sents) > 1:
        subs = [Piece(s, piece.khoan, piece.diem) for s in sents]
        for p, n in zip(subs, counter.count(sents)):
            p.tokens = n
        out = []
        for p in subs:
            if p.tokens <= budget:
                out.append(p)
            else:
                out += [Piece(w, piece.khoan, piece.diem, t)
                        for w, t in _windows_with_counts(p.text, budget, counter, overlap)]
        return out
    # 3) cửa sổ token (bước cuối cùng)
    return [Piece(w, piece.khoan, piece.diem, t)
            for w, t in _windows_with_counts(piece.text, budget, counter, overlap)]


def _windows_with_counts(text, budget, counter, overlap):
    overlap = min(overlap, budget // 4)          # chồng lấn không bao giờ vượt 1/4 cửa sổ
    ws = counter.windows(text, budget, overlap)
    return list(zip(ws, counter.count(ws)))


def build_header(subject: str, chapter: str, title: str, preamble: str = "",
                 max_chapter_chars: int = 160, max_title_chars: int = 400) -> str:
    parts = []
    # Không dùng chapter_title trong header: ở Bộ Pháp điển trường này gộp nhiều tiêu đề Mục liên tiếp,
    # khi cắt ngắn dễ mang tiêu đề Mục khác chủ đề vào embedding. Chương vẫn được giữ trong metadata.
    ctx = subject.strip()
    if ctx:
        parts.append(ctx)
    parts.append(title.strip()[:max_title_chars])
    if preamble:
        parts.append(preamble.strip())
    return "\n".join(parts)


def chunk_article(content: str, title: str, subject: str = "", chapter: str = "", *,
                  counter: TokenCounter, max_tokens: int = 512, overlap: int = 48,
                  max_preamble_tokens: int = 80, safety: int = 8) -> list[Chunk]:
    """Chia 1 Điều thành các chunk. max_tokens tính cả header (không gồm special tokens)."""
    content = (content or "").strip()
    if not content:
        return []
    header = build_header(subject, chapter, title)
    full = f"{header}\n{content}"
    full_tok, header_tok = counter.count([full, header])
    if full_tok <= max_tokens - safety:
        return [Chunk(content, full, None, None, [], full_tok)]

    # Điều dài -> tách Khoản
    sp = split_khoan(content)
    preamble = ""
    if sp:
        preamble, items = sp
        pieces = [Piece(body, label) for label, body in items]
    else:
        pieces = [Piece(content)]

    # câu dẫn ngắn -> đưa vào header của mọi chunk; dài -> thành mảnh riêng
    if preamble:
        (pre_tok,) = counter.count([preamble])
        if pre_tok <= max_preamble_tokens:
            header = build_header(subject, chapter, title, preamble)
        else:
            pieces = [Piece(preamble)] + pieces
    (header_tok,) = counter.count([header])
    budget = max(32, max_tokens - header_tok - safety)

    for p, n in zip(pieces, counter.count([p.text for p in pieces])):
        p.tokens = n
    refined: list[Piece] = []
    for p in pieces:
        refined += _refine(p, budget, counter, overlap)

    # Ghép các mảnh liên tiếp (packing)
    chunks: list[Chunk] = []
    cur: list[Piece] = []
    cur_tok = 0

    def flush():
        nonlocal cur, cur_tok
        if not cur:
            return
        body = " ".join(p.text for p in cur)
        ks = [p.khoan for p in cur if p.khoan]
        ds = sorted({(p.khoan or "", p.diem) for p in cur if p.diem},
                    key=lambda kd: (int(kd[0]) if kd[0].isdigit() else 0,
                                    DIEM_LETTERS.index(kd[1]) if kd[1] in DIEM_LETTERS else 99))
        ds = [f"{k}{d}" for k, d in ds]
        chunks.append(Chunk(body, f"{header}\n{body}", ks[0] if ks else None,
                            ks[-1] if ks else None, ds, 0))
        cur, cur_tok = [], 0

    for p in refined:
        if cur and cur_tok + p.tokens + 1 > budget:
            flush()
        cur.append(p)
        cur_tok += p.tokens + 1
    flush()

    for c, n in zip(chunks, counter.count([c.embed_text for c in chunks])):
        c.tokens = n
    return chunks


# --------------------------------------------------------------------------------------
# Parse ghi chú nguồn: "(Điều 26 Bộ luật số 45/2019/QH14 Lao động ngày 20/11/2019 của Quốc hội,
#                       có hiệu lực thi hành kể từ ngày 01/01/2021)"
# --------------------------------------------------------------------------------------
DOC_TYPES = ["Thông tư liên tịch", "Nghị quyết liên tịch", "Bộ luật", "Luật", "Nghị định",
             "Thông tư", "Nghị quyết", "Pháp lệnh", "Quyết định", "Hiến pháp", "Lệnh", "Chỉ thị"]
_DOC_TYPE_RE = re.compile(r"(" + "|".join(re.escape(t) for t in DOC_TYPES) + r")\s+số\s+(\d+/\d{4}/[\w\-]+)")
_ARTICLE_RE = re.compile(r"\(?\s*((?:Khoản\s+\d+\s+)?Điều\s+\d+[a-zđ]?)", re.IGNORECASE)
_EFFECTIVE_RE = re.compile(r"hiệu lực(?: thi hành)?\s+(?:kể\s+)?từ ngày\s+(\d{1,2}/\d{1,2}/\d{4})")
_ISSUED_RE = re.compile(r"ngày\s+(\d{1,2}/\d{1,2}/\d{4})\s+của")


def _iso(d: Optional[str]) -> Optional[str]:
    if not d:
        return None
    dd, mm, yy = d.split("/")
    return f"{yy}-{int(mm):02d}-{int(dd):02d}"


def parse_source_note(note: str) -> dict:
    note = note or ""
    m_doc = _DOC_TYPE_RE.search(note)
    m_art = _ARTICLE_RE.search(note)
    m_eff = _EFFECTIVE_RE.search(note)
    m_iss = _ISSUED_RE.search(note)
    all_so_hieu = sorted({m.group(2) for m in _DOC_TYPE_RE.finditer(note)})
    return {
        "origin_article": m_art.group(1) if m_art else None,
        "origin_doc_type": m_doc.group(1) if m_doc else None,
        "origin_so_hieu": m_doc.group(2) if m_doc else None,
        "origin_issued_date": _iso(m_iss.group(1)) if m_iss else None,
        "origin_effective_date": _iso(m_eff.group(1)) if m_eff else None,
        "all_so_hieu": all_so_hieu,
    }


def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _first_vbpl_link(links) -> Optional[str]:
    try:
        for x in list(links or []):
            href = (x.get("href") if isinstance(x, dict) else None) or ""
            if "vbpl.vn" in href:
                return href
    except Exception:
        pass
    return None


def build_chunk_records(rows, counter: TokenCounter, *, max_tokens: int = 512, overlap: int = 48,
                        batch: int = 2048, progress=None) -> list[dict]:
    """rows: iterable các dict có các cột của config `documents` (Bộ Pháp điển).

    Fast path: đếm token cho cả lô "header + nội dung"; Điều nào vừa ngân sách thì thành 1 chunk
    ngay, chỉ Điều dài mới đi qua chunk_article(). Trả về list dict, mỗi dict là 1 chunk.
    """
    rows = list(rows)
    headers = [build_header(r.get("subject_title_vi") or "", r.get("chapter_title") or "",
                            r.get("article_title") or "") for r in rows]
    fulls = [f"{h}\n{(r.get('content_text') or '').strip()}" for h, r in zip(headers, rows)]
    full_tok: list[int] = []
    for i in range(0, len(fulls), batch):
        full_tok += counter.count(fulls[i:i + batch])

    out: list[dict] = []
    it = range(len(rows)) if progress is None else progress(range(len(rows)))
    for i in it:
        r = rows[i]
        content = (r.get("content_text") or "").strip()
        if not content:
            continue
        if full_tok[i] <= max_tokens - 8:
            chunks = [Chunk(content, fulls[i], None, None, [], full_tok[i])]
        else:
            chunks = chunk_article(content, r.get("article_title") or "", r.get("subject_title_vi") or "",
                                   r.get("chapter_title") or "", counter=counter,
                                   max_tokens=max_tokens, overlap=overlap)
        meta = parse_source_note(r.get("source_note_text") or "")
        base = {
            "record_id": r["record_id"],
            "article_id": r.get("article_id"),
            "article_title": r.get("article_title"),
            "chapter_title": (r.get("chapter_title") or "")[:500],
            "subject_id": r.get("subject_id"),
            "subject_title_vi": r.get("subject_title_vi"),
            "topic_id": r.get("topic_id"),
            "topic_number": r.get("topic_number"),
            "topic_title_vi": r.get("topic_title_vi"),
            "source_note_text": r.get("source_note_text"),
            "related_note_text": (r.get("related_note_text") or "")[:2000],
            "source_url": r.get("source_url"),
            "source_vbpl_url": _first_vbpl_link(r.get("source_links")),
            **meta,
        }
        n = len(chunks)
        for j, c in enumerate(chunks):
            out.append({
                "chunk_id": f"{r['record_id']}-{j:03d}",
                "chunk_index": j,
                "n_chunks": n,
                "khoan_from": c.khoan_from,
                "khoan_to": c.khoan_to,
                "diem": c.diem,
                "tokens": c.tokens,
                "body": c.body,
                "embed_text": c.embed_text,
                "text_hash": text_hash(c.embed_text),
                **base,
            })
    return out
