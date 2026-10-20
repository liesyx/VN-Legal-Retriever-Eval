"""Tạo đáp án (nhãn bạc) cho bộ câu hỏi chuẩn — chép nguyên phần tạo nhãn của legal_eval.py (legal-eval-v5)
trong notebook benchmark, để bộ câu hỏi dựng lại ra đúng như benchmark (494 câu test với seed 42).

Câu hỏi thật của người dân (tmquan/pbgdpl-vn-legal-qna) + câu trả lời của cán bộ Bộ Tư pháp:
  1. bóc trích dẫn "khoản 1 Điều 241 Bộ luật Dân sự năm 2015" trong câu trả lời -> (số hiệu, Điều) có trong Pháp điển
  2. không ghi năm: chọn phiên bản luật có hiệu lực tại ngày gửi câu hỏi
  3. kiểm chứng nội dung: chỉ giữ Điều mà câu trả lời thật sự nói tới (khớp tiêu đề / chép lại đoạn đặc trưng),
     ngưỡng tự hiệu chỉnh trên phần dev để nhận nhầm ≤ 2%
build_golden() gói cả quy trình (chế độ "validated" của benchmark).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from legal_core.text import article_key, dieu_number, norm, syllables

EVAL_VERSION = "legal-eval-v5"


def to_iso_date(x) -> Optional[str]:
    """Đưa ngày về chuỗi 'YYYY-MM-DD' để so với ngày hiệu lực. Nhận str (ISO hoặc dd/mm/yyyy),
    datetime/date, pandas.Timestamp, numpy.datetime64, số giây/mili-giây epoch. Không đọc được -> None."""
    import datetime as _dt
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except TypeError:
        pass
    if hasattr(x, "strftime"):                                   # datetime, date, pandas.Timestamp
        try:
            return x.strftime("%Y-%m-%d")
        except ValueError:                                        # NaT
            return None
    if isinstance(x, np.datetime64):
        return None if np.isnat(x) else str(x.astype("datetime64[D]"))
    if isinstance(x, (int, np.integer, float, np.floating)):
        v = float(x)
        if v > 1e11:                                              # mili-giây
            v /= 1000.0
        try:
            return _dt.datetime.utcfromtimestamp(v).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return None
    t = str(x).strip()
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", t)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return None


DOC_TYPES = ["thông tư liên tịch", "nghị quyết liên tịch", "bộ luật", "luật", "nghị định",
             "thông tư", "nghị quyết", "pháp lệnh", "quyết định", "hiến pháp"]
_TYPE_ALT = "|".join(re.escape(t) for t in DOC_TYPES)
_SO_HIEU = r"\d+/\d{4}/[\w\-]+"


# --------------------------------------------------------------------------------------
# Catalog văn bản từ ghi chú nguồn của Pháp điển
# --------------------------------------------------------------------------------------
_NOTE_FULL = re.compile(r"(" + _TYPE_ALT + r")\s+số\s+(" + _SO_HIEU + r")\s+(.+?)\s+ngày\s+\d{1,2}/\d{1,2}/\d{4}")


@dataclass
class DocInfo:
    so_hieu: str
    doc_type: str
    name: str
    year: int
    effective: Optional[str]   # ISO date, ngày hiệu lực sớm nhất thấy trong corpus


def build_catalog(notes: Iterable[str], so_hieus: Iterable[Optional[str]],
                  doc_types: Iterable[Optional[str]], effectives: Iterable[Optional[str]]):
    """Trả về (docs: so_hieu->DocInfo, by_name: (type,name)->[DocInfo])."""
    names: dict[str, str] = {}
    types: dict[str, str] = {}
    eff: dict[str, str] = {}
    for note, sh, dt, ef in zip(notes, so_hieus, doc_types, effectives):
        if not sh:
            continue
        sh_l = sh.lower()
        if dt:
            types.setdefault(sh_l, norm(dt))
        if ef and (sh_l not in eff or ef < eff[sh_l]):
            eff[sh_l] = ef
        if sh_l not in names:
            m = _NOTE_FULL.search(norm(note or ""))
            if m and m.group(2).lower() == sh_l:
                names[sh_l] = m.group(3).strip(" ,;")
    docs: dict[str, DocInfo] = {}
    by_name: dict[tuple, list] = defaultdict(list)
    for sh_l, dt in types.items():
        try:
            year = int(sh_l.split("/")[1])
        except (IndexError, ValueError):
            continue
        info = DocInfo(sh_l, dt, names.get(sh_l, ""), year, eff.get(sh_l))
        docs[sh_l] = info
        if info.name:
            by_name[(dt, info.name)].append(info)
    return docs, dict(by_name)


# --------------------------------------------------------------------------------------
# Trích dẫn trong câu trả lời
# --------------------------------------------------------------------------------------
_ART_GROUP = (r"(?:các\s+)?điều\s+(\d+[a-zđ]?)"
              r"((?:\s*(?:,|;|và|hoặc|đến)\s*(?:điều\s+)?\d+[a-zđ]?\b)*)")
_CITE_RE = re.compile(_ART_GROUP + r"\s*,?\s*(?:của\s+|tại\s+|thuộc\s+)?(" + _TYPE_ALT + r")\b(?=(.{0,160}))")   # phần đuôi chỉ "nhìn trước", không ăn mất trích dẫn kế tiếp


@dataclass
class Citation:
    raw: str
    articles: list
    doc_type: str
    so_hieu: Optional[str]      # đã resolve (chữ thường) hoặc None
    reason: str                 # "ok" hoặc lý do bị loại


class CitationResolver:
    """require_explicit_version=True ("strict"): chỉ nhận trích dẫn có ghi NĂM hoặc SỐ HIỆU.
    Trích dẫn kiểu "Điều 202 Bộ luật Hình sự" (không ghi năm) rất hay dùng số Điều của bộ luật CŨ
    (vd. BLHS 1999) nên không thể suy ra phiên bản một cách đáng tin cậy."""
    def __init__(self, docs: dict, by_name: dict, require_explicit_version: bool = False):
        self.require_explicit = require_explicit_version
        self.docs = docs
        self.names_by_type: dict[str, list] = defaultdict(list)
        for (dt, name) in by_name:
            self.names_by_type[dt].append(name)
        for dt in self.names_by_type:
            self.names_by_type[dt].sort(key=len, reverse=True)   # khớp tên dài nhất trước
        self.by_name = by_name

    def _resolve(self, doc_type: str, tail: str, date: Optional[str]):
        tail = tail.lstrip(" ,")
        m = re.match(r"(?:số\s+)?(" + _SO_HIEU + r")", tail)
        if m:
            sh = m.group(1).lower()
            return (sh, "ok") if sh in self.docs else (None, "so_hieu_not_in_corpus")
        name = next((n for n in self.names_by_type.get(doc_type, [])
                     if tail.startswith(n) and (len(tail) == len(n) or not tail[len(n)].isalnum())), None)
        if name is None:
            return None, "name_not_in_corpus"
        rest = tail[len(name):]
        ym = re.match(r"\s*,?\s*(?:năm\s+)?(\d{4})\b", rest)
        cands = self.by_name[(doc_type, name)]
        if ym:
            year = int(ym.group(1))
            hit = [c for c in cands if c.year == year]
            return (hit[0].so_hieu, "ok") if hit else (None, "version_not_in_corpus")
        # Không ghi năm
        if self.require_explicit:
            return None, "no_year_given"
        # (chế độ loose) chọn phiên bản có hiệu lực tại ngày gửi câu hỏi
        if date:
            valid = [c for c in cands if c.effective and c.effective <= date]
            if valid:
                return max(valid, key=lambda c: c.effective).so_hieu, "ok"
            return None, "version_newer_than_answer"
        if len(cands) == 1:
            return cands[0].so_hieu, "ok"
        return None, "ambiguous_version"

    def extract(self, answer: str, date: Optional[str] = None) -> list[Citation]:
        text = norm(answer)
        out = []
        for m in _CITE_RE.finditer(text):
            arts = [m.group(1)] + re.findall(r"\d+[a-zđ]?", m.group(2) or "")
            if re.search(r"\bđến\b", m.group(2) or "") and len(arts) == 2:
                a, b = (int(re.match(r"\d+", x).group()) for x in arts)
                if 0 < b - a <= 30:                              # "Điều 10 đến Điều 15"
                    arts = [str(i) for i in range(a, b + 1)]
            doc_type = m.group(3)
            sh, reason = self._resolve(doc_type, m.group(4), date)
            out.append(Citation(m.group(0)[:120], arts, doc_type, sh, reason))
        return out

    def gold_keys(self, answer: str, date: Optional[str], corpus_keys: set, validator=None):
        """Trả về (danh sách khoá Điều đáp án, danh sách Citation).

        validator (tuỳ chọn): ContentValidator — chỉ giữ Điều mà câu trả lời thật sự nói tới
        (chặn trường hợp số Điều của luật cũ trùng số Điều của luật mới nhưng khác nội dung)."""
        cites = self.extract(answer, date)
        feats = validator.features(answer) if validator is not None else None
        keys = []
        for c in cites:
            c.kept = []
            if not c.so_hieu:
                continue
            n_in, n_ok = 0, 0
            for a in c.articles:
                k = article_key(c.so_hieu, a)
                if k not in corpus_keys:
                    continue
                n_in += 1
                if validator is not None and not validator.check(k, feats):
                    continue
                n_ok += 1
                c.kept.append(k)
                if k not in keys:
                    keys.append(k)
            if c.reason == "ok":
                c.reason = "ok" if n_ok else ("content_mismatch" if n_in else "article_not_in_corpus")
        return keys, cites


class ContentValidator:
    """Kiểm chứng một trích dẫn bằng nội dung: giữ Điều nếu
    (a) phần lớn bigram trong TIÊU ĐỀ Điều xuất hiện trong câu trả lời, hoặc
    (b) câu trả lời chép lại ít nhất `min_shingles` đoạn 6 âm tiết liên tiếp của NỘI DUNG Điều
        (sau fit_df: chỉ tính đoạn đặc trưng của Điều, bỏ văn mẫu lặp ở nhiều Điều)."""

    _PREFIX = re.compile(r"^điều\s+[\w.]+\.\s*")

    def __init__(self, titles: dict, bodies: dict, title_ratio: float = 0.5, min_shingles: int = 3,
                 shingle_n: int = 6, max_body_syllables: int = 4000):
        self.titles, self.bodies = titles, bodies
        self.title_ratio, self.min_shingles, self.n, self.max_body = title_ratio, min_shingles, shingle_n, max_body_syllables
        self._cache: dict = {}
        self.common: set = set()

    def fit_df(self, answers, max_df: int = 5):
        """Đánh dấu các đoạn 6 âm tiết 'văn mẫu' (xuất hiện ở > max_df Điều, vd. "thì bị phạt tù từ ... năm",
        "trừ trường hợp luật khác có quy định") để KHÔNG tính là bằng chứng chép lại.
        Chỉ đếm các đoạn có trong ít nhất một câu trả lời nên chạy nhanh và ít RAM."""
        U = set()
        for a in answers:
            U |= self._shingles(syllables(a or ""))
        df = Counter()
        for b in self.bodies.values():
            df.update(self._shingles(syllables(b or "")[: self.max_body]) & U)
        self.common = {h for h, c in df.items() if c > max_df}
        self._cache.clear()
        return len(U), len(self.common)

    @staticmethod
    def _bigrams(toks):
        return {(a, b) for a, b in zip(toks, toks[1:])}

    def _shingles(self, toks):
        n = self.n
        return {hash(tuple(toks[i:i + n])) for i in range(max(0, len(toks) - n + 1))}

    def features(self, answer: str):
        toks = syllables(answer)
        return self._bigrams(toks), self._shingles(toks)

    def _article(self, key):
        if key not in self._cache:
            t = syllables(self._PREFIX.sub("", norm(self.titles.get(key, ""))))
            b = syllables(self.bodies.get(key, ""))[: self.max_body]
            self._cache[key] = (self._bigrams(t), self._shingles(b) - self.common)
        return self._cache[key]

    def score_full(self, key, feats):
        """→ (tỷ lệ cặp từ tiêu đề có trong câu trả lời, số cặp khớp, số đoạn 6 âm tiết chép lại)"""
        a_bi, a_sh = feats
        t_bi, b_sh = self._article(key)
        m = len(t_bi & a_bi)
        return (m / len(t_bi) if t_bi else 0.0), m, len(b_sh & a_sh)

    def score(self, key, feats):
        ratio, _, hits = self.score_full(key, feats)
        return ratio, hits

    def passes(self, ratio, matched, hits, title_ratio=None, min_shingles=None):
        # tiêu đề rất ngắn dễ khớp nhờ chính tên văn bản trong trích dẫn → cần ≥2 cặp từ khớp
        tr = self.title_ratio if title_ratio is None else title_ratio
        ms = self.min_shingles if min_shingles is None else min_shingles
        return (ratio >= tr and matched >= 2) or hits >= ms

    def check(self, key, feats) -> bool:
        return self.passes(*self.score_full(key, feats))


def calibrate_validator(validator, resolver, answers, dates, corpus_keys, max_fpr=0.02, seed=0,
                        ratios=(0.4, 0.5, 0.6, 0.7, 2.0), shingles=(1, 2, 3, 4, 6, 8, 10, 15, 20, 10**9)):
    """Chọn ngưỡng cho ContentValidator bằng dữ liệu thật, không cần chấm tay.
    Dương: trích dẫn có ghi năm/số hiệu (resolver strict) → gần như chắc đúng phiên bản.
    Âm   : cùng câu trả lời nhưng ghép với một Điều NGẪU NHIÊN khác của CÙNG văn bản
           → mô phỏng đúng lỗi "nhầm số Điều do khác phiên bản".
    Chọn cặp ngưỡng có tỷ lệ nhận nhầm (FPR) ≤ max_fpr và giữ được nhiều trích dẫn đúng nhất (TPR).
    ratio=2.0 nghĩa là tắt luật tiêu đề; shingles=10**9 nghĩa là tắt luật chép lại."""
    rng = np.random.default_rng(seed)
    by_doc = defaultdict(list)
    for k in corpus_keys:
        if k and "|" in k and not k.startswith("rec:"):
            by_doc[k.split("|")[0]].append(k)
    pos, neg = [], []
    for ans, d in zip(answers, dates):
        g, _ = resolver.gold_keys(ans, d if isinstance(d, str) else None, corpus_keys)
        if not g:
            continue
        f, gs = validator.features(ans), set(g)
        for k in g:
            pos.append(validator.score_full(k, f))
            cands = by_doc.get(k.split("|")[0], [])
            if len(cands) > len(gs):
                for _ in range(20):
                    j = cands[rng.integers(len(cands))]
                    if j not in gs:
                        neg.append(validator.score_full(j, f)); break
    if not pos or not neg:
        return [], None, len(pos), len(neg)
    P, N = np.array(pos, dtype=float), np.array(neg, dtype=float)
    rows = []
    for r in ratios:
        for sh in shingles:
            ok = lambda X: ((X[:, 0] >= r) & (X[:, 1] >= 2)) | (X[:, 2] >= sh)
            rows.append(dict(title_ratio=r, min_shingles=sh, tpr=float(ok(P).mean()), fpr=float(ok(N).mean())))
    feasible = [x for x in rows if x["fpr"] <= max_fpr]
    best = max(feasible, key=lambda x: (round(x["tpr"], 3), -x["fpr"], x["min_shingles"] < 10**9, x["min_shingles"])) if feasible else min(rows, key=lambda x: (x["fpr"], -x["tpr"]))
    return rows, best, len(P), len(N)


# --------------------------------------------------------------------------------------
# Cả quy trình (= cell "Nạp index & tạo đáp án" của notebook benchmark, chế độ validated)
# --------------------------------------------------------------------------------------
def build_golden(chunks, qa, *, seed: int = 42, dev_frac: float = 0.30, min_question_chars: int = 20,
                 max_fpr: float = 0.02, max_df: int = 5, title_ratio: float = 0.5, min_shingles: int = 3, log=print):
    """chunks: chunks.parquet (DataFrame); qa: split train của tmquan/pbgdpl-vn-legal-qna (config "qa").
    -> (DataFrame item_id, split, lv_name, title, query, gold ; dict thống kê)"""
    rec_key = {r: (article_key(s, d) or f"rec:{r}") for r, s, d in
               zip(chunks.record_id, chunks.origin_so_hieu, chunks["origin_article"].map(dieu_number))}
    chunk_keys = chunks.record_id.map(rec_key).to_numpy(dtype=object)
    corpus_keys = set(chunk_keys)
    key_title = dict(zip(chunk_keys, chunks.article_title))
    art = chunks.drop_duplicates("record_id")
    docs, by_name = build_catalog(art.source_note_text, art.origin_so_hieu, art.origin_doc_type, art.origin_effective_date)
    key_body = chunks.sort_values(["record_id", "chunk_index"]).assign(_k=lambda d: d.record_id.map(rec_key))                      .groupby("_k")["body"].apply(" ".join).to_dict()
    validator = ContentValidator(key_title, key_body, title_ratio=title_ratio, min_shingles=min_shingles)
    loose = CitationResolver(docs, by_name)
    strict = CitationResolver(docs, by_name, require_explicit_version=True)
    log(f"catalog: {len(docs):,} văn bản · {len(by_name):,} tên · {len(corpus_keys):,} khoá Điều")

    qa = qa[qa.question_text.fillna("").str.len() >= min_question_chars].reset_index(drop=True)
    qa = qa.assign(date_iso=qa.date_sent.map(to_iso_date))
    # chia dev/test TRƯỚC khi tạo đáp án: bộ lọc chỉ được hiệu chỉnh trên dev
    qa["split"] = np.where(np.random.default_rng(seed).random(len(qa)) < dev_frac, "dev", "test")
    nU, nC = validator.fit_df(qa.answer_text.fillna(""), max_df=max_df)
    log(f"văn mẫu: {nC:,}/{nU:,} đoạn 6 âm tiết không tính là bằng chứng")
    dev = qa[qa.split == "dev"]
    _, best, nP, nN = calibrate_validator(validator, strict, dev.answer_text, dev.date_iso, corpus_keys, max_fpr=max_fpr, seed=seed)
    if best and best["tpr"] >= 0.5:
        validator.title_ratio, validator.min_shingles = best["title_ratio"], best["min_shingles"]
        log(f"hiệu chỉnh bộ lọc trên dev ({nP:,} dương · {nN:,} âm): tiêu đề ≥{best['title_ratio']:.0%} hoặc "
            f"≥{best['min_shingles']} đoạn chép lại -> giữ {best['tpr']:.1%} Điều đúng, lọt {best['fpr']:.1%} Điều sai")
    else:
        log("hiệu chỉnh không đạt -> dùng ngưỡng mặc định"); best = None

    gold, reasons = [], Counter()
    for ans, date in zip(qa.answer_text, qa.date_iso):
        g, cites = loose.gold_keys(ans, date if isinstance(date, str) else None, corpus_keys, validator=validator)
        gold.append(g); reasons.update(c.reason for c in cites)
    qa["gold"] = gold
    qa["query"] = qa.question_text
    Q = qa[qa.gold.map(len) > 0].reset_index(drop=True)[["item_id", "split", "lv_name", "title", "query", "gold"]]
    stats = dict(eval_version=EVAL_VERSION, questions=len(qa), with_gold=len(Q), dev=int((Q.split == "dev").sum()),
                 test=int((Q.split == "test").sum()), gold_per_question=round(float(Q.gold.map(len).mean()), 2),
                 calibration=best, citation_reasons=dict(reasons.most_common()))
    log(f"có đáp án: {len(Q):,}/{len(qa):,} câu · dev {stats['dev']:,} · test {stats['test']:,}")
    return Q, stats
