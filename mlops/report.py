"""Báo cáo HTML một trang cho mỗi lần chấm (artifact report.html trong MLflow, hiện trong notebook).

Tự chứa: SVG + CSS, không JavaScript, không thư viện ngoài — MLflow hiển thị artifact HTML trong khung bị
chặn script nên chỉ dùng được loại này. Rê chuột lên cột/thanh để xem số (tooltip của trình duyệt);
mọi số cũng có trong bảng "Xem số liệu" bên dưới mỗi biểu đồ.
"""
from __future__ import annotations

import html
import time
from collections import Counter

BENCHMARK = 0.428          # hybrid trên 494 câu test của notebook benchmark (không rerank)

CSS = """
.viz{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s1wash:rgba(42,120,214,.14);--good:#0ca30c;--goodtext:#006300;
--crit:#d03b3b;--warn:#9a5b06;--other:#c3c2b7}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;
--s1wash:rgba(57,135,229,.22);--goodtext:#0ca30c;--warn:#e8ae55;--other:#4a4a46}}
:root[data-theme="dark"] .viz{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s1wash:rgba(57,135,229,.22);--goodtext:#0ca30c;--warn:#e8ae55;--other:#4a4a46}
*{box-sizing:border-box}
body{margin:0;background:var(--page)}
.viz{background:var(--page);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;padding:24px 16px 48px}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px;font-weight:650}
h2{font-size:16px;margin:0 0 2px;font-weight:600}
.sub{color:var(--ink2);margin:0 0 18px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px;margin:0 0 16px}
.lead{color:var(--ink2);margin:2px 0 12px;max-width:72ch}
.verdict{display:flex;gap:10px;align-items:flex-start;padding:12px 16px;border-radius:10px;border:1px solid var(--border);margin:0 0 16px;background:var(--surface)}
.verdict b{font-weight:650}
.icon{flex:none;width:22px;height:22px;border-radius:50%;display:grid;place-items:center;color:#fff;font-weight:700;font-size:13px}
.ok .icon{background:var(--good)} .bad .icon{background:var(--crit)} .exp .icon{background:var(--muted)}
.hero{display:grid;grid-template-columns:minmax(220px,1.2fr) repeat(3,minmax(140px,1fr));gap:12px}
@media (max-width:760px){.hero{grid-template-columns:1fr 1fr}}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.tile .label{color:var(--ink2);font-size:13px}
.tile .value{font-size:30px;font-weight:600;line-height:1.2;margin:4px 0 2px}
.tile.big .value{font-size:52px}
.tile .note{color:var(--muted);font-size:12.5px}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--ink2);font:12px system-ui,-apple-system,"Segoe UI",sans-serif}
svg .val{fill:var(--ink);font-weight:600;paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
svg .halo{paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
svg .muted{fill:var(--muted)}
.grid{stroke:var(--grid);stroke-width:1}
.base{stroke:var(--axis);stroke-width:1}
.mark{fill:var(--s1)} .mark:hover{opacity:.8}
.other{fill:var(--other)}
.wash{fill:var(--s1wash)}
.ref{stroke:var(--ink2);stroke-width:1.5}
.dot{fill:var(--s1);stroke:var(--surface);stroke-width:2}
details{margin-top:10px;color:var(--ink2)}
summary{cursor:pointer;font-size:13px}
table{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.q{max-width:46ch}
.gloss dt{font-weight:600;margin-top:8px} .gloss dd{margin:2px 0 0;color:var(--ink2)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media (max-width:760px){.two{grid-template-columns:1fr}}
.two .card{margin:0}
.scroll{overflow-x:auto} .scroll svg{min-width:560px}
"""

esc = lambda s: html.escape(str(s), quote=True)
pct = lambda x: f"{x * 100:.0f}%"


def _col_path(x, y, w, h, r=4):
    """Cột mọc từ đáy: bo 4px ở đầu dữ liệu, vuông ở đáy."""
    r = min(r, w / 2, h)
    return f"M{x},{y + h}V{y + r}Q{x},{y} {x + r},{y}H{x + w - r}Q{x + w},{y} {x + w},{y + r}V{y + h}Z"


def _bar_path(x, y, w, h, r=4):
    """Thanh ngang mọc từ trái: bo 4px ở đầu phải."""
    r = min(r, h / 2, w)
    return f"M{x},{y}H{x + w - r}Q{x + w},{y} {x + w},{y + r}V{y + h - r}Q{x + w},{y + h} {x + w - r},{y + h}H{x}Z"


def _table(head, rows, num_cols=(), wide_cols=()):
    c = lambda i: "num" if i in num_cols else ("q" if i in wide_cols else "")
    th = "".join(f'<th class="{c(i)}">{esc(h)}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{c(i)}">{esc(v)}</td>' for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def gauge(s: dict, min_ndcg: float, champion: float | None) -> str:
    """nDCG@10 (chấm + khoảng tin cậy) đặt cạnh các mốc: ngưỡng cổng, benchmark, champion."""
    v, lo, hi = s["ndcg_at_10"], s.get("ndcg_at_10_ci_low", s["ndcg_at_10"]), s.get("ndcg_at_10_ci_high", s["ndcg_at_10"])
    refs = [("Ngưỡng cổng", min_ndcg), ("Benchmark", BENCHMARK)] + ([("Champion", champion)] if champion is not None else [])
    vals = [v, lo, hi] + [r[1] for r in refs]
    a, b = max(0.0, round(min(vals) - 0.05, 1)), min(1.0, round(max(vals) + 0.05, 1) + 0.0001)
    W, L, R, top = 640, 20, 20, 64
    X = lambda t: L + (t - a) / (b - a) * (W - L - R)
    out = [f'<svg viewBox="0 0 {W} 150" role="img" aria-label="nDCG@10 {v:.3f}, khoảng tin cậy {lo:.3f} đến {hi:.3f}">']
    t = a
    while t <= b + 1e-9:
        out.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 8}" y2="{top + 40}"/>'
                   f'<text x="{X(t):.1f}" y="{top + 58}" text-anchor="middle" class="muted">{t:.2f}</text>')
        t = round(t + 0.05, 2)
    out.append(f'<rect class="wash" x="{X(lo):.1f}" y="{top + 6}" width="{max(2, X(hi) - X(lo)):.1f}" height="20" rx="4">'
               f'<title>Khoảng tin cậy 95%: {lo:.3f} – {hi:.3f}</title></rect>')
    for i, (name, r) in enumerate(sorted(refs, key=lambda z: z[1])):
        y = 16 if i % 2 == 0 else 34                     # so le nhãn để không chồng nhau
        out.append(f'<line class="ref" x1="{X(r):.1f}" x2="{X(r):.1f}" y1="{y + 6}" y2="{top + 32}"/>'
                   f'<text x="{X(r):.1f}" y="{y}" text-anchor="middle">{esc(name)} {r:.3f}</text>')
    out.append(f'<circle class="dot" cx="{X(v):.1f}" cy="{top + 16}" r="7"><title>nDCG@10 = {v:.4f}</title></circle>'
               f'<text class="val" x="{X(v):.1f}" y="{top + 42}" text-anchor="middle">{v:.3f}</text></svg>')
    return "".join(out)


def _nice(v: float) -> float:
    """Làm tròn trục lên số đẹp (1, 2, 2.5, 5 × 10^k) để vạch chia sạch."""
    import math
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    return next(m * e for m in (1, 2, 2.5, 5, 10) if m * e >= v)


def columns(labels, values, fmt, title_fmt, other=(), label_idx=None, H=230, W=440, vmax=None) -> str:
    """Cột dọc một chuỗi (màu slot 1); cột trong `other` tô xám. Nhãn số chỉ ở các cột được chọn."""
    L, B, T = 40, 34, 22
    n, vmax = len(values), vmax or _nice(max(values))
    step = (W - L - 10) / n
    bw = min(24, step * 0.62)
    Y = lambda v: T + (1 - v / vmax) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for g in range(5):
        gv = vmax * g / 4
        out.append(f'<line class="grid" x1="{L}" x2="{W - 10}" y1="{Y(gv):.1f}" y2="{Y(gv):.1f}"/>'
                   f'<text x="{L - 8}" y="{Y(gv) + 4:.1f}" text-anchor="end" class="muted">{fmt(gv)}</text>')
    out.append(f'<line class="base" x1="{L}" x2="{W - 10}" y1="{H - B}" y2="{H - B}"/>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = L + step * (i + 0.5)
        h = max(0.0, H - B - Y(v))
        cls = "other" if i in other else "mark"
        out.append(f'<g><path class="{cls}" d="{_col_path(cx - bw / 2, Y(v), bw, h) if h > 0 else ""}"/>'
                   f'<rect x="{cx - step / 2:.1f}" y="{T}" width="{step:.1f}" height="{H - B - T}" fill="transparent">'
                   f'<title>{esc(title_fmt(lab, v))}</title></rect></g>'
                   f'<text x="{cx:.1f}" y="{H - B + 18}" text-anchor="middle">{esc(lab)}</text>')
        if label_idx is None or i in label_idx:
            out.append(f'<text class="val" x="{cx:.1f}" y="{Y(v) - 7:.1f}" text-anchor="middle">{esc(fmt(v))}</text>')
    out.append("</svg>")
    return "".join(out)


def domain_bars(items: list[tuple[str, float, int]], mean: float) -> str:
    """Thanh ngang nDCG@10 theo lĩnh vực (đã sắp), đường dọc = trung bình toàn bộ."""
    rowh, L, R, T = 30, 190, 120, 26
    W, H = 900, T + rowh * len(items) + 24
    X = lambda v: L + v * (W - L - R)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for g in (0, .25, .5, .75, 1):
        out.append(f'<line class="grid" x1="{X(g):.1f}" x2="{X(g):.1f}" y1="{T - 6}" y2="{H - 20}"/>'
                   f'<text x="{X(g):.1f}" y="{H - 4}" text-anchor="middle" class="muted">{g:.2f}</text>')
    out.append(f'<line class="ref" x1="{X(mean):.1f}" x2="{X(mean):.1f}" y1="{T - 10}" y2="{H - 20}"/>'
               f'<text x="{X(mean) + 6:.1f}" y="{T - 12}">trung bình {mean:.2f}</text>')
    for i, (name, v, n) in enumerate(items):
        y = T + i * rowh
        out.append(f'<text x="{L - 10}" y="{y + 15}" text-anchor="end">{esc(name)}</text>'
                   f'<path class="mark" d="{_bar_path(X(0), y + 3, max(0.5, X(v) - X(0)), 18)}"><title>{esc(name)}: nDCG@10 {v:.3f} ({n} câu)</title></path>'
                   f'<text class="val" x="{X(v) + 8:.1f}" y="{y + 16}">{v:.2f}</text>'
                   f'<text class="muted halo" x="{X(v) + 44:.1f}" y="{y + 16}">{n} câu</text>')
    out.append("</svg>")
    return "".join(out)


def build(s: dict, rows: list[dict], desc: dict, status: str, why: str, *, min_ndcg: float = 0.40,
          champion: float | None = None, title: str = "") -> str:
    n = len(rows) or 1
    nd = s["ndcg_at_10"]
    hits = [("Top 1", s.get("hit_at_1", 0)), ("Top 3", s.get("hit_at_3", 0)), ("Top 5", s.get("hit_at_5", 0)), ("Top 10", s.get("hit_at_10", 0))]
    ranks = Counter(int(r.get("first_rank") or 0) for r in rows)
    rank_lab = [str(k) for k in range(1, 11)] + ["Ngoài"]
    rank_val = [ranks.get(k, 0) for k in range(1, 11)] + [ranks.get(0, 0)]
    by = {}
    for r in rows:
        by.setdefault(r.get("lv_name") or "(không rõ)", []).append(r["ndcg@10"])
    doms = sorted(((k, sum(v) / len(v), len(v)) for k, v in by.items() if len(v) >= 10), key=lambda z: -z[1])
    small = sum(len(v) for k, v in by.items() if len(v) < 10)
    fails = [r for r in rows if not r.get("hit@10")]

    cls, icon, head = {"passed": ("ok", "✓", "Đạt cổng chất lượng"), "failed": ("bad", "✕", "Không đạt cổng chất lượng")}.get(
        status, ("exp", "i", "Thí nghiệm (không xét cổng / chưa đẩy lên production)"))
    best_dom, worst_dom = (doms[0], doms[-1]) if doms else (None, None)
    when = time.strftime("%d/%m/%Y %H:%M")

    p = [f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>Kết quả chấm tìm Điều luật</title><style>{CSS}</style></head><body><div class="viz"><div class="wrap">',
         f'<h1>{esc(title or "Kết quả chấm tìm Điều luật")}</h1>',
         f'<p class="sub">{esc(desc.get("collection", ""))} · bộ câu hỏi {esc(desc.get("golden", ""))} · {n} câu hỏi thật của người dân · '
         f'{"có rerank" if desc.get("rerank") else "hybrid 0,7 dense + 0,3 BM25"} · {when}</p>',
         f'<div class="verdict {cls}"><span class="icon" aria-hidden="true">{icon}</span><div><b>{head}.</b> {esc(why)}</div></div>',
         '<div class="hero">',
         f'<div class="tile big"><div class="label">Điểm tổng (nDCG@10)</div><div class="value">{nd:.3f}</div>'
         f'<div class="note">0 = luôn sai, 1 = Điều đúng luôn đứng đầu. Benchmark: {BENCHMARK:.3f}</div></div>',
         f'<div class="tile"><div class="label">Đúng ngay kết quả số 1</div><div class="value">{pct(s.get("hit_at_1", 0))}</div>'
         f'<div class="note">số câu có Điều đúng ở vị trí đầu tiên</div></div>',
         f'<div class="tile"><div class="label">Có Điều đúng trong 10 kết quả</div><div class="value">{pct(s.get("hit_at_10", 0))}</div>'
         f'<div class="note">LLM được đưa {8} Điều đầu để trả lời</div></div>',
         f'<div class="tile"><div class="label">Thời gian tìm / câu</div><div class="value">{s.get("latency_p50_s", 0) * 1000:.0f} ms</div>'
         f'<div class="note">trung vị; 95% câu dưới {s.get("latency_p95_s", 0) * 1000:.0f} ms</div></div>',
         '</div>',
         '<div class="card" style="margin-top:16px"><h2>Điểm tổng so với các mốc</h2>'
         '<p class="lead">Chấm là kết quả lần này, dải mờ là khoảng tin cậy 95% (với ' + str(n) + ' câu, điểm có thể lệch chừng ±'
         f'{(s.get("ndcg_at_10_ci_high", nd) - s.get("ndcg_at_10_ci_low", nd)) / 2:.2f} chỉ do may rủi). Cổng chất lượng yêu cầu chấm nằm bên phải vạch "Ngưỡng cổng"'
         + (' và không thấp hơn Champion quá 0,01.' if champion is not None else '.') + '</p>',
         '<div class="scroll">', gauge(s, min_ndcg, champion), '</div></div>',
         '<div class="two">',
         '<div class="card"><h2>Tìm được Điều đúng trong bao nhiêu kết quả đầu?</h2>'
         f'<p class="lead">Cứ 100 câu hỏi thì {s.get("hit_at_10", 0) * 100:.0f} câu có Điều đúng nằm trong 10 kết quả đầu tiên.</p>',
         columns([h[0] for h in hits], [h[1] for h in hits], pct, lambda l, v: f"{l}: {v * 100:.1f}% số câu", vmax=1.0),
         f'<details><summary>Xem số liệu</summary>{_table(["Trong", "Tỉ lệ câu"], [(h[0], f"{h[1] * 100:.1f}%") for h in hits], (1,))}</details></div>',
         '<div class="card"><h2>Điều đúng đầu tiên đứng ở vị trí nào?</h2>'
         f'<p class="lead">{rank_val[0]} câu đúng ngay vị trí 1; {rank_val[-1]} câu ({rank_val[-1] / n * 100:.0f}%) không có Điều đúng trong top 10 (cột xám "Ngoài").</p>',
         columns(rank_lab, rank_val, lambda v: f"{v:.0f}", lambda l, v: f"Vị trí {l}: {v} câu" if l != "Ngoài" else f"Không có trong top 10: {v} câu",
                 other=(10,), label_idx=(0, 10)),
         f'<details><summary>Xem số liệu</summary>{_table(["Vị trí", "Số câu"], list(zip(rank_lab, rank_val)), (1,))}</details></div>',
         '</div>']
    if doms:
        p += ['<div class="card" style="margin-top:16px"><h2>Điểm theo lĩnh vực</h2>'
              f'<p class="lead">Tốt nhất: {esc(best_dom[0])} ({best_dom[1]:.2f}); yếu nhất: {esc(worst_dom[0])} ({worst_dom[1]:.2f}). '
              'Lĩnh vực điểm thấp có thể do tìm kém, hoặc do đáp án trỏ tới luật cũ đã được thay (vd. Luật Đất đai 2013 → 2024) — '
              'xem các câu sai bên dưới để phân biệt.' + (f' Không vẽ {small} câu thuộc lĩnh vực có dưới 10 câu.' if small else '') + '</p>',
              '<div class="scroll">', domain_bars(doms, nd), '</div>',
              f'<details><summary>Xem số liệu</summary>{_table(["Lĩnh vực", "nDCG@10", "Số câu"], [(d, f"{v:.3f}", c) for d, v, c in doms], (1, 2))}</details></div>']
    if fails:
        groups = {}
        for r in fails:
            groups.setdefault(r.get("lv_name") or "", []).append(r)
        order = sorted(groups.values(), key=len, reverse=True)
        sample = [g[i] for i in range(10) for g in order if i < len(g)][:10]
        p += ['<div class="card"><h2>Ví dụ câu không tìm được Điều đúng</h2>'
              f'<p class="lead">{len(fails)} câu ({len(fails) / n * 100:.0f}%) không có Điều đúng trong top 10. 10 ví dụ, rải qua các lĩnh vực sai nhiều nhất (đủ danh sách: failures.csv). '
              'Nếu kết quả #1 hợp lý mà vẫn bị tính sai, khả năng cao là nhãn đáp án lệch phiên bản luật, không phải lỗi tìm kiếm.</p>',
              _table(["Lĩnh vực", "Câu hỏi", "Hệ thống trả về #1", "Đáp án (số hiệu|Điều)"],
                     [(r.get("lv_name", ""), (r.get("query") or "")[:180], r.get("top1_title") or r.get("top3", "").split("|")[0],
                       r.get("gold", "")) for r in sample], wide_cols=(1,)), '</div>']
    p += ['<div class="card gloss"><h2>Đọc các chỉ số thế nào</h2><dl>'
          '<dt>nDCG@10 (điểm tổng)</dt><dd>Điều đúng càng đứng gần đầu danh sách 10 kết quả thì điểm càng cao. Đứng đầu = 1; đứng thứ 3 ≈ 0,5; '
          'không có trong 10 = 0. Điểm tổng là trung bình mọi câu.</dd>'
          '<dt>Hit@k (đúng trong top k)</dt><dd>Tỉ lệ câu có ít nhất một Điều đúng trong k kết quả đầu.</dd>'
          '<dt>MRR@10</dt><dd>Trung bình của 1 / vị trí Điều đúng đầu tiên (đứng đầu = 1, thứ hai = 0,5…).</dd>'
          '<dt>Đáp án lấy từ đâu</dt><dd>Từ trích dẫn trong câu trả lời của cán bộ Bộ Tư pháp ("khoản 1 Điều 241 Bộ luật Dân sự năm 2015"), '
          'đã kiểm chứng nội dung. Đây là nhãn tự động, có sai sót nhỏ — nên dùng để so các phiên bản với nhau hơn là coi là điểm tuyệt đối.</dd>'
          '</dl></div></div></div></body></html>']
    return "".join(p)
