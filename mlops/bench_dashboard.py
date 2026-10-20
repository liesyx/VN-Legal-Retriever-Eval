"""Dashboard HTML tự chứa (không cần thư viện ngoài) cho benchmark truy xuất."""
import json
from pathlib import Path

TEMPLATE = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark truy xuất pháp luật</title>
<style>
:root{
  --surface-0:#f4f3ef; --surface-1:#fcfcfb; --border:#e3e1da; --grid:#ebe9e3;
  --ink-1:#1f1e1c; --ink-2:#55534d; --ink-3:#8a877f;
  --s1:#2a78d6; --fam-lexical:#2a78d6; --fam-sparse:#eb6834; --fam-dense:#1baf7a;
  --fam-hybrid:#eda100; --fam-rerank:#e87ba4; --fam-structured:#6250d6;
  --seq-0:#cde2fb; --seq-1:#86b6ef; --seq-2:#3987e5; --seq-3:#1c5cab; --seq-4:#0d366b;
  --muted-bar:#c9c6bd; --focus:#2a78d6; --good:#008300;
  color-scheme: light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --surface-0:#121211; --surface-1:#1a1a19; --border:#2e2d2a; --grid:#262523;
    --ink-1:#f0efec; --ink-2:#c3c2b7; --ink-3:#8f8d84;
    --s1:#3987e5; --fam-lexical:#3987e5; --fam-sparse:#d95926; --fam-dense:#199e70;
    --fam-hybrid:#c98500; --fam-rerank:#d55181; --fam-structured:#9085e9;
    --seq-0:#1b2b40; --seq-1:#1c4778; --seq-2:#2a6bc0; --seq-3:#5598e7; --seq-4:#b7d3f6;
    --muted-bar:#4a4843; --focus:#5598e7; color-scheme: dark;
  }
}
:root[data-theme="dark"]{
  --surface-0:#121211; --surface-1:#1a1a19; --border:#2e2d2a; --grid:#262523;
  --ink-1:#f0efec; --ink-2:#c3c2b7; --ink-3:#8f8d84;
  --s1:#3987e5; --fam-lexical:#3987e5; --fam-sparse:#d95926; --fam-dense:#199e70;
  --fam-hybrid:#c98500; --fam-rerank:#d55181; --fam-structured:#9085e9;
  --seq-0:#1b2b40; --seq-1:#1c4778; --seq-2:#2a6bc0; --seq-3:#5598e7; --seq-4:#b7d3f6;
  --muted-bar:#4a4843; --focus:#5598e7; color-scheme: dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--ink-1);
  font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;
  font-variant-numeric:tabular-nums}
.wrap{max-width:1240px;margin:0 auto;padding:24px 16px 48px}
header{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink-2);margin:0}
button.theme{background:var(--surface-1);color:var(--ink-1);border:1px solid var(--border);border-radius:8px;
  padding:6px 12px;cursor:pointer;font:inherit}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:26px;font-weight:650;letter-spacing:-.02em}
.kpi .l{color:var(--ink-2);font-size:12.5px}
.kpi .d{color:var(--ink-3);font-size:12px;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.grid{display:grid;grid-template-columns:1fr;gap:16px}
@media (min-width:920px){.grid.two{grid-template-columns:1fr 1fr}}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px;min-width:0}
.card h2{font-size:15px;margin:0 0 2px}
.card .hint{color:var(--ink-2);font-size:12.5px;margin:0 0 10px}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:4px 0 10px}
select{font:inherit;background:var(--surface-1);color:var(--ink-1);border:1px solid var(--border);border-radius:8px;padding:5px 8px}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--border);background:var(--surface-1);
  color:var(--ink-2);border-radius:999px;padding:3px 10px;font-size:12px;cursor:pointer;user-select:none}
.chip[aria-pressed="true"]{color:var(--ink-1);border-color:var(--ink-3)}
.chip[aria-disabled="true"]{opacity:.45;cursor:not-allowed}
.chip svg{flex:none}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--ink-2);font-size:12px;margin-top:6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}
svg text{fill:var(--ink-2);font-size:11.5px}
svg .lbl{fill:var(--ink-1)}
svg .axis line,svg .axis path{stroke:var(--border)}
svg .gridline{stroke:var(--grid)}
.chart{width:100%;overflow:visible}
.mark{transition:opacity .12s}
.mark:hover,.mark:focus{opacity:.78;outline:none}
.tip{position:fixed;pointer-events:none;z-index:9;background:var(--surface-1);color:var(--ink-1);
  border:1px solid var(--border);border-radius:10px;padding:8px 10px;font-size:12.5px;
  box-shadow:0 6px 24px rgba(0,0,0,.14);max-width:340px;display:none}
.tip .t{color:var(--ink-2);font-size:12px;margin-bottom:4px}
.tip .r{display:flex;align-items:center;gap:8px;justify-content:space-between}
.tip .r .n{color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:220px}
.tip .r b{font-weight:650}
table{border-collapse:collapse;width:100%;font-size:12.5px}
.tablewrap{overflow-x:auto}
th,td{padding:7px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th{color:var(--ink-2);font-weight:600;cursor:pointer;position:sticky;top:0;background:var(--surface-1)}
th:first-child,td:first-child{text-align:left}
td.best{font-weight:700}
td .muted{color:var(--ink-3)}
.badge{font-size:11px;border:1px solid var(--border);border-radius:6px;padding:1px 6px;color:var(--ink-2)}
.badge.sig{color:var(--ink-1);border-color:var(--ink-3)}
.note{color:var(--ink-2);font-size:12.5px}
.note li{margin:3px 0}
footer{color:var(--ink-3);font-size:12px;margin-top:24px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1 id="title"></h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <button class="theme" id="themeBtn" type="button" aria-label="Đổi giao diện sáng/tối">◐ Giao diện</button>
  </header>

  <section class="kpis" id="kpis" aria-label="Chỉ số chính"></section>

  <div class="grid">
    <section class="card">
      <h2>So sánh các phương pháp</h2>
      <p class="hint">Xếp theo metric đang chọn · thanh lỗi = khoảng tin cậy 95% (bootstrap) · <b>ns</b> = chưa khác biệt có ý nghĩa với phương án tham chiếu — phương án tốt nhất trên tập dev (Holm, p ≥ 0,05)</p>
      <div class="controls">
        <label for="metricSel" class="note">Metric</label><select id="metricSel"></select>
        <span id="famLegend" class="legend" style="margin:0"></span>
      </div>
      <div id="barChart"></div>
    </section>

    <section class="card">
      <h2>Recall@k — tìm thấy Điều đúng trong top-k</h2>
      <p class="hint">Bấm vào tên để bật/tắt (tối đa 8). Rê chuột để xem giá trị tại từng k.</p>
      <div class="controls" id="lineChips"></div>
      <div id="lineChart"></div>
    </section>

    <div class="grid two">
      <section class="card">
        <h2>Chất lượng ↔ độ trễ</h2>
        <p class="hint">Trục ngang: ms/câu (thang log). Đường nét đứt nối các phương án Pareto — không phương án nào vừa nhanh hơn vừa tốt hơn chúng.</p>
        <div id="scatterChart"></div>
      </section>
      <section class="card">
        <h2>Theo lĩnh vực pháp luật</h2>
        <p class="hint" id="heatHint"></p>
        <div id="heatChart"></div>
      </section>
    </div>

    <div class="grid two">
      <section class="card">
        <h2>Độ sâu rerank</h2>
        <p class="hint" id="rrHint">Rerank top-N chunk của phương án nền; N = 0 là không rerank.</p>
        <div id="rrChart"></div>
      </section>
      <section class="card">
        <h2>Biến thể chunk</h2>
        <p class="hint">Cùng model embedding, khác cách chia văn bản.</p>
        <div id="chunkChart"></div>
      </section>
    </div>

    <div class="grid two">
      <section class="card">
        <h2>Trọng số α của hybrid (tập dev)</h2>
        <p class="hint">α = 0: chỉ BM25 · α = 1: chỉ dense.</p>
        <div id="alphaChart"></div>
      </section>
      <section class="card">
        <h2>Chất lượng nhãn đáp án</h2>
        <p class="hint">Trích dẫn trong câu trả lời → Điều trong Bộ Pháp điển. Chỉ "ok" được dùng làm đáp án.</p>
        <div id="citeChart"></div>
      </section>
    </div>

    <section class="card">
      <h2>Bảng đầy đủ</h2>
      <p class="hint">Bấm tiêu đề cột để sắp xếp. In đậm = cao nhất trong cột.</p>
      <div class="tablewrap"><table id="mainTable"></table></div>
    </section>

    <section class="card">
      <h2>Ví dụ câu tìm sai</h2>
      <p class="hint" id="failHint"></p>
      <div class="tablewrap"><table id="failTable"></table></div>
    </section>

    <section class="card note">
      <h2>Lưu ý khi đọc</h2>
      <ul id="notes"></ul>
    </section>
  </div>
  <footer id="footer"></footer>
</div>
<div class="tip" id="tip" role="status" aria-live="polite"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const NS = "http://www.w3.org/2000/svg";
const FAM = D.families;                       // [{id,label}]
const famIndex = Object.fromEntries(FAM.map((f,i)=>[f.id,i]));
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const famColor = id => css("--fam-" + id) || css("--s1");
const ML = D.metric_labels;
const P = D.meta.primary;
const fmt = (v, d=3) => (v===null || v===undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const pct = v => (v===null||v===undefined) ? "—" : (100*v).toFixed(1) + "%";

function h(tag, attrs, parent, text){
  const e = document.createElement(tag);
  for (const k in (attrs||{})) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function s(tag, attrs, parent, text){
  const e = document.createElementNS(NS, tag);
  for (const k in (attrs||{})) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function clear(node){ while (node.firstChild) node.removeChild(node.firstChild); }
function widthOf(id){ const n = document.getElementById(id); return Math.max(280, n.clientWidth || 600); }

// ---------- tooltip ----------
const tip = document.getElementById("tip");
function showTip(evt, title, rows){
  clear(tip);
  h("div", {class:"t"}, tip, title);
  for (const r of rows){
    const row = h("div", {class:"r"}, tip);
    const left = h("span", {style:"display:flex;align-items:center;gap:6px;min-width:0"}, row);
    if (r.color){ const k = s("svg",{width:14,height:6},left); s("line",{x1:0,y1:3,x2:14,y2:3,stroke:r.color,"stroke-width":3,"stroke-linecap":"round","stroke-dasharray":r.dash||""},k); }
    h("span", {class:"n"}, left, r.name);
    h("b", {}, row, r.value);
  }
  tip.style.display = "block";
  const pad = 14, vw = innerWidth, vh = innerHeight;
  let x = (evt.clientX ?? 0) + pad, y = (evt.clientY ?? 0) + pad;
  const rect = tip.getBoundingClientRect();
  if (x + rect.width > vw - 8) x = (evt.clientX ?? 0) - rect.width - pad;
  if (y + rect.height > vh - 8) y = (evt.clientY ?? 0) - rect.height - pad;
  tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
}
function hideTip(){ tip.style.display = "none"; }
function focusTip(el, title, rows){
  el.setAttribute("tabindex", "0");
  el.addEventListener("focus", () => { const b = el.getBoundingClientRect(); showTip({clientX:b.right, clientY:b.top}, title, rows()); });
  el.addEventListener("blur", hideTip);
}

// ---------- scales ----------
const lin = (d0,d1,r0,r1) => v => r0 + (v-d0)/((d1-d0)||1)*(r1-r0);
function niceMax(v){ if (v <= 0) return 1; const e = Math.pow(10, Math.floor(Math.log10(v))); const m = v/e; return (m<=1?1:m<=2?2:m<=2.5?2.5:m<=5?5:10)*e; }
function ticks(max, n=5){
  const steps = [1,2,2.5,5].flatMap(b=>[-3,-2,-1,0,1,2,3,4].map(e=>b*Math.pow(10,e))).sort((a,b)=>a-b);
  const st = steps.find(x => max/x <= n) || max/n;
  const out = []; for (let v=0; v<=max+1e-9; v+=st) out.push(+v.toFixed(6)); return out;
}
function textW(t){ return 6.4 * String(t).length; }
function hexLum(c){ const m = /^#?([0-9a-f]{6})$/i.exec(c||""); if (!m) return 0.5; const n = parseInt(m[1],16);
  const ch = [(n>>16)&255,(n>>8)&255,n&255].map(v=>{ v/=255; return v<=0.03928? v/12.92 : Math.pow((v+0.055)/1.055,2.4); });
  return 0.2126*ch[0]+0.7152*ch[1]+0.0722*ch[2]; }

// ---------- methods ----------
const methods = D.methods.slice();
const byName = Object.fromEntries(methods.map(m=>[m.name,m]));
const famMembers = {};
methods.slice().sort((a,b)=>a.name.localeCompare(b.name)).forEach(m=>{ (famMembers[m.family] = famMembers[m.family]||[]).push(m.name); });
const DASHES = ["", "6 4", "2 3", "10 3 2 3"];
const dashOf = name => { const fam = byName[name].family; return DASHES[(famMembers[fam].indexOf(name)) % DASHES.length]; };

// ---------- header + KPIs ----------
document.getElementById("title").textContent = D.meta.title;
document.getElementById("subtitle").textContent = D.meta.subtitle;
document.getElementById("footer").textContent = D.meta.footer;
function renderKpis(){
  const k = document.getElementById("kpis"); clear(k);
  const best = methods.slice().sort((a,b)=>b.metrics[P]-a.metrics[P])[0];
  const fastestTie = methods.filter(m => m.sig !== "*").sort((a,b)=>(a.ms??1e9)-(b.ms??1e9))[0];
  const items = [
    [fmt(best.metrics[P]), `${ML[P]||P} cao nhất`, best.name],
    [pct(best.metrics["recall@10"]), "Recall@10 của phương án tốt nhất", "tỷ lệ Điều đúng nằm trong top-10"],
    [pct(Math.max(...methods.map(m=>m.metrics["recall@100"]||0))), "Recall@100 cao nhất", "trần cho reranker / LLM"],
    [fastestTie ? (fastestTie.ms!=null? fastestTie.ms.toFixed(0)+" ms" : "—") : "—", "Nhanh nhất mà chưa thua có ý nghĩa", fastestTie ? fastestTie.name : ""],
    [D.meta.n_test.toLocaleString("vi-VN"), "câu hỏi chấm điểm (test)", `${D.meta.n_gold.toLocaleString("vi-VN")}/${D.meta.n_total.toLocaleString("vi-VN")} câu có đáp án`],
    [String(methods.length), "phương pháp được so sánh", `${D.meta.n_chunks.toLocaleString("vi-VN")} chunk · ${D.meta.n_articles.toLocaleString("vi-VN")} Điều`],
  ];
  for (const [v,l,d] of items){ const c = h("div",{class:"kpi"},k); h("div",{class:"v"},c,v); h("div",{class:"l"},c,l); h("div",{class:"d",title:d},c,d); }
}

// ---------- family legend ----------
function renderFamLegend(){
  const L = document.getElementById("famLegend"); clear(L);
  const used = new Set(methods.map(m=>m.family));
  for (const f of FAM){ if (!used.has(f.id)) continue; const sp = h("span",{},L); h("i",{class:"dot",style:`background:${famColor(f.id)}`},sp); h("span",{},sp,f.label); }
}

// ---------- bar chart ----------
const metricSel = document.getElementById("metricSel");
for (const m of D.metric_order){ const o = h("option",{value:m},metricSel, ML[m]||m); if (m===P) o.selected = true; }
metricSel.addEventListener("change", renderBars);
function renderBars(){
  const box = document.getElementById("barChart"); clear(box);
  const metric = metricSel.value;
  const rows = methods.slice().sort((a,b)=>(b.metrics[metric]??-1)-(a.metrics[metric]??-1));
  const W = widthOf("barChart"), rowH = 26, labelW = Math.min(300, W*0.38), right = 118, top = 6;
  const H = top + rows.length*rowH + 26;
  const maxV = niceMax(Math.max(...rows.map(r=>(r.ci&&r.ci[metric]? r.ci[metric][1] : r.metrics[metric])||0)));
  const x = lin(0, maxV, labelW, W-right);
  const svg = s("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:"img","aria-label":`Biểu đồ cột ${ML[metric]||metric}`},box);
  for (const t of ticks(maxV, 5)){ const xx = x(t); s("line",{class:"gridline",x1:xx,x2:xx,y1:top,y2:H-22},svg); s("text",{x:xx,y:H-6,"text-anchor":"middle"},svg,fmt(t,2)); }
  rows.forEach((r,i)=>{
    const y = top + i*rowH, v = r.metrics[metric] ?? 0, x1 = x(v), bh = 14, yy = y + (rowH-bh)/2, rad = Math.min(4, (x1-labelW)/2);
    s("text",{x:labelW-8,y:y+rowH/2+4,"text-anchor":"end",class:"lbl"},svg,r.name.length>40? r.name.slice(0,39)+"…" : r.name);
    const w = Math.max(0, x1-labelW);
    const d = w <= 0 ? "" : `M${labelW},${yy} H${x1-rad} Q${x1},${yy} ${x1},${yy+rad} V${yy+bh-rad} Q${x1},${yy+bh} ${x1-rad},${yy+bh} H${labelW} Z`;
    const bar = s("path",{d, fill:famColor(r.family), class:"mark"},svg);
    let endX = x1;
    if (r.ci && r.ci[metric]){
      const [lo,hi] = r.ci[metric]; const cy = yy+bh/2;
      s("line",{x1:x(lo),x2:x(hi),y1:cy,y2:cy,stroke:css("--ink-2"),"stroke-width":1.2},svg);
      for (const cx of [x(lo),x(hi)]) s("line",{x1:cx,x2:cx,y1:cy-4,y2:cy+4,stroke:css("--ink-2"),"stroke-width":1.2},svg);
      endX = Math.max(endX, x(hi));
    }
    const lab = s("text",{x:endX+6,y:y+rowH/2+4,class:"lbl"},svg,fmt(v));
    if (r.sig === "ns"){ s("tspan",{dx:6,fill:css("--ink-3")},lab,"ns"); }
    if (r.sig === "(best)"){ s("tspan",{dx:6,fill:css("--ink-3")},lab,"tham chiếu"); }
    const hit = s("rect",{x:0,y:y,width:W,height:rowH,fill:"transparent"},svg);
    const rowsTip = () => [
      {name: ML[metric]||metric, value: fmt(v)},
      ...(r.ci&&r.ci[metric] ? [{name:"KTC 95%", value:`${fmt(r.ci[metric][0])} – ${fmt(r.ci[metric][1])}`}] : []),
      {name:"Recall@10", value: pct(r.metrics["recall@10"])}, {name:"MRR@10", value: fmt(r.metrics["mrr@10"])},
      {name:"ms/câu", value: r.ms!=null? r.ms.toFixed(1) : "—"},
      {name:"p (Holm) so với tham chiếu", value: r.p_holm==null? "—" : (r.p_holm<0.001? "<0,001" : r.p_holm.toFixed(3))},
    ];
    hit.addEventListener("pointermove", e => { bar.style.opacity = .78; showTip(e, `${r.name} · ${r.desc||""}`, rowsTip()); });
    hit.addEventListener("pointerleave", () => { bar.style.opacity = 1; hideTip(); });
    focusTip(hit, r.name, rowsTip);
  });
}

// ---------- line chart: recall@k ----------
const MAX_LINES = 8;
const lineSel = new Set(methods.slice().sort((a,b)=>b.metrics[P]-a.metrics[P]).slice(0,6).map(m=>m.name));
function renderChips(){
  const C = document.getElementById("lineChips"); clear(C);
  methods.slice().sort((a,b)=> famIndex[a.family]-famIndex[b.family] || a.name.localeCompare(b.name)).forEach(m=>{
    const on = lineSel.has(m.name), dis = !on && lineSel.size >= MAX_LINES;
    const c = h("span",{class:"chip",role:"button",tabindex:"0","aria-pressed":String(on),"aria-disabled":String(dis)},C);
    const k = s("svg",{width:18,height:8},c); s("line",{x1:1,y1:4,x2:17,y2:4,stroke:on?famColor(m.family):css("--ink-3"),"stroke-width":2.5,"stroke-linecap":"round","stroke-dasharray":dashOf(m.name)},k);
    h("span",{},c,m.name);
    const toggle = () => { if (on) lineSel.delete(m.name); else if (!dis) lineSel.add(m.name); renderChips(); renderLines(); };
    c.addEventListener("click", toggle); c.addEventListener("keydown", e => { if (e.key===" "||e.key==="Enter"){ e.preventDefault(); toggle(); } });
  });
}
function lineChart(boxId, xs, series, opts){
  const box = document.getElementById(boxId); clear(box);
  const W = widthOf(boxId), H = opts.height || 300, m = {l:54, r: opts.right ?? 16, t:12, b:34};
  const svg = s("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:"img","aria-label":opts.aria},box);
  const yMax = opts.yMax ?? niceMax(Math.max(0.01, ...series.flatMap(sr=>sr.values.filter(v=>v!=null))));
  const yMin = opts.yMin ?? 0;
  const x = i => m.l + (xs.length===1? (W-m.l-m.r)/2 : i*(W-m.l-m.r)/(xs.length-1));
  const y = lin(yMin, yMax, H-m.b, m.t);
  for (const t of ticks(yMax-yMin, 4).map(t=>t+yMin)){ s("line",{class:"gridline",x1:m.l,x2:W-m.r,y1:y(t),y2:y(t)},svg); s("text",{x:m.l-8,y:y(t)+4,"text-anchor":"end"},svg,opts.yFmt? opts.yFmt(t) : fmt(t,2)); }
  xs.forEach((xv,i)=> s("text",{x:x(i),y:H-m.b+18,"text-anchor":"middle"},svg,String(xv)));
  if (opts.xLabel) s("text",{x:(m.l+W-m.r)/2,y:H-2,"text-anchor":"middle"},svg,opts.xLabel);
  const endLabels = [];
  for (const sr of series){
    const pts = sr.values.map((v,i)=> v==null? null : [x(i), y(v)]).filter(Boolean);
    if (pts.length > 1) s("path",{d:"M"+pts.map(p=>p.join(",")).join(" L"),fill:"none",stroke:sr.color,"stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round","stroke-dasharray":sr.dash||""},svg);
    for (const p of pts) s("circle",{cx:p[0],cy:p[1],r:4,fill:sr.color,stroke:css("--surface-1"),"stroke-width":2},svg);
    if (sr.label && pts.length){ const p = pts[pts.length-1]; endLabels.push({x:p[0]+8, y:p[1]+4, text:sr.label}); }
    if (opts.annotate && opts.annotate.series === sr.name){ const i = opts.annotate.index, v = sr.values[i];
      if (v != null){ const ax = x(i), ay = y(v); s("circle",{cx:ax,cy:ay,r:7,fill:"none",stroke:sr.color,"stroke-width":2},svg);
        const tw = textW(opts.annotate.text); const tx = Math.min(ax - tw/2, W - m.r - tw); s("text",{x:Math.max(m.l, tx),y:ay-12,class:"lbl"},svg,opts.annotate.text); } }
  }
  endLabels.sort((a,b)=>a.y-b.y);
  for (let i=1; i<endLabels.length; i++) if (endLabels[i].y - endLabels[i-1].y < 14) endLabels[i].y = endLabels[i-1].y + 14;
  for (const L of endLabels) s("text",{x:L.x,y:L.y,class:"lbl"},svg,L.text);
  const hair = s("line",{x1:0,x2:0,y1:m.t,y2:H-m.b,stroke:css("--ink-3"),"stroke-width":1,opacity:0},svg);
  const overlay = s("rect",{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:"transparent"},svg);
  const pick = e => { const r = svg.getBoundingClientRect(); const px = (e.clientX - r.left) * (W / r.width);
    let best = 0, bd = 1e9; xs.forEach((_,i)=>{ const d = Math.abs(x(i)-px); if (d<bd){bd=d;best=i;} }); return best; };
  overlay.addEventListener("pointermove", e => { const i = pick(e); hair.setAttribute("x1",x(i)); hair.setAttribute("x2",x(i)); hair.setAttribute("opacity",1);
    const rows = series.map(sr=>({name:sr.name, value: sr.values[i]==null? "—" : (opts.yFmt? opts.yFmt(sr.values[i]) : fmt(sr.values[i])), color:sr.color, dash:sr.dash, raw:sr.values[i]??-1}))
                       .sort((a,b)=>b.raw-a.raw);
    showTip(e, `${opts.xName||""} ${xs[i]}`, rows); });
  overlay.addEventListener("pointerleave", () => { hair.setAttribute("opacity",0); hideTip(); });
  return svg;
}
function renderLines(){
  const names = methods.filter(m=>lineSel.has(m.name)).map(m=>m.name);
  lineChart("lineChart", D.ks, names.map(n=>({name:n, values:D.recall_k[n], color:famColor(byName[n].family), dash:dashOf(n)})),
            {aria:"Recall@k theo k", yMax:1, yFmt:v=>pct(v), xName:"k =", xLabel:"k (số Điều trả về)", height:320});
}

// ---------- scatter: quality vs latency ----------
function renderScatter(){
  const box = document.getElementById("scatterChart"); clear(box);
  const pts = methods.filter(m=>m.ms!=null && m.ms>0);
  if (!pts.length){ h("p",{class:"note"},box,"Không có số liệu độ trễ."); return; }
  const W = widthOf("scatterChart"), H = 300, m = {l:48,r:16,t:16,b:36};
  const svg = s("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:"img","aria-label":"Chất lượng theo độ trễ"},box);
  const lx = pts.map(p=>Math.log10(p.ms)); const x0 = Math.floor(Math.min(...lx)), x1 = Math.ceil(Math.max(...lx)+1e-9);
  const x = lin(x0, x1===x0? x0+1 : x1, m.l, W-m.r);
  const ys = pts.map(p=>p.metrics[P]); const yMax = niceMax(Math.max(...ys)); const y = lin(0, yMax, H-m.b, m.t);
  for (let e=x0; e<=x1; e++){ const xx = x(e); s("line",{class:"gridline",x1:xx,x2:xx,y1:m.t,y2:H-m.b},svg); s("text",{x:xx,y:H-m.b+16,"text-anchor":"middle"},svg, (Math.pow(10,e)>=1? Math.pow(10,e) : Math.pow(10,e).toFixed(2)) + " ms"); }
  for (const t of ticks(yMax,4)){ s("line",{class:"gridline",x1:m.l,x2:W-m.r,y1:y(t),y2:y(t)},svg); s("text",{x:m.l-8,y:y(t)+4,"text-anchor":"end"},svg,fmt(t,2)); }
  s("text",{x:(m.l+W-m.r)/2,y:H-2,"text-anchor":"middle"},svg,`ms/câu (log) · trục dọc: ${ML[P]||P}`);
  const sorted = pts.slice().sort((a,b)=>a.ms-b.ms); const pareto = []; let bestY = -1;
  for (const p of sorted){ if (p.metrics[P] > bestY){ pareto.push(p); bestY = p.metrics[P]; } }
  if (pareto.length > 1) s("path",{d:"M"+pareto.map(p=>`${x(Math.log10(p.ms))},${y(p.metrics[P])}`).join(" L"),fill:"none",stroke:css("--ink-3"),"stroke-width":1.5,"stroke-dasharray":"4 4"},svg);
  const pset = new Set(pareto.map(p=>p.name));
  const boxes = []; const free = b => !boxes.some(o => b.x < o.x+o.w && b.x+b.w > o.x && b.y < o.y+o.h && b.y+b.h > o.y);
  const labelOf = new Map();
  for (const p of pts){ const cx = x(Math.log10(p.ms)), cy = y(p.metrics[P]); boxes.push({x:cx-7, y:cy-7, w:14, h:14}); }   // tránh đè lên điểm
  for (const p of pareto.slice().reverse()){                  // ưu tiên nhãn cho điểm tốt nhất
    const cx = x(Math.log10(p.ms)), cy = y(p.metrics[P]); const t = p.name.length>24? p.name.slice(0,23)+"…" : p.name, w = textW(t);
    for (const [dx,dy] of [[9,-8],[9,14],[-9-w,-8],[-9-w,14]]){ const b = {x:cx+dx, y:cy+dy-11, w, h:14};
      if (b.x >= m.l && b.x+b.w <= W-2 && b.y >= 0 && free(b)){ boxes.push(b); labelOf.set(p.name, {x:b.x, y:cy+dy, t}); break; } }
  }
  for (const p of pts){
    const cx = x(Math.log10(p.ms)), cy = y(p.metrics[P]);
    const c = s("circle",{cx,cy,r:6,fill:famColor(p.family),stroke:css("--surface-1"),"stroke-width":2,class:"mark"},svg);
    if (labelOf.has(p.name)){ const L = labelOf.get(p.name); s("text",{x:L.x,y:L.y,class:"lbl"},svg,L.t); }
    const hit = s("circle",{cx,cy,r:14,fill:"transparent"},svg);
    const rows = () => [{name:ML[P]||P, value:fmt(p.metrics[P])},{name:"ms/câu", value:p.ms.toFixed(1)},{name:"Pareto", value: pset.has(p.name)? "có" : "không"}];
    hit.addEventListener("pointermove", e=>{ c.style.opacity=.78; showTip(e, p.name, rows()); });
    hit.addEventListener("pointerleave", ()=>{ c.style.opacity=1; hideTip(); });
    focusTip(hit, p.name, rows);
  }
}

// ---------- heatmap ----------
function seqColor(t){ const st = ["--seq-0","--seq-1","--seq-2","--seq-3","--seq-4"]; const i = Math.min(4, Math.max(0, Math.round(t*4))); return css(st[i]); }
function renderHeat(){
  const box = document.getElementById("heatChart"); clear(box);
  const bd = D.by_domain; if (!bd || !bd.domains.length){ h("p",{class:"note"},box,"Không có dữ liệu."); return; }
  const cols = methods.slice().sort((a,b)=>b.metrics[P]-a.metrics[P]).slice(0,6).map(m=>m.name);
  document.getElementById("heatHint").textContent = `${ML[P]||P} của 6 phương án đứng đầu, theo lĩnh vực có nhiều câu nhất. Màu đậm hơn = cao hơn.`;
  const W = widthOf("heatChart"), labelW = Math.min(170, W*0.34), top = 84, cellH = 30, padR = 70;
  const cw = (W - labelW - padR)/cols.length, H = top + bd.domains.length*cellH + 4;
  const svg = s("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:"img","aria-label":"Heatmap theo lĩnh vực"},box);
  const all = cols.flatMap(c=>bd.values[c]||[]).filter(v=>v!=null); const vmax = Math.max(0.01, ...all);
  cols.forEach((c,j)=>{ const tx = labelW + j*cw + cw/2; s("text",{x:tx,y:top-8,transform:`rotate(-32 ${tx} ${top-8})`,"text-anchor":"start",class:"lbl"},svg,c.length>20? c.slice(0,19)+"…" : c); });
  bd.domains.forEach((d,i)=>{
    const yy = top + i*cellH;
    s("text",{x:labelW-8,y:yy+cellH/2+4,"text-anchor":"end",class:"lbl"},svg, `${d.name} (${d.n})`);
    cols.forEach((c,j)=>{
      const v = (bd.values[c]||[])[i]; const t = v==null? 0 : v/vmax;
      const fill = v==null? css("--grid") : seqColor(t);
      const r = s("rect",{x:labelW+j*cw+1,y:yy+1,width:cw-2,height:cellH-2,rx:4,fill,class:"mark"},svg);
      const ink = hexLum(fill) > 0.36 ? "#161615" : "#ffffff";      // chọn chữ theo độ sáng thật của ô
      s("text",{x:labelW+j*cw+cw/2,y:yy+cellH/2+4,"text-anchor":"middle",style:`fill:${ink}`},svg, v==null? "—" : fmt(v,2));
      const rows = () => [{name:ML[P]||P, value: fmt(v)}, {name:"Số câu", value:String(d.n)}];
      r.addEventListener("pointermove", e=>showTip(e, `${d.name} · ${c}`, rows()));
      r.addEventListener("pointerleave", hideTip);
      focusTip(r, `${d.name} · ${c}`, rows);
    });
  });
}

// ---------- small charts ----------
function renderRerank(){
  const rr = D.rerank_depth || [];
  if (!rr.length){ const b = document.getElementById("rrChart"); clear(b); h("p",{class:"note"},b,"Không chạy rerank."); return; }
  if (D.meta.rerank_base) document.getElementById("rrHint").textContent = `Rerank top-N chunk của "${D.meta.rerank_base}"; N = 0 là không rerank.`;
  const keys = [P, P + "_mix", "recall@10", "mrr@10"].filter((k,i,a)=>a.indexOf(k)===i && rr.some(r=>r[k]!=null));
  const colors = [css("--s1"), css("--fam-rerank"), css("--fam-sparse"), css("--fam-dense")];
  const lab = k => k.endsWith("_mix") ? `${ML[k.slice(0,-4)]||k} (trộn điểm)` : (ML[k]||k);
  lineChart("rrChart", rr.map(r=>r.depth), keys.map((k,i)=>({name:lab(k), values: rr.map(r=>r[k] ?? null), color:colors[i], dash:DASHES[i % DASHES.length], label: lab(k)})),
            {aria:"Độ sâu rerank", xName:"N =", xLabel:"N (số chunk đưa vào reranker)", height:260, right:110});
}
function renderAlpha(){
  const al = D.alpha || [];
  if (!al.length){ const b = document.getElementById("alphaChart"); clear(b); h("p",{class:"note"},b,"Không có dữ liệu."); return; }
  const best = al.reduce((a,b)=> b.score>a.score? b : a, al[0]);
  const name = ML[P]||P;
  lineChart("alphaChart", al.map(a=>a.alpha), [{name, values: al.map(a=>a.score), color: css("--fam-hybrid")}],
            {aria:"Đường α", xName:"α =", xLabel:"α (trọng số dense)", height:240, right:24,
             annotate:{series:name, index: al.indexOf(best), text:`tốt nhất α = ${best.alpha} (${fmt(best.score)})`}});
}
function hbars(boxId, rows, opts){
  const box = document.getElementById(boxId); clear(box);
  if (!rows.length){ h("p",{class:"note"},box,"Không có dữ liệu."); return; }
  const W = widthOf(boxId), rowH = 28, labelW = Math.min(220, W*0.42), right = 70;
  const H = rows.length*rowH + 24, maxV = opts.max ?? niceMax(Math.max(...rows.map(r=>r.value)));
  const x = lin(0, maxV, labelW, W-right);
  const svg = s("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:"img","aria-label":opts.aria},box);
  for (const t of ticks(maxV,4)){ s("line",{class:"gridline",x1:x(t),x2:x(t),y1:0,y2:H-20},svg); s("text",{x:x(t),y:H-4,"text-anchor":"middle"},svg, opts.tickFmt? opts.tickFmt(t) : fmt(t,2)); }
  rows.forEach((r,i)=>{
    const y = i*rowH, bh = 14, yy = y+(rowH-bh)/2, x1 = x(r.value), rad = Math.min(4,(x1-labelW)/2);
    s("text",{x:labelW-8,y:y+rowH/2+4,"text-anchor":"end",class:"lbl"},svg,r.name);
    const d = x1-labelW <= 0 ? "" : `M${labelW},${yy} H${x1-rad} Q${x1},${yy} ${x1},${yy+rad} V${yy+bh-rad} Q${x1},${yy+bh} ${x1-rad},${yy+bh} H${labelW} Z`;
    const bar = s("path",{d,fill:r.color,class:"mark"},svg);
    s("text",{x:x1+6,y:y+rowH/2+4,class:"lbl"},svg, opts.valFmt? opts.valFmt(r.value) : fmt(r.value));
    const hit = s("rect",{x:0,y,width:W,height:rowH,fill:"transparent"},svg);
    hit.addEventListener("pointermove", e=>{ bar.style.opacity=.78; showTip(e, r.name, r.tip()); });
    hit.addEventListener("pointerleave", ()=>{ bar.style.opacity=1; hideTip(); });
    focusTip(hit, r.name, r.tip);
  });
}
function renderChunks(){
  const cv = (D.chunk_variants||[]).slice().sort((a,b)=>b[P]-a[P]);
  hbars("chunkChart", cv.map(c=>({name:c.name, value:c[P], color:css("--fam-dense"),
     tip:()=>[{name:ML[P]||P,value:fmt(c[P])},{name:"Recall@10",value:pct(c["recall@10"])},{name:"Recall@100",value:pct(c["recall@100"])},{name:"Số chunk",value:(c.n_chunks||0).toLocaleString("vi-VN")}]})),
     {aria:"Biến thể chunk"});
}
function renderCites(){
  const cs = (D.citations||[]).slice().sort((a,b)=>b.n-a.n); const tot = cs.reduce((a,b)=>a+b.n,0) || 1;
  const LBL = {ok:"ok — dùng làm đáp án", name_not_in_corpus:"văn bản không có trong Pháp điển", version_not_in_corpus:"phiên bản cũ (năm khác)",
               version_newer_than_answer:"luật hiện hành mới hơn câu trả lời", so_hieu_not_in_corpus:"số hiệu không có trong Pháp điển",
               ambiguous_version:"không rõ phiên bản", article_not_in_corpus:"Điều không có trong Pháp điển",
               no_year_given:"không ghi năm (bị loại ở chế độ strict)", content_mismatch:"nội dung Điều không khớp câu trả lời (sai phiên bản luật)"};
  hbars("citeChart", cs.map(c=>({name:LBL[c.reason]||c.reason, value:c.n/tot, color: c.reason==="ok"? css("--s1") : css("--muted-bar"),
     tip:()=>[{name:"Số trích dẫn",value:c.n.toLocaleString("vi-VN")},{name:"Tỷ lệ",value:pct(c.n/tot)}]})),
     {aria:"Kết quả map trích dẫn", max:1, valFmt:v=>pct(v), tickFmt:v=>pct(v)});
}

// ---------- tables ----------
let sortKey = P, sortDir = -1;
function renderTable(){
  const T = document.getElementById("mainTable"); clear(T);
  const cols = [["name","Phương pháp"],["family","Nhóm"],...D.table_metrics.map(m=>[m, ML[m]||m]),["ms","ms/câu"],["p_holm","p (Holm)"],["sig",""]];
  const thead = h("thead",{},T), tr = h("tr",{},thead);
  for (const [k,l] of cols){ const th = h("th",{scope:"col"},tr, l + (sortKey===k? (sortDir<0?" ↓":" ↑") : ""));
    th.addEventListener("click", ()=>{ if (sortKey===k) sortDir*=-1; else { sortKey=k; sortDir = (k==="name"||k==="family"||k==="ms"||k==="p_holm")? 1 : -1; } renderTable(); }); }
  const val = (m,k) => k==="name"? m.name : k==="family"? (FAM[famIndex[m.family]]||{}).label : k==="ms"? m.ms : k==="p_holm"? m.p_holm : k==="sig"? m.sig : m.metrics[k];
  const rows = methods.slice().sort((a,b)=>{ const va = val(a,sortKey), vb = val(b,sortKey); if (va==null) return 1; if (vb==null) return -1; return (va>vb?1:va<vb?-1:0)*sortDir; });
  const best = Object.fromEntries(D.table_metrics.map(k=>[k, Math.max(...methods.map(m=>m.metrics[k]??-1))]));
  const tb = h("tbody",{},T);
  for (const m of rows){
    const r = h("tr",{},tb);
    const c0 = h("td",{},r); h("i",{class:"dot",style:`background:${famColor(m.family)};margin-right:6px`},c0); h("span",{},c0,m.name);
    if (m.desc) h("div",{class:"muted",style:"font-size:11px;white-space:normal;max-width:340px"},c0,m.desc);
    h("td",{},r,(FAM[famIndex[m.family]]||{}).label||m.family);
    for (const k of D.table_metrics){ const v = m.metrics[k]; h("td",{class: v===best[k]? "best":""},r, fmt(v)); }
    h("td",{},r, m.ms!=null? m.ms.toFixed(1) : "—");
    h("td",{},r, m.p_holm==null? "—" : (m.p_holm<0.001? "<0,001" : m.p_holm.toFixed(3)));
    const ct = h("td",{},r); h("span",{class:"badge"+(m.sig==="*"?" sig":"")},ct, m.sig==="(best)"? "tham chiếu" : m.sig==="*"? "kém hơn *" : "ns");
  }
}
function renderFails(){
  const T = document.getElementById("failTable"); clear(T);
  const f = D.failures || [];
  document.getElementById("failHint").textContent = f.length ? `Một số câu "${D.meta.best}" không có Điều đúng nào trong top-10 (tổng ${D.meta.n_fail.toLocaleString("vi-VN")} câu). Xem đầy đủ trong file failures_*.csv.` : "Không có.";
  if (!f.length) return;
  const tr = h("tr",{},h("thead",{},T)); for (const l of ["Câu hỏi","Lĩnh vực","Đáp án","Hạng Điều đúng","Hệ thống trả về (top-3)"]) h("th",{},tr,l);
  const tb = h("tbody",{},T);
  for (const x of f){ const r = h("tr",{},tb);
    for (const [v,st] of [[x.title,"white-space:normal;min-width:240px;text-align:left"],[x.lv||"",""],[x.gold,"white-space:normal;min-width:160px;text-align:left"],[x.first_rank? String(x.first_rank) : ">100",""],[x.top3,"white-space:normal;min-width:240px;text-align:left"]])
      h("td",{style:st},r,v); }
}
function renderNotes(){ const U = document.getElementById("notes"); clear(U); for (const n of D.notes) h("li",{},U,n); }

// ---------- theme ----------
document.getElementById("themeBtn").addEventListener("click", ()=>{
  const r = document.documentElement, cur = r.getAttribute("data-theme");
  const sysDark = matchMedia("(prefers-color-scheme: dark)").matches;
  const next = cur ? (cur==="dark"? "light" : "dark") : (sysDark? "light" : "dark");
  r.setAttribute("data-theme", next); renderAll();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", renderAll);

function renderAll(){ renderKpis(); renderFamLegend(); renderBars(); renderChips(); renderLines(); renderScatter(); renderHeat(); renderRerank(); renderChunks(); renderAlpha(); renderCites(); renderTable(); renderFails(); renderNotes(); }
let rt; addEventListener("resize", ()=>{ clearTimeout(rt); rt = setTimeout(renderAll, 150); });
renderAll();
window.__dashboardReady = true;
})();
</script>
</body>
</html>
"""


def _clean(x):
    """numpy -> Python, NaN/Inf -> None (JSON.parse không nhận NaN)."""
    import math
    try:
        import numpy as np
        if isinstance(x, np.generic):
            x = x.item()
        elif isinstance(x, np.ndarray):
            x = x.tolist()
    except ImportError:
        pass
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def render(data: dict, path) -> Path:
    payload = json.dumps(_clean(data), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload)
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
