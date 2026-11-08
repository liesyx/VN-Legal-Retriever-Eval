"use strict";

const API = document.querySelector('meta[name="legal-api"]')?.content || "/ask";   // backend legal-api
const FEEDBACK = API.replace(/\/ask$/, "/feedback");
const TIMEOUT_MS = 120000;
const STEP_AT = [0, 6000, 9000]; // n8n trả một lần: mốc chuyển bước ước theo thời gian thực tế
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (s, el = document) => el.querySelector(s);
const thread = $("#thread");
const form = $("#composer");
const input = $("#q");
const send = $("#send");
const tpl = $("#tpl-turn");
const newChat = $("#new-chat");

/* ---------- ngữ cảnh hỏi tiếp: chỉ giữ trong tab này, server không lưu ----------
   Mỗi lượt trả lời thành công thêm {q, a}; lần hỏi sau gửi kèm MAX_TURNS lượt gần nhất.
   n8n ghép câu hỏi tiếp với lịch sử thành câu hỏi đầy đủ rồi mới tìm Điều luật.
   Tải lại trang hoặc bấm "Cuộc hỏi mới" là bắt đầu lại. */
const MAX_TURNS = 4;
const MAX_ANSWER = 1200;     // API nhận tối đa 1500 ký tự mỗi câu trả lời
let history = [];
const PLACEHOLDER = input.placeholder;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const today = new Date().toISOString().slice(0, 10);
const vnDate = (d) => (d ? d.split("-").reverse().join("/") : "");

/* ---------- answer text: paragraphs, lists, **bold**, [n] citations ---------- */
function inline(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|\s)\*(\S[^*]*?)\*(?=\s|$|[.,;:])/g, "$1<em>$2</em>")
    .replace(/\[(\d{1,2})\]/g, '<button type="button" class="cite" data-n="$1" aria-label="Xem Điều luật số $1">$1</button>');
}

function renderAnswer(text) {
  const blocks = String(text || "").trim().split(/\n\s*\n/);
  const BULLET = /^([-*•]|\d+[.)])\s+/;
  const para = (lines) => {
    const plain = lines.join(" ").replace(/^\*(.*)\*$/, "$1");
    const caveat = /thông tin tham khảo/i.test(plain) && plain.length < 160;
    return `<p${caveat ? ' class="caveat"' : ""}>${inline(plain)}</p>`;
  };
  return blocks.map((b) => {
    const out = [];
    let buf = [], list = [];
    const flush = () => {
      if (buf.length) out.push(para(buf));
      if (list.length) out.push("<ul>" + list.map((l) => `<li>${inline(l)}</li>`).join("") + "</ul>");
      buf = []; list = [];
    };
    for (const l of b.split("\n").map((x) => x.trim()).filter(Boolean)) {
      if (BULLET.test(l)) { if (buf.length) { out.push(para(buf)); buf = []; } list.push(l.replace(BULLET, "")); }
      else { if (list.length) flush(); buf.push(l); }
    }
    flush();
    return out.join("");
  }).join("");
}

/* ---------- source slips ---------- */
function slip(s) {
  const name = String(s.title || "").replace(/^Điều\s+\S+\s+/, "");
  const future = s.effective_date && s.effective_date > today;
  const status = s.effective_date
    ? `<span class="status ${future ? "wait" : "ok"}">${future ? "Chưa có hiệu lực, áp dụng từ " : "Hiệu lực từ "}${vnDate(s.effective_date)}</span>`
    : "";
  const body = String(s.text || "").split("\n").slice(1)
    .map((l) => (l.trim() === "…" ? '<span class="elide">[…]</span>' : esc(l))).join("\n").trim();
  const li = document.createElement("li");
  li.className = "slip";
  li.dataset.n = s.n;
  li.innerHTML = `
    <span class="seal" aria-hidden="true">${s.n}</span>
    <p class="slip-cite">${esc(s.citation)}</p>
    <p class="slip-title">${esc(name)}</p>
    ${status}
    ${body ? `<details><summary>Đọc nội dung Điều</summary><div class="slip-text">${body}</div></details>` : ""}`;
  return li;
}

function renderSources(turn, sources, used) {
  const aside = $(".turn-sources", turn);
  if (!sources.length) return;
  const cited = used.length ? sources.filter((s) => used.includes(s.n)) : sources;
  const rest = used.length ? sources.filter((s) => !used.includes(s.n)) : [];
  cited.forEach((s) => $(".slips", aside).append(slip(s)));
  if (rest.length) {
    rest.forEach((s) => $(".slips-more", aside).append(slip(s)));
    $(".more", aside).hidden = false;
  }
  aside.hidden = false;
}

/* ---------- one question/answer turn ---------- */
function startProgress(turn) {
  const steps = [...turn.querySelectorAll(".steps li")];
  const timers = STEP_AT.map((t, i) => setTimeout(() => {
    steps.forEach((el, j) => { el.classList.toggle("done", j < i); el.classList.toggle("active", j === i); });
  }, t));
  return () => timers.forEach(clearTimeout);
}

const forHistory = (answer) => String(answer || "")
  .replace(/\s*\[(\d{1,2})\]/g, "")                       // số trích dẫn [n] chỉ có nghĩa trong lượt đó
  .replace(/\n*[^\n]*thông tin tham khảo[^\n]*$/i, "")   // bỏ câu nhắc cuối
  .replace(/\s+/g, " ").trim().slice(0, MAX_ANSWER);

async function ask(question, turn) {
  const stop = startProgress(turn);
  const progress = $(".progress", turn);
  const failure = $(".failure", turn);
  failure.hidden = true;
  progress.hidden = false;
  setBusy(true);

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  let data = null, problem = "";
  try {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: 5, history: turn._hist }),
      signal: ctrl.signal,
    });
    data = await res.json().catch(() => null);
    if (!res.ok) problem = (data && typeof data.detail === "string" && data.detail) || `Máy chủ tra cứu trả lỗi ${res.status}. Gửi lại sau ít phút.`;
  } catch (e) {
    problem = e.name === "AbortError"
      ? "Hệ thống mất quá 2 phút mà chưa trả lời. Gửi lại câu hỏi, hoặc viết ngắn gọn hơn."
      : "Không kết nối được hệ thống tra cứu. Kiểm tra mạng rồi gửi lại.";
  } finally {
    clearTimeout(timer);
    stop();
    setBusy(false);
  }

  progress.hidden = true;
  $(".vote", turn).hidden = true;
  if (!problem && data && data.ok && data.answer) {
    const answer = $(".answer", turn);
    answer.innerHTML = renderAnswer(data.answer);
    answer.hidden = false;
    renderSources(turn, data.sources || [], data.used_sources || []);
    showUnderstood(turn, question, data.standalone_question, data.rerank_query);
    showUnverified(turn, data.unverified_citations);
    const standalone = String(data.standalone_question || "").trim();
    turn._fb = { trace_id: data.trace_id || null, question, standalone,
                 sources: (data.sources || []).map((x) => x.key).filter(Boolean) };
    if (!turn._done) {                    // gửi lại một lượt đã trả lời thì không thêm lần nữa
      history.push({ q: question, a: forHistory(data.answer) });
      history = history.slice(-MAX_TURNS);
      turn._done = true;
    }
    resetVote(turn);
  } else {
    $(".failure-text", failure).textContent = problem || data?.reply_text || "Chưa tìm thấy Điều luật phù hợp. Thử kể rõ hơn tình huống của bạn.";
    failure.hidden = false;
  }
}

// Câu hỏi tiếp: hiện câu đầy đủ đã ghép ngữ cảnh. Câu đầu (hoặc câu đầy đủ trùng câu gốc): hiện tiêu đề vấn đề pháp lý
// mà LLM rút ra (rerank_query). Không có gì khác câu gốc thì ẩn.
function showUnderstood(turn, question, standalone, title) {
  const el = $(".understood", turn);
  const q = question.trim().toLowerCase();
  const s = String(standalone || "").trim();
  const t = String(title || "").trim();
  const text = (turn._hist.length && s && s.toLowerCase() !== q) ? s
             : (t && t !== "-" && t.toLowerCase() !== q) ? t : "";
  el.hidden = !text;
  if (text) el.innerHTML = `Hệ thống hiểu câu hỏi là: <b>${esc(text)}</b>`;
}

function showUnverified(turn, cites) {
  const el = $(".unverified", turn);
  const list = Array.isArray(cites) ? cites.filter(Boolean) : [];
  el.hidden = !list.length;
  if (list.length) el.textContent = `Câu trả lời có nhắc ${list.join(", ")} nhưng các Điều này không nằm trong danh sách Điều luật hệ thống tìm được. Hãy tự kiểm tra lại trước khi dựa vào.`;
}

function addTurn(question) {
  const turn = tpl.content.firstElementChild.cloneNode(true);
  turn._hist = history.slice(-MAX_TURNS);          // ngữ cảnh tại lúc hỏi (gửi lại câu này vẫn dùng đúng ngữ cảnh đó)
  $(".question", turn).textContent = question;
  $(".retry", turn).addEventListener("click", () => ask(question, turn));
  thread.append(turn);
  document.body.classList.add("has-thread");
  newChat.hidden = false;
  input.placeholder = "Hỏi tiếp, ví dụ: thế nếu chưa làm hợp đồng tặng cho thì sao?";
  turn.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
  ask(question, turn);
}

/* ---------- 👍/👎: gửi về backend, đếm trong Grafana, câu 👎 được gom để bổ sung bộ câu hỏi chuẩn ---------- */
function resetVote(turn) {
  const box = $(".vote", turn);
  box.querySelectorAll(".vote-btn").forEach((b) => { b.disabled = false; b.removeAttribute("aria-pressed"); });
  $(".vote-thanks", box).hidden = true;
  box.hidden = false;
}

thread.addEventListener("click", (e) => {
  const btn = e.target.closest(".vote-btn");
  if (!btn) return;
  const turn = btn.closest(".turn");
  const box = btn.closest(".vote");
  box.querySelectorAll(".vote-btn").forEach((b) => { b.disabled = true; });
  btn.setAttribute("aria-pressed", "true");
  $(".vote-thanks", box).hidden = false;
  fetch(FEEDBACK, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vote: btn.dataset.vote, ...(turn._fb || {}) }),
    keepalive: true,
  }).catch(() => {});
});

/* ---------- citation [n] -> its slip ---------- */
thread.addEventListener("click", (e) => {
  const btn = e.target.closest(".cite");
  if (!btn) return;
  const turn = btn.closest(".turn");
  const target = turn.querySelector(`.slip[data-n="${btn.dataset.n}"]`);
  if (!target) return;
  const more = target.closest(".more");
  if (more) more.open = true;
  const d = $("details", target);
  if (d) d.open = true;
  turn.querySelectorAll(".cite[aria-pressed]").forEach((b) => b.removeAttribute("aria-pressed"));
  btn.setAttribute("aria-pressed", "true");
  target.classList.remove("lit");
  void target.offsetWidth;          // chạy lại hiệu ứng đóng dấu
  target.classList.add("lit");
  target.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "nearest" });
  clearTimeout(target._t);
  target._t = setTimeout(() => target.classList.remove("lit"), 2200);
});

/* ---------- cuộc hỏi mới: xoá ngữ cảnh ---------- */
newChat.addEventListener("click", () => {
  if (!newChat.disabled) {
    history = [];
    thread.replaceChildren();
    document.body.classList.remove("has-thread");
    newChat.hidden = true;
    input.placeholder = PLACEHOLDER;
    input.focus();
  }
});

/* ---------- composer ---------- */
function setBusy(busy) {
  send.disabled = busy;
  newChat.disabled = busy;
  send.textContent = busy ? "Đang tra…" : "Hỏi";
}

function fit() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight + 3, 200) + "px";
}

input.addEventListener("input", () => { fit(); input.setCustomValidity(""); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (send.disabled) return;
  const q = input.value.trim();
  if (q.length < 8) {
    input.setCustomValidity("Viết câu hỏi dài hơn một chút để hệ thống hiểu tình huống.");
    input.reportValidity();
    return;
  }
  input.value = "";
  fit();
  addTurn(q);
});

document.querySelectorAll(".example").forEach((b) =>
  b.addEventListener("click", () => { input.value = b.textContent.trim(); fit(); form.requestSubmit(); }));
