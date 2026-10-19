"use strict";

const API = document.querySelector('meta[name="legal-api"]')?.content || "/ask";   // backend legal-api
const TIMEOUT_MS = 120000;
const STEP_AT = [0, 6000, 9000]; // n8n trả một lần: mốc chuyển bước ước theo thời gian thực tế
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (s, el = document) => el.querySelector(s);
const thread = $("#thread");
const form = $("#composer");
const input = $("#q");
const send = $("#send");
const tpl = $("#tpl-turn");

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
      body: JSON.stringify({ question, k: 5 }),
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
  if (!problem && data && data.ok && data.answer) {
    const answer = $(".answer", turn);
    answer.innerHTML = renderAnswer(data.answer);
    answer.hidden = false;
    renderSources(turn, data.sources || [], data.used_sources || []);
  } else {
    $(".failure-text", failure).textContent = problem || data?.reply_text || "Chưa tìm thấy Điều luật phù hợp. Thử kể rõ hơn tình huống của bạn.";
    failure.hidden = false;
  }
}

function addTurn(question) {
  const turn = tpl.content.firstElementChild.cloneNode(true);
  $(".question", turn).textContent = question;
  $(".retry", turn).addEventListener("click", () => ask(question, turn));
  thread.append(turn);
  document.body.classList.add("has-thread");
  turn.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
  ask(question, turn);
}

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

/* ---------- composer ---------- */
function setBusy(busy) {
  send.disabled = busy;
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
