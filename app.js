/* 虹色のオーブ トレーナー: 画面側。AI と局の進行は worker.js（Python）に任せ、ここは表示・入力・保存（localStorage）だけ */
const VERSION = "2026-10-29b";
const $ = (s) => document.querySelector(s);
const GROUP_NAMES = { pair_v: "縦2マス", pair_lr: "横2マス", pair_diag: "ななめ2マス", quad: "2×2" };
const GROUPS = [["single", "単体"], ["pair", "2マス"], ["quad", "面・みだれ"], ["aim", "ねらい（会心7倍）"], ["temp", "温度・必殺"]];
const GROUP_OF = {};
let META = null, S = null, sel = null, auto = false, aiSeq = 0, ready = false;

// ---------------------------------------------------------------- Worker とのやり取り
const W = new Worker("worker.js?v=" + VERSION);
const pending = new Map();
let rid = 0;
function call(fn, ...args) {
  return new Promise((res, rej) => {
    const id = ++rid;
    pending.set(id, { res, rej });
    W.postMessage({ id, fn, args });
  });
}
W.onmessage = (e) => {
  const m = e.data;
  if (m.type === "progress") { setLoading(m.msg, m.step); return; }
  if (m.type === "ready") { onReady(m.info); return; }
  if (m.type === "fatal") { setLoading("起動に失敗しました: " + m.error + "\n（通信環境を確認して再読み込みしてください）", -1); return; }
  if (m.id) {
    const p = pending.get(m.id); pending.delete(m.id);
    if (!p) return;
    if (m.error) p.rej(new Error(m.error)); else p.res(m.result);
  }
};
W.onerror = (e) => setLoading("起動に失敗しました: " + (e.message || e), -1);
async function api(fn, ...args) {
  let j;
  try { j = await call(fn, ...args); }
  catch (e) { toast("計算に失敗: " + e.message.split("\n").slice(-2).join(" ").slice(0, 160)); throw e; }
  if (j && j.error) { toast(j.error); throw new Error(j.error); }
  return j;
}

function setLoading(msg, step) {
  const el = $("#loading");
  $("#loadMsg").textContent = msg;
  if (step >= 0) $("#loadBar").style.width = Math.round(step / 3 * 100) + "%";
  if (step < 0) el.classList.add("err");
  el.classList.remove("done");
}
async function onReady(info) {
  META = await call("meta");
  META.skill_group.forEach(([g, names]) => names.forEach((n) => (GROUP_OF[n] = g)));
  $("#aiModel").textContent = "学習型 " + (info.model || "");
  buildReal();
  ready = true;
  $("#loading").classList.add("done");
  const saved = loadSave();
  try {
    if (saved && saved.seed) { S = await api("restore", JSON.stringify(saved)); render(); askAI(); }
    else await newGame();
  } catch (e) { await newGame(); }
}

// ---------------------------------------------------------------- 保存（この端末のブラウザだけ）
function loadSave() { try { return JSON.parse(localStorage.getItem("orb_save") || "null"); } catch (e) { return null; } }
function save() { try { if (S && S.save) localStorage.setItem("orb_save", JSON.stringify(S.save)); } catch (e) {} }
function kifuAppend(line) {
  if (!line) return;
  try {
    const arr = JSON.parse(localStorage.getItem("orb_kifu") || "[]");
    arr.push(new Date().toISOString().slice(0, 19).replace("T", " ") + "\t" + line);
    while (arr.length > 3000) arr.shift();
    localStorage.setItem("orb_kifu", JSON.stringify(arr));
    $("#kifuCount").textContent = arr.length;
  } catch (e) {}
}
function kifuCount() { try { return JSON.parse(localStorage.getItem("orb_kifu") || "[]").length; } catch (e) { return 0; } }
$("#kifuDl").addEventListener("click", () => {
  let arr = [];
  try { arr = JSON.parse(localStorage.getItem("orb_kifu") || "[]"); } catch (e) {}
  const head = "日時\t商材\t-\tseed\t手番\t特技\t対象\tメモ";
  const blob = new Blob(["﻿" + head + "\n" + arr.join("\n") + "\n"], { type: "text/tab-separated-values;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "orb_kifu.tsv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ---------------------------------------------------------------- 表示
function toast(m) { const t = $("#toast"); t.textContent = m; t.classList.add("on"); clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove("on"), 2600); }
function pct(p) { return (p * 100).toFixed(p >= 0.995 ? 0 : 1) + "%"; }
function cellSub(c) {
  if (c.pinned) return "★ 本会心済（理想値で停止）";
  if (c.st === "over") return "はみ出し +" + (-c.rem_hi);
  if (c.st === "zone") return "範囲内（上限まで +" + c.rem_hi + "）";
  return "残り +" + c.rem_lo + "〜+" + c.rem_hi;
}
function renderBoard(el, cells, regen, opts) {
  opts = opts || {}; el.innerHTML = "";
  cells.forEach((c) => {
    const d = document.createElement("div");
    d.className = "cell " + c.st + (regen === c.name ? " regen" : "");
    d.style.gridRow = c.row + 1; d.style.gridColumn = c.col + 1; d.dataset.name = c.name;
    d.innerHTML = `<span class="n">${c.name}</span><span class="range">${c.lo}〜${c.hi}</span><div class="val">${c.v}</div><div class="sub">${cellSub(c)}</div>` +
      (c.err != null ? `<span class="err" style="color:${c.err ? "var(--red)" : "var(--green)"}">誤差 ${c.err}</span>` : "");
    if (opts.onClick) d.addEventListener("click", () => opts.onClick(c.name));
    el.appendChild(d);
  });
}
function render() {
  if (!S) return;
  $("#turnPill").textContent = "手番 " + S.turn + (S.finished ? "・終了" : "") + "・seed " + S.seed;
  $("#temp").textContent = S.temp; $("#focus").textContent = S.focus; $("#focusMax").textContent = S.focus_max;
  $("#focusBar").style.width = (100 * S.focus / S.focus_max) + "%";
  $("#hissatsu").textContent = S.hissatsu; $("#hissatsu").style.color = S.hissatsu_charged ? "var(--gold)" : "";
  $("#regen").textContent = S.regen ? S.regen + "（200℃の倍数で戻る）" : "—";
  renderBoard($("#board"), S.cells, S.regen, { onClick: (n) => onCell(n) });
  const sb = $("#skillsBox"); sb.innerHTML = "";
  GROUPS.forEach(([g, label]) => {
    const h = document.createElement("div"); h.className = "grp"; h.textContent = label; sb.appendChild(h);
    const grid = document.createElement("div"); grid.className = "skills";
    S.skills.filter((s) => GROUP_OF[s.name] === g).forEach((s) => {
      const b = document.createElement("button");
      b.className = "sk" + (s.name === "必殺" ? " hissatsu" : "") + (sel === s.name ? " on" : "");
      b.disabled = !s.ok; b.innerHTML = `<b>${s.name}</b><span>${s.desc}</span><em>${s.cost || ""}</em>`; b.dataset.skill = s.name;
      b.addEventListener("click", () => onSkill(s)); grid.appendChild(b);
    });
    sb.appendChild(grid);
  });
  $("#undo").disabled = !S.can_undo; $("#finish").disabled = S.finished; $("#playTop").disabled = S.finished;
  const lg = $("#log"); lg.innerHTML = S.log.map((l) => `<div>${l.replace(/</g, "&lt;")}</div>`).join(""); lg.scrollTop = lg.scrollHeight;
  $("#kifuCount").textContent = kifuCount();
  renderSel();
  save();
}
function renderSel() {
  const bar = $("#selbar");
  document.querySelectorAll(".cell").forEach((c) => c.classList.remove("pick"));
  $("#groups").innerHTML = "";
  if (!sel) { bar.style.display = "none"; return; }
  const sk = S.skills.find((s) => s.name === sel);
  bar.style.display = "flex"; $("#selName").textContent = sel;
  if (sk.tg === "cell") { $("#selHint").textContent = "→ マスを選ぶ"; document.querySelectorAll(".cell").forEach((c) => c.classList.add("pick")); }
  else {
    $("#selHint").textContent = "→ " + GROUP_NAMES[sk.tg] + "を選ぶ";
    (S.groups[sk.tg] || []).forEach((gname) => {
      const ch = document.createElement("button"); ch.className = "chip"; ch.textContent = gname.split("").join("+");
      ch.addEventListener("click", () => doMove(sel, gname)); $("#groups").appendChild(ch);
    });
  }
}
function onSkill(s) {
  if (!s.ok) return;
  if (s.tg === "none" || s.tg === "rand4") { doMove(s.name, ""); return; }
  sel = sel === s.name ? null : s.name; render();
  if (sel && window.innerWidth <= 860) $("#selbar").scrollIntoView({ behavior: "smooth", block: "start" });
}
function onCell(n) { if (!sel) return; const sk = S.skills.find((s) => s.name === sel); if (sk.tg !== "cell") return; doMove(sel, n); }
$("#selCancel").addEventListener("click", () => { sel = null; render(); });

let busy = false;
async function doMove(skill, targets, byAI) {
  if (busy || !S || S.finished) return;
  busy = true;
  try { S = await api("move", skill, targets || "", !!byAI); kifuAppend(S.kifu); sel = null; render(); askAI(); }
  finally { busy = false; }
}
async function askAI() {
  const seq = ++aiSeq;
  $("#recs").innerHTML = '<div class="wait">考え中</div>'; $("#pNow").textContent = "—"; $("#pVal").textContent = "—";
  if (S.finished) { $("#recs").innerHTML = '<div class="hint-text">局は終了しました。</div>'; $("#sticky").classList.remove("on"); return; }
  try {
    const t0 = performance.now();
    const ai = await api("ai");
    if (seq !== aiSeq) return;
    S.ai = ai; renderRecs($("#recs"), ai, { onPick: (m) => playRec(m) });
    $("#pNow").textContent = pct(ai.now); $("#pVal").textContent = pct(ai.value);
    $("#aiTime").textContent = ((performance.now() - t0) / 1000).toFixed(1) + " 秒";
    highlight(ai.moves[0]); sticky(ai.moves[0]);
    if (auto) setTimeout(() => autoStep(), 350);
  } catch (e) { if (seq === aiSeq) $("#recs").innerHTML = '<div class="hint-text">計算に失敗しました。</div>'; }
}
function renderRecs(el, ai, opts) {
  el.innerHTML = "";
  if (!ai.moves.length) { el.innerHTML = '<div class="hint-text">打てる手がありません。しあげてください。</div>'; return; }
  ai.moves.forEach((m, i) => {
    const d = document.createElement("div"); d.className = "rec" + (i === 0 ? " top" : "");
    d.innerHTML = `<div class="rank">${i + 1}</div><div class="mv">${m.label}<small>${m.exact ? "厳密解" : "評価関数"}${m.skill === null ? "・ここで終える" : ""}</small></div><div class="p">${pct(m.p)}<small>大成功</small></div><div class="pb"><i style="width:${Math.round(m.p * 100)}%"></i></div>`;
    if (opts.onPick) d.addEventListener("click", () => opts.onPick(m));
    el.appendChild(d);
  });
}
function sticky(m) {
  const s = $("#sticky");
  if (!m || S.finished) { s.classList.remove("on"); return; }
  $("#stMv").textContent = m.label; $("#stP").textContent = pct(m.p); s.classList.add("on");
}
$("#stPlay").addEventListener("click", () => { if (S.ai && S.ai.moves.length) playRec(S.ai.moves[0]); });
function highlight(m) {
  document.querySelectorAll(".cell.hint,.sk.hint").forEach((e) => e.classList.remove("hint"));
  if (!m || !m.skill) return;
  const b = document.querySelector(`.sk[data-skill="${m.skill}"]`); if (b) b.classList.add("hint");
  (m.targets || []).forEach((n) => { const c = document.querySelector(`.cell[data-name="${n}"]`); if (c) c.classList.add("hint"); });
}
async function playRec(m) { if (S.finished) return; if (m.skill === null) { await finish(); return; } await doMove(m.skill, (m.targets || []).join(""), true); }
$("#playTop").addEventListener("click", () => { if (S.ai && S.ai.moves.length) playRec(S.ai.moves[0]); });
$("#autoBtn").addEventListener("click", () => { auto = !auto; $("#autoBtn").textContent = auto ? "■ 停止" : "▶ 最後まで AI"; if (auto && S.ai) autoStep(); });
async function autoStep() {
  if (!auto || !S || S.finished || !S.ai) { auto = false; $("#autoBtn").textContent = "▶ 最後まで AI"; return; }
  await playRec(S.ai.moves[0]);
}
async function finish() {
  if (busy) return;
  busy = true;
  try {
    S = await api("finish"); kifuAppend(S.kifu); render();
    $("#recs").innerHTML = '<div class="hint-text">局は終了しました。</div>'; $("#sticky").classList.remove("on");
    auto = false; $("#autoBtn").textContent = "▶ 最後まで AI";
    const s = S.summary;
    $("#resGrade").textContent = s.grade;
    $("#resText").textContent = `誤差合計 ${s.err} ・ 残り集中 ${s.focus} ・ ${s.turns} 手（AI の手 ${s.ai_used}）`;
    $("#result").classList.add("on");
  } finally { busy = false; }
}
$("#finish").addEventListener("click", () => { if (confirm("ここでしあげますか？")) finish(); });
$("#undo").addEventListener("click", async () => { if (busy) return; S = await api("undo"); sel = null; render(); askAI(); });
async function newGame() {
  const u = new URL(location.href);   // seed の指定は URL の ?seed=N で（局の再現用）
  const qs = parseInt(u.searchParams.get("seed") || "0", 10) || 0;
  S = await api("new_game", qs); if (qs) { u.searchParams.delete("seed"); history.replaceState(null, "", u.toString()); }
  sel = null; auto = false; $("#autoBtn").textContent = "▶ 最後まで AI"; $("#result").classList.remove("on");
  render(); askAI();
}
$("#newGame").addEventListener("click", newGame); $("#resNew").addEventListener("click", newGame);
$("#resClose").addEventListener("click", () => $("#result").classList.remove("on"));

// ---------------------------------------------------------------- 実機モード
function buildReal() {
  const t = $("#rTemp");
  if (!t.options.length) for (let v = 3000; v >= 0; v -= 50) { const o = document.createElement("option"); o.value = v; o.textContent = v + "℃"; if (v === 1000) o.selected = true; t.appendChild(o); }
  const box = $("#rCells"); box.innerHTML = "";
  META.cells.forEach((c) => {
    const d = document.createElement("div"); d.className = "cellin";
    d.innerHTML = `<div class="top"><b>${c.name}</b><span>${c.lo}〜${c.hi}</span></div><div class="row"><input type="number" min="0" max="400" value="0" data-c="${c.name}" inputmode="numeric"><label class="pin"><input type="checkbox" data-p="${c.name}">本会心済</label></div>`;
    box.appendChild(d);
  });
}
function readReal() {
  const cells = {};
  META.cells.forEach((c) => { cells[c.name] = { v: +document.querySelector(`[data-c="${c.name}"]`).value || 0, pinned: document.querySelector(`[data-p="${c.name}"]`).checked }; });
  return { temp: +$("#rTemp").value, focus: +$("#rFocus").value, hissatsu: $("#rHiss").value, cells, turn: 1 };
}
$("#askBtn").addEventListener("click", async () => {
  $("#rRecs").innerHTML = '<div class="wait">考え中</div>'; $("#rPNow").textContent = "—"; $("#rPVal").textContent = "—";
  try { const ai = await api("advise", JSON.stringify(readReal())); renderRecs($("#rRecs"), ai, {}); $("#rPNow").textContent = pct(ai.now); $("#rPVal").textContent = pct(ai.value); }
  catch (e) { $("#rRecs").innerHTML = '<div class="hint-text">計算に失敗しました。</div>'; }
});
$("#rReset").addEventListener("click", () => {
  $("#rTemp").value = 1000; $("#rFocus").value = 253; $("#rHiss").value = "none";
  document.querySelectorAll("[data-c]").forEach((i) => (i.value = 0)); document.querySelectorAll("[data-p]").forEach((i) => (i.checked = false));
  $("#rRecs").innerHTML = ""; $("#rPNow").textContent = "—"; $("#rPVal").textContent = "—";
});
$("#tabSim").addEventListener("click", () => { if (S && S.ai && !S.finished) sticky(S.ai.moves[0]); $("#tabSim").classList.add("on"); $("#tabReal").classList.remove("on"); $("#viewSim").style.display = ""; $("#viewReal").style.display = "none"; });
$("#tabReal").addEventListener("click", () => { $("#sticky").classList.remove("on"); $("#tabReal").classList.add("on"); $("#tabSim").classList.remove("on"); $("#viewReal").style.display = ""; $("#viewSim").style.display = "none"; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { sel = null; render(); } });
