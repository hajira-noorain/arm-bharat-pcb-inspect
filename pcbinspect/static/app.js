// PCB Inspection Station dashboard. Plain JS, no network beyond this station.
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const L = window.Labels;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const words = (s) => String(s || "").replaceAll("_", " ");
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private window: not remembered */ } } };

async function api(path, opts = {}) {
  const isForm = opts.body instanceof FormData;
  const r = await fetch(path, { ...opts, headers: isForm ? {} : { "Content-Type": "application/json", ...(opts.headers || {}) } });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `${r.status}`);
  return body;
}

let toastTimer;
function toast(text, undoEvent) {
  const t = $("#toast"), u = $("#toast-undo");
  $("#toast-text").textContent = text; t.style.display = "flex";
  u.hidden = !undoEvent;
  u.onclick = async () => {
    await api(`/api/undo/${undoEvent}`, { method: "POST" }).catch((e) => toast(e.message));
    t.style.display = "none"; if (current && !current.upload) refreshCurrent();
  };
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.style.display = "none", 7000);
}

// ======================= screens =======================
const TITLES = { inspect: "Inspect", log: "Log", trends: "Trends", teach: "Teach a board", eyetest: "Eye test",
  rules: "Customer rules", ask: "Ask the log" };
let screen = "inspect";
function show(name) {
  screen = name;
  $$(".side nav button").forEach((b) => { if (b.dataset.screen === name) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
  $$("section[data-screen]").forEach((s) => s.hidden = s.dataset.screen !== name);
  $("#title").textContent = TITLES[name];
  $("#source").hidden = !(state?.demo && name === "inspect");
  $("#side").classList.remove("open"); $("#menu-btn").setAttribute("aria-expanded", "false");
  if (name === "log") loadLog();
  if (name === "trends") loadTrends();
  if (name === "teach") loadLearn();
  if (name === "eyetest") loadEyetest();
  if (name === "rules") loadRules();
  if (name !== "inspect") setLive(false);
}
$$(".side nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.screen)));
$("#menu-btn").addEventListener("click", () => {
  const open = $("#side").classList.toggle("open");
  $("#menu-btn").setAttribute("aria-expanded", open ? "true" : "false");
});

// ======================= station state =======================
let state = null, current = null, samples = [], sampleIdx = -1, source = "camera", busy = false;

async function loadState() {
  state = await api("/api/state");
  const sel = $("#design"), keep = sel.value || store.get("pcb-design");
  sel.innerHTML = state.designs.map((d) => `<option value="${esc(d.name)}">${esc(d.name)}${d.calibrated ? "" : " (not calibrated)"}</option>`).join("");
  if (keep && state.designs.some((d) => d.name === keep)) sel.value = keep;
  syncAlpha();
  const set = (id, ok, text) => { const el = $(id); el.className = ok ? "on" : "off"; el.textContent = text; };
  set("#st-vision", !!state.vision.detector || state.vision.backbone, state.vision.detector ? `Vision · ${state.vision.detector === "executorch" ? "ExecuTorch" : state.vision.detector}` : "Vision · anomaly only");
  set("#st-llm", !!state.llm, state.llm ? `${L.modelName(state.llm)} · ExecuTorch` : "Llama not loaded");
  set("#st-camera", !!state.camera, state.demo ? "Source · test boards" : state.camera ? `Camera · ${state.camera}` : "No camera");
  $("#llm-banner").hidden = !!state.llm;
  $("#disclosure").hidden = !state.disclosure; $("#disclosure").textContent = state.disclosure || "";
  $("#source").hidden = !(state.demo && screen === "inspect");
  $('#source [data-src="camera"]').hidden = !state.camera || state.camera === "replay";
  if (busy) $("#start-go").textContent = state.queue > 1 ? `Waiting for ${state.queue - 1} ahead` : "Inspecting";
  if (!current) renderStart();
}

// Only offer miss rates the calibration set can support: alpha needs at least 1/alpha - 1 defective calibration
// boards, otherwise the guarantee is empty and nothing would pass automatically.
function syncAlpha() {
  const d = state?.designs.find((x) => x.name === $("#design").value);
  const opts = [...$("#alpha").options];
  opts.forEach((o) => {
    const a = parseFloat(o.value), need = Math.ceil(1 / a) - 1;
    o.disabled = !d || d.n_defective < need;
    o.textContent = `${Math.round(a * 100)}%${o.disabled ? " –" : ""}`;
    o.title = o.disabled ? "Not enough calibration boards for this design to promise this miss rate" : "";
  });
  if ($("#alpha").selectedOptions[0]?.disabled) { const ok = opts.find((o) => !o.disabled); if (ok) $("#alpha").value = ok.value; }
}
$("#design").addEventListener("change", () => { store.set("pcb-design", $("#design").value); syncAlpha(); if (screen !== "inspect") show(screen); });

$("#operator").value = store.get("pcb-operator") || "";
$("#operator").addEventListener("change", () => store.set("pcb-operator", $("#operator").value.trim()));

// ======================= shift figures =======================
async function loadStats() {
  const s = await api("/api/stats").catch(() => null);
  if (!s) return;
  const pct = (n) => s.boards ? `${Math.round(100 * n / s.boards)}%` : "";
  const sec = (ms, d) => ms == null ? "–" : (ms / 1000).toFixed(d);
  $("#kpis").innerHTML = [
    ["", "Boards this shift", s.boards, ""], ["pass", "Passed", s.passed, pct(s.passed)],
    ["check", "Sent to a person", s.review, pct(s.review)], ["fail", "Failed", s.failed, pct(s.failed)],
    ["", "Verdict time, median", sec(s.verdict_ms_median, 2), s.verdict_ms_median == null ? "" : "s"],
    ["", "Advice time, median", sec(s.advice_ms_median, 1), s.advice_ms_median == null ? "" : "s"],
  ].map(([cls, label, v, extra]) => `<div class="kpi ${cls}"><small>${label}</small><b class="num">${v}</b><em>${extra}</em></div>`).join("");
}

// ======================= board source (demo) =======================
async function loadSamples() {
  if (!state?.demo) { samples = []; return; }
  samples = await api("/api/samples").catch(() => []);
  const label = (s) => s.kind === "assembled" ? "Assembled" : ({ deeppcb: "DeepPCB", hripcb: "PKU", mendeley: "Real photo" }[s.design] || s.design);
  $("#thumbs").innerHTML = samples.map((s, i) => `<button class="thumb" type="button" data-i="${i}" aria-pressed="false" title="${esc(s.design)}: ${s.kind === "assembled" ? (s.good ? "good board" : "faulty board") : `${s.marked} marked defect${s.marked === 1 ? "" : "s"}`}">
    <img alt="" loading="lazy" src="/api/samples/${encodeURIComponent(s.file)}.jpg"><span>${esc(label(s))}</span></button>`).join("");
  $("#strip-note").textContent = samples.length ? "never used in training" : "No test boards installed";
  $("#gallery").innerHTML = samples.slice(0, 12).map((s, i) => `<button class="tile" type="button" data-i="${i}">
    <img alt="" loading="lazy" src="/api/samples/${encodeURIComponent(s.file)}.jpg">
    <span class="t"><span>${s.kind === "assembled" ? "Assembled" : "Bare"} · ${esc(label(s))}</span><b>${s.good ? "good" : s.kind === "assembled" ? "faulty" : `${s.marked} defect${s.marked === 1 ? "" : "s"}`}</b></span></button>`).join("");
}
function setSource(src) {
  source = src;
  $$("#source button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.src === src ? "true" : "false"));
  $("#strip").hidden = !(state?.demo && src !== "camera" && current);
  if (src === "upload") $("#file").click();
  renderStart();
}
$$("#source button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.src)));
$("#thumbs").addEventListener("click", (e) => { const t = e.target.closest(".thumb"); if (t) inspectSample(+t.dataset.i); });
$("#gallery").addEventListener("click", (e) => { const t = e.target.closest(".tile"); if (t) inspectSample(+t.dataset.i); });
for (const id of ["#file", "#file2"]) $(id).addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) inspectUpload(f); });
$$("label.upload, #start-upload").forEach((l) => l.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("input", l).click(); } }));

function renderStart() {
  if (current) return;
  const demo = state?.demo;
  $("#start").hidden = false; $("#result").hidden = true;
  $("#start-title").textContent = demo ? "Try the station on a test board" : state && !state.designs.length ? "No board design yet" : "Ready";
  $("#start-text").textContent = demo
    ? "These boards were never used for training. Pick one, or upload your own photo. The verdict comes first; Llama's advice follows."
    : state && !state.designs.length ? "Teach a board design first: Setup, Teach a board." : "Place a board against the corner stops and press Inspect.";
  $("#start-go").textContent = demo ? "Inspect a random test board" : "Inspect";
  $("#start-upload").hidden = !demo;
  $("#start-keys").hidden = !!demo;
  $("#gallery").hidden = !demo || !samples.length;
}

// ======================= inspecting =======================
async function run(task, label) {
  if (busy) return;
  busy = true;
  const btns = $$("[data-go]").concat([$("#start-go")]);
  btns.forEach((b) => { b.disabled = true; b.dataset.label = b.textContent; b.textContent = label; });
  try { const r = await task(); current = r; render(r, true); poll(r); loadStats(); }
  catch (e) { toast(`Inspection did not run: ${e.message}`); }
  finally { busy = false; btns.forEach((b) => { b.disabled = false; if (b.dataset.label) b.textContent = b.dataset.label; }); }
}
const alpha = () => parseFloat($("#alpha").value);
function inspectCamera() {
  return run(() => api("/api/inspect", { method: "POST", body: JSON.stringify({ design: $("#design").value, alpha: alpha() }) }), "Inspecting");
}
function inspectSample(i) {
  if (!samples.length) return toast("No test boards are installed on this station.");
  sampleIdx = ((i % samples.length) + samples.length) % samples.length;
  const s = samples[sampleIdx];
  $$("#thumbs .thumb").forEach((t) => t.setAttribute("aria-pressed", +t.dataset.i === sampleIdx ? "true" : "false"));
  $("#strip").hidden = false; setSourceQuiet("samples");
  if ([...$("#design").options].some((o) => o.value === s.design)) { $("#design").value = s.design; syncAlpha(); }
  return run(() => api("/api/inspect", { method: "POST", body: JSON.stringify({ design: s.design, sample: s.file, alpha: alpha() }) }), "Inspecting");
}
function inspectUpload(file) {
  if (!/^image\/(jpeg|png)$/.test(file.type)) return toast("Upload a JPEG or PNG photo.");
  if (file.size > 10e6) return toast("The photo is larger than 10 MB.");
  const fd = new FormData(); fd.append("file", file);
  $("#strip").hidden = false; setSourceQuiet("upload");
  return run(() => api("/api/inspect/upload", { method: "POST", body: fd }), "Inspecting your photo");
}
function setSourceQuiet(src) { source = src; $$("#source button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.src === src ? "true" : "false")); }
function inspectNext() {
  if (state?.demo && source === "upload") return $("#file").click();
  if (state?.demo) return inspectSample(sampleIdx < 0 ? Math.floor(Math.random() * Math.max(samples.length, 1)) : sampleIdx + 1);
  return inspectCamera();
}
$("#start-go").addEventListener("click", inspectNext);
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || screen !== "inspect") return;
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON", "A", "LABEL"].includes(document.activeElement.tagName)) return;
  e.preventDefault(); inspectNext();
});

let pollTimer = null;
function poll(r) {
  clearInterval(pollTimer);
  if (!r.advice_pending) return;
  let n = 0;
  const url = r.upload ? `/api/upload/${r.token}` : `/api/inspection/${r.id}`;
  pollTimer = setInterval(async () => {
    n++;
    const fresh = await api(url).catch(() => null);
    if (!fresh || current !== r && (current?.id !== r.id || current?.token !== r.token)) return clearInterval(pollTimer);
    if (!fresh.advice_pending || n > 150) {
      clearInterval(pollTimer);
      fresh.advice_timed_out = fresh.advice_pending;
      Object.assign(fresh, { known: r.known, sample: r.sample });
      current = fresh; renderAdvice(fresh); loadStats();
    }
  }, 2000);
}
async function refreshCurrent() {
  if (!current || current.upload) return;
  const keep = { known: current.known, sample: current.sample };
  current = Object.assign(await api(`/api/inspection/${current.id}`), keep); render(current, false);
}

// ======================= the result =======================
let ipcClass = 2, showAll = false, hot = null, view = "board";
const num = (d) => parseInt(String(d.defect_key || d.id).replace(/\D/g, ""), 10) || d.id;

function verdictWords(r) {
  const n = r.defects.length, s = n === 1 ? "" : "s";
  const cr = r.check_reason || "";
  if (r.verdict === "PASS") return ["Board passed", "No defect found at the chosen miss rate."];
  if (r.verdict === "FAIL") {
    const worst = r.defects[0];
    return ["board failed", `${n} defect${s} found. Most serious: ${worst ? esc(L.typeName(worst.cls).toLowerCase()) : "see the list"}.`];
  }
  if (r.upload) return ["look at this board", `${n ? `${n} possible defect${s} marked` : "Nothing found"}. Not calibrated for this board, so a person decides; sizes are in pixels.`];
  if (/outside the calibrated envelope/.test(cr)) return ["take the picture again", esc(cr.split(": ").slice(1).join(": ") || cr)];
  if (/golden comparison refused/.test(cr)) return ["re-seat the board", "It did not line up with this design's reference picture, so it was not compared. Put it against the corner stops and inspect again."];
  return ["look at this board", `${n ? `${n} possible defect${s} marked` : "The station is not sure"}. Look at it yourself, then pass or fail it.`];
}

function render(r, fresh) {
  $("#start").hidden = true; $("#result").hidden = false;
  if (fresh) { showAll = false; hot = null; view = "board"; setLive(false); }
  const v = $("#verdict");
  v.className = `card verdict ${r.verdict}${fresh ? " fresh" : ""}`;
  const [line, what] = verdictWords(r);
  const word = L.VERDICT[r.verdict].word;
  $("#vword").innerHTML = `<b>${word}</b> · ${line}`;
  const ms = r.latency?.total_to_verdict;
  $("#vline").innerHTML = `${what} &nbsp;·&nbsp; ${r.upload ? "your photo" : `board ${esc(r.board_id)}`}${ms ? ` · verdict in ${Math.round(ms)} ms` : ""}`;
  renderActions(r); renderBoard(r); renderFindings(r); renderAdvice(r); renderKnown(r); renderTitleblock(r);
}

function renderActions(r) {
  const next = state?.demo ? (source === "upload" ? "Upload another" : "Next test board") : "Inspect next";
  const decide = !r.upload && r.verdict === "REVIEW" && !r.final_verdict
    ? `<div class="decision"><input id="review-reason" maxlength="200" placeholder="reason (optional)" aria-label="Reason for the decision">
       <button class="btn pass" data-review="PASS" type="button">Pass board</button><button class="btn fail" data-review="FAIL" type="button">Fail board</button></div>`
    : r.final_verdict ? `<span class="meta">Decided by ${esc(r.decided_by || "a person")}: ${r.final_verdict === "PASS" ? "passed" : "failed"}${r.decision_reason ? ` (${esc(r.decision_reason)})` : ""}</span>` : "";
  $("#vactions").innerHTML = `${decide}<button class="btn primary" data-go type="button">${next}</button>`;
  $("[data-go]", $("#vactions")).addEventListener("click", inspectNext);
}

function imgUrl(r) { return r.upload ? `/api/upload/${r.token}.jpg` : `/api/image/${r.id}.jpg`; }
function renderBoard(r) {
  const views = [["board", "Board"]];
  if (!r.upload) views.push(["marks", "Station marks"]);
  if (state?.camera && state.camera !== "replay" && !state.demo) views.push(["live", "Live camera"]);
  $("#views").innerHTML = views.map(([k, t]) => `<button type="button" data-view="${k}" aria-pressed="${view === k}">${t}</button>`).join("");
  if (view === "live") return;
  const src = view === "marks" ? `/api/overlay/${r.id}.jpg` : imgUrl(r);
  $("#stage").innerHTML = `<img id="board-img" alt="${view === "marks" ? "The station's own markings" : "Inspected board with numbered findings"}" src="${src}">`;
  $("#live-note").textContent = "";
  const img = $("#board-img");
  const draw = () => { if (view === "board") drawBalloons(r, img.naturalWidth, img.naturalHeight); };
  img.complete ? draw() : img.addEventListener("load", draw, { once: true });
  img.addEventListener("error", () => { $("#stage").innerHTML = `<div class="empty">The picture is no longer available.</div>`; }, { once: true });
}
$("#views").addEventListener("click", (e) => {
  const b = e.target.closest("[data-view]"); if (!b || !current) return;
  view = b.dataset.view;
  if (view === "live") { setLive(true); $$("#views button").forEach((x) => x.setAttribute("aria-pressed", x === b)); }
  else { setLive(false); renderBoard(current); }
});

function drawBalloons(r, W, H) {
  if (!W || !H || !r.defects.length) return;
  const R = Math.max(W, H) * 0.021, sw = Math.max(1.2, R * 0.15), fs = R * 1.1;
  const g = r.defects.map((d) => {
    const n = num(d), w = d.x2 - d.x1, h = d.y2 - d.y1;
    const cx = Math.min(W - R - 1, d.x2 + R * 0.35), cy = Math.max(R + 1, d.y1 - R * 0.35);
    return `<g data-n="${n}"><rect class="box${hot === n ? " hot" : ""}" x="${d.x1}" y="${d.y1}" width="${w}" height="${h}" stroke-width="${sw}"/>
      <g class="balloon${hot === n ? " hot" : ""}"><circle cx="${cx}" cy="${cy}" r="${R}" stroke-width="${sw}"/><text x="${cx}" y="${cy}" font-size="${fs}">${n}</text></g>
      <circle class="hit" cx="${cx}" cy="${cy}" r="${R * 1.4}" fill="transparent"/></g>`;
  }).join("");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = g;
  const img = $("#board-img");
  svg.style.width = `${img.clientWidth}px`; svg.style.height = `${img.clientHeight}px`;
  svg.style.left = `${img.offsetLeft}px`; svg.style.top = `${img.offsetTop}px`; svg.style.inset = "auto";
  $("#stage svg")?.remove(); $("#stage").appendChild(svg);
  svg.addEventListener("mouseover", (e) => { const n = e.target.closest("[data-n]")?.dataset.n; if (n) setHot(+n); });
  svg.addEventListener("click", (e) => { const n = e.target.closest("[data-n]")?.dataset.n; if (n) { setHot(+n); $(`#findings tr[data-n="${n}"]`)?.scrollIntoView({ block: "nearest" }); } });
}
window.addEventListener("resize", () => { if (current && view === "board") { const img = $("#board-img"); if (img?.complete) drawBalloons(current, img.naturalWidth, img.naturalHeight); } });

function setHot(n) {
  hot = n;
  $$("#stage [data-n]").forEach((g) => { const on = +g.dataset.n === n; $$(".box, .balloon", g).forEach((x) => x.classList.toggle("hot", on)); });
  $$("#findings tr[data-n]").forEach((tr) => tr.classList.toggle("hot", +tr.dataset.n === n));
}

const ipcCell = (d) => {
  const v = (d.ipc || {})[`class${ipcClass}`];
  if (!v) return `<span class="meta">–</span>`;
  return v === "reject" ? `<span class="pill reject">Reject</span>` : v === "uncertain" ? `<span class="pill unsure">Unsure</span>` : `<span class="pill ok">OK</span>`;
};
function sizeText(d) {
  const loss = /(\d+)% of the width lost/.exec(d.severity_reason || "");
  const base = d.size_mm != null ? `${d.size_mm.toFixed(2)} mm` : d.size_px != null ? `${d.size_px} px` : "–";
  return loss ? `${base} · ${loss[1]}% width` : base;
}
function renderFindings(r) {
  $("#findings-card").hidden = !r.defects.length;
  if (!r.defects.length) return;
  const rows = showAll ? r.defects : r.defects.slice(0, 6);
  const fb = (d) => r.upload ? "" : d.operator_label ? `<span class="labelled">${d.operator_label === "false_alarm" ? "false alarm" : "confirmed"}</span>`
    : `<button class="btn small" type="button" data-fb="confirmed" data-d="${d.id}">Confirm</button> <button class="btn small" type="button" data-fb="false_alarm" data-d="${d.id}">False alarm</button>`;
  $("#findings").innerHTML = `<table class="findings"><thead><tr><th>#</th><th>Defect</th><th>Size</th><th>Where</th><th>Class ${ipcClass}</th><th class="conf">Conf.</th><th></th></tr></thead><tbody>
    ${rows.map((d) => { const n = num(d); return `<tr data-n="${n}" tabindex="0" class="${hot === n ? "hot" : ""}">
      <td><span class="bn">${n}</span></td><td class="name">${esc(L.typeName(d.cls))}${d.part && d.part !== "part" ? ` <span class="meta">(${esc(d.part)})</span>` : ""}</td>
      <td>${esc(sizeText(d))}</td><td>${esc(L.place(d.location))}</td><td>${ipcCell(d)}</td><td class="conf">${Math.round(d.confidence * 100)}%</td>
      <td class="fb">${fb(d)}</td></tr>
      <tr class="why" data-why="${n}" hidden><td></td><td colspan="6">${esc(L.plainReason(d.severity_reason))}</td></tr>`; }).join("")}
    </tbody></table>${r.defects.length > 6 ? `<button class="btn link" type="button" id="show-all" style="margin-top:8px">${showAll ? "Show fewer" : `Show all ${r.defects.length}`}</button>` : ""}`;
}
$("#findings").addEventListener("mouseover", (e) => { const tr = e.target.closest("tr[data-n]"); if (tr) setHot(+tr.dataset.n); });
$("#findings").addEventListener("focusin", (e) => { const tr = e.target.closest("tr[data-n]"); if (tr) setHot(+tr.dataset.n); });
$("#findings").addEventListener("click", (e) => {
  if (e.target.id === "show-all") { showAll = !showAll; renderFindings(current); return; }
  if (e.target.closest("[data-fb]")) return;
  const tr = e.target.closest("tr[data-n]"); if (!tr) return;
  const why = $(`tr[data-why="${tr.dataset.n}"]`, $("#findings")); if (why && why.textContent.trim()) why.hidden = !why.hidden;
});
$("#findings").addEventListener("keydown", (e) => {
  const tr = e.target.closest("tr[data-n]");
  if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); const why = $(`tr[data-why="${tr.dataset.n}"]`, $("#findings")); if (why && why.textContent.trim()) why.hidden = !why.hidden; }
});
$("#ipc-class").addEventListener("click", (e) => {
  const b = e.target.closest("[data-class]"); if (!b) return;
  ipcClass = +b.dataset.class;
  $$("#ipc-class button").forEach((x) => { x.setAttribute("aria-pressed", x === b ? "true" : "false"); x.textContent = x.dataset.class === b.dataset.class ? `Class ${x.dataset.class}` : x.dataset.class; });
  if (current) renderFindings(current);
});

// steps with the same defect type, cause and action become one line ("Shorts 1, 3, 5, 6 · ...")
function groupAdvice(steps, defects) {
  const byKey = new Map(defects.map((d) => [String(d.defect_key || `D${d.id}`), d]));
  const groups = new Map();
  for (const s of steps) {
    const d = byKey.get(String(s.defect));
    const cls = d ? d.cls : "";
    const k = `${cls}|${s.cause_id}|${s.action_id}`;
    if (!groups.has(k)) groups.set(k, { ids: [], cls, action: s.action, cause: s.cause, step: s.process_step, cause_id: s.cause_id, action_id: s.action_id });
    groups.get(k).ids.push(d ? num(d) : String(s.defect));
  }
  return [...groups.values()].map((g) => ({ ...g, ids: g.ids.sort((a, b) => a - b) }));
}
window.groupAdvice = groupAdvice;

function renderAdvice(r) {
  $("#advice-card").hidden = !r.defects.length;
  if (!r.defects.length) return;
  const a = r.advice?.[r.advice.length - 1];
  if (!a) {
    $("#advice-by").textContent = "";
    $("#advice").innerHTML = `<p class="${r.advice_timed_out ? "" : "pending"}">${r.advice_timed_out ? "Advice did not arrive. The verdict is not affected: follow your standard procedure for these defects." : `${state?.llm ? L.modelName(state.llm) : "The station"} is writing advice`}</p>`;
    return;
  }
  const adv = a.advice, llama = a.backend === "llama-executorch", swapped = llama && adv.summary_by === "template";
  $("#advice-by").textContent = llama ? `${L.modelName(a.model)} · ${(a.latency_ms / 1000).toFixed(1)} s` : "Standard advice";
  const groups = groupAdvice(adv.steps || [], r.defects);
  const note = llama ? `${swapped ? "Causes and actions chosen" : "Written"} by ${esc(L.modelName(a.model))} on this device. Every cause and action is checked against the defect knowledge base.${swapped ? ` Llama's summary sentence was replaced by the standard one${a.rejected_reason ? ` (${esc(a.rejected_reason)})` : ""}.` : ""}`
    : /busy with newer boards/.test(a.rejected_reason || "") ? "Standard advice from the defect knowledge base: Llama was busy with newer boards on this shared demo."
    : `From the defect knowledge base${a.rejected_reason ? `. Llama's answer was not used: ${esc(a.rejected_reason)}` : ""}.`;
  const rate = r.upload ? "" : a.operator_rating == null
    ? `Useful? <button class="btn small" type="button" data-rate="1" data-a="${a.id}">Yes</button> <button class="btn small" type="button" data-rate="-1" data-a="${a.id}">No</button>`
    : `Rated ${a.operator_rating > 0 ? "useful" : "not useful"}.`;
  $("#advice").innerHTML = `<p class="summary">${esc(adv.summary)}</p>
    ${groups.map((g) => `<div class="adv-group"><b>${esc(L.typeName(g.cls, g.ids.length))} ${g.ids.join(", ")}</b> · ${esc(g.action)}<br>
      <span>Likely cause: ${esc(g.cause)}</span><div class="kb" hidden>${esc(g.cause_id)}, ${esc(g.action_id)}${g.step ? `, ${esc(String(g.step).split(":")[0])}` : ""}</div></div>`).join("")}
    <p class="meta">${note} <button class="btn link" type="button" id="kb-toggle">details</button></p>
    <div class="actions meta">${rate}</div>`;
}
$("#advice").addEventListener("click", (e) => { if (e.target.id === "kb-toggle") $$("#advice .kb").forEach((k) => k.hidden = !k.hidden); });

function renderKnown(r) {
  const k = r.known;
  $("#known-card").hidden = !k;
  if (!k) return;
  $("#known").innerHTML = k.marked != null
    ? `<span><b>Known answer</b> · the dataset marks ${k.marked} defect${k.marked === 1 ? "" : "s"}</span><span>found <b>${k.found} of ${k.marked}</b></span><span>${k.extra} extra call${k.extra === 1 ? "" : "s"}</span><span class="meta">test boards only</span>`
    : `<span><b>Known answer</b> · ${k.good ? "a good board" : "a faulty board"}</span><span>the verdict ${k.verdict_agrees ? "<b>agrees</b>" : "does not agree"}</span><span class="meta">test boards only</span>`;
}

function renderTitleblock(r) {
  const g = r.gate;
  const cal = r.upload ? "not calibrated for this board" : !g ? "not calibrated" : g.planted ? `test defects drawn on ${g.n_good} captures` : `${g.n_defective + g.n_good} boards`;
  const cells = [["Design", r.upload ? `${r.design} detector` : r.design], ["Scale", r.mm_per_px ? `${(+r.mm_per_px).toFixed(4)} mm/px` : "not known"],
    ["Miss rate", r.alpha ? `${Math.round(r.alpha * 100)}%` : "–"], ["Calibrated on", cal], ["Evidence", r.evidence || "–"]];
  $("#titleblock").innerHTML = cells.map(([k, v]) => `<div><small>${k}</small>${esc(v)}</div>`).join("");
}

// feedback, decisions, ratings
document.addEventListener("click", async (e) => {
  const fb = e.target.closest("[data-fb]"), rv = e.target.closest("[data-review]"), rt = e.target.closest("[data-rate]"), ack = e.target.closest("[data-ack]");
  try {
    if (fb) {
      fb.disabled = true;
      const res = await api("/api/feedback", { method: "POST", body: JSON.stringify({ inspection_id: current.id, defect_id: +fb.dataset.d, action: fb.dataset.fb }) });
      if (res.accepted === false) { toast(res.message); fb.disabled = false; return; }
      toast(`${fb.dataset.fb === "false_alarm" ? "Marked as a false alarm" : "Confirmed"}. ${res.learned ? res.learned[0].toUpperCase() + res.learned.slice(1) + "." : ""}`, res.event_id);
      refreshCurrent();
    } else if (rv) {
      const who = $("#operator").value.trim();
      if (!who) { toast("Enter your name in Operator at the top first."); $("#operator").focus(); return; }
      await api("/api/review", { method: "POST", body: JSON.stringify({ inspection_id: current.id, verdict: rv.dataset.review, operator: who, reason: ($("#review-reason") || {}).value || null }) });
      refreshCurrent(); loadStats();
    } else if (rt) {
      await api(`/api/advice/${rt.dataset.a}/rate/${rt.dataset.rate}`, { method: "POST" });
      const keep = { known: current.known, sample: current.sample };
      current = Object.assign(await api(`/api/inspection/${current.id}`), keep); renderAdvice(current);
    } else if (ack) {
      await api(`/api/alerts/${ack.dataset.ack}/ack`, { method: "POST" }); loadTrends();
    }
  } catch (err) { toast(err.message); }
});

// live camera view (real camera only): one frame at a time, paused when not visible
let liveOn = false, liveTimer = null;
function setLive(on) {
  liveOn = on; clearTimeout(liveTimer);
  if (on) { $("#stage").innerHTML = '<img id="live-img" alt="Live camera view">'; $("#live-note").textContent = "Live camera: place the board against the corner stops, then inspect."; liveTick(); }
}
function liveTick() {
  if (!liveOn) return;
  if (document.hidden || screen !== "inspect") { liveTimer = setTimeout(liveTick, 1000); return; }
  const next = new Image();
  next.onload = () => { if (liveOn && $("#live-img")) $("#live-img").src = next.src; liveTimer = setTimeout(liveTick, 400); };
  next.onerror = () => { $("#live-note").textContent = "The camera is not giving a picture."; liveTimer = setTimeout(liveTick, 2500); };
  next.src = `/api/preview.jpg?t=${Date.now()}`;
}

// ======================= log =======================
let logRows = [];
async function loadLog() {
  logRows = await api("/api/log?limit=200").catch((e) => { toast(e.message); return []; });
  const designs = [...new Set(logRows.map((r) => r.design))].sort();
  const keep = $("#f-design").value;
  $("#f-design").innerHTML = `<option value="">All designs</option>` + designs.map((d) => `<option>${esc(d)}</option>`).join("");
  $("#f-design").value = designs.includes(keep) ? keep : "";
  drawLog();
}
function drawLog() {
  const v = $("#f-verdict").value, d = $("#f-design").value, b = $("#f-board").value.trim().toLowerCase();
  const rows = logRows.filter((r) => (!v || r.verdict === v) && (!d || r.design === d) && (!b || r.board_id.toLowerCase().includes(b)));
  $("#log-empty").hidden = logRows.length > 0;
  $("#f-count").textContent = logRows.length ? `${rows.length} of ${logRows.length}` : "";
  $("#log-body").innerHTML = rows.map((r) => `<tr class="row" tabindex="0" data-id="${r.id}" title="Open this board">
    <td>${new Date(r.ts * 1000).toLocaleString()}</td><td>${esc(r.board_id)}</td><td>${esc(r.design)}</td>
    <td><span class="pill v-${r.verdict}">${L.VERDICT[r.verdict].word}</span></td><td>${r.n_defects}</td>
    <td>${r.final_verdict ? `${r.final_verdict === "PASS" ? "passed" : "failed"}${r.decided_by ? ` by ${esc(r.decided_by)}` : ""}${r.decision_reason ? `: ${esc(r.decision_reason)}` : ""}` : ""}</td>
    <td>${r.advice_by === "llama-executorch" ? "Llama" : r.advice_by === "template" ? "standard text" : ""}</td></tr>`).join("");
}
["#f-verdict", "#f-design"].forEach((s) => $(s).addEventListener("change", drawLog));
$("#f-board").addEventListener("input", drawLog);
async function openLogged(id) {
  const r = await api(`/api/inspection/${id}`).catch((e) => { toast(e.message); return null; });
  if (!r) return;
  show("inspect"); current = r; render(r, true); poll(r);
}
$("#log-body").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) openLogged(tr.dataset.id); });
$("#log-body").addEventListener("keydown", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openLogged(tr.dataset.id); } });

// ======================= trends =======================
async function loadTrends() {
  const d = await api(`/api/spc/${encodeURIComponent($("#design").value)}`).catch((e) => { toast(e.message); return null; });
  if (!d) return;
  $("#alerts").innerHTML = d.alerts.map((a) => `<div class="alert"><span><strong>Process drift:</strong> ${esc(a.message)}</span><button class="btn small" type="button" data-ack="${a.id}">Acknowledge</button></div>`).join("");
  drawPareto(d.pareto); drawPChart(d.p_chart, d.boards);
}
function tipFor(container) {
  let tip = $(".tip", container);
  if (!tip) { tip = document.createElement("div"); tip.className = "tip"; container.appendChild(tip); }
  return { show: (e, html) => { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = `${e.clientX + 14}px`; tip.style.top = `${e.clientY + 14}px`; }, hide: () => tip.style.display = "none" };
}
function drawPareto(rows) {
  const box = $("#pareto");
  if (!rows.length) { box.innerHTML = `<p class="meta">No defects logged for this design yet.</p>`; return; }
  const W = 560, rowH = 32, left = 140, right = 96, H = rows.length * rowH + 6, max = rows[0][1];
  const x = (v) => (W - left - right) * v / max;
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Defect counts by type">${rows.map(([cls, n, cum], i) => `
    <g data-i="${i}"><rect x="0" y="${i * rowH}" width="${W}" height="${rowH}" fill="transparent"/>
    <text x="${left - 10}" y="${i * rowH + 21}" text-anchor="end" font-size="14" fill="#16201C">${esc(L.typeName(cls, 2))}</text>
    <rect x="${left}" y="${i * rowH + 7}" width="${Math.max(2, x(n))}" height="18" rx="3" fill="#1D3F8F"/>
    <text x="${left + x(n) + 7}" y="${i * rowH + 21}" font-size="14" fill="#16201C" font-family="Archivo Narrow">${n}</text>
    <text x="${W}" y="${i * rowH + 21}" text-anchor="end" font-size="13" fill="#5E6A66" font-family="Archivo Narrow">${Math.round(cum * 100)}% so far</text></g>`).join("")}</svg>`;
  const tip = tipFor($("#pareto-panel"));
  $$("g[data-i]", box).forEach((g) => { const [cls, n, cum] = rows[+g.dataset.i];
    g.addEventListener("mousemove", (e) => tip.show(e, `${esc(L.typeName(cls, 2))}: ${n}<br>${Math.round(cum * 100)}% of all defects up to here`)); g.addEventListener("mouseleave", tip.hide); });
}
function drawPChart(pc, boards) {
  const box = $("#pchart");
  if (!pc.p.length) { box.innerHTML = `<p class="meta">Needs at least 20 inspected boards (${boards} so far).</p>`; return; }
  const W = 560, H = 240, Lm = 44, R = 70, T = 12, B = 30;
  const ymax = Math.min(1, Math.max(pc.ucl, ...pc.p, 0.05) * 1.1);
  const x = (i) => Lm + (pc.p.length === 1 ? (W - Lm - R) / 2 : i * (W - Lm - R) / (pc.p.length - 1));
  const y = (v) => T + (H - T - B) * (1 - v / ymax);
  const out = new Set(pc.out_of_control);
  const path = pc.p.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("");
  const ticks = [0, ymax / 2, ymax].map((v) => `<line x1="${Lm}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#E6EBF2"/><text x="${Lm - 6}" y="${y(v) + 5}" text-anchor="end" font-size="12" fill="#5E6A66" font-family="Archivo Narrow">${Math.round(v * 100)}%</text>`).join("");
  const ref = (v, name) => `<line x1="${Lm}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#5E6A66" stroke-dasharray="5 4"/><text x="${W - R + 6}" y="${y(v) + 5}" font-size="12" fill="#5E6A66" font-family="Archivo Narrow">${name} ${Math.round(v * 100)}%</text>`;
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of boards not passed per group of 20">${ticks}${ref(pc.ucl, "upper")}${pc.lcl > 0 ? ref(pc.lcl, "lower") : ""}
    <line x1="${Lm}" x2="${W - R}" y1="${y(pc.pbar)}" y2="${y(pc.pbar)}" stroke="#9DB0DA"/>
    <path d="${path}" fill="none" stroke="#1D3F8F" stroke-width="2"/>
    ${pc.p.map((v, i) => `<g data-i="${i}"><circle cx="${x(i)}" cy="${y(v)}" r="14" fill="transparent"/><circle cx="${x(i)}" cy="${y(v)}" r="${out.has(i) ? 6 : 4}" fill="${out.has(i) ? "#C62828" : "#1D3F8F"}" stroke="#fff" stroke-width="2"/>${out.has(i) ? `<text x="${x(i)}" y="${y(v) - 12}" text-anchor="middle" font-size="12" fill="#C62828">out of control</text>` : ""}</g>`).join("")}
    <text x="${Lm}" y="${H - 6}" font-size="12" fill="#5E6A66">oldest</text><text x="${W - R}" y="${H - 6}" text-anchor="end" font-size="12" fill="#5E6A66">latest</text></svg>`;
  const tip = tipFor($("#pchart-panel"));
  $$("g[data-i]", box).forEach((g) => { const i = +g.dataset.i;
    g.addEventListener("mousemove", (e) => tip.show(e, `Boards ${i * 20 + 1} to ${i * 20 + 20}: ${Math.round(pc.p[i] * 100)}% not passed${out.has(i) ? "<br>Outside the control limits" : ""}`)); g.addEventListener("mouseleave", tip.hide); });
}

// ======================= ask =======================
$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#question").value.trim(); if (!q) return;
  $("#ask-out").innerHTML = `<p class="pending answer">Looking it up</p>`;
  try {
    const r = await api("/api/ask", { method: "POST", body: JSON.stringify({ question: q }) });
    const cols = r.rows.length ? Object.keys(r.rows[0]) : [];
    $("#ask-out").innerHTML = `<p class="answer">${esc(r.answer)}</p>
      <p class="meta">Answered from the "${esc(words(r.query))}" query${r.route === "llama" ? ", chosen by Llama" : ""}; ${r.summary_by === "llama" ? "summary written by Llama and checked against these rows" : "shown directly from the rows"}.</p>
      ${cols.length ? `<table><thead><tr>${cols.map((c) => `<th>${esc(words(c))}</th>`).join("")}</tr></thead><tbody>${r.rows.map((row) => `<tr>${cols.map((c) => `<td>${esc(row[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>` : ""}`;
  } catch (err) { $("#ask-out").innerHTML = `<p class="answer">The question could not be answered: ${esc(err.message)}</p>`; }
});

// ======================= teach =======================
const tKind = () => { const bare = $("#t-kind").value === "bare"; $("#t-bare").hidden = $("#t-bare-finish").hidden = !bare; $("#t-assembled").hidden = bare; };
$("#t-kind").addEventListener("change", tKind); tKind();
const tName = () => { const n = $("#t-name").value.trim(); if (!/^[A-Za-z0-9_-]{2,40}$/.test(n)) { $("#teach-out").textContent = "Give the design a name first: letters, digits, - and _."; $("#t-name").focus(); return null; } return n; };
$("#teach-btn").addEventListener("click", async () => {
  const name = tName(); if (!name) return;
  const b = $("#teach-btn"); b.disabled = true; $("#teach-out").textContent = "Capturing good boards and learning";
  try {
    const r = await api("/api/enrol", { method: "POST", body: JSON.stringify({ name, kind: $("#t-kind").value, images: +$("#t-n").value }) });
    $("#teach-out").textContent = `Learned ${r.design} from ${r.images} good boards in ${r.seconds} s. ${r.note}`;
    await loadState(); $("#design").value = r.design;
  } catch (err) { $("#teach-out").textContent = `Could not learn the design: ${err.message}`; }
  finally { b.disabled = false; }
});
$("#tb-capture").addEventListener("click", async () => {
  const name = tName(); if (!name) return;
  const b = $("#tb-capture"); b.disabled = true;
  try {
    const r = await api("/api/teach/capture", { method: "POST", body: JSON.stringify({ name }) });
    $("#tb-cancel").hidden = false;
    $("#teach-out").textContent = !r.ok ? `Not saved: ${r.reason}` : r.golden ? "Saved as the reference picture. Now lift the board, put it down again and capture once more." : `${r.captures} good captures saved. Capture another, or finish.`;
    $("#tb-finish").disabled = !(r.captures >= 2);
  } catch (err) { $("#teach-out").textContent = `Could not capture: ${err.message}`; }
  finally { b.disabled = false; }
});
$("#tb-cancel").addEventListener("click", async () => {
  const name = tName(); if (!name) return;
  await api("/api/teach/cancel", { method: "POST", body: JSON.stringify({ name }) }).catch((e) => toast(e.message));
  $("#teach-out").textContent = "Cleared. Start with a good board."; $("#tb-finish").disabled = true; $("#tb-cancel").hidden = true;
});
$("#tb-finish").addEventListener("click", async () => {
  const name = tName(); if (!name) return;
  const mm = parseFloat($("#t-mm").value);
  try { await api("/api/teach/finish", { method: "POST", body: JSON.stringify({ name, mm_per_px: isFinite(mm) && mm > 0 ? mm : null }) }); }
  catch (err) { $("#teach-out").textContent = err.message; return; }
  $("#tb-capture").disabled = $("#tb-finish").disabled = true; $("#tb-progress").hidden = false;
  $("#teach-out").textContent = "Drawing test defects on your pictures and checking which ones the station finds";
  const tick = async () => {
    const r = await api(`/api/teach/${encodeURIComponent(name)}`).catch(() => ({ running: true }));
    if (r.running) { $("#tb-progress").value = r.progress || 0; setTimeout(tick, 1500); return; }
    $("#tb-progress").hidden = true; $("#tb-capture").disabled = false; $("#tb-cancel").hidden = true;
    if (!r.done) { $("#teach-out").textContent = `Teaching did not finish: ${r.reason}`; $("#tb-finish").disabled = false; return; }
    const lims = Object.values(r.limits || {}).map((x) => `<li>${esc(x)}</li>`).join("");
    $("#teach-out").innerHTML = `Learned <b>${esc(r.design)}</b> from ${r.captures} good captures in ${r.seconds} s (scale from ${esc(r.scale_from)}). ` +
      (r.can_fail ? "It can now pass and fail this board on its own. " : "It will send boards with a difference to a person: capture more good boards to let it fail them itself. ") +
      `Measured on test defects drawn onto your own board, not on real defects:<ul>${lims}</ul>` + (r.notes || []).map(esc).join(" ");
    await loadState(); $("#design").value = r.design; syncAlpha();
  };
  setTimeout(tick, 1000);
});
async function loadLearn() {
  const d = $("#design").value; if (!d) return;
  const s = await api(`/api/learn/${encodeURIComponent(d)}`).catch(() => null); if (!s) return;
  $("#learn-counts").textContent = `${d}: ${s.confirmed} confirmed defects and ${s.false_alarms} false alarms collected. Active version: ${s.active_version ?? "detector only"}.`;
  $("#learn-hist").innerHTML = s.versions.map((v) => `<tr><td>${new Date(v.ts * 1000).toLocaleString()}</td><td>${esc(v.version)}</td><td>${v.active ? "switched on" : esc(v.note || "kept previous version")}${v.metrics?.cross_validation?.ok ? `: ${Math.round(v.metrics.cross_validation.false_alarms_removed * 100)}% of false alarms removed, ${Math.round(v.metrics.cross_validation.recall_on_confirmed * 100)}% of real defects kept` : ""}</td></tr>`).join("");
  const st = s.status;
  $("#learn-btn").disabled = !!st?.running;
  $("#learn-out").textContent = st?.running ? "Updating the model and re-checking the miss-rate calibration" : st ? (st.promoted ? `Version ${st.version} switched on in ${st.seconds} s.` : `Kept the previous version: ${st.reason}.`) : "";
  if (st?.running) setTimeout(loadLearn, 2000);
}
$("#learn-btn").addEventListener("click", async () => {
  try { await api(`/api/learn/${encodeURIComponent($("#design").value)}`, { method: "POST" }); loadLearn(); }
  catch (err) { $("#learn-out").textContent = `Could not start the update: ${err.message}`; }
});

// ======================= eye test =======================
const TYPE_NAMES = { open: "Opens", short: "Shorts", pin_hole: "Pinholes", spurious_copper: "Copper islands", mouse_bite: "Mouse bites (nicks)", spur: "Spurs" };
function renderSheets(captured, total) {
  const n = total ?? Math.max(10, ...captured);
  $("#et-sheets").innerHTML = Array.from({ length: n + 1 }, (_, k) => `<span class="${captured.includes(k) ? "got" : ""}" title="${k === 0 ? "defect-free sheet" : "test sheet"}">${k === 0 ? "0 golden" : k}</span>`).join("");
}
function renderLimit(t, v) {
  const n = v.table.length;
  const cells = v.table.map((r) => `<div class="${r.lcb95 >= 0.9 ? "ok" : (r.significantly_low ? "low" : "")}">${r.hits}/${r.n}<small>${r.size_mm} mm</small></div>`).join("");
  const no = !["ok", "below_range"].includes(v.status);
  return `<div class="limit"><div class="what">${esc(TYPE_NAMES[t] || words(t))}</div><p class="say ${no ? "no" : ""}">${esc(v.sentence)}</p><div class="hits" style="--n:${n}">${cells}</div></div>`;
}
async function loadEyetest() {
  const design = $("#design").value; if (!design) return;
  const r = await api(`/api/eyetest/${encodeURIComponent(design)}`).catch(() => null); if (!r) return;
  renderSheets(r.captured, r.latest?.sheets_total);
  $("#et-demo").hidden = !(state?.eyetest_demo?.length);
  const Lt = r.latest;
  if (!Lt) { $("#et-result").innerHTML = `<p class="sub">No eye test has been run for this design yet.</p>`; return; }
  $("#et-result").innerHTML = `${Lt.source === "simulated photos" ? '<p class="chk warn"><b>Simulated photos, not this camera.</b> Photograph the printed card to measure this station.</p>' : ""}${Lt.calibrated_on ? '<p class="chk warn"><b>Measured on test defects drawn onto pictures of this board</b> when it was taught, not on the printed card and not on real defects.</p>' : ""}
    <p class="sub">${esc(new Date(Lt.ts * 1000).toLocaleString())}. ${Lt.n_trials} planted defects${Lt.calibrated_on ? ` in ${Lt.n_windows} picture windows` : ` on ${Lt.sheets_read.length} sheets`}, judged as the station judges boards (${esc(Lt.evidence)}, ${Math.round(Lt.alpha * 100)}% miss setting).${Lt.false_calls_per_coupon == null ? "" : ` False calls: ${Lt.false_calls_per_coupon.toFixed(2)} per pattern.`} Each cell: found / planted at that size; green means at least 90% at 95% confidence.</p>
    ${Object.entries(Lt.types).map(([t, v]) => renderLimit(t, v)).join("")}`;
}
async function etCapture(path) {
  const r = await api("/api/eyetest/capture", { method: "POST", body: JSON.stringify({ design: $("#design").value, path }) });
  renderSheets(r.captured);
  $("#et-msg").textContent = r.ok ? `Captured sheet ${r.sheet}${r.golden ? " (defect-free)" : ""}. ${r.missing?.length ? `Still to capture: ${r.missing.join(", ")}.` : "All sheets captured."}`
    : `Sheet not read: ${r.reason}. Make sure all four corner squares and the top centre square are in view.`;
  return r;
}
$("#et-capture").addEventListener("click", () => etCapture(null).catch((e) => toast(e.message)));
$("#et-demo").addEventListener("click", async () => { for (const p of state.eyetest_demo) { try { await etCapture(p); } catch (e) { toast(e.message); return; } } });
$("#et-reset").addEventListener("click", async () => { await api(`/api/eyetest/reset/${encodeURIComponent($("#design").value)}`, { method: "POST" }).catch((e) => toast(e.message)); $("#et-msg").textContent = ""; loadEyetest(); });
$("#et-finish").addEventListener("click", async () => {
  const b = $("#et-finish"); b.disabled = true; $("#et-msg").textContent = "Reading every sheet";
  try { await api("/api/eyetest/finish", { method: "POST", body: JSON.stringify({ design: $("#design").value, alpha: alpha() }) }); $("#et-msg").textContent = "Done."; loadEyetest(); }
  catch (e) { $("#et-msg").textContent = `Not read: ${e.message}`; }
  finally { b.disabled = false; }
});

// ======================= customer rules =======================
const CAP_TEXT = { guaranteed: ["ok", "The station can guarantee this"], limited: ["warn", "Partly guaranteed"],
  not_guaranteed: ["bad", "Not guaranteed by this station"], unmeasured: ["warn", "Not yet measured: run the eye test"] };
let lastRead = null;
function readerBlock(rd, i) {
  if (!rd.rule) return `<div class="reader"><div class="who">${esc(rd.reader)}</div><p class="rule">Could not read this requirement.</p></div>`;
  const v = rd.verify, c = rd.capability, ct = CAP_TEXT[c.status] || ["warn", c.status];
  const checks = v.status === "verified" ? `<p class="chk ok">Checks passed: the number, defect, comparison and action match the text.</p>` : `<p class="chk ${v.status === "check" ? "warn" : "bad"}">${esc(v.problems.join("; "))}</p>`;
  return `<div class="reader"><div class="who">Read by ${rd.reader === "llama" ? "Llama on this station" : "the keyword reader"}</div>
    <p class="rule">${esc(rd.sentence)}</p>${checks}<p class="chk ${ct[0]}">${esc(ct[1])}: ${esc(c.reason)}</p>
    <button class="btn primary" type="button" data-use="${i}">Use this rule</button></div>`;
}
$("#rule-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const clause = $("#clause").value.trim(); if (!clause) return;
  $("#rule-out").innerHTML = `<p class="pending">Reading</p>`;
  try {
    const r = await api("/api/rules/compile", { method: "POST", body: JSON.stringify({ design: $("#design").value, clause }) });
    lastRead = r;
    const single = r.readers.filter((x) => x.rule).length < 2 && !r.readers.some((x) => x.reader === "llama" && x.rule);
    const head = single ? `<p class="chk warn">Llama is not available, so only the keyword reader read this. Check the rule before using it.</p>`
      : r.readers_agree ? `<p class="chk ok">Both readers agree${r.status === "verified" ? " and the checks pass" : ", but check the warnings"}.</p>`
      : `<p class="chk warn">The readers disagree. Choose the one that matches the requirement, or rephrase it.</p>`;
    const showR = r.readers_agree ? [r.readers[0]] : r.readers;
    $("#rule-out").innerHTML = head + showR.map((rd) => readerBlock(rd, r.readers.indexOf(rd))).join("");
  } catch (e) { $("#rule-out").innerHTML = `<p class="chk bad">${esc(e.message)}</p>`; }
});
$("#rule-out").addEventListener("click", async (ev) => {
  const i = ev.target.dataset.use; if (i === undefined || !lastRead) return;
  const rd = lastRead.readers[+i];
  try {
    await api("/api/rules", { method: "POST", body: JSON.stringify({ design: $("#design").value, clause: lastRead.clause, rule: rd.rule, reader: rd.reader }) });
    $("#rule-out").innerHTML = `<p class="chk ok">Rule saved. It applies from the next inspection.</p>`; $("#clause").value = ""; loadRules();
  } catch (e) { toast(e.message); }
});
async function loadRules() {
  const rows = await api(`/api/rules/${encodeURIComponent($("#design").value)}`).catch(() => []);
  $("#rule-list").innerHTML = rows.length ? `<table><thead><tr><th>Rule</th><th>From the requirement</th><th>Can the station keep it?</th><th></th></tr></thead><tbody>${rows.map((r) => {
    const ct = CAP_TEXT[r.capability?.status] || ["warn", ""];
    return `<tr><td><b>${esc(r.sentence)}</b></td><td class="meta">"${esc(r.clause)}"</td><td><span class="chk ${ct[0]}" style="display:inline-block">${esc(ct[1])}</span>${r.capability?.reason ? `<div class="meta">${esc(r.capability.reason)}</div>` : ""}</td>
      <td><button class="btn small" type="button" data-del="${r.id}">Remove</button></td></tr>`; }).join("")}</tbody></table>`
    : `<p class="sub">No customer rules yet: boards are judged with the IPC-style table for Class ${state?.designs.find((d) => d.name === $("#design").value)?.customer_class ?? 2}.</p>`;
}
$("#rule-list").addEventListener("click", async (ev) => {
  const id = ev.target.dataset.del; if (!id) return;
  await api(`/api/rules/${id}`, { method: "DELETE" }).catch((e) => toast(e.message)); loadRules();
});

// ======================= start =======================
(async () => {
  try {
    await loadState();
    await loadSamples();
    if (state.demo) setSourceQuiet("samples");
    $("#strip").hidden = true;      // before the first result the start screen's gallery is the picker
    renderStart(); loadStats();
  } catch (e) { toast(`Station not reachable: ${e.message}`); }
  setInterval(() => loadState().catch(() => {}), 3000);
  setInterval(loadStats, 15000);
})();
