/* mediasum web 前端。无构建步骤，纯 DOM 操作。
   进度由后端解析 CLI 日志后以 SSE 推送，前端只负责呈现。 */

const STAGES = [
  ["S1", "读取"], ["S2", "抽音"], ["S3", "转写"], ["S4", "标点"],
  ["S5", "抽帧"], ["S6", "说话人"], ["S7", "归纳"], ["S8", "产出"],
];
const CORE = ["S1", "S3", "S4", "S5", "S7", "S8"];
const EXTRA = ["S2", "S6"];
const MAX_LOG_LINES = 2000;

const $ = (id) => document.getElementById(id);

let job = null;          // 当前任务状态
let shownPct = 0;        // 已展示的最大进度：分母会随条件阶段出现而变大，
                         // 不夹一下会出现 16.7% → 14.3% 的倒退，看着像出错
let es = null;           // EventSource
let tick = null;         // 计时器
let busy = false;

/* ------------------------------------------------------------------ 阶段 */

function initStages() {
  $("stages").innerHTML = "";
  for (const [tag, label] of STAGES) {
    const el = document.createElement("span");
    el.className = "stage idle";
    el.id = "st-" + tag;
    el.innerHTML = `<span class="mk"></span><span class="nm">${label}</span>`;
    $("stages").appendChild(el);
  }
}

function paintStages() {
  if (!job) return;
  for (const [tag] of STAGES) {
    const el = $("st-" + tag);
    if (!el) continue;
    const st = job.stages[tag] || "idle";
    el.className = "stage " + st;
  }
}

function calcPct() {
  if (!job) return 0;
  if (job.state === "done") return 100;
  const extras = job.extras || [];
  const order = CORE.concat(extras.filter((t) => EXTRA.includes(t)));
  const denom = order.length || 1;
  let acc = 0;
  for (const tag of order) {
    const st = job.stages[tag] || "idle";
    if (st === "done" || st === "skip") { acc += 1; continue; }
    if (st === "running") {
      const s = job.sub || {};
      const frac = s.total ? Math.min(0.9, s.cur / s.total) : 0;
      return Math.min(99, ((acc + frac) / denom) * 100);
    }
    break;
  }
  return Math.min(99, (acc / denom) * 100);
}

function paintProgress() {
  const pct = calcPct();
  shownPct = pct > shownPct ? pct : shownPct;
  $("bar").style.width = shownPct.toFixed(1) + "%";
  paintStages();
  paintStat(shownPct);
}

function fmtDur(sec) {
  sec = Math.round(sec);
  if (sec < 60) return sec + "s";
  if (sec < 3600) return Math.floor(sec / 60) + "m" + String(sec % 60).padStart(2, "0") + "s";
  return Math.floor(sec / 3600) + "h" + String(Math.floor((sec % 3600) / 60)).padStart(2, "0") + "m";
}

function paintStat(pct) {
  const el = $("stat");
  el.className = "stat";
  if (!job) { el.textContent = "待命"; return; }

  const st = job.state;
  if (st === "queued") {
    el.textContent = "排队中，前面还有任务在跑";
    return;
  }
  if (st === "running") {
    const running = STAGES.find(([t]) => job.stages[t] === "running");
    const name = running ? running[1] : "启动";
    const sub = (job.sub && job.sub.label) ? " · " + job.sub.label : "";
    const used = job.startedAt ? fmtDur((Date.now() / 1000) - job.startedAt) : "00:00";
    el.textContent = `${name}${sub} · ${pct.toFixed(0)}% · 已用 ${used}`;
    return;
  }
  if (st === "done") {
    el.className = "stat ok";
    el.textContent = `完成 · 用时 ${job.elapsed ? fmtDur(job.elapsed) : "—"}`;
    return;
  }
  if (st === "cancelled") {
    el.className = "stat";
    el.textContent = "已取消。中间结果留在工作目录，重跑同一文件会复用缓存";
    return;
  }
  el.className = "stat err";
  el.textContent = "失败：" + (job.error || "未知错误");
}

/* ------------------------------------------------------------------ 日志 */

function appendLog(text, cls) {
  const box = $("log");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;

  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  box.appendChild(line);

  while (box.childElementCount > MAX_LOG_LINES) box.firstElementChild.remove();
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function clearLog() {
  $("log").innerHTML = "";
}

/* ------------------------------------------------------------------ 产物 */

function paintOutputs() {
  const wrap = $("outputs");
  wrap.innerHTML = "";
  if (!job || !job.outputs) return;
  for (const o of job.outputs) {
    const a = document.createElement("a");
    a.className = "out";
    a.href = `/api/jobs/${job.id}/file/${encodeURIComponent(o.name)}`;
    a.target = "_blank";
    a.rel = "noopener";
    const kb = o.size / 1024;
    const size = kb > 1024 ? (kb / 1024).toFixed(1) + " MB" : Math.round(kb) + " KB";
    a.innerHTML = `<span>${o.label}</span><span class="sz">${size}</span>`;
    wrap.appendChild(a);
  }
}

/* ------------------------------------------------------------------ 事件 */

function onEvent(ev) {
  switch (ev.kind) {
    case "queued":
      job.state = "queued";
      paintProgress();
      break;

    case "start":
      job.state = "running";
      job.startedAt = Date.now() / 1000;
      if (!tick) tick = setInterval(() => paintStat(calcPct()), 1000);
      paintProgress();
      break;

    case "stage_start":
      // 后端会把「已判定跳过」的阶段一并带过来，避免前端各自推断
      if (ev.stages) job.stages = { ...job.stages, ...ev.stages };
      job.stages[ev.stage] = "running";
      if (EXTRA.includes(ev.stage) && !job.extras.includes(ev.stage)) job.extras.push(ev.stage);
      job.sub = { label: "", cur: 0, total: 0 };
      paintProgress();
      break;

    case "stage_done":
      if (ev.stages) job.stages = { ...job.stages, ...ev.stages };
      job.stages[ev.stage] = "done";
      job.sub = { label: "", cur: 0, total: 0 };
      paintProgress();
      break;

    case "sub":
      job.stages[ev.stage] = "running";
      job.sub = { label: ev.label, cur: ev.cur, total: ev.total };
      paintProgress();
      break;

    case "title":
      job.title = ev.text;
      $("meta").textContent = `处理中 · ${ev.text}`;
      break;

    case "log":
      appendLog(ev.text, ev.text.startsWith("$ ") ? "cmd" : "");
      break;

    case "ok":
      if (ev.out_dir) job.outDir = ev.out_dir;
      break;

    case "end":
    case "close":
      finish(ev);
      break;
  }
}

function finish(ev) {
  job.state = ev.state || "done";
  job.stages = ev.stages || job.stages;
  job.outputs = ev.outputs || [];
  job.error = ev.error || "";
  job.elapsed = ev.elapsed || 0;
  job.outDir = ev.out_dir || job.outDir;
  if (job.state === "done") {
    job.stages = Object.fromEntries(Object.keys(job.stages).map((k) => {
      const v = job.stages[k];
      return [k, v === "idle" ? "skip" : v];
    }));
  }
  if (job.state === "done") { shownPct = 100; $("bar").style.width = "100%"; }
  paintProgress();
  paintOutputs();
  setBusy(false);
  if (tick) { clearInterval(tick); tick = null; }
  if (job.state === "done") $("meta").textContent = `完成 · ${job.title || ""}`;
  if (es) { es.close(); es = null; }
}

/* ------------------------------------------------------------------ 流程 */

function setBusy(on) {
  busy = on;
  $("btnStart").disabled = on;
  $("btnCancel").disabled = !on;
}

function resetState() {
  job = { id: null, state: "idle", stages: {}, extras: [], sub: {}, outputs: [] };
  for (const [tag] of STAGES) job.stages[tag] = "idle";
  clearLog();
  $("outputs").innerHTML = "";
  $("bar").style.width = "0%";
  shownPct = 0;
  if (tick) { clearInterval(tick); tick = null; }
}

async function start() {
  const src = $("src").value.trim();
  if (!src) {
    $("src").focus();
    $("srcHint").textContent = "先填文件路径。";
    $("srcHint").classList.add("warn");
    return;
  }
  $("srcHint").classList.remove("warn");
  $("srcHint").textContent = "支持 mp4 / mov / mkv / mp3 / m4a / wav 等；不经过上传，后端直接读本地文件。";

  resetState();
  setBusy(true);

  const opts = {
    transcript: $("optTranscript").checked,
    html: $("optHtml").checked,
    abstract: $("optAbstract").checked,
    detail: $("optDetail").value,
    diar: $("optDiar").value,
    crop: $("optCrop").value.trim(),
  };
  if (!opts.transcript && !opts.html && !opts.abstract) {
    appendLog("三份产物全部关闭，这一次不会有输出。", "err");
    setBusy(false);
    return;
  }

  try {
    const r = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ src, opts }),
    });
    const data = await r.json();
    if (!r.ok) {
      appendLog("提交失败：" + (data.error || r.status), "err");
      setBusy(false);
      return;
    }
    job.id = data.id;
    $("meta").textContent = "已提交，准备中…";
    attach(data.id);
  } catch (e) {
    appendLog("提交失败：" + e, "err");
    setBusy(false);
  }
}

async function cancel() {
  if (!job || !job.id) return;
  appendLog("已请求中断…");
  try {
    await fetch(`/api/jobs/${job.id}/cancel`, { method: "POST" });
  } catch (e) {
    appendLog("取消失败：" + e, "err");
  }
}

function attach(jobId) {
  if (es) es.close();
  es = new EventSource(`/api/jobs/${jobId}/events?since=0`);
  es.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch { return; }
    onEvent(ev);
  };
  es.onerror = () => {
    // 服务被关掉或连接中断：不要无限重连，提示一次后收起
    if (es) { es.close(); es = null; }
    if (job && job.state === "running") {
      appendLog("与服务的连接中断。若终端已关闭，任务也会随之终止。", "err");
      setBusy(false);
    }
  };
}

/* ------------------------------------------------------------------ 初始化 */

async function initUI() {
  initStages();
  resetState();
  try {
    const cfg = await (await fetch("/api/config")).json();
    $("optTranscript").checked = cfg.outputs.transcript;
    $("optHtml").checked = cfg.outputs.detail_html;
    $("optAbstract").checked = cfg.outputs.abstract;
    $("optDetail").value = cfg.detail;
    $("optDiar").value = cfg.diar;
    $("optCrop").value = cfg.crop || "";
    $("meta").textContent = `产物落在 ${cfg.output_dir}`;
  } catch (e) {
    $("meta").textContent = "配置读取失败";
  }

  try {
    const media = await (await fetch("/api/media")).json();
    const sel = $("recent");
    for (const f of media.files) {
      const o = document.createElement("option");
      o.value = f.path;
      o.textContent = f.name;
      sel.appendChild(o);
    }
    if (!media.files.length) sel.options[0].textContent = "无最近文件";
  } catch { /* 列表失败不影响主流程 */ }

  // 页面刷新后接管正在跑的任务
  try {
    const data = await (await fetch("/api/jobs")).json();
    const live = (data.jobs || []).find((j) => j.state === "running" || j.state === "queued");
    if (live) {
      job = { ...live, extras: live.extras || [], sub: live.sub || {}, outputs: live.outputs || [] };
      job.startedAt = live.elapsed ? Date.now() / 1000 - live.elapsed : null;
      $("src").value = live.src;
      if (live.state === "running" && !tick) tick = setInterval(() => paintStat(calcPct()), 1000);
      paintProgress();
      paintOutputs();
      setBusy(live.state !== "queued" ? true : true);
      attach(live.id);
    }
  } catch { /* 无历史任务 */ }
}

$("btnStart").addEventListener("click", start);
$("btnCancel").addEventListener("click", cancel);
$("src").addEventListener("keydown", (e) => { if (e.key === "Enter") start(); });
$("recent").addEventListener("change", (e) => {
  if (e.target.value) $("src").value = e.target.value;
});
$("btnFold").addEventListener("click", () => {
  const box = $("log").parentElement;
  box.classList.toggle("folded");
  $("btnFold").textContent = box.classList.contains("folded") ? "展开" : "折叠";
});

initUI();
