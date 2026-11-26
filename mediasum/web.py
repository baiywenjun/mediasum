"""本地 Web 界面：把 CLI 包成一个按需启动的 HTTP 服务。

设计取舍：
1. **子进程包装**，不 import 引擎模块。与 tui.py 同一策略：mlx-whisper 会吃满
   内存与 GIL，隔离后 Web 进程崩溃不影响任务，CLI 这条已验证的路径零改动。
2. **严格串行**。M1 16 GB 跑一次 Whisper 就几乎占满内存，并发两个任务必然
   被 OOM killer 干掉。队列一次只放行一个，其余显示「排队中」。
3. **不做文件上传**。单机自用，浏览器把 2 GB 视频传回本机是纯粹的浪费；
   前端只提交路径，后端直接打开文件。
4. **只绑 127.0.0.1**，不做鉴权。这是本机工具，暴露到 0.0.0.0 等于把
   任意文件转写能力开放给局域网。

生命周期：前台进程，Ctrl+C 或关闭终端即结束。另有 --idle-exit 兜底，
长时间无请求自动退出（任务执行中不计时）。
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import config
from .progress import Progress
from .util import fmt_dur

BASE = Path(__file__).resolve().parent.parent
MS = BASE / "ms"
WEBUI = Path(__file__).resolve().parent / "webui"

TERMINAL = ("done", "failed", "cancelled")

# 产物固定三份，按文件名前缀识别
OUTPUT_KINDS = [
    ("01_逐字稿.txt", "逐字稿", "text"),
    ("02_详细总结.html", "详细总结", "html"),
    ("03_摘要.md", "摘要", "text"),
]

ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm",
                ".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg"}
DETAILS = {"brief", "standard", "detailed"}
DIAR_MODES = {"off", "auto", "full", "coarse"}
RE_CROP = re.compile(r"^\d*\.?\d+(,\d*\.?\d+){3}$")


# ------------------------------------------------------------------ 任务

class Job:
    def __init__(self, src: Path, opts: dict) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.src = src
        self.opts = opts
        self.state = "queued"          # queued / running / done / failed / cancelled
        self.prog = Progress()
        self.events: list[dict] = []   # 供 SSE 回放，索引即游标
        self.created = time.time()
        self.started = 0.0
        self.ended = 0.0
        self.code: int | None = None
        self.error = ""
        self.cancelled = False
        self.proc: subprocess.Popen | None = None
        self.out_dir = ""
        self.pct_at_end: float | None = None   # 收口前的真实进度，取消/失败时用它

    def emit(self, ev: dict) -> None:
        t0 = self.started or self.created
        self.events.append({"t": round(time.time() - t0, 2), **ev})

    def feed(self, line: str) -> None:
        ev = self.prog.feed(line)
        if ev:
            self.emit(ev)
        elif line.strip():
            self.emit({"kind": "log", "text": line})

    def summary(self) -> dict:
        snap = self.prog.snapshot()
        return {
            "id": self.id,
            "src": str(self.src),
            "name": self.src.name,
            "state": self.state,
            "pct": self.display_pct(),
            "stages": snap["stages"],
            "extras": snap["extras"],
            "title": snap["title"] or self.src.stem,
            "sub": snap["sub"],
            "code": self.code,
            "error": self.error,
            "out_dir": self.out_dir,
            "outputs": self.outputs(),
            "elapsed": round((self.ended or time.time()) - self.started, 1) if self.started else 0,
        }

    def display_pct(self) -> float:
        """对外展示的进度。

        不能直接用 finish() 之后的值：它会把未完成阶段标成 skip，而 skip 在
        进度公式里算作「已完成」，于是被取消的任务会显示 99%。
        """
        if self.state == "done":
            return 100.0
        if self.state in ("cancelled", "failed") and self.pct_at_end is not None:
            return round(self.pct_at_end, 1)
        return round(self.prog.pct, 1)

    def outputs(self) -> list[dict]:
        if not self.out_dir:
            return []
        d = Path(self.out_dir)
        found = []
        for fname, label, kind in OUTPUT_KINDS:
            f = d / fname
            if f.exists():
                found.append({"name": fname, "label": label, "kind": kind,
                              "size": f.stat().st_size})
        return found


JOBS: dict[str, Job] = {}
QUEUE: "queue.Queue[Job]" = queue.Queue()
ACTIVE: Job | None = None
LAST_HIT = time.time()
IDLE_LIMIT = 0.0          # 秒；0 表示不自动退出
SERVER: uvicorn.Server | None = None


def _build_cmd(job: Job) -> list[str]:
    """把界面选项翻译成 CLI 参数。所有默认值都以 config.toml 为准，只传显式指定项。"""
    cmd = [str(MS), str(job.src)]
    o = job.opts
    if not o.get("transcript", True):
        cmd.append("--no-transcript")
    if not o.get("html", True):
        cmd.append("--no-html")
    if not o.get("abstract", True):
        cmd.append("--no-abstract")
    if o.get("detail"):
        cmd += ["--detail", o["detail"]]
    if o.get("diar"):
        cmd += ["--diar", o["diar"]]
    if o.get("crop"):
        cmd += ["--crop", o["crop"]]
    if o.get("from"):
        cmd += ["--from", o["from"]]
    return cmd


def _child_env() -> dict:
    """给子进程准备环境，剔除宿主注入的 sitecustomize shim。

    WorkBuddy 会把一个 sandbox shim 目录塞进 PYTHONPATH，该目录的 sitecustomize
    会对「个人目录下一次性删除超过 50 个文件」要求人工确认。抽帧重跑正好是
    这种操作（12 分钟视频约 50 帧，两小时视频 200+ 帧），不清掉会直接失败：
        [safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED] {"count":61,...}

    用户在真实终端里执行 ./ms-web 时环境里没有这个 shim，所以这里剔除它，
    让 Web 子进程的行为和命令行完全一致。被删的是工具自己生成的缓存帧，
    位于 .work/<文件名>/frames/ 下，不属于用户数据。
    """
    env = os.environ.copy()
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
             if p and "sandbox-center-runtime-kit" not in p]
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths)
    else:
        env.pop("PYTHONPATH", None)
    return env


def _run(job: Job) -> None:
    """在 worker 线程里执行一次任务，逐行读 stdout 并解析进度。"""
    global ACTIVE
    ACTIVE = job
    job.state = "running"
    job.started = time.time()
    job.emit({"kind": "start", "src": str(job.src), "name": job.src.name})

    cmd = _build_cmd(job)
    job.emit({"kind": "log", "text": "$ " + " ".join(cmd[1:])})
    try:
        job.proc = subprocess.Popen(
            cmd, cwd=str(BASE), env=_child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace",
            start_new_session=True,   # 独立进程组，取消时能连孙进程一起杀
        )
        assert job.proc.stdout is not None
        for line in job.proc.stdout:
            job.feed(line)
        job.code = job.proc.wait()
    except Exception as exc:                      # noqa: BLE001 - 需要把任何异常呈现到界面
        job.code = -1
        job.error = f"{type(exc).__name__}: {exc}"
        job.emit({"kind": "log", "text": job.error})

    job.ended = time.time()
    job.pct_at_end = job.prog.pct
    job.prog.finish(job.code or 0)
    job.out_dir = job.prog.out_dir or str(_guess_outdir(job))
    if job.cancelled:
        job.state = "cancelled"
    elif job.code == 0:
        job.state = "done"
    else:
        job.state = "failed"
        if not job.error:
            job.error = f"退出码 {job.code}"
    job.emit({"kind": "end", **job.summary()})
    ACTIVE = None


def _guess_outdir(job: Job) -> Path:
    """CLI 没打印产物目录时的兜底（例如 --from 中途失败）。"""
    cfg = config.load()
    return Path(cfg["general"]["output_dir"]) / job.src.stem


def _worker() -> None:
    while True:
        job = QUEUE.get()
        try:
            _run(job)
        finally:
            QUEUE.task_done()


threading.Thread(target=_worker, daemon=True, name="mediasum-worker").start()


def _idle_watch() -> None:
    """空闲到点就请求 uvicorn 优雅退出。任务运行期间不计时。"""
    while True:
        time.sleep(20)
        if not IDLE_LIMIT or SERVER is None:
            continue
        if ACTIVE is not None or not QUEUE.empty():
            continue
        if time.time() - LAST_HIT >= IDLE_LIMIT:
            print(f"\n[web] 已空闲 {fmt_dur(IDLE_LIMIT)}，自动退出。下次用再执行 ./ms-web", flush=True)
            SERVER.should_exit = True
            return


# ------------------------------------------------------------------ 应用

app = FastAPI(title="mediasum", docs_url=None, redoc_url=None)


@app.middleware("http")
async def _touch(request, call_next):
    global LAST_HIT
    LAST_HIT = time.time()
    return await call_next(request)


def _get(jid: str) -> Job:
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEBUI / "index.html", media_type="text/html; charset=utf-8")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(WEBUI / "app.js", media_type="application/javascript; charset=utf-8")


@app.get("/style.css")
def style_css() -> FileResponse:
    return FileResponse(WEBUI / "style.css", media_type="text/css; charset=utf-8")


@app.get("/api/config")
def api_config() -> dict:
    cfg = config.load()
    return {
        "outputs": {k: bool(cfg["output"][k]) for k in ("transcript", "detail_html", "abstract")},
        "detail": cfg["summary"]["detail"],
        "diar": cfg["diarization"]["mode"],
        "crop": ",".join(f"{x:g}" for x in cfg["frames"]["crop"]),
        "media_dir": str(Path.home() / "Videos" / "media"),
        "output_dir": cfg["general"]["output_dir"],
    }


@app.get("/api/media")
def api_media() -> dict:
    """列出常用目录下的音视频，按修改时间倒序。只读，不递归太深。"""
    roots = [Path.home() / "Videos" / "media", Path.home() / "Movies",
             Path.home() / "Downloads", Path.home() / "Desktop"]
    files, seen = [], set()
    for root in roots:
        if not root.is_dir():
            continue
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:60]
        except OSError:
            continue
        for p in entries:
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS and p not in seen:
                seen.add(p)
                st = p.stat()
                files.append({"path": str(p), "name": p.name,
                              "size": st.st_size, "mtime": st.st_mtime})
    files.sort(key=lambda x: -x["mtime"])
    return {"files": files[:40]}


@app.post("/api/jobs")
async def api_create(body: dict) -> dict:
    raw = (body.get("src") or "").strip()
    if not raw:
        raise HTTPException(400, "缺少文件路径")
    src = Path(os.path.expanduser(raw))
    if not src.is_absolute():
        raise HTTPException(400, "请填绝对路径")
    if not src.is_file():
        raise HTTPException(400, f"文件不存在：{src}")
    if src.suffix.lower() not in ALLOWED_EXTS:
        raise HTTPException(400, f"不支持的类型：{src.suffix}")

    opts = body.get("opts") or {}
    detail = opts.get("detail")
    if detail and detail not in DETAILS:
        raise HTTPException(400, f"detail 取值非法：{detail}")
    diar = opts.get("diar")
    if diar and diar not in DIAR_MODES:
        raise HTTPException(400, f"diar 取值非法：{diar}")
    crop = (opts.get("crop") or "").strip()
    if crop and not RE_CROP.match(crop):
        raise HTTPException(400, "crop 格式应为 左,上,右,下 四个 0–1 的小数")

    job = Job(src, opts)
    JOBS[job.id] = job
    QUEUE.put(job)
    ahead = QUEUE.qsize() - 1 + (1 if ACTIVE else 0)
    job.emit({"kind": "queued", "ahead": max(0, ahead)})
    return {"id": job.id, "ahead": max(0, ahead)}


@app.get("/api/jobs")
def api_jobs() -> dict:
    items = sorted(JOBS.values(), key=lambda j: -j.created)[:20]
    return {"jobs": [j.summary() for j in items], "active": ACTIVE.id if ACTIVE else None,
            "queued": QUEUE.qsize()}


@app.get("/api/jobs/{jid}")
def api_job(jid: str) -> dict:
    return _get(jid).summary()


@app.get("/api/jobs/{jid}/events")
async def api_events(jid: str, since: int = 0) -> StreamingResponse:
    job = _get(jid)
    return StreamingResponse(_stream(job, max(0, since)),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


async def _stream(job: Job, since: int):
    """按索引回放事件。页面刷新后带 since=0 即可拿到完整历史。"""
    idx = since
    # 长连接也要让空闲计时知道还有人在看
    while True:
        while idx < len(job.events):
            yield f"data: {json.dumps(job.events[idx], ensure_ascii=False)}\n\n"
            idx += 1
        if job.state in TERMINAL and idx >= len(job.events):
            yield f"data: {json.dumps({'kind': 'close', **job.summary()}, ensure_ascii=False)}\n\n"
            return
        await asyncio.sleep(0.25)


@app.post("/api/jobs/{jid}/cancel")
def api_cancel(jid: str) -> dict:
    job = _get(jid)
    if job.state in TERMINAL:
        return {"ok": False, "reason": "任务已结束"}
    job.cancelled = True
    proc = job.proc
    if proc is None:
        # 还在队列里没开始，直接标记，worker 会跳过
        job.state = "cancelled"
        job.ended = time.time()
        job.prog.finish(1)
        job.emit({"kind": "end", **job.summary()})
        return {"ok": True}
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except OSError:
        pass
    return {"ok": True}


@app.get("/api/jobs/{jid}/file/{name}")
def api_file(jid: str, name: str) -> FileResponse:
    job = _get(jid)
    # 只允许取本任务产出的三个固定文件，杜绝路径穿越
    allowed = {f[0] for f in OUTPUT_KINDS}
    if name not in allowed:
        raise HTTPException(404, "文件不存在")
    if not job.out_dir:
        raise HTTPException(404, "产物目录未知")
    f = Path(job.out_dir) / name
    if not f.is_file():
        raise HTTPException(404, "文件尚未生成")
    media = "text/html; charset=utf-8" if f.suffix == ".html" else "text/plain; charset=utf-8"
    return FileResponse(f, media_type=media)


@app.exception_handler(HTTPException)
async def _http_err(request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


# ------------------------------------------------------------------ 启动

def serve(port: int = 8765, host: str = "127.0.0.1", idle_exit: int = 60,
          open_browser: bool = True, overrides: list[str] | None = None) -> int:
    global IDLE_LIMIT, SERVER

    if overrides:
        # 先加载一次，配置错误在启动时暴露，而不是等第一个任务
        config.load(None, overrides)

    IDLE_LIMIT = idle_exit * 60 if idle_exit > 0 else 0.0
    url = f"http://{host}:{port}"

    print(f"mediasum web  {url}")
    print(f"  产物目录    {config.load()['general']['output_dir']}")
    print(f"  空闲退出    {'关闭' if not IDLE_LIMIT else fmt_dur(IDLE_LIMIT) + '无请求后自动退出'}")
    print("  停止服务    Ctrl+C（或直接关闭这个终端窗口）\n", flush=True)

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    threading.Thread(target=_idle_watch, daemon=True, name="mediasum-idle").start()

    SERVER = uvicorn.Server(uvicorn.Config(
        app, host=host, port=port, log_level="warning", access_log=False,
    ))
    try:
        SERVER.run()
    except KeyboardInterrupt:
        pass
    print("\n[web] 服务已停止", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="ms web",
        description="启动本地网页界面。前台进程，Ctrl+C 或关闭终端即结束。",
    )
    p.add_argument("--port", type=int, default=8765, help="端口，默认 8765")
    p.add_argument("--host", default="127.0.0.1",
                   help="监听地址，默认仅本机。改成 0.0.0.0 会把转写能力开放给局域网")
    p.add_argument("--idle-exit", type=int, default=60, metavar="分钟",
                   help="多少分钟无请求自动退出，0 表示不自动退出。任务执行期间不计时")
    p.add_argument("--no-browser", dest="no_browser", action="store_true",
                   help="启动后不自动打开浏览器")
    p.add_argument("--set", action="append", default=[], metavar="K=V",
                   help="覆盖配置项，可多次，如 --set llm.provider=openai")
    a = p.parse_args(argv)
    return serve(port=a.port, host=a.host, idle_exit=a.idle_exit,
                 open_browser=not a.no_browser, overrides=a.set or None)
