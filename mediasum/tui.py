"""终端界面（TUI）。

设计取向：本文件只是一个外壳，通过子进程调用 ms，不 import 引擎模块。
好处是 TUI 崩溃不影响任务、mlx-whisper 的内存与 GIL 不污染界面进程，
CLI 那条已经验证过的路径也不会被改动。

    ms-tui                     启动界面
    ms-tui <音视频路径>         启动并预填文件
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import (Button, Checkbox, Footer, Header, Input, Label,
                             OptionList, ProgressBar, RichLog, Select, Static)
from textual.widgets.option_list import Option

from . import config
from .util import fmt_dur

BASE = Path(__file__).resolve().parent.parent
MS = BASE / "ms"
MEDIA = Path.home() / "Videos" / "media"
EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm",
        ".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg"}

# 阶段清单固定为全量，未执行的阶段显示为跳过
STAGES = [("S1", "读取"), ("S2", "抽音"), ("S3", "转写"), ("S4", "标点"),
          ("S5", "抽帧"), ("S6", "说话人"), ("S7", "归纳"), ("S8", "产出")]
CORE_STAGES = ["S1", "S3", "S4", "S5", "S7", "S8"]
EXTRA_STAGES = ["S2", "S6"]

RE_DONE = re.compile(r"^\[(S\d)\] 完成，耗时 (.+)$")
RE_START = re.compile(r"^\[(S\d)\] ▶ (.+)$")
RE_FRAME = re.compile(r"帧 (\d+)/(\d+)")
RE_MAP = re.compile(r"第 (\d+)/(\d+) 段")
RE_TITLE = re.compile(r"^\[S7\] 标题：(.+)$")
RE_OK = re.compile(r"^\[OK\] 全部完成，(.+?)(?: · 产物目录 (.+))?$")

CSS = """
Screen { background: $surface; }
#pathrow { height: 3; }
#path { width: 1fr; }
#pathrow Button { width: 12; }
#files { height: 4; display: none; border: round $primary; }
.row { height: 3; }
.k { width: 12; height: 3; content-align: left middle; color: $text-muted; }
.k2 { width: 10; height: 3; content-align: left middle; color: $text-muted; margin-left: 2; }
.row Checkbox { width: 21; }
.row Select { width: 18; }
#crop { width: 32; }
#hint { width: 1fr; height: 3; content-align: left middle; color: $text-muted; }
#actions { height: 3; }
#actions Button { width: 19; margin: 0 1 0 0; }
#stages { height: 1; padding: 0 1; }
#bar { height: 1; }
#stat { height: 1; padding: 0 1; color: $text-muted; }
#log { height: 1fr; border: round $primary; }
"""


class MediasumApp(App):
    """音视频转笔记的终端界面。"""

    TITLE = "mediasum"
    SUB_TITLE = "音视频转笔记"
    CSS = CSS
    BINDINGS = [
        ("ctrl+r", "run", "开始"),
        ("ctrl+k", "stop", "取消"),
        ("ctrl+o", "open_out", "产物目录"),
        ("q", "quit", "退出"),
    ]

    def __init__(self, preset: str | None = None) -> None:
        super().__init__()
        # 注意：不要用 _running 作属性名，Textual 的 MessagePump 占用了它，
        # 框架启动时会把它置为 True，导致「有任务在跑」的守卫永远为真。
        self.preset = preset
        self.cfg = self._load_cfg()
        self.state = {tag: "idle" for tag, _ in STAGES}
        self.extras: set[str] = set()
        self.sub_frac = 0.0
        self.sub_label = ""
        self.out_dir = Path(self.cfg["general"]["output_dir"])
        self._proc: subprocess.Popen | None = None
        self._cancelled = False
        self._t0 = 0.0
        self._busy = False

    @staticmethod
    def _load_cfg() -> dict:
        try:
            return config.load()
        except SystemExit:
            return dict(config.DEFAULTS)

    # ---------------------------------------------------------------- 布局
    def compose(self) -> ComposeResult:
        o, s, d = self.cfg["output"], self.cfg["summary"], self.cfg["diarization"]
        yield Header(show_clock=False)

        with Horizontal(id="pathrow"):
            yield Input(value=self.preset or "", id="path",
                        placeholder="视频或音频路径（可把文件拖进终端窗口）")
            yield Button("扫描", id="scan")
        yield OptionList(id="files")

        with Horizontal(classes="row"):
            yield Label("产物", classes="k")
            yield Checkbox("逐字稿", value=bool(o["transcript"]), id="cb_tr")
            yield Checkbox("详细 HTML", value=bool(o["detail_html"]), id="cb_html")
            yield Checkbox("摘要", value=bool(o["abstract"]), id="cb_abs")

        with Horizontal(classes="row"):
            yield Label("详细程度", classes="k")
            yield Select([("简要", "brief"), ("标准", "standard"), ("详尽", "detailed")],
                         value=s["detail"], allow_blank=False, id="sel_detail")
            yield Label("说话人", classes="k2")
            yield Select([("关闭", "off"), ("自动", "auto"), ("全量", "full"),
                          ("粗分", "coarse")],
                         value=d["mode"], allow_blank=False, id="sel_diar")

        crop = ",".join(f"{x:g}" for x in self.cfg["frames"]["crop"])
        with Horizontal(classes="row"):
            yield Label("画面裁剪", classes="k")
            yield Input(value=crop, id="crop",
                        placeholder="左,上,右,下 如 0.02,0.07,0.78,0.95")
            yield Static("留空=全画面；直播录屏建议填，否则 OCR 会读进聊天区与二维码",
                         id="hint")

        with Horizontal(id="actions"):
            yield Button("开始处理", id="run", variant="primary")
            yield Button("取消", id="stop", variant="error", disabled=True)
            yield Button("打开产物目录", id="open", disabled=True)
            yield Button("打开 HTML", id="openhtml", disabled=True)

        yield Static("", id="stages")
        yield ProgressBar(total=100, show_percentage=False, show_eta=False, id="bar")
        yield Static("", id="stat")
        yield RichLog(id="log", markup=True, wrap=True, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_stages()
        self.set_interval(1.0, self._tick)
        log = self.query_one("#log", RichLog)
        log.write(f"[dim]引擎 {BASE / 'ms'} · 产物写到 {self.out_dir}[/]")
        if MEDIA.is_dir():
            n = sum(1 for p in MEDIA.iterdir() if p.suffix.lower() in EXTS)
            log.write(f"[dim]{MEDIA} 下有 {n} 个音视频，点「扫描」列出[/]")
        else:
            log.write(f"[yellow]{MEDIA} 不存在，可直接把路径粘到输入框[/]")

    # ---------------------------------------------------------------- 渲染
    def _refresh_stages(self) -> None:
        parts = []
        for tag, name in STAGES:
            st = self.state[tag]
            if st == "done":
                parts.append(f"[green]{tag} ✓[/]")
            elif st == "running":
                parts.append(f"[bold cyan]{tag} {name}[/]")
            elif st == "skip":
                parts.append(f"[dim]{tag} −[/]")
            else:
                parts.append(f"[dim]{tag}[/]")
        self.query_one("#stages", Static).update("   ".join(parts))

    def _pct(self) -> float:
        """S2/S6 是条件阶段（纯音频无抽帧、说话人关闭时不出现），
        所以分母用「常态六阶段 + 本次实际出现的条件阶段」，否则进度条会卡住不动。"""
        stages = CORE_STAGES + [t for t in EXTRA_STAGES if t in self.extras]
        if not stages:
            return 0.0
        acc = 0.0
        for tag in stages:
            st = self.state[tag]
            if st in ("done", "skip"):
                acc += 1.0
            elif st == "running":
                acc += min(0.95, self.sub_frac)
        return min(100.0, acc / len(stages) * 100)

    def _refresh_progress(self) -> None:
        pct = self._pct()
        self.query_one("#bar", ProgressBar).update(progress=pct)
        elapsed = fmt_dur(time.time() - self._t0) if self._t0 else "00:00"
        bits = [f"{pct:.0f}%", f"已用 {elapsed}"]
        if self.sub_label:
            bits.insert(0, self.sub_label)
        self.query_one("#stat", Static).update(" · ".join(bits))

    def _tick(self) -> None:
        if self._busy:
            self._refresh_progress()

    def _log(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)

    # ---------------------------------------------------------------- 事件
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self.query_one("#path", Input).value = event.option_id
            self.query_one("#files", OptionList).display = False
            self.query_one("#path", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "scan":
            self.action_scan()
        elif bid == "run":
            self.action_run()
        elif bid == "stop":
            self.action_stop()
        elif bid == "open":
            self.action_open_out()
        elif bid == "openhtml":
            self.action_open_html()

    def action_scan(self) -> None:
        ol = self.query_one("#files", OptionList)
        ol.clear_options()
        if not MEDIA.is_dir():
            self._log(f"[yellow]{MEDIA} 不存在[/]")
            return
        files = [p for p in MEDIA.iterdir()
                 if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith(".")]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            self._log(f"[yellow]{MEDIA} 里没有可处理的音视频[/]")
            return
        for p in files[:40]:
            mb = p.stat().st_size / 1048576
            ol.add_option(Option(f"{p.name}   {mb:.0f} MB", id=str(p)))
        ol.display = True
        ol.focus()

    # ---------------------------------------------------------------- 执行
    def _build_cmd(self) -> list[str] | None:
        src_text = self.query_one("#path", Input).value.strip().strip("'\"")
        if not src_text:
            self._log("[red]请先填写文件路径[/]")
            return None
        src = Path(src_text).expanduser()
        if not src.exists():
            self._log(f"[red]文件不存在：{src}[/]")
            return None

        cmd = [str(MS), str(src)]
        if not self.query_one("#cb_tr", Checkbox).value:
            cmd.append("--no-transcript")
        if not self.query_one("#cb_html", Checkbox).value:
            cmd.append("--no-html")
        if not self.query_one("#cb_abs", Checkbox).value:
            cmd.append("--no-abstract")

        detail = self.query_one("#sel_detail", Select).value
        diar = self.query_one("#sel_diar", Select).value
        if isinstance(detail, str):
            cmd += ["--detail", detail]
        if isinstance(diar, str):
            cmd += ["--diar", diar]
        crop = self.query_one("#crop", Input).value.strip()
        if crop:
            cmd += ["--crop", crop]

        self.out_dir = Path(self.cfg["general"]["output_dir"]) / src.stem
        return cmd

    def action_run(self) -> None:
        if self._busy:
            self._log("[yellow]已有任务在跑，先取消再开始[/]")
            return
        cmd = self._build_cmd()
        if not cmd:
            return

        self.state = {tag: "idle" for tag, _ in STAGES}
        self.extras = set()
        self.sub_frac = 0.0
        self.sub_label = ""
        self._t0 = time.time()
        self._busy = True
        self._cancelled = False
        self._refresh_stages()
        self._refresh_progress()

        self.query_one("#run", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        self.query_one("#open", Button).disabled = True
        self.query_one("#openhtml", Button).disabled = True

        self._log(f"[dim]$ {' '.join(cmd)}[/]")
        self._run_job(cmd)

    @work(thread=True, exclusive=True)
    def _run_job(self, cmd: list[str]) -> None:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(BASE), env=env,
                start_new_session=True,
            )
        except Exception as exc:
            self.call_from_thread(self._finish, 1, f"无法启动：{exc}")
            return

        self._proc = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            self.call_from_thread(self._on_line, line.rstrip("\n"))
        code = proc.wait()
        self._proc = None
        self.call_from_thread(self._finish, code, "")

    def _on_line(self, line: str) -> None:
        if not line.strip():
            return
        self.query_one("#log", RichLog).write(escape(line))

        m = RE_OK.search(line)
        if m:
            self.sub_label = ""
            self._refresh_progress()
            return

        m = RE_FRAME.search(line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            self.state["S5"] = "running"
            self.sub_frac = cur / total if total else 0.0
            self.sub_label = f"帧 {cur}/{total}"
            self._refresh_stages()
            self._refresh_progress()
            return

        m = RE_MAP.search(line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            self.state["S7"] = "running"
            self.sub_frac = cur / total if total else 0.0
            self.sub_label = f"归纳 {cur}/{total} 块"
            self._refresh_stages()
            self._refresh_progress()
            return

        m = RE_DONE.match(line)
        if m:
            self.state[m.group(1)] = "done"
            self.sub_frac = 0.0
            self.sub_label = ""
            self._refresh_stages()
            self._refresh_progress()
            return

        m = RE_TITLE.match(line)
        if m:
            self.query_one(Header).sub_title = m.group(1)[:28]
            return

        m = RE_START.match(line)
        if m:
            tag = m.group(1)
            if tag in EXTRA_STAGES:
                self.extras.add(tag)
            self.state[tag] = "running"
            self.sub_frac = 0.0
            self.sub_label = ""
            self._refresh_stages()
            self._refresh_progress()

    def _finish(self, code: int, err: str) -> None:
        self._busy = False
        for tag, _ in STAGES:
            st = self.state[tag]
            if st == "running":
                self.state[tag] = "done" if code == 0 else "skip"
            elif st == "idle":
                # 从未开始 = 本次路径不需要（纯音频无抽帧、说话人关闭等）
                self.state[tag] = "skip"
        self._refresh_stages()
        self._refresh_progress()

        self.query_one("#run", Button).disabled = False
        self.query_one("#stop", Button).disabled = True
        if code == 0:
            self.query_one("#bar", ProgressBar).update(progress=100)
            self.query_one("#open", Button).disabled = False
            self.query_one("#openhtml", Button).disabled = False
            self._log(f"[green]完成，{fmt_dur(time.time() - self._t0)}[/]")
            self.bell()
        elif self._cancelled:
            self._log("[yellow]已取消。中间结果留在工作目录，重跑同一文件会复用缓存[/]")
        else:
            self._log(f"[red]失败（退出码 {code}）{(' ' + err) if err else ''}[/]")

    # ---------------------------------------------------------------- 动作
    def action_stop(self) -> None:
        p = self._proc
        if not p or p.poll() is not None:
            self._log("[dim]没有正在运行的任务[/]")
            return
        self._cancelled = True
        self._log("[yellow]已请求中断…[/]")
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    def action_open_out(self) -> None:
        if self.out_dir.is_dir():
            subprocess.run(["open", str(self.out_dir)], check=False)
        else:
            self._log(f"[yellow]产物目录还不存在：{self.out_dir}[/]")

    def action_open_html(self) -> None:
        for name in ("02_详细总结.html",):
            p = self.out_dir / name
            if p.is_file():
                subprocess.run(["open", str(p)], check=False)
                return
        self._log("[yellow]没有找到详细 HTML[/]")

    def action_quit(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._log("[yellow]任务仍在运行，按 Ctrl+K 先取消[/]")
            return
        self.exit()


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    preset = args[0] if args and not args[0].startswith("-") else None
    MediasumApp(preset=preset).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
