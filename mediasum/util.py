"""通用工具：日志、时间格式化、子进程封装、阶段计时。"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_QUIET = False
_TIMINGS: list[tuple[str, float]] = []


def set_quiet(flag: bool) -> None:
    global _QUIET
    _QUIET = flag


def log(msg: str, tag: str = "") -> None:
    if _QUIET:
        return
    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}{msg}", flush=True)


def warn(msg: str) -> None:
    print(f"警告：{msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"错误：{msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


class stage:
    """阶段计时：进入时打印标题，退出时打印耗时并计入汇总表。"""

    def __init__(self, tag: str, title: str) -> None:
        self.tag = tag
        self.title = title
        self.t0 = 0.0

    def __enter__(self) -> "stage":
        self.t0 = time.time()
        # ▶ 是给解析层的锚点：TUI 和 Web 都靠它区分「阶段开始」和普通日志。
        # 没有这个标记时，`[S3] 2 个原始分段` 这类普通日志会被误判成阶段开始，
        # 导致子进度被反复清零。改这个字符必须同步 progress.py 与 tui.py 的正则。
        log(f"▶ {self.title}", self.tag)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        dt = time.time() - self.t0
        if exc_type is None:
            log(f"完成，耗时 {fmt_dur(dt)}", self.tag)
        _TIMINGS.append((f"{self.tag} {self.title}", dt))
        return False


def timing_report() -> str:
    if not _TIMINGS:
        return ""
    total = sum(d for _, d in _TIMINGS)
    rows = [f"  {name:<34s} {fmt_dur(d):>10s}" for name, d in _TIMINGS]
    rows.append(f"  {'合计':<34s} {fmt_dur(total):>10s}")
    return "阶段耗时：\n" + "\n".join(rows)


def fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m"


def ts(sec: float, ms: bool = False) -> str:
    """秒 → HH:MM:SS 或 HH:MM:SS,mmm（SRT 用）。"""
    if sec < 0:
        sec = 0
    total = int(sec)
    h, m, s = total // 3600, total % 3600 // 60, total % 60
    if ms:
        milli = int(round((sec - total) * 1000))
        if milli > 999:
            milli = 999
        return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_ts(text: str) -> float:
    """HH:MM:SS / MM:SS / 纯秒数 → 秒。无法解析返回 0。"""
    try:
        parts = [float(x) for x in str(text).strip().split(":")]
    except (ValueError, AttributeError):
        return 0.0
    while len(parts) < 3:
        parts.insert(0, 0.0)
    if len(parts) > 3:
        return 0.0
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def run(cmd: list[str], check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    """执行子进程，失败时把 stderr 尾部带进异常信息。"""
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(
            f"命令失败（{p.returncode}）：{' '.join(cmd[:3])} …\n" + "\n".join(tail)
        )
    return p


def human_size(num: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024:
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}TB"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
