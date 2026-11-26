"""媒体层：ffmpeg 探测、音轨抽取、关键帧抽取。

全部走 ffmpeg 单一入口，不依赖 ffprobe —— 本机 ffmpeg 是手工安装的
6.0 版本，旁边没有 ffprobe，用 `ffmpeg -i` 解析足以拿到时长和分辨率。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .util import ensure_dir, log, run, warn

_CANDIDATES = (
    Path.home() / "Documents" / "software" / "ffmpeg" / "ffmpeg",
    Path("/opt/homebrew/bin/ffmpeg"),
    Path("/usr/local/bin/ffmpeg"),
)


def ffmpeg_bin() -> str:
    for cand in _CANDIDATES:
        if cand.exists():
            return str(cand)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("未找到 ffmpeg，请先安装或修改 media.py 中的候选路径")


FFMPEG = None


def ffmpeg() -> str:
    global FFMPEG
    if FFMPEG is None:
        FFMPEG = ffmpeg_bin()
    return FFMPEG


def probe(src: Path) -> dict:
    """解析 `ffmpeg -i` 的 stderr，拿到时长、是否含视频/音频、分辨率。"""
    p = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(src)],
                       capture_output=True, text=True)
    out = p.stderr or ""
    if "Duration:" not in out:
        raise RuntimeError(f"无法读取媒体信息（文件损坏或格式不支持）：{src}")

    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    duration = 0.0
    if m:
        h, mi, se = m.groups()
        duration = int(h) * 3600 + int(mi) * 60 + float(se)

    has_video = bool(re.search(r"Stream #\d+:\d+.*: Video:", out))
    has_audio = bool(re.search(r"Stream #\d+:\d+.*: Audio:", out))

    width = height = 0
    vm = re.search(r": Video:.*?, (\d{2,5})x(\d{2,5})", out)
    if vm:
        width, height = int(vm.group(1)), int(vm.group(2))

    return {
        "duration": duration,
        "has_video": has_video,
        "has_audio": has_audio,
        "width": width,
        "height": height,
    }


def extract_audio(src: Path, wav: Path) -> Path:
    """抽 16 kHz 单声道 PCM。显式解码一次，后续所有环节复用同一份音频。"""
    ensure_dir(wav.parent)
    run([ffmpeg(), "-nostdin", "-y", "-i", str(src), "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
    if not wav.exists() or wav.stat().st_size == 0:
        raise RuntimeError("音轨抽取失败，文件可能没有音频流")
    return wav


def _crop_filter(crop: list[float]) -> str:
    """crop 为 [左, 上, 右, 下] 的比例，裁成 PPT 正文区域。"""
    if not crop or len(crop) != 4:
        return ""
    l, t, r, b = crop
    if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
        raise ValueError(f"crop 必须是 [左,上,右,下] 的 0–1 比例且左<右、上<下，收到 {crop}")
    return f"crop=iw*{r - l:.4f}:ih*{b - t:.4f}:iw*{l:.4f}:ih*{t:.4f},"


def _scene_pass(src: Path, outdir: Path, cfg: dict) -> list[dict]:
    """按画面变化抽帧，能抓住 PPT 翻页的瞬间。"""
    chain = [f"select='gt(scene,{cfg['scene_threshold']})'"]
    crop = _crop_filter(cfg["crop"]).rstrip(",")
    if crop:
        chain.append(crop)
    chain.append("showinfo")
    chain.append(f"scale={cfg['width']}:-2")
    vf = ",".join(chain)
    pattern = outdir / "s_%05d.jpg"
    p = subprocess.run(
        [ffmpeg(), "-nostdin", "-y", "-i", str(src), "-vf", vf,
         "-vsync", "vfr", "-q:v", str(cfg["jpeg_quality"]), str(pattern)],
        capture_output=True, text=True,
    )
    times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", p.stderr or "")]
    files = sorted(outdir.glob("s_*.jpg"))
    n = min(len(times), len(files))
    return [{"t": times[i], "path": files[i], "src": "scene"} for i in range(n)]


def _interval_pass(src: Path, outdir: Path, cfg: dict, duration: float) -> list[dict]:
    """定时补帧：纯静态 PPT 长时间不翻页时，场景检测只给一帧，
    靠这个兜底保证每个时间段都有画面可对照。"""
    interval = float(cfg["interval"])
    if interval <= 0 or duration <= interval:
        return []
    vf = _crop_filter(cfg["crop"]) + f"fps=1/{interval:g},showinfo"
    vf += f",scale={cfg['width']}:-2"
    pattern = outdir / "i_%05d.jpg"
    p = subprocess.run(
        [ffmpeg(), "-nostdin", "-y", "-i", str(src), "-vf", vf,
         "-vsync", "vfr", "-q:v", str(cfg["jpeg_quality"]), str(pattern)],
        capture_output=True, text=True,
    )
    times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", p.stderr or "")]
    files = sorted(outdir.glob("i_*.jpg"))
    n = min(len(times), len(files))
    return [{"t": times[i], "path": files[i], "src": "interval"} for i in range(n)]


def extract_frames(src: Path, outdir: Path, cfg: dict, duration: float) -> list[dict]:
    ensure_dir(outdir)
    for old in list(outdir.glob("*.jpg")):
        old.unlink()

    scene = _scene_pass(src, outdir, cfg)
    every = _interval_pass(src, outdir, cfg, duration)
    frames = sorted(scene + every, key=lambda f: f["t"])
    log(f"场景帧 {len(scene)} + 定时帧 {len(every)} = {len(frames)}", "S5")
    return frames
