"""画面文字识别：走 macOS 原生 Vision 框架（ocrmac），不走视觉大模型。

PPT 讲解类视频里画面信息的绝大部分是文字，Vision 单帧约 0.3–0.4 秒，
而 7B 视觉模型单帧要数秒，几十帧就是分钟级的差距。
"""

from __future__ import annotations

import re
from pathlib import Path

from .util import log


def ocr_image(path: Path, languages: list[str], min_conf: float) -> str:
    try:
        from ocrmac import ocrmac
    except ImportError:
        raise RuntimeError("未安装 ocrmac（uv pip install ocrmac）")

    try:
        ann = ocrmac.OCR(
            str(path),
            recognition_level="accurate",
            language_preference=list(languages),
        ).recognize()
    except Exception as exc:
        return f"(OCR 失败: {exc})"

    lines, seen = [], set()
    for text, conf, _box in ann:
        s = (text or "").strip()
        if not s or conf < min_conf or len(s) < 2 or s in seen:
            continue
        seen.add(s)
        lines.append(s)
    return "\n".join(lines)


def _drop_noise(text: str, patterns: list[str]) -> str:
    """滤掉电商/直播界面元素。

    课程前几十秒常是售卖页或直播间，OCR 会读出「¥666 立即购买」「好评度 96%」
    这类内容，它们会作为「幻灯片文字」混进归纳材料，把模型带偏。
    """
    if not patterns:
        return text
    regs = []
    for p in patterns:
        try:
            regs.append(re.compile(p))
        except re.error:
            continue
    kept = [ln for ln in text.splitlines()
            if ln.strip() and not any(r.search(ln) for r in regs)]
    return "\n".join(kept)


def ocr_frames(frames: list[dict], cfg: dict) -> list[dict]:
    f = cfg["frames"]
    patterns = f.get("ocr_blocklist") or []
    for i, fr in enumerate(frames, 1):
        raw = ocr_image(fr["path"], f["ocr_languages"], float(f["ocr_min_confidence"]))
        fr["ocr"] = _drop_noise(raw, patterns)
        log(f"帧 {i}/{len(frames)} {int(fr['t'])}s → {len(fr['ocr'])} 字", "S5")

    before = len(frames)
    kept = [x for x in frames if len(x["ocr"]) >= int(f["min_ocr_chars"])]
    if len(kept) != before:
        log(f"丢弃 {before - len(kept)} 张无文字帧（过渡/动画/纯界面）", "S5")
    return kept
