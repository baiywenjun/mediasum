#!/usr/bin/env python3
"""原型验证脚本：视频 → 逐字稿 + 关键帧 OCR + 详细 HTML + 摘要。

只为验证产物形态，不做说话人分离（单人讲解场景不需要）。

已被 mediasum/ 包完全取代，仅作历史留存。正式用法见 README，命令为 ./ms。
本文件不再维护，字段和产物格式都与正式管线不一致。
"""

import base64
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib import request as urlreq

ROOT = Path(__file__).resolve().parent
MLX = ROOT / ".venv" / "bin" / "mlx_whisper"
WORK = Path.home() / "Videos" / "notes" / "prototype"
MODEL = "mlx-community/whisper-large-v3-turbo"
OLLAMA = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:latest"

INITIAL_PROMPT = (
    "这是一段计算机网络课程讲解，涉及 IP 地址、子网掩码、MAC 地址、"
    "集线器、交换机、路由器、局域网、以太网、数据链路层、网络层、"
    "TCP/IP、OSI 模型等术语。以下是普通话的讲解，请使用简体中文，"
    "并使用规范的标点符号断句。"
)

PUNCT_MODEL = ROOT / "models" / "punct" / "model.int8.onnx"
CJK_PUNCT = "，。？！；：、"
_PUNC = None

MAX_FRAMES = 24
SCENE_THRESHOLD = 0.18

# 直播录屏类需要裁掉聊天区/工具栏，格式 (左, 上, 右, 下) 比例；None 为不裁剪
CROP = None


def log(msg: str) -> None:
    print(f"[{msg}]", flush=True)


def ffmpeg_path() -> str:
    for cand in (
        Path.home() / "Documents/software/ffmpeg/ffmpeg",
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
    ):
        if cand.exists():
            return str(cand)
    subprocess.run(["which", "ffmpeg"], check=True)
    return "ffmpeg"


FFMPEG = ffmpeg_path()


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def probe(src: Path) -> dict:
    p = subprocess.run(
        [FFMPEG, "-i", str(src)], capture_output=True, text=True
    )
    out = p.stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    secs = 0.0
    if dur:
        h, m, s = dur.groups()
        secs = int(h) * 3600 + int(m) * 60 + float(s)
    has_video = "Stream #" in out and "Video:" in out
    return {"duration": secs, "has_video": has_video}


def extract_audio(src: Path, wav: Path) -> None:
    log("S2 抽音轨")
    run([FFMPEG, "-nostdin", "-y", "-i", str(src), "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])


def transcribe(wav: Path, outdir: Path) -> dict:
    log("S3 转写（首次会下载约 3 GB 模型）")
    run([str(MLX), str(wav),
         "--model", MODEL,
         "--language", "zh",
         "--output-dir", str(outdir),
         "--output-format", "json",
         "--initial-prompt", INITIAL_PROMPT,
         "--condition-on-previous-text", "False",
         "--verbose", "False"])
    data = json.loads((outdir / f"{wav.stem}.json").read_text(encoding="utf-8"))
    log(f"S3 完成，{len(data.get('segments', []))} 段")
    return data


def extract_frames(src: Path, fdir: Path) -> list:
    log("S5 场景检测抽帧")
    fdir.mkdir(parents=True, exist_ok=True)
    for old in fdir.glob("*.jpg"):
        old.unlink()
    vf = f"select='gt(scene,{SCENE_THRESHOLD})',showinfo"
    if CROP:
        l, t, r, b = CROP
        vf += f",crop=iw*({r}-{l}):ih*({b}-{t}):iw*{l}:ih*{t}"
    vf += ",scale=1280:-2"
    p = subprocess.run(
        [FFMPEG, "-nostdin", "-y", "-i", str(src),
         "-vf", vf, "-vsync", "vfr", "-q:v", "3", str(fdir / "f_%04d.jpg")],
        capture_output=True, text=True,
    )
    times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", p.stderr)]
    files = sorted(fdir.glob("f_*.jpg"))
    n = min(len(times), len(files))
    log(f"S5 检出 {n} 帧")
    return [{"t": times[i], "path": files[i]} for i in range(n)]


def dhash(path: Path, size: int = 8) -> int:
    from PIL import Image
    resample = getattr(Image, "Resampling", Image).LANCZOS
    img = Image.open(path).convert("L").resize((size + 1, size), resample)
    px = list(img.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
    return bits


def dedupe(frames: list, threshold: int = 10) -> list:
    kept, hashes = [], []
    for f in frames:
        try:
            h = dhash(f["path"])
        except Exception:
            kept.append(f)
            continue
        if all(bin(h ^ o).count("1") > threshold for o in hashes):
            hashes.append(h)
            kept.append(f)
    if len(kept) > MAX_FRAMES:
        step = len(kept) / MAX_FRAMES
        kept = [kept[int(i * step)] for i in range(MAX_FRAMES)]
    log(f"S5 去重后保留 {len(kept)} 帧")
    return kept


def ocr_frame(path: Path) -> str:
    from ocrmac import ocrmac
    try:
        ann = ocrmac.OCR(
            str(path),
            recognition_level="accurate",
            language_preference=["zh-Hans", "en-US"],
        ).recognize()
    except Exception as exc:
        return f"(OCR 失败: {exc})"
    lines = []
    seen = set()
    for t, conf, _ in ann:
        s = (t or "").strip()
        if not s or conf < 0.35 or len(s) < 2 or s in seen:
            continue
        seen.add(s)
        lines.append(s)
    return "\n".join(lines)


def ocr_all(frames: list) -> None:
    log("S5 OCR")
    for i, f in enumerate(frames, 1):
        f["ocr"] = ocr_frame(f["path"])
        print(f"  frame {i}/{len(frames)} {int(f['t'])}s → {len(f['ocr'])} 字", flush=True)


def ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def load_punct():
    """加载中文标点恢复模型（CT-Transformer，INT8 ONNX）。

    注意：这个模型对已有标点的文本会重复标注（产生「。。」「？?」），
    所以调用方必须先判断文本是否已有中文标点，只在纯无标点文本上使用。
    """
    global _PUNC
    if _PUNC is None:
        if not PUNCT_MODEL.exists():
            log(f"警告：标点模型不存在 {PUNCT_MODEL}，退化为规则补标点")
            _PUNC = False
        else:
            import sherpa_onnx
            cfg = sherpa_onnx.OfflinePunctuationConfig(
                model=sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=str(PUNCT_MODEL), num_threads=2,
                )
            )
            _PUNC = sherpa_onnx.OfflinePunctuation(cfg)
            log("S3 标点模型已加载")
    return _PUNC


def to_fullwidth(text: str) -> str:
    text = re.sub(r"(?<!\d),(?!\d)", "，", text)
    return (text.replace("?", "？").replace("!", "！")
            .replace(";", "；").replace("(", "（").replace(")", "）"))


def restore_punct(text: str, punc) -> str:
    body = (text or "").strip()
    if not body:
        return body

    plain = re.sub(r"[\s,，。？！；：、!?;:.,]", "", body)
    if len(plain) < 4:
        return body

    if any(p in body for p in CJK_PUNCT):
        # 已有中文标点：只统一全角，不再送模型（否则会重复标注）
        fixed = to_fullwidth(body)
        if fixed and fixed[-1] not in "。？！":
            fixed += "。"
        return fixed

    if punc:
        out = punc.add_punctuation(body).strip()
        if out:
            return to_fullwidth(out)
    return body + "。"


def punctuate_segments(data: dict) -> dict:
    punc = load_punct()
    changed = 0
    for s in data.get("segments", []):
        raw = s.get("text", "")
        out = restore_punct(raw, punc)
        if out != raw.strip():
            changed += 1
        s["text"] = out
    log(f"S3 标点处理 {changed}/{len(data.get('segments', []))} 段")
    return data


def parse_ts(text: str) -> float:
    try:
        parts = [float(x) for x in str(text).split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        return 0.0


def merge_segments(data: dict, max_gap: float = 1.5, min_len: float = 10.0) -> dict:
    """Whisper 按停顿切段，中文口语会产生大量 1–3 秒的碎片，读起来像断句残片。
    这里按「间隔小且当前段还不够长」合并，目标每段 10 秒以上。"""
    out, cur = [], None
    for s in data.get("segments", []):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        if cur is None:
            cur = {"start": s["start"], "end": s["end"], "text": text}
        elif s["start"] - cur["end"] <= max_gap and (cur["end"] - cur["start"]) < min_len:
            cur["text"] += text
            cur["end"] = s["end"]
        else:
            out.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": text}
    if cur:
        out.append(cur)
    merged = dict(data)
    merged["segments"] = out
    log(f"S3 段落合并 {len(data.get('segments', []))} → {len(out)} 段")
    return merged


def ollama(prompt: str, timeout: int = 900) -> str:
    body = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": 24576},
    }).encode()
    req = urlreq.Request(
        f"{OLLAMA}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def build_transcript(data: dict) -> str:
    return "\n".join(
        f"[{ts(s['start'])}] {s['text'].strip()}"
        for s in data.get("segments", [])
    )


def summarize(transcript: str, frames: list, duration: float, transcript_segments: list) -> dict:
    log("S7 归纳")
    excerpt = transcript[:14000]
    shot_list = "\n".join(
        f"- {ts(f['t'])}: {f['ocr'][:200].replace(chr(10), ' / ')}"
        for f in frames
    )
    prompt = f"""你是课程笔记助理。基于以下逐字稿和幻灯片文字，输出 JSON（不要解释、不要 markdown 代码块）：

{{
  "title": "课程标题",
  "abstract": "摘要正文",
  "chapters": [
    {{"time": "HH:MM:SS", "title": "章节标题", "points": ["要点1", "要点2"]}}
  ],
  "terms": [{{"term": "术语", "desc": "一句话解释"}}]
}}

硬约束：
1. 视频总时长是 {ts(duration)}。所有 chapters 的 time 必须小于该值，不得编造未出现的内容。
2. abstract 必须是 450–580 字之间的连续段落，写清讲了什么、分几部分、核心结论。不要罗列要点。
3. chapters 分 4–8 章，每章 3–6 条 points，每条不超过 40 字。
4. terms 取 5–10 个，只收逐字稿或幻灯片里真实出现的。

逐字稿：
{excerpt}

幻灯片文字：
{shot_list}
"""
    raw = ollama(prompt)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise RuntimeError(f"模型未返回 JSON：{raw[:300]}")
    s = json.loads(m.group(0))

    chapters = s.get("chapters", [])
    kept = [c for c in chapters if 0 <= parse_ts(c.get("time", "0:00:00")) <= duration]
    if len(kept) != len(chapters):
        log(f"S7 丢弃 {len(chapters) - len(kept)} 个超出时长的章节")
    # 模型倾向输出 00:02:00 这类整数时间，吸附到真实段落起点
    starts = [x["start"] for x in transcript_segments]
    for c in kept:
        if starts:
            c["time"] = ts(min(starts, key=lambda x: abs(x - parse_ts(c["time"]))))
    s["chapters"] = kept

    if len(s.get("abstract", "")) < 250:
        log("S7 摘要偏短，二次扩写")
        s["abstract"] = ollama(
            "把下面的课程摘要扩写成 450–580 字的连续段落，保留原意、补充具体内容，"
            "只输出摘要正文：\n" + s.get("abstract", "")
        ).strip()
    return s


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def render(info: dict, data: dict, frames: list, summary: dict, out: Path) -> None:
    log("S8 渲染 HTML")
    cards = []
    for f in frames:
        near = [s for s in data["segments"] if abs(s["start"] - f["t"]) < 60]
        speech = " ".join(s["text"].strip() for s in near)[:400]
        b64 = img_b64(f["path"])
        cards.append(f"""
    <section class="shot">
      <div class="shot-head"><span class="t">{ts(f['t'])}</span></div>
      <div class="shot-body">
        <img src="data:image/jpeg;base64,{b64}" alt="关键画面 {ts(f['t'])}">
        <div class="shot-text">
          <p class="ocr">{f['ocr'][:600] or '（本帧无可识别文字）'}</p>
          <p class="speech">讲解：{speech or '（无对应语音）'}</p>
        </div>
      </div>
    </section>""")

    chap = "\n".join(
        f"<li><b>{c.get('time', '')}</b> {c.get('title', '')}<ul>"
        + "".join(f"<li>{p}</li>" for p in c.get("points", []))
        + "</ul></li>"
        for c in summary.get("chapters", [])
    )
    terms = "\n".join(
        f"<li><b>{t.get('term','')}</b> — {t.get('desc','')}</li>"
        for t in summary.get("terms", [])
    )
    transcript_html = build_transcript(data).replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{summary.get('title', '课程笔记')}</title>
<style>
:root{{--ink:#1a1a1a;--sub:#666;--line:#e5e5e5;--bg:#fafaf8;--accent:#185fa5}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.75 -apple-system,"PingFang SC","Hiragino Sans GB",sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:48px 28px 80px}}
h1{{font-size:30px;line-height:1.3;margin:0 0 12px;letter-spacing:-.5px}}
.meta{{color:var(--sub);font-size:14px;padding-bottom:24px;border-bottom:1px solid var(--line)}}
h2{{font-size:19px;margin:44px 0 16px;padding-left:11px;border-left:3px solid var(--accent)}}
.abstract{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:22px 24px;
font-size:15.5px;line-height:1.9}}
ul{{padding-left:20px}} li{{margin:6px 0}}
.shot{{margin:26px 0;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}}
.shot-head{{padding:9px 16px;border-bottom:1px solid var(--line);background:#f6f6f3}}
.t{{font-variant-numeric:tabular-nums;font-size:13px;color:var(--accent);font-weight:500}}
.shot-body{{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr)}}
.shot-body img{{width:100%;display:block;background:#000}}
.shot-text{{padding:16px 18px;font-size:14px;line-height:1.7}}
.ocr{{white-space:pre-wrap;margin:0 0 12px;padding:10px 12px;background:#f6f6f3;
border-radius:7px;font-size:13px;color:#333}}
.speech{{margin:0;color:var(--sub)}}
details{{margin-top:44px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 22px}}
summary{{cursor:pointer;font-weight:500}}
.transcript{{font-size:14px;color:#333;line-height:2;max-height:560px;overflow:auto;margin-top:14px}}
@media(max-width:720px){{.shot-body{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>{summary.get('title', '课程笔记')}</h1>
<div class="meta">{info['name']} · 时长 {ts(info['duration'])} · 转写 {len(data['segments'])} 段 ·
关键帧 {len(frames)} 张 · 模型 {MODEL.split('/')[-1]}</div>

<h2>摘要</h2>
<div class="abstract">{summary.get('abstract', '')}</div>

<h2>章节详解</h2>
<ul>{chap}</ul>

<h2>术语</h2>
<ul>{terms}</ul>

<h2>关键画面与讲解对照</h2>
{''.join(cards)}

<details><summary>展开完整逐字稿</summary>
<div class="transcript">{transcript_html}</div></details>
</div></body></html>"""

    out.write_text(html, encoding="utf-8")
    log(f"S8 完成 {out} ({out.stat().st_size / 1048576:.1f} MB)")


def main() -> None:
    global CROP
    argv = sys.argv[1:]
    if not argv:
        sys.exit("用法: prototype.py <文件> [--reuse] [--crop=l,t,r,b]")

    src = Path(argv[0]).expanduser().resolve()
    if not src.exists():
        sys.exit(f"文件不存在: {src}")
    for a in argv:
        if a.startswith("--crop="):
            CROP = tuple(float(x) for x in a.split("=", 1)[1].split(","))
            log(f"裁剪区域 {CROP}")
    reuse = "--reuse" in argv

    name = src.stem
    work = WORK / name
    work.mkdir(parents=True, exist_ok=True)

    info = probe(src)
    info["name"] = name
    log(f"S1 {name} 时长 {ts(info['duration'])} 含视频={info['has_video']}")

    js = work / f"{name}.json"
    if reuse and js.exists():
        data = json.loads(js.read_text(encoding="utf-8"))
        log("S3 复用已有转写")
    else:
        wav = work / f"{name}.wav"
        extract_audio(src, wav)
        data = transcribe(wav, work)

    data = merge_segments(data)
    data = punctuate_segments(data)
    transcript = build_transcript(data)
    (work / "01_逐字稿.txt").write_text(transcript, encoding="utf-8")
    log(f"逐字稿 {len(transcript)} 字 / {len(data['segments'])} 段")

    frames = []
    if info["has_video"]:
        frames = dedupe(extract_frames(src, work / "frames"))
        ocr_all(frames)
        before = len(frames)
        frames = [f for f in frames if len(f["ocr"]) >= 10]
        if len(frames) != before:
            log(f"S5 丢弃 {before - len(frames)} 张无文字帧（过渡画面）")

    summary = summarize(transcript, frames, info["duration"], data["segments"])
    (work / "03_摘要.md").write_text(
        f"# {summary.get('title','')}\n\n{summary.get('abstract','')}\n",
        encoding="utf-8",
    )
    render(info, data, frames, summary, work / "02_详细总结.html")
    log(f"全部完成 → {work}")


if __name__ == "__main__":
    main()
