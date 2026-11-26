"""产物渲染：逐字稿（txt/md/srt）、摘要（md）、详细版（内嵌图片的 HTML）。"""

from __future__ import annotations

import base64
import html as htmllib
from pathlib import Path

from .util import human_size, log, ts


def _esc(text: str) -> str:
    return htmllib.escape(text or "", quote=False)


# ---------------------------------------------------------------- 逐字稿

def render_transcript(paras: list[dict], info: dict, cfg: dict) -> str:
    show = cfg["output"]["show_speaker"]
    if cfg["output"]["transcript_format"] == "md":
        out = [
            f"# {info['name']} 逐字稿", "",
            f"- 时长：{ts(info['duration'])}",
            f"- 段落：{len(paras)}",
            "",
            "---", "",
        ]
        for p in paras:
            who = f"**{p['speaker']}**：" if show and p.get("speaker") else ""
            out.append(f"`{ts(p['start'])}` {who}{p['text']}")
            out.append("")
        return "\n".join(out)

    out = []
    for p in paras:
        who = f"{p['speaker']}：" if show and p.get("speaker") else ""
        out.append(f"[{ts(p['start'])}] {who}{p['text']}")
    return "\n".join(out) + "\n"


def render_srt(paras: list[dict], cfg: dict) -> str:
    show = cfg["output"]["show_speaker"]
    blocks = []
    for i, p in enumerate(paras, 1):
        who = f"{p['speaker']}：" if show and p.get("speaker") else ""
        blocks.append(
            f"{i}\n{ts(p['start'], ms=True)} --> {ts(p['end'], ms=True)}\n{who}{p['text']}\n"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------- 摘要

def render_abstract(summary: dict, info: dict, cfg: dict) -> str:
    lines = [f"# {summary.get('title', '未命名')}", ""]
    lines.append(summary.get("abstract", "").strip())
    lines.append("")
    meta = (f"> 来源：{info['name']} · 时长 {ts(info['duration'])} · "
            f"{info['generated']}")
    lines.append(meta)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 详细 HTML

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#fafaf8;color:#1a1a1a;
font:16px/1.75 -apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:44px 26px 72px}
h1{font-size:29px;line-height:1.32;margin:0 0 10px;letter-spacing:-.4px}
.meta{color:#666;font-size:13.5px;padding-bottom:22px;border-bottom:1px solid #e5e5e5;
display:flex;flex-wrap:wrap;gap:6px 16px}
h2{font-size:18px;margin:40px 0 14px;padding-left:11px;border-left:3px solid #185fa5}
.abstract{background:#fff;border:1px solid #e5e5e5;border-radius:12px;padding:20px 22px;
font-size:15.5px;line-height:1.92}
ul{padding-left:20px}li{margin:5px 0}
.chapter{margin:14px 0;padding:14px 16px;background:#fff;border:1px solid #e5e5e5;border-radius:10px}
.chapter .h{display:flex;gap:10px;align-items:baseline;margin-bottom:6px}
.chapter .t{font-variant-numeric:tabular-nums;font-size:12.5px;color:#185fa5;
background:#eef4fa;border-radius:5px;padding:2px 7px;white-space:nowrap}
.chapter .n{font-weight:600;font-size:15.5px}
.chapter ul{margin:4px 0 0}
.terms{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px;
list-style:none;padding:0}
.terms li{background:#fff;border:1px solid #e5e5e5;border-radius:9px;padding:10px 13px;font-size:14px}
.terms b{color:#185fa5}
.shot{margin:20px 0;background:#fff;border:1px solid #e5e5e5;border-radius:11px;overflow:hidden}
.shot-head{padding:8px 14px;border-bottom:1px solid #e5e5e5;background:#f6f6f3;
display:flex;gap:10px;align-items:center}
.shot-head .t{font-variant-numeric:tabular-nums;font-size:12.5px;color:#185fa5;font-weight:500}
.shot-head .sp{font-size:12px;color:#888}
.shot-body{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr)}
.shot-body img{width:100%;display:block;background:#000}
.shot-text{padding:14px 16px;font-size:13.5px;line-height:1.68}
.ocr{white-space:pre-wrap;margin:0 0 10px;padding:9px 11px;background:#f6f6f3;border-radius:7px;
font-size:12.5px;color:#333;max-height:280px;overflow:auto}
.speech{margin:0;color:#666}
blockquote{margin:8px 0;padding:10px 14px;border-left:3px solid #ddd;color:#333;
background:#fff;border-radius:0 8px 8px 0;font-size:14.5px}
details{margin-top:36px;background:#fff;border:1px solid #e5e5e5;border-radius:12px;padding:14px 20px}
summary{cursor:pointer;font-weight:500}
.transcript{font-size:13.5px;color:#333;line-height:1.95;max-height:600px;overflow:auto;margin-top:12px}
.transcript .row{display:flex;gap:10px;padding:3px 0;border-bottom:1px dashed #f0f0ee}
.transcript .ts{color:#185fa5;font-variant-numeric:tabular-nums;white-space:nowrap;font-size:12.5px}
.transcript .who{color:#7a4fb5;font-size:12.5px;white-space:nowrap}
footer{margin-top:40px;padding-top:16px;border-top:1px solid #e5e5e5;color:#999;font-size:12.5px}
@media(max-width:720px){.shot-body{grid-template-columns:1fr}.wrap{padding:28px 16px 56px}}
"""


def render_html(summary: dict, paras: list[dict], frames: list[dict],
                info: dict, cfg: dict) -> str:
    o = cfg["output"]
    detail = cfg["summary"]["detail"]
    parts: list[str] = []

    # 摘要
    parts.append('<h2>摘要</h2><div class="abstract">%s</div>'
                 % _esc(summary.get("abstract", "")).replace("\n", "<br>"))

    # 章节
    chapters = summary.get("chapters") or []
    if chapters and detail != "brief":
        items = []
        for c in chapters:
            pts = "".join(f"<li>{_esc(p)}</li>" for p in c.get("points") or [])
            items.append(
                '<div class="chapter"><div class="h">'
                f'<span class="t">{_esc(c.get("time", ""))}</span>'
                f'<span class="n">{_esc(c.get("title", ""))}</span></div>'
                f"<ul>{pts}</ul></div>"
            )
        parts.append("<h2>内容脉络</h2>" + "".join(items))

    # 关键原话与未决问题（detailed 才有）
    if summary.get("quotes"):
        qs = "".join(f"<blockquote>{_esc(q)}</blockquote>" for q in summary["quotes"])
        parts.append("<h2>关键原话</h2>" + qs)
    if summary.get("questions"):
        qs = "".join(f"<li>{_esc(q)}</li>" for q in summary["questions"])
        parts.append(f"<h2>待确认</h2><ul>{qs}</ul>")

    # 术语
    if summary.get("terms") and detail != "brief":
        ts_ = []
        for t in summary["terms"]:
            if not isinstance(t, dict):
                continue
            ts_.append(f'<li><b>{_esc(t.get("term", ""))}</b> — {_esc(t.get("desc", ""))}</li>')
        if ts_:
            parts.append('<h2>术语</h2><ul class="terms">' + "".join(ts_) + "</ul>")

    # 画面与讲解对照
    if frames and o["embed_frames"]:
        parts.append('<h2>关键画面与讲解对照</h2>' + "".join(
            _shot(f, paras) for f in frames
        ))

    # 逐字稿
    if o["transcript"]:
        rows = []
        for p in paras:
            who = (f'<span class="who">{_esc(p["speaker"])}</span>'
                   if o["show_speaker"] and p.get("speaker") else "")
            rows.append(f'<div class="row"><span class="ts">{ts(p["start"])}</span>'
                        f'{who}<span>{_esc(p["text"])}</span></div>')
        parts.append(
            f'<details><summary>展开完整逐字稿（{len(paras)} 段）</summary>'
            f'<div class="transcript">{"".join(rows)}</div></details>'
        )

    meta = [
        f"<span>{_esc(info['name'])}</span>",
        f"<span>时长 {ts(info['duration'])}</span>",
        f"<span>{len(paras)} 段逐字稿</span>",
        f"<span>模型 {_esc(info['llm_label'])}</span>",
    ]
    if frames:
        meta.append(f"<span>关键帧 {len(frames)} 张</span>")
    if info.get("speakers"):
        meta.append(f"<span>说话人 {info['speakers']} 位</span>")
    meta.append(f"<span>{_esc(info['generated'])}</span>")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(summary.get("title", "总结"))}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{_esc(summary.get("title", "总结"))}</h1>
<div class="meta">{"".join(meta)}</div>
{"".join(parts)}
<footer>由 mediasum 生成 · 转写 {_esc(info.get("asr_model", ""))} · 归纳 {_esc(info["llm_label"])}</footer>
</div></body></html>
"""


def _shot(frame: dict, paras: list[dict]) -> str:
    """一帧画面 + 该时刻附近的讲解原文，用于「看图对照听」。"""
    near = [p for p in paras if abs(p["start"] - frame["t"]) < 60]
    speech = " ".join(p["text"] for p in near)[:420]
    who = ""
    if near and near[0].get("speaker"):
        who = f'<span class="sp">{_esc(near[0]["speaker"])}</span>'
    b64 = base64.b64encode(Path(frame["path"]).read_bytes()).decode()
    ocr = _esc(frame.get("ocr", "")) or "（本帧无可识别文字）"
    return (
        '<div class="shot"><div class="shot-head">'
        f'<span class="t">{ts(frame["t"])}</span>{who}</div>'
        '<div class="shot-body">'
        f'<img src="data:image/jpeg;base64,{b64}" alt="{ts(frame["t"])} 画面">'
        f'<div class="shot-text"><div class="ocr">{ocr}</div>'
        f'<p class="speech">讲解：{_esc(speech) or "（无对应语音）"}</p></div>'
        "</div></div>"
    )


def write_outputs(workdir: Path, summary: dict, paras: list[dict], frames: list[dict],
                  info: dict, cfg: dict) -> list[Path]:
    o = cfg["output"]
    written: list[Path] = []

    if o["transcript"]:
        ext = o["transcript_format"]
        path = workdir / f"01_逐字稿.{ext}"
        path.write_text(render_transcript(paras, info, cfg), encoding="utf-8")
        written.append(path)
        if o["srt"]:
            srt = workdir / "01_逐字稿.srt"
            srt.write_text(render_srt(paras, cfg), encoding="utf-8")
            written.append(srt)

    if o["detail_html"]:
        path = workdir / "02_详细总结.html"
        path.write_text(render_html(summary, paras, frames, info, cfg), encoding="utf-8")
        log(f"HTML {human_size(path.stat().st_size)}", "S8")
        written.append(path)

    if o["abstract"]:
        path = workdir / "03_摘要.md"
        path.write_text(render_abstract(summary, info, cfg), encoding="utf-8")
        written.append(path)

    return written
