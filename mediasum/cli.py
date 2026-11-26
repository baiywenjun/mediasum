"""命令行入口与流程编排。

    ms <音视频> [选项]                 完整处理
    ms rerun <音视频|工作目录> --from punct   从某一步重跑（复用缓存）
    ms probe <音视频>                  只看媒体信息
    ms models download-diar            下载说话人分离模型
    ms config                          打印生效配置
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from . import asr, config, diarize, llm as llm_mod, media, ocr, render, summarize
from .util import (die, ensure_dir, fmt_dur, human_size, log, set_quiet,
                   stage, timing_report, ts, warn)

STAGES = ["asr", "punct", "frames", "diarize", "summary", "render"]

DIAR_REPO_HINT = {
    "segmentation": "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    "embedding": "csukuangfj/speaker-embedding-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
}

# 标点恢复模型：阿里达摩院 CT-Transformer 的中英版，INT8 量化，约 72 MB。
# 三个文件必须放在同一目录，缺 tokens.json 会在加载时报错。
PUNCT_REPO = "ranger810/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"
PUNCT_FILES = ["model.int8.onnx", "tokens.json", "config.yaml"]


# ------------------------------------------------------------------ 公共

def _paths(src: Path, cfg: dict, out_override: str | None) -> tuple[Path, Path]:
    name = src.stem
    work = Path(cfg["general"]["work_dir"]) / name
    out = Path(out_override).expanduser() if out_override else Path(cfg["general"]["output_dir"]) / name
    ensure_dir(work)
    ensure_dir(out)
    return work, out


def _read_json(path: Path, what: str):
    if not path.exists():
        die(f"缺少 {what} 缓存（{path}），无法从该步骤继续。去掉 --from 从头跑一次")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _should(start: str | None, step: str) -> bool:
    if start is None:
        return True
    if start not in STAGES:
        die(f"--from 只能是 {'/'.join(STAGES)}")
    return STAGES.index(step) >= STAGES.index(start)


def _llm_label(cfg: dict) -> str:
    p = cfg["llm"]["provider"]
    return f"{p}/{cfg['llm'][p]['model']}"


# ------------------------------------------------------------------ run

def cmd_run(args) -> int:
    cfg = config.load(args.config, args.set)
    if args.detail:
        cfg["summary"]["detail"] = args.detail
    for flag, path in (("no_html", ("output", "detail_html")),
                       ("no_transcript", ("output", "transcript")),
                       ("no_abstract", ("output", "abstract"))):
        if getattr(args, flag, False):
            cfg[path[0]][path[1]] = False
    if args.crop:
        cfg["frames"]["crop"] = [float(x) for x in args.crop.split(",")]
    if args.diar:
        cfg["diarization"]["mode"] = args.diar
    if args.srt:
        cfg["output"]["srt"] = True
    if args.format:
        cfg["output"]["transcript_format"] = args.format
    config.validate(cfg)

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        die(f"文件不存在：{src}")

    work, out = _paths(src, cfg, args.out)
    start = args.start

    print(config.describe(cfg), flush=True)
    if args.dry_run:
        log(f"输入 {src}\n工作目录 {work}\n产物目录 {out}\n起始步骤 {start or 'asr'}（演练模式，未执行）")
        return 0

    t_begin = time.time()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    with stage("S1", "读取媒体信息"):
        info = media.probe(src)
        info["name"] = src.stem
        info["source"] = str(src)
        log(f"{src.name} · 时长 {ts(info['duration'])} · "
            f"{'含视频 ' + str(info.get('width')) + 'x' + str(info.get('height')) if info['has_video'] else '纯音频'}")
        if not info["has_audio"]:
            die("该文件没有音频流，无法转写")
        if cfg["frames"]["enabled"] and not info["has_video"]:
            log("纯音频输入，跳过抽帧")

    wav = work / "audio.wav"
    diar_on = cfg["diarization"]["mode"] != "off"
    # 抽帧不需要 wav，只有转写和说话人分离需要
    need_wav = _should(start, "asr") or (diar_on and _should(start, "diarize"))
    if need_wav and not wav.exists():
        with stage("S2", "抽取音轨（16 kHz 单声道）"):
            media.extract_audio(src, wav)

    # --- 转写
    asr_json = work / "asr.json"
    if _should(start, "asr"):
        with stage("S3", "语音转写"):
            data = asr.transcribe(wav, work, cfg)
            _write_json(asr_json, data)
    else:
        data = _read_json(asr_json, "转写结果")
        log(f"复用转写结果：{len(data.get('segments', []))} 段", "S3")

    # --- 标点与分段
    seg_json = work / "segments.json"
    punct_stats: dict = {}
    if _should(start, "punct"):
        with stage("S4", "标点恢复与分段"):
            punct = asr.Punctuator(cfg)
            paras = asr.build_paragraphs(data.get("segments", []), punct, cfg)
            st = asr.paragraph_stats(paras)
            log(f"{st['count']} 段，共 {st['chars']} 字，"
                f"平均 {st['avg_sec']:.1f}s / {st['avg_chars']:.0f} 字", "S4")
            log(f"标点来源：模型 {punct.stats['model']} · 全角统一 {punct.stats['fullwidth']} "
                f"· 规则兜底 {punct.stats['fallback']}", "S4")
            punct_stats = punct.stats
            _write_json(seg_json, {"paragraphs": paras, "punct_stats": punct_stats})
    else:
        cached = _read_json(seg_json, "分段结果")
        paras = cached["paragraphs"]
        punct_stats = cached.get("punct_stats", {})
        log(f"复用分段结果：{len(paras)} 段", "S4")

    # --- 抽帧与 OCR
    frames_json = work / "frames.json"
    frames: list[dict] = []
    if info["has_video"] and cfg["frames"]["enabled"]:
        if _should(start, "frames"):
            with stage("S5", "关键帧与画面文字"):
                raw_frames = media.extract_frames(src, work / "frames", cfg["frames"], info["duration"])
                raw_frames = _dedupe(raw_frames, cfg)
                frames = ocr.ocr_frames(raw_frames, cfg)
                frames = _collapse_bursts(frames)
                _write_json(frames_json, [
                    {"t": f["t"], "path": str(f["path"]), "src": f["src"], "ocr": f["ocr"]}
                    for f in frames
                ])
        else:
            cached = _read_json(frames_json, "关键帧") if frames_json.exists() else []
            frames = [{"t": c["t"], "path": Path(c["path"]), "src": c.get("src", ""),
                       "ocr": c["ocr"]} for c in cached if Path(c["path"]).exists()]
            log(f"复用关键帧：{len(frames)} 张", "S5")
    elif info["has_video"]:
        log("配置关闭了抽帧，详细总结将只依据语音", "S5")

    # --- 说话人
    mode = cfg["diarization"]["mode"]
    speakers: list[str] = []
    if mode != "off" and _should(start, "diarize"):
        with stage("S6", f"说话人分离（{mode}）"):
            if not diarize.models_ready(cfg):
                warn("分离模型未就绪，跳过。执行 `ms models download-diar` 后可启用")
            elif not wav.exists():
                warn("缺少中间音频，跳过说话人分离（去掉 --from 重跑一次）")
            else:
                turns = diarize.diarize(wav, cfg)
                speakers = diarize.attach(paras, turns, cfg)
                if mode == "auto" and len(speakers) < 2:
                    log("只检出 1 位说话人，按单人处理", "S6")
                    for p in paras:
                        p.pop("speaker", None)
                    speakers = []
                else:
                    log(f"说话人：{', '.join(speakers) or '无'}", "S6")
                _write_json(seg_json, {"paragraphs": paras, "punct_stats": punct_stats})
    elif mode != "off":
        cached = _read_json(seg_json, "分段结果")["paragraphs"]
        for i, p in enumerate(paras):
            p["speaker"] = cached[i].get("speaker", "") if i < len(cached) else ""
        speakers = sorted({p["speaker"] for p in paras if p.get("speaker")})

    # --- 归纳
    summary_json = work / "summary.json"
    if _should(start, "summary"):
        with stage("S7", f"归纳（{_llm_label(cfg)}）"):
            client = llm_mod.LLM(cfg)
            summary = summarize.summarize(paras, frames, info["duration"], cfg, client)
            _write_json(summary_json, summary)
            body = re.sub(r"[\s，。？！；：、]", "", summary.get("abstract", ""))
            log(f"标题：{summary.get('title')}", "S7")
            log(f"摘要 {len(body)} 字（不含标点），"
                f"{len(summary.get('chapters') or [])} 章", "S7")
    else:
        summary = _read_json(summary_json, "归纳结果")
        log("复用归纳结果", "S7")

    # --- 渲染
    info.update({
        "generated": generated,
        "llm_label": _llm_label(cfg),
        "asr_model": cfg["asr"]["model"].split("/")[-1],
        "speakers": len(speakers) if speakers else 0,
    })
    with stage("S8", "写出产物"):
        written = render.write_outputs(out, summary, paras, frames, info, cfg)

    if not cfg["general"]["keep_audio"] and wav.exists() and not args.keep_audio:
        wav.unlink()
    if not cfg["general"]["keep_frames"] and not args.keep_frames:
        shutil.rmtree(work / "frames", ignore_errors=True)

    total = time.time() - t_begin
    log(f"全部完成，{fmt_dur(total)} · 产物目录 {out}", "OK")
    for p in written:
        print(f"  {p.name:<22s} {human_size(p.stat().st_size):>8s}", flush=True)
    report = timing_report()
    if report and not args.quiet:
        print("\n" + report, flush=True)
    return 0


def _collapse_bursts(frames: list[dict], window: float = 3.0) -> list[dict]:
    """把同一瞬间的连拍合成一张。

    PPT 翻页动画会在 1–2 秒内产生 4–5 个画面变化，场景检测会全部抓下来，
    占掉大量帧配额却只表达一页内容。这里按时间簇分组，只留 OCR 文字最多
    （也就是内容最完整）的那张。
    """
    if not frames:
        return frames
    groups: list[list[dict]] = []
    for f in frames:
        if groups and f["t"] - groups[-1][0]["t"] <= window:
            groups[-1].append(f)
        else:
            groups.append([f])

    kept = [max(g, key=lambda x: len(x.get("ocr", ""))) for g in groups]
    if len(kept) != len(frames):
        log(f"同簇连拍合并 {len(frames)} → {len(kept)} 帧", "S5")
    return kept


def _dedupe(frames: list[dict], cfg: dict) -> list[dict]:
    """按感知哈希去重，再按视频长度限制总量。"""
    f = cfg["frames"]
    try:
        from PIL import Image
    except ImportError:
        warn("未安装 Pillow，跳过去重")
        return frames[: int(f["max_frames"])]

    def dhash(path: Path, size: int = 8) -> int:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img = Image.open(path).convert("L").resize((size + 1, size), resample)
        px = list(img.getdata())
        bits = 0
        for r in range(size):
            for c in range(size):
                bits = (bits << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
        return bits

    kept, hashes = [], []
    for fr in frames:
        try:
            h = dhash(fr["path"])
        except Exception:
            kept.append(fr)
            continue
        if all(bin(h ^ o).count("1") > int(f["dedupe_distance"]) for o in hashes):
            hashes.append(h)
            kept.append(fr)

    duration = frames[-1]["t"] if frames else 0
    soft = max(12, int(duration / 60 * float(f["frames_per_minute"])) + 1)
    cap = min(int(f["max_frames"]), soft)
    if len(kept) > cap:
        step = len(kept) / cap
        kept = [kept[int(i * step)] for i in range(cap)]
        log(f"按上限抽稀到 {cap} 帧（原 {len(frames)} 帧）", "S5")
    else:
        log(f"去重后 {len(kept)} 帧（原 {len(frames)} 帧，上限 {cap}）", "S5")
    return kept


# ------------------------------------------------------------------ 其它命令

def cmd_probe(args) -> int:
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        die(f"文件不存在：{src}")
    info = media.probe(src)
    print(f"文件：{src}")
    print(f"大小：{human_size(src.stat().st_size)}")
    print(f"时长：{ts(info['duration'])}（{info['duration']:.1f}s）")
    print(f"视频：{'有 ' + str(info['width']) + 'x' + str(info['height']) if info['has_video'] else '无'}")
    print(f"音频：{'有' if info['has_audio'] else '无'}")
    if info["has_video"] and info["width"]:
        print(f"\n裁剪建议（若画面右侧/底部有聊天区或水印）：")
        print(f"  ms \"{src}\" --crop 0.02,0.07,{0.78:.2f},0.95  # 比例依次为 左,上,右,下")
    return 0


def cmd_rerun(args) -> int:
    """rerun 与 run 同构，只是要求显式指定 --from（默认从标点步骤起）。"""
    if not args.start:
        args.start = "punct"
    return cmd_run(args)


def _pull(label: str, dest: Path, repo: str, fname: str, force: bool) -> None:
    """下载单个模型文件；已存在且未指定 --force 时跳过。"""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"{label} 已存在：{dest}（{human_size(dest.stat().st_size)}）")
        return
    ensure_dir(dest.parent)
    log(f"下载{label} {repo}::{fname}", "dl")
    _hf_download(repo, fname, dest)


def cmd_models(args) -> int:
    cfg = config.load(args.config, args.set)
    target = args.target
    if target not in ("download-diar", "download-punct", "download-all"):
        die("用法：ms models download-diar | download-punct | download-all")

    if target in ("download-punct", "download-all"):
        # 目标目录由 config 里的模型路径推导，保证下载位置和运行时读取位置一致
        pdir = Path(cfg["punctuation"]["model"]).parent
        for fname in PUNCT_FILES:
            _pull(f"标点模型 {fname}", pdir / fname, PUNCT_REPO, fname, args.force)

    if target in ("download-diar", "download-all"):
        d = cfg["diarization"]
        targets = [
            ("分割模型", d["segmentation_model"], DIAR_REPO_HINT["segmentation"]),
            ("声纹模型", d["embedding_model"], DIAR_REPO_HINT["embedding"]),
        ]
        for label, dest, repo_file in targets:
            repo, _, fname = repo_file.rpartition("/")
            _pull(label, dest, repo, fname, args.force)

    if target in ("download-diar", "download-all"):
        print("完成。把 config.toml 里 diarization.mode 设为 auto 或 full 即可启用说话人分离。")
    else:
        print("完成。标点模型已就绪，punctuation.enabled 默认即为 true，无需改配置。")
    return 0


def _hf_download(repo: str, fname: str, dest: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die("未安装 huggingface_hub（uv pip install huggingface_hub）")
    try:
        p = hf_hub_download(repo_id=repo, filename=fname)
    except Exception as exc:
        warn(f"从 {repo} 下载失败：{exc}\n可尝试设置 HF_ENDPOINT=https://hf-mirror.com")
        raise SystemExit(1)
    shutil.copy(p, dest)
    print(f"  → {dest}（{human_size(dest.stat().st_size)}）")


def cmd_config(args) -> int:
    cfg = config.load(args.config, args.set)
    print(config.describe(cfg))
    print(f"工具根目录：{config.ROOT}")
    for section in ("general", "asr", "punctuation", "paragraph", "diarization",
                    "frames", "llm", "summary", "output"):
        print(f"\n[{section}]")
        for k, v in cfg[section].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    print(f"  {k}.{k2} = {v2!r}")
            else:
                print(f"  {k} = {v!r}")
    return 0


# ------------------------------------------------------------------ 参数

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ms", description="音视频转文字稿 + 详细总结 + 摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  ms ~/Videos/a.mp4                          完整处理
  ms a.mp4 --detail detailed --crop 0.02,0.07,0.78,0.95
  ms a.mp4 --set llm.provider=openai         临时切到云端模型
  ms a.mp4 --from summary                    只重跑归纳（调 prompt 时常用）
  ms a.wav --no-html --no-transcript         只要摘要
""")
    p.add_argument("input", nargs="?", help="音视频文件路径")
    p.add_argument("--config", help="配置文件路径（默认工具目录下 config.toml）")
    p.add_argument("--set", action="append", metavar="K=V",
                   help="覆盖配置项，可多次，如 --set summary.detail=brief")
    p.add_argument("--out", help="产物输出目录（默认 ~/Videos/notes/<文件名>）")
    p.add_argument("--from", dest="start", choices=STAGES, help="从指定步骤开始，复用之前的缓存")
    p.add_argument("--detail", choices=["brief", "standard", "detailed"], help="详细程度")
    p.add_argument("--crop", help="画面裁剪比例 左,上,右,下（0–1）")
    p.add_argument("--diar", choices=["off", "auto", "full", "coarse"], help="说话人分离模式")
    p.add_argument("--format", choices=["txt", "md"], help="逐字稿格式")
    p.add_argument("--srt", action="store_true", help="额外输出 SRT 字幕")
    p.add_argument("--no-html", dest="no_html", action="store_true", help="不生成详细 HTML")
    p.add_argument("--no-transcript", dest="no_transcript", action="store_true", help="不生成逐字稿")
    p.add_argument("--no-abstract", dest="no_abstract", action="store_true", help="不生成摘要")
    p.add_argument("--keep-audio", dest="keep_audio", action="store_true", help="保留中间 wav")
    p.add_argument("--keep-frames", dest="keep_frames", action="store_true", help="保留抽帧图片")
    p.add_argument("--dry-run", action="store_true", help="只打印配置与路径，不执行")
    p.add_argument("--force", action="store_true",
                   help="models 子命令用：即使模型已存在也重新下载")
    p.add_argument("--quiet", action="store_true", help="只输出错误")
    p.add_argument("-v", "--version", action="version", version="mediasum 0.2.0")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    # tui / web 前置分发：它们各自有交互参数，不该走 argparse
    if argv and argv[0] == "tui":
        from .tui import main as tui_main
        return tui_main(argv[1:])
    if argv and argv[0] == "web":
        from .web import main as web_main
        return web_main(argv[1:])

    # 子命令在位置参数之前处理，保持 `ms probe xxx` 这类写法的直觉
    command = None
    if argv and argv[0] in ("probe", "models", "config", "run", "rerun", "help"):
        command = argv.pop(0)
        if command == "help":
            build_parser().print_help()
            return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    set_quiet(args.quiet)

    if command == "probe":
        if not args.input:
            die("用法：ms probe <音视频>")
        return cmd_probe(args)
    if command == "config":
        return cmd_config(args)
    if command == "models":
        args.target = args.input or ""
        return cmd_models(args)
    if command == "rerun":
        if not args.input:
            die("用法：ms rerun <音视频> --from <步骤>")
        return cmd_rerun(args)

    if not args.input:
        parser.print_help()
        return 1
    try:
        return cmd_run(args)
    except KeyboardInterrupt:
        warn("已中断")
        return 130
    except Exception as exc:
        die(str(exc))
        return 1
