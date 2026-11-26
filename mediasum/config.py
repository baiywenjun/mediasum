"""配置加载：内置默认值 ← config.toml ← 命令行 --set 覆盖。

三层来源的优先级依次递增，任何一层缺失都退回上一层，因此删掉
config.toml 里的某一行不会让程序崩掉，只是回落到默认值。
"""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

from .util import die, log

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.toml"

DEFAULTS: dict[str, Any] = {
    "general": {
        "output_dir": "~/Videos/notes",
        "work_dir": "~/Videos/notes/.work",
        "keep_audio": False,
        "keep_frames": True,
    },
    "asr": {
        "model": "mlx-community/whisper-large-v3-turbo",
        "language": "zh",
        "initial_prompt": "",
        "condition_on_previous_text": False,
    },
    "punctuation": {
        "enabled": True,
        "model": "models/punct/model.int8.onnx",
        "force_fullwidth": True,
        "tail_period": True,
    },
    "paragraph": {
        "merge_gap": 1.5,
        "chunk_max": 30.0,
        "target": 12.0,
        "max": 22.0,
        "max_chars": 120,
        "separate_sentence": False,
    },
    "diarization": {
        "mode": "off",
        "threshold": 0.5,
        "min_speakers": -1,
        "max_speakers": 8,
        "segmentation_model": "models/diar/segmentation.onnx",
        "embedding_model": "models/diar/embedding.onnx",
        "coarse": True,
    },
    "frames": {
        "enabled": True,
        "scene_threshold": 0.18,
        "interval": 120.0,
        "max_frames": 48,
        "frames_per_minute": 3.0,
        "dedupe_distance": 10,
        "crop": [],
        "width": 1280,
        "jpeg_quality": 4,
        "min_ocr_chars": 10,
        "ocr_languages": ["zh-Hans", "en-US"],
        "ocr_min_confidence": 0.35,
        "ocr_blocklist": [
            r"¥\s*\d", r"￥\s*\d", "好评度", "立即购买", "已售",
            "正在看", "点赞", "扫码", "关注公众号",
        ],
    },
    "llm": {
        "provider": "ollama",
        "temperature": 0.3,
        "timeout": 900,
        "ollama": {
            "base_url": "http://127.0.0.1:11434",
            "model": "qwen2.5:latest",
            "num_ctx": 24576,
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "json_mode": True,
            "max_tokens": 4096,
        },
    },
    "summary": {
        "detail": "standard",
        "map_reduce": True,
        "chunk_chars": 0,
        "abstract_min": 450,
        "abstract_max": 580,
        "chapters_min": 4,
        "chapters_max": 8,
        "points_per_chapter": 5,
        "terms_count": 8,
    },
    "output": {
        "transcript": True,
        "detail_html": True,
        "abstract": True,
        "transcript_format": "txt",
        "srt": False,
        "embed_frames": True,
        "show_speaker": True,
    },
}

PATH_KEYS = {
    ("general", "output_dir"),
    ("general", "work_dir"),
    ("punctuation", "model"),
    ("diarization", "segmentation_model"),
    ("diarization", "embedding_model"),
}


def _merge(base: dict, over: dict) -> dict:
    """递归合并，over 中的标量覆盖 base，字典继续下钻。"""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(raw: str) -> Any:
    """--set 的值是字符串，按字面量推断类型。"""
    s = raw.strip()
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_coerce(x) for x in inner.split(",")] if inner else []
    return s


def apply_override(cfg: dict, expr: str) -> None:
    """把 a.b.c=value 写进配置树，路径不存在时直接报错而不是静默忽略。"""
    if "=" not in expr:
        die(f"--set 需要 key=value 形式，收到：{expr}")
    key, _, value = expr.partition("=")
    parts = [p.strip() for p in key.split(".") if p.strip()]
    if not parts:
        die(f"--set 的键为空：{expr}")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            die(f"--set 未知配置项：{key}（{p} 不存在）")
        node = node[p]
    if parts[-1] not in node:
        die(f"--set 未知配置项：{key}")
    node[parts[-1]] = _coerce(value)


def _resolve_paths(cfg: dict) -> None:
    for section, key in PATH_KEYS:
        raw = cfg[section][key]
        if not raw:
            continue
        p = Path(os.path.expandvars(os.path.expanduser(str(raw))))
        # 相对路径按工具根目录解析，避免受启动时 cwd 影响
        if not p.is_absolute():
            p = ROOT / p
        cfg[section][key] = p


def load(config_path: str | None = None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG
    if config_path and not path.exists():
        die(f"配置文件不存在：{path}")
    if path.exists():
        try:
            with path.open("rb") as fh:
                user = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            die(f"配置文件语法错误（{path}）：{exc}")
        unknown = [f"{s}.{k}" for s, v in user.items() if isinstance(v, dict)
                   for k in v if s not in DEFAULTS or k not in DEFAULTS.get(s, {})]
        if unknown:
            log(f"提示：以下配置项未识别，已忽略 —— {', '.join(sorted(unknown))}", "cfg")
        cfg = _merge(cfg, user)
    for expr in overrides or []:
        apply_override(cfg, expr)
    _resolve_paths(cfg)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    para = cfg["paragraph"]
    if para["target"] <= 0 or para["max"] < para["target"]:
        die("paragraph.target 必须大于 0，且 paragraph.max 不得小于 target")
    if para["chunk_max"] < para["max"]:
        die("paragraph.chunk_max 不得小于 paragraph.max（否则句子会被块边界切断）")
    if int(para["max_chars"]) < 20:
        die("paragraph.max_chars 不建议小于 20 字")

    mode = cfg["diarization"]["mode"]
    if mode not in ("off", "auto", "full", "coarse"):
        die(f"diarization.mode 只能是 off/auto/full/coarse，收到：{mode}")

    provider = cfg["llm"]["provider"]
    if provider not in ("ollama", "openai"):
        die(f"llm.provider 只能是 ollama/openai，收到：{provider}")
    if provider == "openai":
        env = cfg["llm"]["openai"]["api_key_env"]
        if not os.environ.get(env):
            die(f"llm.provider=openai 但环境变量 {env} 未设置。"
                f"执行 export {env}=sk-... 后重试")

    detail = cfg["summary"]["detail"]
    if detail not in ("brief", "standard", "detailed"):
        die(f"summary.detail 只能是 brief/standard/detailed，收到：{detail}")

    smin, smax = cfg["summary"]["abstract_min"], cfg["summary"]["abstract_max"]
    if smin > smax:
        die("summary.abstract_min 不得大于 abstract_max")

    if cfg["output"]["transcript_format"] not in ("txt", "md"):
        die("output.transcript_format 只能是 txt 或 md")

    if not any(cfg["output"][k] for k in ("transcript", "detail_html", "abstract")):
        die("output 里至少要开启一项产物（transcript / detail_html / abstract）")


def describe(cfg: dict) -> str:
    """打印生效配置中真正影响行为的项，便于复现问题。"""
    o, s, l = cfg["output"], cfg["summary"], cfg["llm"]
    provider = l["provider"]
    model = l[provider]["model"]
    outs = [k for k in ("transcript", "detail_html", "abstract") if o[k]]
    return (
        f"配置：产物={','.join(outs)} 详细程度={s['detail']} "
        f"模型={provider}/{model} 说话人={cfg['diarization']['mode']} "
        f"标点={'on' if cfg['punctuation']['enabled'] else 'off'}"
    )
