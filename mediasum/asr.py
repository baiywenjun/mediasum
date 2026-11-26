"""语音转写与标点恢复。

三层标点策略（按执行顺序）：
  第一层 Whisper 侧：initial_prompt 里显式要求使用规范标点，能救回约七成；
  第二层 标点模型：CT-Transformer 转 ONNX（INT8，纯 CPU），给完全无标点的
        文本补全角标点，这是准确率的主要来源；
  第三层 规则兜底：半角转全角、段尾补句号，保证输出形态统一。

分流是必须的：第二层模型对「已有标点的文本」会重复标注（产生「。。」「？?」），
所以只有完全无中文标点的文本才送模型，其余只做全角统一。

段落重组也在这里做。顺序是「先按时间聚块 → 整块标点 → 按句号切句 →
把句子拼成目标时长的段落」，这样段落边界一定落在句末，不会像早期版本
那样把一句话切在两段之间。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .config import ROOT
from .util import log, warn

DEFAULT_PROMPT = "以下是普通话讲解，请使用简体中文，并使用规范的标点符号断句。"

CJK_PUNCT = "，。？！；：、"
SENT_END = "。？！…"
# 计算「纯文本」时剥离的字符：标点与空白都不占语义位
_STRIP = re.compile(r"[\s，。？！；：、,?!;:…\-—·\"'“”‘’（）()《》〈〉【】\[\]]")


def mlx_bin() -> Path:
    cand = ROOT / ".venv" / "bin" / "mlx_whisper"
    if cand.exists():
        return cand
    raise RuntimeError(f"未找到 mlx_whisper：{cand}（先在工具目录执行 uv pip install mlx-whisper）")


def _model_cached(repo: str) -> bool:
    base = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "hub"
    return (base / ("models--" + repo.replace("/", "--"))).exists()


def _whisper_env(model: str) -> dict:
    """优先离线加载已缓存的模型。

    这台机器走了 HTTP 代理，mlx-whisper 启动时会去 HuggingFace 校验权重，
    代理返回 502 后它的处理是「Skipping <file> due to ProxyError」——
    退出码仍是 0，只是什么也没产出。所以有缓存时必须强制离线。
    """
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if _model_cached(model):
        env["HF_HUB_OFFLINE"] = "1"
    return env


def transcribe(wav: Path, workdir: Path, cfg: dict) -> dict:
    """调用 mlx-whisper 生成 JSON（含逐段起止时间）。"""
    asr = cfg["asr"]
    prompt = (asr.get("initial_prompt") or DEFAULT_PROMPT).strip()
    cmd = [
        str(mlx_bin()), str(wav),
        "--model", asr["model"],
        "--language", asr["language"],
        "--output-dir", str(workdir),
        "--output-format", "json",
        "--initial-prompt", prompt,
        "--condition-on-previous-text", "True" if asr["condition_on_previous_text"] else "False",
        "--verbose", "False",
    ]
    from .util import run

    js = workdir / f"{wav.stem}.json"
    if js.exists():
        js.unlink()

    proc = run(cmd, env=_whisper_env(asr["model"]))

    if not js.exists():
        # 离线模式失败（首次运行尚无缓存）时退回联网，仍失败才报错
        env2 = _whisper_env(asr["model"])
        if env2.pop("HF_HUB_OFFLINE", None):
            log("离线加载失败，改为联网获取模型", "S3")
            proc = run(cmd, env=env2)
    if not js.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        hint = ""
        if any("Proxy" in x or "Skipping" in x for x in tail):
            hint = ("\n看起来是网络代理拦截了模型下载。若模型已缓存，"
                    "可设 HF_HUB_OFFLINE=1 重试；否则挂代理或设 HF_ENDPOINT=https://hf-mirror.com")
        raise RuntimeError(f"转写未产出 JSON：{js}\n" + "\n".join(tail) + hint)
    data = json.loads(js.read_text(encoding="utf-8"))
    segs = data.get("segments") or []
    if not segs:
        raise RuntimeError("转写结果为空，请检查音频是否含有效人声")
    log(f"{len(segs)} 个原始分段，{sum(len(s['text']) for s in segs)} 字", "S3")
    return data


class Punctuator:
    """第二、三层标点的封装。模型缺失时自动降级为纯规则处理。"""

    def __init__(self, cfg: dict) -> None:
        p = cfg["punctuation"]
        self.enabled = bool(p["enabled"])
        self.fullwidth = bool(p["force_fullwidth"])
        self.tail_period = bool(p["tail_period"])
        self.model_path = Path(p["model"])
        self._model = None
        self._loaded = False
        self.stats = {"model": 0, "fullwidth": 0, "fallback": 0}

    def _load(self):
        if self._loaded:
            return self._model
        self._loaded = True
        if not self.enabled:
            self._model = None
            return None
        if not self.model_path.exists():
            warn(f"标点模型不存在：{self.model_path}，降级为规则补标点")
            self._model = None
            return None
        try:
            import sherpa_onnx
        except ImportError:
            warn("未安装 sherpa-onnx，降级为规则补标点（uv pip install sherpa-onnx）")
            self._model = None
            return None
        cfg = sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(
                ct_transformer=str(self.model_path), num_threads=2,
            )
        )
        self._model = sherpa_onnx.OfflinePunctuation(cfg)
        log(f"标点模型已加载（{self.model_path.name}）", "S4")
        return self._model

    def _to_fullwidth(self, text: str) -> str:
        # 数字里的千分位逗号（1,000）不能被转成全角
        text = re.sub(r"(?<!\d),(?!\d)", "，", text)
        table = {"?": "？", "!": "！", ";": "；", ":": "：", "(": "（", ")": "）"}
        for half, full in table.items():
            text = text.replace(half, full)
        return text

    def restore(self, text: str) -> str:
        body = (text or "").strip()
        if not body:
            return body

        plain = _STRIP.sub("", body)
        if len(plain) < 4:
            return self._add_tail(body)

        if any(ch in body for ch in CJK_PUNCT):
            # 已有中文标点：模型会重复标注，只做全角统一
            self.stats["fullwidth"] += 1
            fixed = self._to_fullwidth(body) if self.fullwidth else body
            return self._add_tail(fixed)

        model = self._load()
        if model is not None:
            try:
                out = (model.add_punctuation(body) or "").strip()
                # 模型只应插入标点；纯文本长度出现明显变化说明输出异常
                if out and abs(len(_STRIP.sub("", out)) - len(plain)) <= max(2, len(plain) * 0.05):
                    self.stats["model"] += 1
                    return self._add_tail(self._to_fullwidth(out) if self.fullwidth else out)
                warn(f"标点模型输出异常，改用规则处理：{body[:30]}…")
            except Exception as exc:  # 模型异常不应中断整条管线
                warn(f"标点模型调用失败：{exc}")

        self.stats["fallback"] += 1
        return self._add_tail(body)

    def _add_tail(self, text: str) -> str:
        if not self.tail_period:
            return text
        s = text.rstrip()
        if not s:
            return s
        if s[-1] in "。？！…”）":
            return s
        return s + "。"


def _plain_len(text: str) -> int:
    return len(_STRIP.sub("", text))


def _chunk(segments: list[dict], gap: float, chunk_max: float) -> list[list[dict]]:
    """把 Whisper 的碎片段聚成块，块是标点模型的处理单元。

    块要足够长才能让模型判断句界（短句缺乏上下文），又不能太长以免
    单次推理变慢；chunk_max 默认 30 秒。
    """
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        item = {"start": float(s["start"]), "end": float(s["end"]), "text": text}
        if not cur:
            cur = [item]
            continue
        within_gap = item["start"] - cur[-1]["end"] <= gap
        short_enough = (cur[-1]["end"] - cur[0]["start"]) < chunk_max
        if within_gap and short_enough:
            cur.append(item)
        else:
            chunks.append(cur)
            cur = [item]
    if cur:
        chunks.append(cur)
    return chunks


def _join(chunk: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """拼接块内文本（中文不加分隔符），同时记录每段在纯文本中的区间。"""
    parts, spans, pos = [], [], 0
    for s in chunk:
        t = s["text"].strip()
        n = _plain_len(t)
        spans.append((pos, pos + n))
        parts.append(t)
        pos += n
    return "".join(parts), spans


def _split_sentences(text: str) -> list[str]:
    out, buf = [], ""
    for ch in text:
        buf += ch
        if ch in SENT_END:
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def _time_at(chunk: list[dict], spans: list[tuple[int, int]], pos: int, is_end: bool) -> float:
    """把纯文本字符位置换算成时间。

    Whisper 偶尔会把几十秒连续的讲话合并成一个长段（开场白最典型），
    若直接取该段的 start/end，块内所有句子都会得到同一个时间戳。
    这里按字符数在段内做线性插值，误差取决于语速是否均匀，
    对本场景（均匀讲解）足够。
    """
    for (a, b), s in zip(spans, chunk):
        if b <= a:
            continue
        if b > pos or (is_end and b >= pos):
            ratio = (min(max(pos, a), b) - a) / (b - a)
            return s["start"] + ratio * (s["end"] - s["start"])
    return chunk[-1]["end"]


def _timed_sentences(chunk: list[dict], spans: list[tuple[int, int]],
                     marked: str) -> list[dict]:
    """把标点后的句子映射回时间段（按纯文本字符位置对齐）。"""
    timed: list[dict] = []
    cursor = 0
    for sent in _split_sentences(marked):
        idx = marked.find(sent, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(sent)
        lo, hi = _plain_len(marked[:idx]), _plain_len(marked[: idx + len(sent)])
        if hi <= lo:
            continue
        start = _time_at(chunk, spans, lo, is_end=False)
        end = _time_at(chunk, spans, hi - 1, is_end=True)
        if end <= start:
            end = min(start + 0.5, chunk[-1]["end"])
        timed.append({"text": sent, "start": start, "end": end})
    return timed


def _group(sentences: list[dict], target: float, limit: float,
           max_chars: int, per_sentence: bool) -> list[dict]:
    """句子聚成段落。

    时长和字数必须同时约束：语速快的人一分钟能说 450 字，只按时长切
    会得到一百多字的巨型段落；反过来只管字数，遇到语速慢又会产生
    长达半分钟、读起来拖沓的段。
    """
    if per_sentence:
        return [dict(s) for s in sentences]

    paras: list[dict] = []
    cur: list[dict] = []
    for s in sentences:
        if cur:
            span = cur[-1]["end"] - cur[0]["start"]
            chars = sum(_plain_len(x["text"]) for x in cur)
            would = s["end"] - cur[0]["start"]
            would_chars = chars + _plain_len(s["text"])
            if span >= target or would > limit or would_chars > max_chars:
                paras.append(_merge(cur))
                cur = []
        cur.append(s)
    if cur:
        paras.append(_merge(cur))
    return paras


def _merge(sents: list[dict]) -> dict:
    return {
        "start": sents[0]["start"],
        "end": sents[-1]["end"],
        "text": "".join(s["text"] for s in sents),
    }


def build_paragraphs(raw_segments: list[dict], punct: Punctuator, cfg: dict) -> list[dict]:
    p = cfg["paragraph"]
    chunks = _chunk(raw_segments, float(p["merge_gap"]), float(p["chunk_max"]))
    paras: list[dict] = []
    for chunk in chunks:
        text, spans = _join(chunk)
        if not text:
            continue
        marked = punct.restore(text)
        sents = _timed_sentences(chunk, spans, marked)
        if not sents:
            # 时间映射失败时退回整块，至少不丢内容
            paras.append({"start": chunk[0]["start"], "end": chunk[-1]["end"],
                          "text": marked})
            continue
        paras.extend(_group(sents, float(p["target"]), float(p["max"]),
                            int(p["max_chars"]), bool(p["separate_sentence"])))
    log(f"{len(raw_segments)} 碎片段 → {len(chunks)} 块 → {len(paras)} 段", "S4")
    return paras


def paragraph_stats(paras: list[dict]) -> dict:
    if not paras:
        return {"count": 0, "chars": 0, "avg_sec": 0.0, "avg_chars": 0}
    chars = [_plain_len(p["text"]) for p in paras]
    spans = [p["end"] - p["start"] for p in paras]
    return {
        "count": len(paras),
        "chars": sum(chars),
        "avg_sec": sum(spans) / len(spans),
        "avg_chars": sum(chars) / len(chars),
    }
