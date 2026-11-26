"""说话人分离：sherpa-onnx 打包的 pyannote 分割 + 3D-Speaker 声纹嵌入。

不用官方 pyannote 3.1 的原因是它是 gated model，要在 HuggingFace 申请权限；
这里用的 ONNX 版本只有三十几兆，纯 CPU 可跑，无需 token，中文场景用
campplus 声纹模型效果足够。

三种模式：
  off    不分离（默认，单人讲解场景省 10 分钟）
  auto   先做一次轻量探测，只有检出 2 人以上才启用
  full   严格按声纹分到每个人
  coarse 分出主要说话人，把零散短发言合并为「其他人」
"""

from __future__ import annotations

import wave
from pathlib import Path

from .util import log, warn

MIN_TURN = 0.4


def _read_wav(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise RuntimeError("说话人分离要求 16 kHz 单声道 PCM，请用 extract_audio 产出")
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0, rate


def models_ready(cfg: dict) -> bool:
    d = cfg["diarization"]
    return Path(d["segmentation_model"]).exists() and Path(d["embedding_model"]).exists()


def diarize(wav: Path, cfg: dict) -> list[dict]:
    """返回按时间排序的 [{start, end, speaker}]；失败时返回空列表而不是抛错。"""
    import sherpa_onnx as so

    d = cfg["diarization"]
    if not models_ready(cfg):
        warn("说话人分离模型缺失，跳过（先执行 ms models download-diar）")
        return []

    try:
        audio, rate = _read_wav(wav)
    except Exception as exc:
        warn(f"读取音频失败，跳过说话人分离：{exc}")
        return []

    min_spk = int(d["min_speakers"])
    max_spk = int(d["max_speakers"])
    num_clusters = min_spk if min_spk == max_spk and min_spk > 0 else -1

    config = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(d["segmentation_model"]),
            ),
            num_threads=2,
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(
            model=str(d["embedding_model"]), num_threads=2,
        ),
        clustering=so.FastClusteringConfig(
            num_clusters=num_clusters, threshold=float(d["threshold"]),
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = so.OfflineSpeakerDiarization(config)
    if sd.sample_rate != rate:
        warn(f"采样率不匹配（{rate} vs {sd.sample_rate}），跳过说话人分离")
        return []

    # sort_by_start_time() 返回的是 list，num_speakers 只能在排序前从 Result 取
    raw_result = sd.process(audio.tolist())
    n_speakers = raw_result.num_speakers
    turns = [
        {"start": float(s.start), "end": float(s.end), "speaker": int(s.speaker)}
        for s in raw_result.sort_by_start_time()
        if float(s.end) - float(s.start) >= MIN_TURN
    ]
    log(f"检出 {n_speakers} 位说话人 / {len(turns)} 个发言片段", "S6")
    if max_spk > 0:
        over = sorted({t["speaker"] for t in turns})
        if len(over) > max_spk:
            warn(f"检出人数 {len(over)} 超过上限 {max_spk}，结果可能过分割")
    return turns


def attach(paragraphs: list[dict], turns: list[dict], cfg: dict) -> list[str]:
    """按时间重叠给每段指派说话人，返回按出现顺序的标签列表。"""
    if not turns:
        return []

    durations: dict[int, float] = {}
    for t in turns:
        durations[t["speaker"]] = durations.get(t["speaker"], 0.0) + (t["end"] - t["start"])

    order = sorted(durations, key=durations.get, reverse=True)
    # 时长最长的认定为「说话人 1」，主讲通常就是这位
    label_of = {spk: f"说话人 {i + 1}" for i, spk in enumerate(order)}

    if cfg["diarization"]["coarse"] and len(order) > 1:
        total = sum(durations.values()) or 1.0
        minor = [s for s in order if durations[s] / total < 0.15]
        if minor and len(minor) < len(order):
            for s in minor:
                label_of[s] = "其他人"

    for p in paragraphs:
        overlap: dict[int, float] = {}
        for t in turns:
            a, b = max(p["start"], t["start"]), min(p["end"], t["end"])
            if b > a:
                overlap[t["speaker"]] = overlap.get(t["speaker"], 0.0) + (b - a)
        p["speaker"] = label_of[max(overlap, key=overlap.get)] if overlap else ""

    labels: list[str] = []
    seen = set()
    for p in paragraphs:
        s = p.get("speaker", "")
        if s and s not in seen:
            seen.add(s)
            labels.append(s)
    return labels


def speaker_count(paragraphs: list[dict]) -> int:
    return len({p.get("speaker") for p in paragraphs if p.get("speaker")})
