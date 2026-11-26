"""归纳层：把逐字稿与画面文字整理成结构化笔记。

长音频必然超出上下文，所以默认走 map-reduce：先按块提炼要点（map），
再拿要点加少量原文生成最终结构（reduce）。章节时间戳一律以真实段落
起点为候选，模型只能从中挑选，选完再做一次就近吸附——模型自己编
整数分钟（00:02:00、00:04:00）是这个环节最常见的失真。
"""

from __future__ import annotations

import re

from .llm import LLM, LLMError
from .util import log, parse_ts, ts, warn

SYSTEM = "你是严谨的课程与技术会议笔记助理，只依据给定材料作答，不添加材料外的推测。"


def _fmt_time(sec: float) -> str:
    return ts(sec)


def _para_lines(paras: list[dict], show_speaker: bool) -> str:
    lines = []
    for p in paras:
        who = f"{p['speaker']}：" if show_speaker and p.get("speaker") else ""
        lines.append(f"[{_fmt_time(p['start'])}] {who}{p['text']}")
    return "\n".join(lines)


def _frame_lines(frames: list[dict]) -> str:
    if not frames:
        return "（无画面）"
    out = []
    for f in frames:
        text = " / ".join(x.strip() for x in f["ocr"].splitlines() if x.strip())[:300]
        out.append(f"- {_fmt_time(f['t'])}：{text}")
    return "\n".join(out)


def _cn_len(text: str) -> int:
    """统计有效字符数（剥离标点与空白），用于校验摘要长度。"""
    return len(re.sub(r"[\s，。？！；：、“”‘’（）()《》〈〉…—\-,.?!;:]", "", text or ""))


def _chunk_paras(paras: list[dict], size: int) -> list[list[dict]]:
    chunks, cur, n = [], [], 0
    for p in paras:
        if cur and n + len(p["text"]) > size:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(p)
        n += len(p["text"])
    if cur:
        chunks.append(cur)
    return chunks


def _plan_chunks(paras: list[dict], duration: float, cfg: dict) -> list[list[dict]]:
    """决定分几块做 map。

    小块是必要的：把两小时逐字稿整段丢给 7B 模型，它只会总结开头
    一两分钟（注意力衰减），章节全挤在片头是这个问题最典型的表现。
    改成每 3 分钟一块，块数上限 16，模型拿到的是「带时间范围的分段要点」，
    覆盖全程的概率大幅提高。
    """
    configured = int(cfg["summary"]["chunk_chars"])
    if configured > 0:
        return _chunk_paras(paras, configured)
    total = sum(len(p["text"]) for p in paras)
    target = max(3, min(16, int(duration // 180) + 1))
    return _chunk_paras(paras, max(800, total // target))


def _map_chunk(paras: list[dict], cfg: dict, llm: LLM, idx: int, total: int) -> dict:
    span = f"{_fmt_time(paras[0]['start'])}–{_fmt_time(paras[-1]['end'])}"
    prompt = f"""下面是长音视频第 {idx}/{total} 段（{span}）的逐字稿。请提炼这一段的内容，输出 JSON：
{{"points": ["这一段的要点，每条不超过 40 字"], "terms": ["这一段出现的专业术语"]}}

要求：points 取 4–8 条，按讲述顺序；terms 只收确实出现的术语，没有就给空列表。
只输出 JSON。

逐字稿：
{_para_lines(paras, cfg['output']['show_speaker'])}
"""
    data = llm.chat_json(prompt, SYSTEM)
    points = [str(x).strip() for x in data.get("points", []) if str(x).strip()]
    terms = [str(x).strip() for x in data.get("terms", []) if str(x).strip()]
    log(f"第 {idx}/{total} 段 → {len(points)} 条要点", "S7")
    return {"span": span, "start": paras[0]["start"], "points": points, "terms": terms}


def _schema(detail: str) -> str:
    base = """{
  "title": "标题（不超过 24 字，写清主题，不要写「课程笔记」这种空话）",
  "abstract": "摘要",
  "chapters": [
    {"time": "HH:MM:SS", "title": "章节标题", "points": ["要点"]}
  ],
  "terms": [{"term": "术语", "desc": "一句话解释"}]
}"""
    if detail == "detailed":
        base = base.rstrip()[:-1].rstrip() + """,
  "quotes": ["原文中的关键论断，逐字引用，最多 6 条"],
  "questions": ["讲者提出但未展开、或仍然悬而未决的问题"]
}"""
    return base


def _requirements(detail: str, duration: float, cfg: dict) -> str:
    s = cfg["summary"]
    lines = [
        f"1. 音视频总时长 {ts(duration)}，chapters 的 time 必须从下面给出的「时间候选」里挑，"
        f"不得超出总时长，也不得自造时间。",
        f"2. abstract 必须是 {s['abstract_min']}–{s['abstract_max']} 字的连续段落："
        f"先讲这段内容在解决什么问题，再讲分了哪几个部分，最后给结论或价值。不要罗列条目。",
        f"3. chapters 取 {s['chapters_min']}–{s['chapters_max']} 章，按时间顺序；"
        f"每章 {max(3, s['points_per_chapter'] - 2)}–{s['points_per_chapter']} 条 points，每条不超过 40 字。",
        f"4. terms 取 {s['terms_count']} 个左右，只收材料里真实出现的。",
        "5. 全部用简体中文，不要出现「本视频」「up 主」这类口水说法，直接讲内容。",
    ]
    if detail == "brief":
        lines = [lines[0], lines[1], "2. 不要输出 chapters 和 terms。"]
    elif detail == "detailed":
        lines.append("6. quotes 逐字引用原文，不要改写；questions 只收材料里确实存疑的点，没有就给空列表。")
    return "\n".join(lines)


def summarize(paras: list[dict], frames: list[dict], duration: float,
              cfg: dict, llm: LLM) -> dict:
    s = cfg["summary"]
    detail = s["detail"]
    full = _para_lines(paras, cfg["output"]["show_speaker"])

    chunks = _plan_chunks(paras, duration, cfg) if s["map_reduce"] else []
    use_map = bool(s["map_reduce"]) and len(chunks) > 1

    maps: list[dict] = []
    if use_map:
        log(f"分 {len(chunks)} 块提炼要点（每块约 {int(duration / len(chunks) / 60) or 1} 分钟）", "S7")
        for i, ch in enumerate(chunks, 1):
            maps.append(_map_chunk(ch, cfg, llm, i, len(chunks)))

    best = None
    for attempt in (1, 2):
        data = _reduce_once(maps, full, frames, paras, duration, cfg, llm,
                            retry=attempt > 1)
        data = _postprocess(data, paras, duration, cfg, llm)
        if best is None or data["_coverage"] > best["_coverage"]:
            best = data
        if detail == "brief" or not data["chapters"] or data["_coverage"] >= 0.55:
            break
        warn(f"章节只覆盖到时长 {data['_coverage'] * 100:.0f}%，重试一次")

    best.pop("_coverage", None)
    if detail != "brief" and maps and (
        not best["chapters"]
        or max(parse_ts(c["time"]) for c in best["chapters"]) < duration * 0.55
    ):
        warn("模型仍未覆盖全片，改用分块要点重建章节骨架")
        best["chapters"] = _skeleton_from_maps(maps, cfg)

    # 摘要长度放在最后统一校准，避免重试时反复调用扩写
    best["abstract"] = _fit_abstract(best["abstract"], paras, detail, cfg, llm)
    return best


def _reduce_once(maps: list[dict], full: str, frames: list[dict], paras: list[dict],
                 duration: float, cfg: dict, llm: LLM, retry: bool) -> dict:
    s = cfg["summary"]
    detail = s["detail"]

    if maps:
        timeline = "\n".join(
            f"[{m['span']}] " + "；".join(m["points"]) for m in maps
        )
        material = "分段要点（按时间顺序，覆盖全片）：\n" + timeline
        material += "\n\n开头原文：\n" + _para_lines(paras[:5], False)
        if len(paras) > 10:
            material += "\n\n结尾原文：\n" + _para_lines(paras[-5:], False)
        term_hint = "、".join(dict.fromkeys(t for m in maps for t in m["terms"]))[:400]
        if term_hint:
            material += f"\n\n术语候选：{term_hint}"
        cand = [m["start"] for m in maps]
    else:
        material = "逐字稿：\n" + full
        cand = [p["start"] for p in paras]

    cand_text = "、".join(_fmt_time(c) for c in _pick_candidates(sorted(cand), 60))

    extra = ""
    if retry:
        extra = ("\n【重要】上一次的输出只覆盖了开头部分。这次必须按时间轴从头到尾写成，"
                 "最后一章的时间要接近片尾，不能只总结前几分钟。")

    prompt = f"""基于下面的材料，输出一份结构化笔记，严格按 JSON 格式（不要解释、不要代码块）：

{_schema(detail)}

硬约束：
{_requirements(detail, duration, cfg)}
6. chapters 必须覆盖整段内容：第一章接近 00:00:00，最后一章接近片尾 {ts(duration)}，
   中间不得出现大段空白（相邻章节间隔不应超过总时长的 1/5）。{extra}

时间候选（chapter 的 time 只能从这里选）：{cand_text}

画面文字（幻灯片/板书 OCR，供交叉印证）：
{_frame_lines(frames)}

材料：
{material}
"""
    return llm.chat_json(prompt, SYSTEM)


def _skeleton_from_maps(maps: list[dict], cfg: dict) -> list[dict]:
    """模型始终只覆盖开头时的确定性兜底：直接用分块要点拼章节。

    标题取自该块第一条要点，不如模型起的好，但保证结构覆盖全片。
    """
    s = cfg["summary"]
    n = len(maps)
    want = min(int(s["chapters_max"]), max(int(s["chapters_min"]), min(n, 6)))
    step = n / want
    out = []
    for i in range(want):
        lo = int(i * step)
        hi = max(lo + 1, int((i + 1) * step))
        grp = maps[lo:hi]
        if not grp:
            continue
        points = [p for m in grp for p in m["points"]][: int(s["points_per_chapter"])]
        out.append({
            "time": ts(grp[0]["start"]),
            "title": (points[0][:22] if points else f"第 {i + 1} 部分"),
            "points": points,
        })
    return out


def _pick_candidates(starts: list[float], limit: int) -> list[float]:
    if len(starts) <= limit:
        return starts
    step = len(starts) / limit
    return [starts[int(i * step)] for i in range(limit)]


def _postprocess(data: dict, paras: list[dict], duration: float,
                 cfg: dict, llm: LLM) -> dict:
    s = cfg["summary"]
    detail = s["detail"]
    title = str(data.get("title") or "未命名").strip()
    abstract = str(data.get("abstract") or "").strip()

    chapters = data.get("chapters") or []
    if detail == "brief":
        chapters = []
    kept = []
    for c in chapters:
        if not isinstance(c, dict):
            continue
        t = parse_ts(c.get("time", "0:00:00"))
        if 0 <= t <= duration + 1:
            kept.append(c)
    dropped = len(chapters) - len(kept)
    if dropped > 0:
        log(f"丢弃 {dropped} 个时间越界的章节", "S7")

    starts = sorted(p["start"] for p in paras) or [0.0]
    for c in kept:
        t = parse_ts(c.get("time", "0:00:00"))
        c["time"] = ts(min(starts, key=lambda x: abs(x - t)))
        c["points"] = [str(x).strip() for x in (c.get("points") or []) if str(x).strip()]
    kept.sort(key=lambda c: parse_ts(c["time"]))

    out = {"title": title, "abstract": abstract, "chapters": kept,
           "terms": data.get("terms") or []}
    if detail == "detailed":
        out["quotes"] = [str(x).strip() for x in (data.get("quotes") or []) if str(x).strip()]
        out["questions"] = [str(x).strip() for x in (data.get("questions") or []) if str(x).strip()]
    out["_coverage"] = (
        max(parse_ts(c["time"]) for c in kept) / duration
        if kept and duration > 0 else 0.0
    )
    return out


def _fit_abstract(abstract: str, paras: list[dict], detail: str,
                  cfg: dict, llm: LLM) -> str:
    s = cfg["summary"]
    lo, hi = int(s["abstract_min"]), int(s["abstract_max"])
    n = _cn_len(abstract)

    if n < lo:
        log(f"摘要 {n} 字偏短，要求扩写到 {lo}–{hi} 字", "S7")
        context = _para_lines(paras[:40], False)[:5000]
        # 7B 模型第一轮常写到 400 字上下就停，带具体差距反馈再推一轮
        for attempt in range(2):
            if _cn_len(abstract) >= lo:
                break
            gap = lo - _cn_len(abstract)
            tip = (f"你上一次只写了 {_cn_len(abstract)} 字，还差 {gap} 字，"
                   f"这次必须写满 {lo} 字以上。") if attempt else ""
            try:
                more = llm.chat(
                    f"把下面的摘要扩写到 {lo}–{hi} 字，必须是连续段落，补充具体内容"
                    f"（分成哪几个部分、各自讲了什么），不要新增材料之外的事实，"
                    f"不要分条，只输出摘要正文。{tip}\n\n"
                    f"原摘要：{abstract}\n\n原文参考：\n{context}"
                ).strip()
            except LLMError as exc:
                warn(f"扩写失败，保留原摘要：{exc}")
                break
            if _cn_len(more) <= _cn_len(abstract):
                break
            abstract = more
        log(f"扩写后 {_cn_len(abstract)} 字", "S7")

    n = _cn_len(abstract)
    if n > hi:
        log(f"摘要 {n} 字超出上限 {hi}，要求压缩", "S7")
        try:
            shorter = llm.chat(
                f"把下面的摘要压缩到 {lo}–{hi} 字，保留核心结构与结论，"
                f"必须是一段连续文字，只输出摘要正文：\n\n{abstract}"
            ).strip()
            if 0 < _cn_len(shorter) < n:
                abstract = shorter
        except LLMError as exc:
            warn(f"压缩失败：{exc}")

    # 模型经常只差几十字就压不住，这里用确定性截断兜底：按完整句子取舍
    if _cn_len(abstract) > hi:
        cut = _trim_to_sentences(abstract, hi)
        log(f"按句子截断到 {_cn_len(cut)} 字", "S7")
        abstract = cut
    return abstract


def _trim_to_sentences(text: str, limit: int) -> str:
    parts = [p for p in re.split(r"(?<=[。？！])", text.strip()) if p.strip()]
    out = ""
    for p in parts:
        if _cn_len(out + p) > limit:
            break
        out += p
    out = out.strip()
    if not out:
        # 单句就超标，只能硬截断并在末尾补句号
        out = text.strip()[:limit].rstrip("，、；：") + "。"
    return out
