"""把 CLI 的 stdout 解析成结构化进度。

CLI 本身只打印文本行，Web 界面需要知道「现在在哪个阶段、子进度到几成」，
才能画进度条。解析规则集中在这里一份。

注：`tui.py` 里内嵌了等价的一套正则（34–48 行）。之所以没有合并，是因为
TUI 那套已经过端到端验证，改动它的收益低于引入回归的风险。若将来要统一，
把 tui.py 的 _on_line 改成消费本模块的 Progress.feed() 即可。

依赖前提：`mediasum/util.py` 的 log() 输出不带 ANSI 颜色码（已确认），
所以这里可以按纯文本匹配。若将来给 CLI 加了彩色输出，解析会全部失效，
届时需要在 spawn 子进程时设置 NO_COLOR=1。
"""

from __future__ import annotations

import re

# 阶段清单固定为全量。未出现的阶段表示「本次路径不需要」，
# 比如纯音频没有抽帧、说话人关闭时没有 S6。
STAGES = [("S1", "读取"), ("S2", "抽音"), ("S3", "转写"), ("S4", "标点"),
          ("S5", "抽帧"), ("S6", "说话人"), ("S7", "归纳"), ("S8", "产出")]

# 每次运行必然出现的阶段，用来算进度分母
CORE_STAGES = ["S1", "S3", "S4", "S5", "S7", "S8"]
# 条件阶段：出现才计入分母，否则进度会卡在固定比例上不动
EXTRA_STAGES = ["S2", "S6"]

RE_START = re.compile(r"^\[(S\d)\] ▶ (.+)$")
RE_DONE = re.compile(r"^\[(S\d)\] 完成，耗时 (.+)$")
RE_FRAME = re.compile(r"帧 (\d+)/(\d+)")
RE_MAP = re.compile(r"第 (\d+)/(\d+) 段")
RE_TITLE = re.compile(r"^\[S7\] 标题：(.+)$")
RE_OK = re.compile(r"^\[OK\] 全部完成，(.+?)(?: · 产物目录 (.+))?$")

ORDER = [tag for tag, _ in STAGES]


class Progress:
    """累积一份运行的进度状态，并给出可推送的事件。"""

    def __init__(self) -> None:
        self.stages = {tag: "idle" for tag, _ in STAGES}
        self.extras: set[str] = set()
        self.title = ""
        # 子进度：抽帧与归纳这类「块内 x/y」，决定进度条在当前阶段的推进比例
        self.sub_label = ""
        self.sub_cur = 0
        self.sub_total = 0
        self.done = False
        self.out_dir = ""

    # ---------------------------------------------------------------- 解析

    def feed(self, line: str) -> dict | None:
        """吃一行日志，返回一条事件；无状态变化时返回 None。"""
        line = line.rstrip("\r\n")
        if not line.strip():
            return None

        m = RE_OK.search(line)
        if m:
            self.done = True
            self.sub_label, self.sub_cur, self.sub_total = "", 0, 0
            self.out_dir = (m.group(2) or "").strip()
            return {"kind": "ok", "text": m.group(1), "out_dir": self.out_dir}

        m = RE_FRAME.search(line)
        if m:
            return self._sub("S5", f"帧 {m.group(1)}/{m.group(2)}",
                             int(m.group(1)), int(m.group(2)))

        m = RE_MAP.search(line)
        if m:
            return self._sub("S7", f"归纳 {m.group(1)}/{m.group(2)} 块",
                             int(m.group(1)), int(m.group(2)))

        m = RE_DONE.match(line)
        if m:
            tag = m.group(1)
            self.stages[tag] = "done"
            self.sub_label, self.sub_cur, self.sub_total = "", 0, 0
            return {"kind": "stage_done", "stage": tag,
                    "label": dict(STAGES).get(tag, tag), "elapsed": m.group(2).strip(),
                    "stages": dict(self.stages)}

        if RE_TITLE.match(line):
            self.title = RE_TITLE.match(line).group(1).strip()
            return {"kind": "title", "text": self.title}

        m = RE_START.match(line)
        if m:
            tag = m.group(1)
            self._skip_earlier(tag)
            if tag in EXTRA_STAGES:
                self.extras.add(tag)
            self.stages[tag] = "running"
            self.sub_label, self.sub_cur, self.sub_total = "", 0, 0
            return {"kind": "stage_start", "stage": tag, "label": m.group(2).strip(),
                    "stages": dict(self.stages)}

        return None

    def _skip_earlier(self, tag: str) -> None:
        """阶段按 S1→S8 严格顺序执行，某阶段开始即意味着它之前的阶段不会再跑。

        不这么做会有个真实后果：纯音频没有抽帧，S5 永远停在 idle，
        进度计算在 S5 处中断，整个归纳阶段（可能几十秒）进度条定格不动。
        """
        idx = ORDER.index(tag)
        for prev in ORDER[:idx]:
            if self.stages[prev] == "idle":
                self.stages[prev] = "skip"

    def _sub(self, tag: str, label: str, cur: int, total: int) -> dict:
        self.stages[tag] = "running"
        self.sub_label = label
        self.sub_cur, self.sub_total = cur, total
        return {"kind": "sub", "stage": tag, "label": label, "cur": cur, "total": total}

    # ---------------------------------------------------------------- 汇总

    def finish(self, code: int) -> None:
        """进程结束后收口：把未完成的阶段标成 skip，避免 UI 上永远转圈。"""
        for tag, _ in STAGES:
            st = self.stages[tag]
            if st == "running":
                self.stages[tag] = "done" if code == 0 else "skip"
            elif st == "idle":
                self.stages[tag] = "skip"

    @property
    def denominator(self) -> int:
        """进度分母：常态阶段 + 本次实际出现的条件阶段。"""
        return len(CORE_STAGES) + len(self.extras)

    @property
    def pct(self) -> float:
        acc = 0
        for tag in CORE_STAGES + sorted(self.extras):
            st = self.stages[tag]
            if st in ("done", "skip"):
                acc += 1
            elif st == "running":
                frac = (self.sub_cur / self.sub_total) if self.sub_total else 0.0
                return min(99.0, (acc + min(0.9, frac)) / self.denominator * 100)
            else:
                break
        if self.done:
            return 100.0
        return min(99.0, acc / self.denominator * 100)

    def snapshot(self) -> dict:
        return {
            "stages": self.stages,
            "extras": sorted(self.extras),
            "title": self.title,
            "sub": {"label": self.sub_label, "cur": self.sub_cur, "total": self.sub_total},
            "pct": round(self.pct, 1),
            "done": self.done,
            "out_dir": self.out_dir,
        }
