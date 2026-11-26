"""大模型调用层：本地 Ollama 与任意 OpenAI 兼容端点共用一套接口。

切换方式只有一个配置项 `llm.provider`，调用方不感知差异。
"""

from __future__ import annotations

import json
import os
import re
import socket
from urllib import error as urlerr
from urllib import request as urlreq

from .util import log, warn


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["llm"]
        self.provider = self.cfg["provider"]
        self.timeout = int(self.cfg["timeout"])
        self.calls = 0

    # ---- 对外接口 -------------------------------------------------
    def chat(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        self.calls += 1
        if self.provider == "ollama":
            return self._ollama(prompt, system, json_mode)
        return self._openai(prompt, system, json_mode)

    def chat_json(self, prompt: str, system: str | None = None) -> dict:
        """要求模型返回 JSON，容忍代码块包裹；解析失败时让模型自修一次。"""
        raw = self.chat(prompt, system, json_mode=True)
        obj = _extract_json(raw)
        if obj is not None:
            return obj
        warn("模型返回非 JSON，尝试让模型重新修正一次")
        fix = (
            "把下面的内容改写成合法 JSON，只输出 JSON 本身，不要任何解释或代码块：\n"
            + raw[:4000]
        )
        obj = _extract_json(self.chat(fix, json_mode=True))
        if obj is None:
            raise LLMError(f"模型两次都未返回合法 JSON：{raw[:200]}")
        return obj

    def probe(self) -> tuple[bool, str]:
        try:
            text = self.chat("只回答两个字：正常", json_mode=False)
            return ("正常" in text or bool(text.strip())), text.strip()[:40]
        except LLMError as exc:
            return False, str(exc)

    # ---- 后端实现 -------------------------------------------------
    def _ollama(self, prompt: str, system: str | None, json_mode: bool) -> str:
        c = self.cfg["ollama"]
        body: dict = {
            "model": c["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(self.cfg["temperature"]),
                "num_ctx": int(c["num_ctx"]),
            },
        }
        if system:
            body["system"] = system
        if json_mode:
            body["format"] = "json"
        data = self._post(f"{c['base_url'].rstrip('/')}/api/generate", body, headers={})
        try:
            return data["response"]
        except (KeyError, TypeError):
            raise LLMError(f"Ollama 返回结构异常：{str(data)[:200]}")

    def _openai(self, prompt: str, system: str | None, json_mode: bool) -> str:
        c = self.cfg["openai"]
        key_env = c["api_key_env"]
        key = os.environ.get(key_env, "")
        if not key:
            raise LLMError(f"环境变量 {key_env} 未设置")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict = {
            "model": c["model"],
            "messages": messages,
            "temperature": float(self.cfg["temperature"]),
            "max_tokens": int(c["max_tokens"]),
        }
        if json_mode and c["json_mode"]:
            body["response_format"] = {"type": "json_object"}

        data = self._post(
            f"{c['base_url'].rstrip('/')}/chat/completions",
            body,
            headers={"Authorization": f"Bearer {key}"},
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"接口返回结构异常：{str(data)[:200]}")

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urlreq.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urlreq.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urlerr.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"接口返回 {exc.code}：{detail}")
        except socket.timeout:
            raise LLMError(f"请求超时（{self.timeout}s）：{url}。长文本请调大 llm.timeout")
        except urlerr.URLError as exc:
            hint = ("本地模型服务未启动？执行 `ollama serve` 或打开 Ollama 应用"
                    if self.provider == "ollama" else "检查网络或 base_url")
            raise LLMError(f"无法连接 {url}：{exc.reason}。{hint}")
        except json.JSONDecodeError:
            raise LLMError(f"响应不是 JSON：{url}")


def _extract_json(text: str) -> dict | None:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
