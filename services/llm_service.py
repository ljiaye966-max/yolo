from __future__ import annotations

import json
import os
from typing import Any


SYSTEM_PROMPT = """你是一个用于教学原型的安全数据分析助手。只根据提供的结构化 JSON 总结统计结果。不得推断人员身份、事故原因、责任归属或现场真实违规；所有事件只能称为疑似违规并建议人工复核。若字段缺失，明确说明未提供。不要输出 API Key。"""


def generate_llm_report(data: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return {"mode": "deterministic_fallback", "text": _fallback(data)}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None); response = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.1, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(data, ensure_ascii=False)}])
        return {"mode": "llm", "text": response.choices[0].message.content or "模型未返回内容"}
    except Exception as exc: return {"mode": "fallback_after_error", "text": _fallback(data), "error_type": type(exc).__name__}


def _fallback(data: dict[str, Any]) -> str:
    total = sum(int(item.get("count", 0)) for item in (data.get("event_summary") or [])); return f"本次结构化记录包含 {total} 条事件。事件仅表示模型产生的疑似结果，需结合原始图像、时间连续性和现场人员复核；未提供的信息不作推断。"
