"""LangChain-backed Ollama chat helpers for the TaoCrop web UI."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = "http://localhost:11434"

RoleName = Literal["user", "assistant", "system"]


@dataclass(slots=True)
class ChatTurn:
    role: RoleName
    content: str


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _request_json(
    path: str,
    *,
    base_url: str = OLLAMA_BASE_URL,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        _api_url(base_url, path),
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.URLError as exc:  # pragma: no cover - network failure
        raise ConnectionError(f"Could not reach Ollama at {base_url}") from exc
    if not payload.strip():
        return {}
    return json.loads(payload)


def list_models(*, base_url: str = OLLAMA_BASE_URL) -> list[dict[str, Any]]:
    payload = _request_json("/api/tags", base_url=base_url)
    models = payload.get("models") or []
    if not isinstance(models, list):
        return []
    return [model for model in models if isinstance(model, dict)]


def model_info(model: str, *, base_url: str = OLLAMA_BASE_URL) -> dict[str, Any]:
    payload = _request_json("/api/show", base_url=base_url, method="POST", body={"model": model})
    parameters = str(payload.get("parameters") or "")
    context_length = None
    match = re.search(r"num_ctx\s+(\d+)", parameters)
    if match:
        context_length = int(match.group(1))

    running = _request_json("/api/ps", base_url=base_url)
    for item in running.get("models", []) if isinstance(running.get("models"), list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("name") or item.get("model")) == model and item.get("context_length") is not None:
            context_length = int(item["context_length"])
            break

    return {
        "model": model,
        "context_length": context_length,
        "parameters": parameters,
        "capabilities": payload.get("capabilities") or [],
        "details": payload.get("details") or {},
        "raw": payload,
    }


def build_dashboard_context(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    monthly_features = result.get("monthly_features") or []
    feature_groups = []
    for group in result.get("feature_groups") or []:
        if not isinstance(group, dict):
            continue
        features = []
        for feature in (group.get("features") or [])[:2]:
            if not isinstance(feature, dict):
                continue
            series = feature.get("series") or []
            features.append(
                {
                    "label": feature.get("label"),
                    "description": feature.get("description"),
                    "latest_value": feature.get("latest_value"),
                    "recent": [item.get("value") for item in series[-3:] if isinstance(item, dict)],
                }
            )
        feature_groups.append(
            {
                "group": group.get("group"),
                "label": group.get("label"),
                "features": features,
            }
        )
        if len(feature_groups) >= 3:
            break
    return {
        "headline": result.get("headline") or {},
        "summary": {
            "best_model": (result.get("summary") or {}).get("best_model") or {},
            "holdout": (result.get("summary") or {}).get("holdout") or {},
        },
        "drivers": (result.get("drivers") or [])[:3],
        "feature_groups": feature_groups,
        "yield_series": [
            {
                "month_label": item.get("month_label"),
                "predicted_yield": item.get("predicted_yield"),
            }
            for item in (result.get("yield_series") or [])[-6:]
            if isinstance(item, dict)
        ],
        "monthly_features": [
            {
                "month_label": item.get("month_label"),
                "predicted_yield": item.get("predicted_yield"),
            }
            for item in monthly_features[-4:]
            if isinstance(item, dict)
        ],
        "feature_importance": [
            {
                "label": item.get("label"),
                "importance": item.get("importance"),
            }
            for item in (result.get("feature_importance") or [])[:5]
            if isinstance(item, dict)
        ],
    }


def _to_messages(turns: Iterable[dict[str, Any] | ChatTurn]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in turns:
        if isinstance(turn, ChatTurn):
            role = turn.role
            content = turn.content
        else:
            role = str(turn.get("role") or "user")
            content = str(turn.get("content") or "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def _system_prompt(dashboard_context: dict[str, Any]) -> str:
    context_blob = json.dumps(dashboard_context, ensure_ascii=False, default=str, separators=(",", ":"))
    return (
        "You are TaoCrop Chat, a practical farm assistant.\n"
        "Use the provided dashboard snapshot as the source of truth.\n"
        "Answer clearly in 2 to 5 sentences. Do not reply with one word.\n"
        "Avoid technical jargon unless the user asks for it, and explain model-related ideas in plain language.\n"
        "If the answer is not available in the snapshot, say what is missing instead of guessing.\n"
        "Dashboard snapshot:\n"
        f"{context_blob}"
    )


def _extract_stats(message: AIMessage) -> dict[str, Any]:
    usage = getattr(message, "usage_metadata", None) or {}
    response = getattr(message, "response_metadata", None) or {}
    input_tokens = usage.get("input_tokens") or response.get("prompt_eval_count")
    output_tokens = usage.get("output_tokens") or response.get("eval_count")
    prompt_duration = response.get("prompt_eval_duration")
    eval_duration = response.get("eval_duration")
    total_duration = response.get("total_duration")
    duration_ns = eval_duration or total_duration
    tokens_per_second = None
    if output_tokens and duration_ns:
        try:
            tokens_per_second = float(output_tokens) / (float(duration_ns) / 1_000_000_000.0)
        except ZeroDivisionError:
            tokens_per_second = None
    if tokens_per_second is not None and not (0 < tokens_per_second < 1000):
        tokens_per_second = None
    context_length = response.get("context_length") or response.get("num_ctx")
    return {
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "tokens_per_second": tokens_per_second,
        "prompt_duration_ns": prompt_duration,
        "eval_duration_ns": eval_duration,
        "total_duration_ns": total_duration,
        "context_length": context_length,
        "response_metadata": response,
        "usage_metadata": usage,
    }


def chat_with_ollama(
    *,
    model: str,
    messages: Iterable[dict[str, Any] | ChatTurn],
    dashboard_context: dict[str, Any] | None = None,
    base_url: str = OLLAMA_BASE_URL,
) -> dict[str, Any]:
    dashboard_context = dashboard_context or {}
    prompt_messages = [SystemMessage(content=_system_prompt(dashboard_context))]
    prompt_messages.extend(_to_messages(messages))
    chat = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.4,
        validate_model_on_init=False,
    )
    reply = chat.invoke(prompt_messages)
    if not isinstance(reply, AIMessage):
        content = getattr(reply, "content", str(reply))
        reply = AIMessage(content=str(content))
    text = getattr(reply, "text", None)
    extracted = text if isinstance(text, str) and text.strip() else ""
    if not extracted:
        content = reply.content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("content") or item.get("output")
                    if text_value:
                        parts.append(str(text_value))
            extracted = "\n".join(part for part in parts if part).strip()
        elif isinstance(content, str):
            extracted = content.strip()
        else:
            extracted = str(content).strip()
    if not extracted:
        extracted = "I could not generate a reply from the model."
    if extracted != reply.content:
        reply = AIMessage(
            content=extracted,
            response_metadata=reply.response_metadata,
            usage_metadata=reply.usage_metadata,
        )
    stats = _extract_stats(reply)
    return {
        "model": model,
        "reply": reply.content,
        "stats": stats,
    }
