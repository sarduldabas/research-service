"""Thin LLM layer: Azure OpenAI / Microsoft Foundry (Responses API) or a deterministic mock.

Everything is configured through environment variables (see deckpilot-deploy/.env.example):
  LLM_PROVIDER            mock | azure
  AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT   connection + default model
  <ROLE>_MODEL            optional model override per agent role (INTAKE, RESEARCH, SEARCH, BUILDER)
  LLM_REASONING_EFFORT    default reasoning effort (none | low | medium | high ...); blank = model default
  <ROLE>_REASONING_EFFORT optional override per agent role
  LLM_TIMEOUT             seconds per model call (default 180)
  LLM_MAX_OUTPUT_TOKENS   optional cap on output tokens

The agent loop lives here: call the model, validate the JSON against a Pydantic schema,
and on failure feed the validation error back and retry.
"""
import json
import logging
import os
import re
import time
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger("llm")
T = TypeVar("T", bound=BaseModel)

DEFAULT_ROLE = "RESEARCH"
_NO_EFFORT = {"", "default", "unset", "off"}


class LLMError(RuntimeError):
    pass


def _env(name: str) -> str:
    """Environment value without whitespace or accidental surrounding quotes (common in .env files)."""
    return os.getenv(name, "").strip().strip("\"'").strip()


def provider() -> str:
    return (_env("LLM_PROVIDER") or "mock").lower()


_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            base_url=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_env("AZURE_OPENAI_API_KEY"),
            timeout=float(_env("LLM_TIMEOUT") or "180"),
            max_retries=2,
        )
    return _client


def deployment(role: str | None = None) -> str:
    """Model/deployment name for a role: <ROLE>_MODEL if set, otherwise AZURE_OPENAI_DEPLOYMENT."""
    role = (role or DEFAULT_ROLE).upper()
    name = _env(f"{role}_MODEL") or _env("AZURE_OPENAI_DEPLOYMENT")
    if not name:
        raise LLMError("no model configured: set AZURE_OPENAI_DEPLOYMENT (or %s_MODEL)" % role)
    return name


def effort(role: str | None = None) -> str:
    role = (role or DEFAULT_ROLE).upper()
    value = _env(f"{role}_REASONING_EFFORT") or _env("LLM_REASONING_EFFORT")
    return "" if value.lower() in _NO_EFFORT else value.lower()


def reasoning_kwargs(role: str | None = None) -> dict:
    """Reasoning effort for GPT-5/GPT-6 style models; empty for models without reasoning."""
    value = effort(role)
    return {"reasoning": {"effort": value}} if value else {}


def _call_kwargs(role: str | None) -> dict:
    kwargs = reasoning_kwargs(role)
    cap = _env("LLM_MAX_OUTPUT_TOKENS")
    if cap.isdigit():
        kwargs["max_output_tokens"] = int(cap)
    return kwargs


def _log_usage(role, model, started, reply):
    usage = getattr(reply, "usage", None)
    log.info(
        "llm role=%s model=%s effort=%s %.1fs tokens in=%s out=%s",
        (role or DEFAULT_ROLE), model, effort(role) or "default", time.monotonic() - started,
        getattr(usage, "input_tokens", "?"), getattr(usage, "output_tokens", "?"),
    )


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates ``` fences / prose)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model reply")
    return json.loads(text[start : end + 1])


def ask_json(system: str, user: str, model_cls: Type[T], max_attempts: int = 3, role: str | None = None) -> T:
    """Agent loop: ask -> validate -> (feed error back -> retry)."""
    system = system.replace("{schema}", json.dumps(model_cls.model_json_schema()))
    prompt = user
    last_error = "unknown"
    for attempt in range(1, max_attempts + 1):
        started = time.monotonic()
        model = deployment(role)
        try:
            reply = client().responses.create(
                model=model, instructions=system, input=prompt, **_call_kwargs(role)
            )
        except Exception as exc:  # network / auth / quota
            raise LLMError(f"LLM call failed ({model}): {exc}") from exc
        _log_usage(role, model, started, reply)
        try:
            if getattr(reply, "status", "") == "incomplete":
                raise ValueError("reply was cut off (output limit reached); be more concise")
            return model_cls.model_validate(extract_json(reply.output_text))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:600]
            log.warning("attempt %s returned invalid JSON: %s", attempt, last_error)
            prompt = (
                f"{user}\n\nYour previous reply was rejected: {last_error}\n"
                "Return ONLY one valid JSON object that matches the schema."
            )
    raise LLMError(f"model did not return valid JSON after {max_attempts} attempts: {last_error}")


def web_search(prompt: str, role: str = "SEARCH") -> tuple[str, list[dict]]:
    """Grounded answer via the Foundry/Azure OpenAI `web_search` tool (Responses API).

    Returns (answer_text, [{"title":..., "url":...}, ...]) built from url_citation annotations.
    """
    model = deployment(role)
    started = time.monotonic()
    try:
        reply = client().responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=prompt,
            include=["web_search_call.action.sources"],
            **_call_kwargs(role),
        )
    except Exception as exc:
        raise LLMError(f"web search call failed ({model}): {exc}") from exc
    _log_usage(role, model, started, reply)

    sources: list[dict] = []
    seen: set[str] = set()
    for item in reply.output:
        if getattr(item, "type", "") != "message":
            continue
        for part in getattr(item, "content", None) or []:
            for ann in getattr(part, "annotations", None) or []:
                if getattr(ann, "type", "") == "url_citation" and ann.url not in seen:
                    seen.add(ann.url)
                    sources.append({"title": getattr(ann, "title", "") or ann.url, "url": ann.url})
    return reply.output_text, sources
