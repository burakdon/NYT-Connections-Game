"""Minimal Claude Messages API client using Python's standard library."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-6"
TEMPERATURE_DEPRECATED_MODEL_PREFIXES = ("claude-opus-4-7",)


class ClaudeError(RuntimeError):
    """Raised when the Claude API request fails."""


def load_env_files() -> None:
    """Load .env-style files without adding a third-party dependency."""

    for name in (".env.local", ".env"):
        path = ROOT_DIR / name
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)


def create_ssl_context() -> ssl.SSLContext:
    """Create an HTTPS context that works with python.org macOS builds."""

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def model_accepts_temperature(model: str) -> bool:
    """Return false for Claude models that reject the temperature parameter."""

    return not any(model.startswith(prefix) for prefix in TEMPERATURE_DEPRECATED_MODEL_PREFIXES)


def build_messages_payload(
    *,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float | None,
    model: str,
) -> dict[str, Any]:
    """Build a Claude Messages API payload with model-specific parameters."""

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    if temperature is not None and model_accepts_temperature(model):
        payload["temperature"] = temperature

    return payload


def create_request(payload: dict[str, Any], api_key: str) -> urllib.request.Request:
    """Create a Claude Messages API request."""

    return urllib.request.Request(
        ANTHROPIC_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )


def call_claude(
    *,
    system: str,
    user: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    model: str | None = None,
    timeout: int = 120,
) -> str:
    """Call Claude and return the combined text response."""

    load_env_files()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ClaudeError("Missing ANTHROPIC_API_KEY. Add it to .env.local or your shell.")

    selected_model = model or os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
    payload = build_messages_payload(
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
        model=selected_model,
    )
    request = create_request(payload, api_key)

    try:
        with urllib.request.urlopen(request, timeout=timeout, context=create_ssl_context()) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        if "temperature" in payload and "temperature" in body and "deprecated" in body:
            payload = dict(payload)
            payload.pop("temperature", None)
            retry_request = create_request(payload, api_key)
            try:
                with urllib.request.urlopen(
                    retry_request,
                    timeout=timeout,
                    context=create_ssl_context(),
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as retry_error:
                retry_body = retry_error.read().decode("utf-8", errors="replace")
                raise ClaudeError(f"Claude API returned {retry_error.code}: {retry_body}") from retry_error
            except urllib.error.URLError as retry_error:
                raise ClaudeError(f"Could not reach Claude API: {retry_error}") from retry_error
        else:
            raise ClaudeError(f"Claude API returned {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise ClaudeError(f"Could not reach Claude API: {error}") from error

    blocks = data.get("content", [])
    text_blocks = [block.get("text", "") for block in blocks if block.get("type") == "text"]
    text = "\n".join(block for block in text_blocks if block).strip()

    if not text:
        raise ClaudeError(f"Claude returned no text content: {data}")

    return text
