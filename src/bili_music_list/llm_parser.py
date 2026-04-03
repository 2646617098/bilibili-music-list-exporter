from __future__ import annotations

import json
import time
from typing import Any, Optional

import requests

from .models import SongCandidate, VideoItem


class LlmSongParser:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 60,
        retries: int = 3,
        delay_ms: int = 800,
        max_tokens: int = 1200,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = max(retries, 1)
        self.delay_ms = max(delay_ms, 0)
        self.max_tokens = max(max_tokens, 128)

    def extract(self, video: VideoItem) -> list[SongCandidate]:
        parsed = self._chat_json(
            system_prompt=(
                "抽取歌曲名和歌手。"
                "只返回 JSON："
                "{\"songs\":[{\"song_title\":\"\",\"artist\":\"\",\"confidence\":0.0,\"reason\":\"\"}]}。"
                "无法判断返回 {\"songs\":[]}。"
            ),
            user_payload={
                "title": video.title,
                "intro": video.intro,
                "page_title": video.page_title,
                "description": video.description,
                "uploader": video.uploader,
                "url": video.url,
            },
        )
        items = parsed.get("songs", []) if isinstance(parsed, dict) else []
        candidates: list[SongCandidate] = []
        for item in items:
            title = (item.get("song_title") or "").strip()
            if not title:
                continue
            candidates.append(
                SongCandidate(
                    song_title=title,
                    artist=(item.get("artist") or "").strip() or None,
                    confidence=float(item.get("confidence") or 0.7),
                    reason=(item.get("reason") or "llm parsed").strip(),
                    source_text=video.title,
                    extractor="llm",
                )
            )
        return candidates

    def normalize_existing_song(self, row: dict[str, str]) -> dict[str, Any]:
        parsed = self._chat_json(
            system_prompt=(
                "清洗歌曲名和歌手。"
                "删除节目名、live、mv、歌词版、括号噪声、无关描述。"
                "只返回 JSON："
                "{\"song_title\":\"\",\"artist\":\"\",\"confidence\":0.0,\"reason\":\"\"}。"
                "歌手不确定时 artist 置空。"
            ),
            user_payload=row,
        )
        return {
            "song_title": (parsed.get("song_title") or row.get("song_title") or "").strip(),
            "artist": (parsed.get("artist") or "").strip(),
            "confidence": float(parsed.get("confidence") or row.get("confidence") or 0.7),
            "reason": (parsed.get("reason") or "llm normalized").strip(),
        }

    def normalize_existing_songs(self, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        parsed = self._chat_json(
            system_prompt=(
                "批量清洗歌曲名和歌手。"
                "删除节目名、live、mv、歌词版、括号噪声、无关描述。"
                "只返回 JSON："
                "{\"items\":[{\"index\":0,\"song_title\":\"\",\"artist\":\"\",\"confidence\":0.0,\"reason\":\"\"}]}。"
                "index 对应输入下标。歌手不确定时 artist 置空。"
            ),
            user_payload={"items": rows},
        )
        return _normalize_batch_result(parsed, rows, default_reason="llm normalized")

    def normalize_for_simple_output(self, row: dict[str, str]) -> dict[str, Any]:
        parsed = self._chat_json(
            system_prompt=(
                "严格清洗歌曲名和歌手。"
                "只保留真实歌名和主要歌手。"
                "删除翻唱说明、live、mv、歌词版、节目名、合集名、版本描述、括号内容、up主信息。"
                "不确定歌名则 song_title 置空；不确定歌手则 artist 置空。"
                "只返回 JSON："
                "{\"song_title\":\"\",\"artist\":\"\",\"confidence\":0.0,\"reason\":\"\"}。"
            ),
            user_payload=row,
        )
        return {
            "song_title": (parsed.get("song_title") or "").strip(),
            "artist": (parsed.get("artist") or "").strip(),
            "confidence": float(parsed.get("confidence") or row.get("confidence") or 0.7),
            "reason": (parsed.get("reason") or "llm simple normalized").strip(),
        }

    def normalize_for_simple_output_batch(self, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        parsed = self._chat_json(
            system_prompt=(
                "批量严格清洗歌曲名和歌手。"
                "只保留真实歌名和主要歌手。"
                "删除翻唱说明、live、mv、歌词版、节目名、合集名、版本描述、括号内容、up主信息。"
                "不确定歌名则 song_title 置空；不确定歌手则 artist 置空。"
                "只返回 JSON："
                "{\"items\":[{\"index\":0,\"song_title\":\"\",\"artist\":\"\",\"confidence\":0.0,\"reason\":\"\"}]}。"
                "index 对应输入下标。"
            ),
            user_payload={"items": rows},
        )
        return _normalize_batch_result(parsed, rows, default_reason="llm simple normalized")

    def _chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        content = self._request_content(payload)
        parsed = parse_json_content(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response is not a JSON object")
        return parsed

    def _request_content(self, payload: dict[str, Any]) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                content = self._single_chat_request(payload)
                if not content.strip():
                    # Some OpenAI-compatible gateways ignore `response_format`
                    # and return empty `content`. Retry once without it.
                    fallback_payload = dict(payload)
                    fallback_payload.pop("response_format", None)
                    fallback_payload["messages"] = [
                        *fallback_payload["messages"],
                        {
                            "role": "user",
                            "content": "只输出JSON对象，不要解释，不要markdown代码块。",
                        },
                    ]
                    content = self._single_chat_request(fallback_payload)
                if not content.strip():
                    raise ValueError("LLM response content is empty")
                if self.delay_ms:
                    time.sleep(self.delay_ms / 1000)
                return content
            except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                sleep_seconds = (self.delay_ms / 1000 if self.delay_ms else 0.5) * attempt
                time.sleep(max(sleep_seconds, 0.5))
        if last_error is None:
            raise RuntimeError("LLM request failed with unknown error")
        raise last_error

    def _single_chat_request(self, payload: dict[str, Any]) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(_content_part_to_text(item) for item in content)
        if isinstance(content, str) and content.strip():
            return content
        # Compatibility fallback for providers that populate non-standard fields.
        alt_fields = (
            message.get("text"),
            message.get("output_text"),
            choices[0].get("text"),
            choices[0].get("output_text"),
            message.get("reasoning_content"),
        )
        for value in alt_fields:
            if isinstance(value, str) and value.strip():
                return value
        return ""


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            inner = "\n".join(lines[1:-1]).strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("Unable to parse JSON content", text, 0)


def _content_part_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "value"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return ""


def _normalize_batch_result(
    parsed: dict[str, Any],
    rows: list[dict[str, str]],
    default_reason: str,
) -> list[dict[str, Any]]:
    items = parsed.get("items", []) if isinstance(parsed, dict) else []
    normalized_by_index: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(rows):
            row = rows[index]
            normalized_by_index[index] = {
                "song_title": (item.get("song_title") or row.get("song_title") or "").strip(),
                "artist": (item.get("artist") or "").strip(),
                "confidence": float(item.get("confidence") or row.get("confidence") or 0.7),
                "reason": (item.get("reason") or default_reason).strip(),
            }
    return [
        normalized_by_index.get(
            index,
            {
                "song_title": (row.get("song_title") or "").strip(),
                "artist": (row.get("artist") or "").strip(),
                "confidence": float(row.get("confidence") or 0.7),
                "reason": "batch fallback to original row",
            },
        )
        for index, row in enumerate(rows)
    ]
