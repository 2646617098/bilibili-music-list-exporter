from __future__ import annotations

import json
from typing import Any, Optional

import requests

from .models import SongCandidate, VideoItem


class LlmSongParser:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def extract(self, video: VideoItem) -> list[SongCandidate]:
        parsed = self._chat_json(
            system_prompt=(
                "你是一个音乐信息抽取助手。"
                "请从给定的 B 站视频信息里抽取歌曲名和歌手。"
                "只返回 JSON。"
                "格式为 {\"songs\": [{\"song_title\": \"\", \"artist\": \"\", \"confidence\": 0.0, \"reason\": \"\"}]}。"
                "如果无法判断，返回 {\"songs\": []}。"
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
                "你是一个音乐列表清洗助手。"
                "请根据已有的粗解析结果和来源视频标题，输出标准化后的歌曲名和歌手。"
                "尽量去掉无关描述、作词作曲信息、设备信息、括号噪声、节目名。"
                "只返回 JSON，格式为 "
                "{\"song_title\": \"\", \"artist\": \"\", \"confidence\": 0.0, \"reason\": \"\"}。"
                "如果无法判断歌手，artist 设为空字符串。"
            ),
            user_payload=row,
        )
        return {
            "song_title": (parsed.get("song_title") or row.get("song_title") or "").strip(),
            "artist": (parsed.get("artist") or "").strip(),
            "confidence": float(parsed.get("confidence") or row.get("confidence") or 0.7),
            "reason": (parsed.get("reason") or "llm normalized").strip(),
        }

    def _chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
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
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response is not a JSON object")
        return parsed
