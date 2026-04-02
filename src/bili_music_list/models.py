from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class VideoItem:
    favorite_id: int
    favorite_title: str
    video_id: int
    bvid: str
    title: str
    intro: str
    page_title: str
    description: str
    uploader: str
    url: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SongCandidate:
    song_title: str
    artist: Optional[str]
    confidence: float
    reason: str
    source_text: str
    extractor: str

    def key(self) -> tuple[str, str]:
        return (self.song_title.strip().lower(), (self.artist or "").strip().lower())


@dataclass
class SongMatch:
    song_title: str
    artist: Optional[str]
    confidence: float
    extractor: str
    reason: str
    bvid: str
    video_title: str
    video_url: str
    uploader: str
    source_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
