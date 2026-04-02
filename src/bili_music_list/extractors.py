from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Optional

from .models import SongCandidate, VideoItem

BRACKETS_PATTERN = re.compile(r"[【\[].*?[】\]]")
MV_MARKERS_PATTERN = re.compile(
    r"(?i)\b(official\s+video|official\s+mv|mv|m/v|music\s+video|中字|歌词版|完整版|4k|hd|p\d+)\b"
)
SONG_IN_QUOTES_PATTERN = re.compile(r"[《「“\"](?P<title>[^》」”\"]{1,80})[》」”\"]")
ARTIST_HINT_PATTERN = re.compile(
    r"(?i)(?:by|cover|artist|feat\.?|ft\.?|演唱|歌手|原唱|翻唱|曲：|词：)\s*[:：]?\s*(?P<artist>[^|/／,，;；]+)"
)
ARTIST_LABEL_PATTERN = re.compile(r"^(?:演唱|歌手|原唱|翻唱|artist|by|cover)\s*$", re.IGNORECASE)
SEPARATOR_PATTERN = re.compile(r"\s*(?:[-|｜/／]|—|–|:|：|·)\s*")
NOISE_PATTERN = re.compile(r"\s+")


def extract_song_candidates(video: VideoItem) -> list[SongCandidate]:
    texts = _candidate_texts(video)
    candidates: list[SongCandidate] = []
    seen: set[tuple[str, str]] = set()
    global_artist = _first_artist_hint(text for _, text in texts)
    for label, text in texts:
        if not text:
            continue
        for candidate in _extract_from_text(text, label):
            if not candidate.artist and global_artist:
                candidate.artist = global_artist
                candidate.reason = f"{candidate.reason}; artist filled from other fields"
            if candidate.key() in seen:
                continue
            seen.add(candidate.key())
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.confidence, reverse=True)


def _candidate_texts(video: VideoItem) -> Iterable[tuple[str, str]]:
    return (
        ("title", video.title),
        ("intro", video.intro),
        ("page_title", video.page_title),
        ("description", video.description),
    )


def _extract_from_text(text: str, label: str) -> list[SongCandidate]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    candidates: list[SongCandidate] = []

    quoted = SONG_IN_QUOTES_PATTERN.search(normalized)
    if quoted:
        title = _cleanup_title(quoted.group("title"))
        if title:
            candidates.append(
                SongCandidate(
                    song_title=title,
                    artist=_extract_artist(normalized),
                    confidence=0.96 if label == "title" else 0.88,
                    reason=f"{label} contains quoted song title",
                    source_text=text,
                    extractor="heuristic",
                )
            )

    parts = [part.strip() for part in SEPARATOR_PATTERN.split(normalized) if part.strip()]
    if len(parts) >= 2:
        title_artist = _from_split_parts(parts, label, text)
        if title_artist:
            candidates.append(title_artist)

    if not candidates:
        fallback = _extract_fallback(normalized, label, text)
        if fallback:
            candidates.append(fallback)
    return candidates


def _from_split_parts(parts: list[str], label: str, source_text: str) -> Optional[SongCandidate]:
    first, second = parts[0], parts[1]
    if ARTIST_LABEL_PATTERN.fullmatch(first):
        artist, title = second, None
    elif _looks_like_artist(first) and not _looks_like_artist(second):
        artist, title = first, second
    else:
        title, artist = first, second if _looks_like_artist(second) or _looks_like_probable_artist(second) else None
    if title is None:
        return None
    title = _cleanup_title(title)
    artist = _cleanup_artist(artist) if artist else _extract_artist(source_text)
    if not title or _looks_like_noise(title):
        return None
    return SongCandidate(
        song_title=title,
        artist=artist,
        confidence=0.89 if label == "title" else 0.77,
        reason=f"{label} split by common separators",
        source_text=source_text,
        extractor="heuristic",
    )


def _extract_fallback(text: str, label: str, source_text: str) -> Optional[SongCandidate]:
    cleaned = _cleanup_title(text)
    if not cleaned or _looks_like_noise(cleaned):
        return None
    return SongCandidate(
        song_title=cleaned,
        artist=_extract_artist(source_text),
        confidence=0.55 if label == "title" else 0.35,
        reason=f"{label} fallback title guess",
        source_text=source_text,
        extractor="heuristic",
    )


def _normalize_text(text: str) -> str:
    text = BRACKETS_PATTERN.sub(" ", text)
    text = MV_MARKERS_PATTERN.sub(" ", text)
    text = NOISE_PATTERN.sub(" ", text).strip()
    return text


def _extract_artist(text: str) -> Optional[str]:
    match = ARTIST_HINT_PATTERN.search(text)
    if not match:
        return None
    return _cleanup_artist(match.group("artist"))


def _cleanup_title(text: str) -> str:
    cleaned = text.strip(" -|｜/／—–:：·")
    cleaned = cleaned.replace("投稿", "").strip()
    cleaned = NOISE_PATTERN.sub(" ", cleaned)
    return cleaned


def _cleanup_artist(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = NOISE_PATTERN.sub(" ", text.strip(" -|｜/／—–:：·"))
    return cleaned or None


def _looks_like_artist(text: str) -> bool:
    text = text.lower()
    return any(token in text for token in ("feat", "ft.", "cover", "翻唱", "演唱", "artist", "by "))


def _looks_like_probable_artist(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if len(text) > 24:
        return False
    if any(token in text for token in (" ", "乐队", "band", "团", "周杰伦", "五月天")):
        return True
    if re.fullmatch(r"[A-Za-z0-9._\u4e00-\u9fff]+", text) is not None:
        return True
    return False


def _looks_like_noise(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("收藏夹", "合集", "混剪", "歌单", "播放列表"))


def _first_artist_hint(texts: Iterable[str]) -> Optional[str]:
    for text in texts:
        artist = _extract_artist(text)
        if artist:
            return artist
    return None
