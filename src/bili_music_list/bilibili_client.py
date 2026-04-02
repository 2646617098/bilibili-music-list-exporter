from __future__ import annotations

from http.cookies import SimpleCookie
from typing import Any, Optional

import requests

from .models import VideoItem


class BilibiliClient:
    def __init__(self, cookie: Optional[str] = None, timeout: int = 20) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/135.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.bilibili.com/",
            }
        )
        if cookie:
            self.session.headers["Cookie"] = cookie
            self.session.headers["X-CSRFToken"] = self._extract_cookie_value(cookie, "bili_jct")
        self.timeout = timeout

    @staticmethod
    def _extract_cookie_value(cookie_text: str, name: str) -> str:
        cookie = SimpleCookie()
        cookie.load(cookie_text)
        morsel = cookie.get(name)
        return morsel.value if morsel else ""

    def fetch_favorite_videos(
        self,
        media_id: int,
        fetch_detail: bool = False,
    ) -> tuple[str, list[VideoItem], list[dict[str, str]]]:
        videos: list[VideoItem] = []
        failed_videos: list[dict[str, str]] = []
        page = 1
        favorite_title = ""
        while True:
            payload = self._get_json(
                "https://api.bilibili.com/x/v3/fav/resource/list",
                params={
                    "media_id": media_id,
                    "pn": page,
                    "ps": 20,
                    "order": "mtime",
                    "type": 0,
                    "tid": 0,
                    "platform": "web",
                },
            )
            data = payload.get("data") or {}
            info = data.get("info") or {}
            favorite_title = info.get("title") or favorite_title or str(media_id)
            medias = data.get("medias") or []
            if not medias:
                break
            for media in medias:
                bvid = media.get("bvid")
                if not bvid:
                    continue
                detail = {}
                detail_error = None
                if fetch_detail:
                    detail, detail_error = self.fetch_video_detail(bvid)
                if detail_error:
                    failed_videos.append(
                        {
                            "bvid": bvid,
                            "video_title": media.get("title") or "",
                            "uploader": (media.get("upper") or {}).get("name") or "",
                            "video_url": f"https://www.bilibili.com/video/{bvid}",
                            "error": detail_error,
                        }
                    )
                videos.append(
                    VideoItem(
                        favorite_id=media_id,
                        favorite_title=favorite_title,
                        video_id=media.get("id") or 0,
                        bvid=bvid,
                        title=media.get("title") or "",
                        intro=media.get("intro") or "",
                        page_title=detail.get("title") or media.get("title") or "",
                        description=detail.get("desc") or media.get("intro") or "",
                        uploader=(detail.get("owner") or {}).get("name")
                        or (media.get("upper") or {}).get("name")
                        or "",
                        url=f"https://www.bilibili.com/video/{bvid}",
                        raw={"favorite": media, "detail": detail},
                    )
                )
            if not data.get("has_more"):
                break
            page += 1
        return favorite_title, videos, failed_videos

    def fetch_video_detail(self, bvid: str) -> tuple[dict[str, Any], Optional[str]]:
        try:
            payload = self._get_json(
                "https://api.bilibili.com/x/web-interface/view",
                params={"bvid": bvid},
            )
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            print(f"[warn] skip video detail {bvid}: HTTP {status_code}")
            return {}, f"HTTP {status_code}"
        except RuntimeError as exc:
            print(f"[warn] skip video detail {bvid}: {exc}")
            return {}, str(exc)
        return payload.get("data") or {}, None

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        code = payload.get("code", -1)
        if code != 0:
            message = payload.get("message") or payload.get("msg") or "unknown error"
            raise RuntimeError(f"Bilibili API error {code}: {message}")
        return payload
