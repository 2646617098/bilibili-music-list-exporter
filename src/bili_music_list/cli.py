from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

try:
    from .bilibili_client import BilibiliClient
    from .extractors import extract_song_candidates
    from .llm_parser import LlmSongParser
    from .models import SongMatch, VideoItem
except ImportError:
    # Support running as a standalone script (e.g. PyInstaller entry).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bili_music_list.bilibili_client import BilibiliClient
    from bili_music_list.extractors import extract_song_candidates
    from bili_music_list.llm_parser import LlmSongParser
    from bili_music_list.models import SongMatch, VideoItem


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(_resolve_cli_args(parser))

    if args.input_csv:
        run_ai_refine_mode(args)
        return

    cookie = load_cookie(args.cookie, args.cookie_file)
    client = BilibiliClient(cookie=cookie, timeout=args.timeout)
    favorite_title, videos, failed_videos = client.fetch_favorite_videos(
        args.media_id,
        fetch_detail=args.with_detail,
    )

    llm_parser = build_llm_parser(args, allow_fallback=True)
    effective_parser = args.parser
    if args.parser == "hybrid" and llm_parser is None:
        effective_parser = "heuristic"
        print("[warn] 未提供 LLM 参数，hybrid 已自动退回 heuristic")

    matches = build_song_matches(videos, effective_parser, llm_parser, args.include_unmatched)
    unique = dedupe_matches(matches, mode=args.dedupe_by)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        write_json(output_path, favorite_title, unique)
    else:
        write_csv(output_path, unique)
    simple_output_path = build_simple_output_path(output_path)
    simple_rows = build_simple_rows(unique, llm_parser)
    write_simple_song_list(simple_output_path, simple_rows)

    failed_path = build_failed_output_path(output_path)
    write_failed_videos(failed_path, favorite_title, failed_videos)

    print(f"收藏夹: {favorite_title}")
    print(f"视频数: {len(videos)}")
    print(f"歌曲数: {len(unique)}")
    print(f"失败详情数: {len(failed_videos)}")
    print(f"歌曲输出: {output_path.resolve()}")
    print(f"精简输出: {simple_output_path.resolve()}")
    print(f"失败列表: {failed_path.resolve()}")


def run_ai_refine_mode(args: argparse.Namespace) -> None:
    llm_parser = build_llm_parser(args, allow_fallback=False)
    input_path = Path(args.input_csv)
    output_path = Path(args.ai_output or build_ai_output_path(input_path))
    simple_output_path = build_simple_output_path(output_path)
    rows = read_csv_rows(input_path)

    def on_progress(processed: int, total: int, partial_rows: list[dict[str, str]]) -> None:
        deduped_rows = dedupe_refined_rows(partial_rows)
        write_refined_csv(output_path, deduped_rows)
        write_simple_song_list(simple_output_path, deduped_rows)
        print(f"[progress] 已完成 {processed}/{total} 行，已写入 {len(deduped_rows)} 条记录")

    refined_rows = refine_rows_with_ai(
        rows,
        llm_parser,
        batch_size=args.llm_batch_size,
        progress_callback=on_progress,
    )
    write_refined_csv(output_path, refined_rows)
    write_simple_song_list(simple_output_path, refined_rows)
    print(f"输入文件: {input_path.resolve()}")
    print(f"AI 输出: {output_path.resolve()}")
    print(f"精简输出: {simple_output_path.resolve()}")
    print(f"记录数: {len(refined_rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 Bilibili 收藏夹中导出歌曲列表，或对现有 CSV 做 AI 二次清洗")
    parser.add_argument("--media-id", type=int, help="收藏夹 media_id")
    parser.add_argument("--cookie", help="完整 Cookie 字符串，适用于私有收藏夹")
    parser.add_argument("--cookie-file", help="保存 Cookie 字符串的文本文件")
    parser.add_argument("--parser", choices=["heuristic", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--llm-base-url", help="兼容 OpenAI chat/completions 的基础地址，例如 https://api.openai.com/v1")
    parser.add_argument("--llm-api-key", help="LLM API Key")
    parser.add_argument("--llm-model", help="LLM 模型名")
    parser.add_argument("--llm-retries", type=int, default=3, help="LLM 请求失败时的重试次数")
    parser.add_argument("--llm-delay-ms", type=int, default=800, help="两次 LLM 请求之间的最小间隔（毫秒）")
    parser.add_argument("--llm-batch-size", type=int, default=20, help="CSV AI 清洗时每批发送给 LLM 的行数")
    parser.add_argument("--llm-max-tokens", type=int, default=1200, help="单次 LLM 返回的最大 token 数")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--with-detail", action="store_true", help="额外请求每个视频详情接口，可能更准但更容易触发风控")
    parser.add_argument("--include-unmatched", action="store_true", help="保留未识别歌曲的视频记录")
    parser.add_argument("--dedupe-by", choices=["title", "title-artist"], default="title-artist")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--output", default="output/music_list.csv")
    parser.add_argument("--input-csv", help="对现有 CSV 做 AI 二次解析")
    parser.add_argument("--ai-output", help="AI 二次解析后的输出路径")
    return parser


def _resolve_cli_args(parser: argparse.ArgumentParser) -> list[str]:
    raw_args = sys.argv[1:]
    if _is_frozen_executable():
        _print_frozen_parameter_guide()
    if raw_args:
        return raw_args
    if _is_frozen_executable():
        print("未检测到命令行参数。")
        print("请粘贴参数后回车，例如：--media-id 123 --parser heuristic --output output/music_list.csv")
        print("直接回车将显示帮助并退出。")
        typed = input("> ").strip()
        if not typed:
            parser.print_help()
            return []
        return shlex.split(typed, posix=False)
    return []


def _is_frozen_executable() -> bool:
    return bool(getattr(sys, "frozen", False))


def _pause_if_frozen() -> None:
    if _is_frozen_executable():
        try:
            input("\n按回车键退出...")
        except EOFError:
            pass


def _print_frozen_parameter_guide() -> None:
    print("=" * 72)
    print("bili-music-list 参数说明 / Parameter Guide")
    print("- 核心 / Core:")
    print("  --media-id <id>              要解析的收藏夹ID，进入网页版B站，然后进入对应收藏夹，url上面的fid=123456就是 / Favorite folder id")
    print("  --parser heuristic|llm|hybrid 解析模式 / Parse mode")
    print("  --output <path>              输出文件 / Output file")
    print("  --format csv|json            输出格式 / Output format")
    print("  --cookie <text>              B站Cookie / Bilibili cookie")
    print("  --cookie-file <path>         B站Cookie文件 / Cookie file")
    print("  --with-detail                拉详情 / Fetch detail API")
    print("  --include-unmatched          保留未识别项 / Keep unmatched")
    print("  --dedupe-by title|title-artist 去重方式 / Dedupe mode")
    print("- AI 清洗 / AI refine:")
    print("  --input-csv <path>           输入CSV / Input CSV")
    print("  --ai-output <path>           AI输出CSV / AI output CSV")
    print("  --llm-base-url <url>         模型接口地址 / API base url")
    print("  --llm-api-key <key>          模型密钥 / API key")
    print("  --llm-model <name>           模型名 / Model name")
    print("  --llm-batch-size <n>         批大小 / Batch size")
    print("  --llm-retries <n>            重试次数 / Retry count")
    print("  --llm-delay-ms <ms>          请求间隔 / Request delay")
    print("  --llm-max-tokens <n>         最大输出token / Max output tokens")
    print("  --timeout <sec>              请求超时秒数 / Timeout seconds")
    print("- 帮助 / Help:")
    print("  --help                       显示完整帮助 / Show full help")
    print("=" * 72)
    print("- 示例 / Example:")
    print("- 解析为csv文件，设置media-id即可，需要先将收藏夹设置为公开，或者附带上cookie参数")
    print("  --media-id 123456 --parser heuristic --output output/music_list.csv")
    print("- 如果csv中不准，还可以使用ai进一步解析csv文件，示例用的国内模型快速解析，只需要key换成自己的就行")
    print("  --input-csv output/music_list.csv --ai-output output/music_list_ai.csv --llm-base-url "
          "https://api.siliconflow.cn/v1 --llm-model Qwen/Qwen2.5-7B-Instruct --llm-batch-size "
          "8 --llm-retries 1 --llm-delay-ms 200 --llm-max-tokens 250 --timeout 45 --llm-api-key YOUR_KEY")
    print("=" * 72)


def build_llm_parser(args: argparse.Namespace, allow_fallback: bool) -> Optional[LlmSongParser]:
    if args.llm_base_url and args.llm_api_key and args.llm_model:
        return LlmSongParser(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            timeout=args.timeout,
            retries=args.llm_retries,
            delay_ms=args.llm_delay_ms,
            max_tokens=args.llm_max_tokens,
        )
    if allow_fallback:
        return None
    raise SystemExit("需要提供 --llm-base-url --llm-api-key --llm-model")


def load_cookie(cookie: Optional[str], cookie_file: Optional[str]) -> Optional[str]:
    if cookie:
        return cookie.strip()
    if cookie_file:
        return Path(cookie_file).read_text(encoding="utf-8").strip()
    return None


def build_song_matches(
    videos: list[VideoItem],
    parser_mode: str,
    llm_parser: Optional[LlmSongParser],
    include_unmatched: bool,
) -> list[SongMatch]:
    matches: list[SongMatch] = []
    for video in videos:
        candidates = []
        if parser_mode in {"heuristic", "hybrid"}:
            candidates.extend(extract_song_candidates(video))
        if parser_mode in {"llm", "hybrid"} and llm_parser is not None:
            try:
                candidates.extend(llm_parser.extract(video))
            except Exception as exc:
                print(f"[warn] LLM 解析失败 {video.bvid}: {exc}")

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        if candidates:
            best = candidates[0]
            matches.append(
                SongMatch(
                    song_title=best.song_title,
                    artist=best.artist,
                    confidence=best.confidence,
                    extractor=best.extractor,
                    reason=best.reason,
                    bvid=video.bvid,
                    video_title=video.title,
                    video_url=video.url,
                    uploader=video.uploader,
                    source_text=best.source_text,
                )
            )
        elif include_unmatched:
            matches.append(
                SongMatch(
                    song_title="[UNMATCHED]",
                    artist=None,
                    confidence=0.0,
                    extractor="none",
                    reason="no song detected",
                    bvid=video.bvid,
                    video_title=video.title,
                    video_url=video.url,
                    uploader=video.uploader,
                    source_text=video.title,
                )
            )
    return matches


def dedupe_matches(matches: list[SongMatch], mode: str) -> list[SongMatch]:
    bucket: dict[str, SongMatch] = {}
    for match in matches:
        key = match.song_title.strip().lower()
        if mode == "title-artist":
            key = f"{key}::{(match.artist or '').strip().lower()}"
        existing = bucket.get(key)
        if existing is None or match.confidence > existing.confidence:
            bucket[key] = match
    return sorted(bucket.values(), key=lambda item: (-item.confidence, item.song_title.lower()))


def write_csv(path: Path, matches: list[SongMatch]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "song_title",
                "artist",
                "confidence",
                "source_bvid",
                "source_video_title",
                "source_video_url",
                "uploader",
            ],
        )
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    "song_title": match.song_title,
                    "artist": match.artist or "",
                    "confidence": match.confidence,
                    "source_bvid": match.bvid,
                    "source_video_title": match.video_title,
                    "source_video_url": match.video_url,
                    "uploader": match.uploader,
                }
            )


def write_json(path: Path, favorite_title: str, matches: list[SongMatch]) -> None:
    payload = {
        "favorite_title": favorite_title,
        "songs": [match.to_dict() for match in matches],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_failed_output_path(output_path: Path) -> Path:
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        return output_path.with_name(f"{output_path.stem}_failed_videos.json")
    return output_path.with_name(f"{output_path.stem}_failed_videos.csv")


def build_simple_output_path(output_path: Path) -> Path:
    suffix = output_path.suffix.lower() or ".csv"
    return output_path.with_name(f"{output_path.stem}_simple{suffix}")


def build_simple_rows(
    matches: list[SongMatch],
    llm_parser: Optional[LlmSongParser],
) -> list[dict[str, str]]:
    rows = [
        {
            "song_title": match.song_title,
            "artist": match.artist or "",
            "confidence": str(match.confidence),
            "source_bvid": match.bvid,
            "source_video_title": match.video_title,
            "source_video_url": match.video_url,
            "uploader": match.uploader,
        }
        for match in matches
    ]
    if llm_parser is None:
        print("[warn] 未提供 LLM 参数，精简输出使用原始解析结果")
        return rows
    return refine_rows_for_simple_output(rows, llm_parser, batch_size=20)


def write_simple_song_list(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        for row in rows:
            title = (row.get("song_title") or "").strip()
            if not title:
                continue
            artist = (row.get("artist") or "").strip()
            line = f"{title} {artist}".strip() if artist else title
            file.write(f"{line}\n")


def write_failed_videos(path: Path, favorite_title: str, failed_videos: list[dict[str, str]]) -> None:
    if path.suffix.lower() == ".json":
        payload = {"favorite_title": favorite_title, "failed_videos": failed_videos}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["bvid", "video_title", "uploader", "video_url", "error"],
        )
        writer.writeheader()
        for item in failed_videos:
            writer.writerow(item)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def refine_rows_with_ai(
    rows: list[dict[str, str]],
    llm_parser: LlmSongParser,
    batch_size: int = 20,
    progress_callback: Optional[Callable[[int, int, list[dict[str, str]]], None]] = None,
) -> list[dict[str, str]]:
    refined: list[dict[str, str]] = []
    size = max(batch_size, 1)
    for batch_start in range(0, len(rows), size):
        batch = rows[batch_start : batch_start + size]
        try:
            normalized_batch = llm_parser.normalize_existing_songs(batch)
        except Exception as exc:
            print(f"[warn] AI 批量解析失败，第 {batch_start + 1}-{batch_start + len(batch)} 行: {exc}")
            normalized_batch = []
            for offset, row in enumerate(batch, start=1):
                try:
                    normalized_batch.append(llm_parser.normalize_existing_song(row))
                except Exception as item_exc:
                    print(f"[warn] AI 二次解析失败，第 {batch_start + offset} 行: {item_exc}")
                    normalized_batch.append(
                        {
                            "song_title": (row.get("song_title") or "").strip(),
                            "artist": (row.get("artist") or "").strip(),
                            "confidence": float(row.get("confidence") or 0),
                            "reason": "fallback to original row",
                        }
                    )
        for row, normalized in zip(batch, normalized_batch):
            refined.append(
                {
                    "song_title": normalized["song_title"],
                    "artist": normalized["artist"],
                    "confidence": normalized["confidence"],
                    "source_bvid": row.get("source_bvid") or "",
                    "source_video_title": row.get("source_video_title") or "",
                    "source_video_url": row.get("source_video_url") or "",
                    "uploader": row.get("uploader") or "",
                    "reason": normalized["reason"],
                }
            )
        if progress_callback is not None:
            progress_callback(min(batch_start + len(batch), len(rows)), len(rows), refined)
    return dedupe_refined_rows(refined)


def refine_rows_for_simple_output(
    rows: list[dict[str, str]],
    llm_parser: LlmSongParser,
    batch_size: int = 20,
) -> list[dict[str, str]]:
    refined: list[dict[str, str]] = []
    size = max(batch_size, 1)
    for batch_start in range(0, len(rows), size):
        batch = rows[batch_start : batch_start + size]
        try:
            normalized_batch = llm_parser.normalize_for_simple_output_batch(batch)
        except Exception as exc:
            print(f"[warn] 精简输出 AI 批量清洗失败，第 {batch_start + 1}-{batch_start + len(batch)} 行: {exc}")
            normalized_batch = []
            for offset, row in enumerate(batch, start=1):
                try:
                    normalized_batch.append(llm_parser.normalize_for_simple_output(row))
                except Exception as item_exc:
                    print(f"[warn] 精简输出 AI 清洗失败，第 {batch_start + offset} 行: {item_exc}")
                    normalized_batch.append(
                        {
                            "song_title": (row.get("song_title") or "").strip(),
                            "artist": (row.get("artist") or "").strip(),
                            "confidence": float(row.get("confidence") or 0),
                            "reason": "fallback to original row",
                        }
                    )
        for row, normalized in zip(batch, normalized_batch):
            if not normalized["song_title"]:
                continue
            refined.append(
                {
                    "song_title": normalized["song_title"],
                    "artist": normalized["artist"],
                    "confidence": normalized["confidence"],
                    "source_bvid": row.get("source_bvid") or "",
                    "source_video_title": row.get("source_video_title") or "",
                    "source_video_url": row.get("source_video_url") or "",
                    "uploader": row.get("uploader") or "",
                    "reason": normalized["reason"],
                }
            )
    return dedupe_refined_rows(refined)


def dedupe_refined_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for row in rows:
        key = f"{row['song_title'].strip().lower()}::{row['artist'].strip().lower()}"
        current = deduped.get(key)
        if current is None or float(row["confidence"]) > float(current["confidence"]):
            deduped[key] = row
    return sorted(deduped.values(), key=lambda item: item["song_title"].lower())


def write_refined_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "song_title",
                "artist",
                "confidence",
                "source_bvid",
                "source_video_title",
                "source_video_url",
                "uploader",
                "reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_ai_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_ai.csv")


if __name__ == "__main__":
    try:
        main()
        if _is_frozen_executable():
            print("\n执行完成 / Completed successfully.")
            _pause_if_frozen()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if _is_frozen_executable():
            if code == 0:
                print("\n程序结束 / Program finished.")
            else:
                print(f"\n程序异常退出 / Program exited with error code {code}.")
            _pause_if_frozen()
        raise
    except Exception as exc:
        if _is_frozen_executable():
            print(f"\n错误 / Error: {exc}")
            _pause_if_frozen()
            sys.exit(1)
        traceback.print_exc()
        raise
