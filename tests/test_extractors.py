from bili_music_list.extractors import extract_song_candidates
from bili_music_list.models import VideoItem


def make_video(title: str, intro: str = "", description: str = "") -> VideoItem:
    return VideoItem(
        favorite_id=1,
        favorite_title="fav",
        video_id=1,
        bvid="BV1xx411c7mD",
        title=title,
        intro=intro,
        page_title=title,
        description=description,
        uploader="tester",
        url="https://www.bilibili.com/video/BV1xx411c7mD",
    )


def test_extract_quoted_title() -> None:
    candidates = extract_song_candidates(make_video("周杰伦《晴天》MV"))
    assert candidates[0].song_title == "晴天"


def test_extract_split_title_and_artist() -> None:
    candidates = extract_song_candidates(make_video("青花瓷 - 周杰伦"))
    assert candidates[0].song_title == "青花瓷"
    assert candidates[0].artist == "周杰伦"


def test_extract_artist_from_intro() -> None:
    candidates = extract_song_candidates(
        make_video("起风了", intro="演唱：买辣椒也用券")
    )
    assert candidates[0].song_title == "起风了"
    assert candidates[0].artist == "买辣椒也用券"
