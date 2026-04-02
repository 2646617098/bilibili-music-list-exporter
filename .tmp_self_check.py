from bili_music_list.extractors import extract_song_candidates
from bili_music_list.models import VideoItem

def make_video(title, intro='', description=''):
    return VideoItem(
        favorite_id=1,
        favorite_title='fav',
        video_id=1,
        bvid='BV1xx411c7mD',
        title=title,
        intro=intro,
        page_title=title,
        description=description,
        uploader='tester',
        url='https://www.bilibili.com/video/BV1xx411c7mD',
    )

c1 = extract_song_candidates(make_video('周杰伦《晴天》MV'))
assert c1 and c1[0].song_title == '晴天', c1

c2 = extract_song_candidates(make_video('青花瓷 - 周杰伦'))
assert c2 and c2[0].song_title == '青花瓷', c2
assert c2[0].artist == '周杰伦', c2

c3 = extract_song_candidates(make_video('起风了', intro='演唱：买辣椒也用券'))
assert c3 and any(item.song_title == '起风了' and item.artist == '买辣椒也用券' for item in c3), c3

print('self-check passed')
