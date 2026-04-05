# Bilibili Favorites Music List

从零开始构建的 Python 项目：读取指定的 Bilibili 收藏夹，抓取其中所有视频，结合规则解析和可选 LLM 解析，输出一份歌曲清单。

## 功能

- 读取指定收藏夹 `media_id`
- 支持公开收藏夹，也支持通过 Cookie 访问私有收藏夹
- 拉取每个视频的标题、简介，默认不请求详情接口以降低风控
- 使用规则解析歌曲名/歌手
- 可选接入兼容 OpenAI `chat/completions` 的 LLM 做二次抽取
- 输出 `CSV` 或 `JSON`
- 自动按歌曲去重
- 额外输出一个精简文件：每行仅保留“歌曲名 歌手”，优先使用 AI 清洗结果

## 项目结构

```text
src/bili_music_list/
  bilibili_client.py   # B 站收藏夹/视频详情抓取
  extractors.py        # 规则解析器
  llm_parser.py        # 可选 LLM 解析器
  cli.py               # 命令行入口
tests/
  test_extractors.py
```

## 安装

推荐先安装 Python 3.10+，然后执行：

```bash
pip install -e .
```

## 使用

### 0. 图形化界面（推荐普通用户）

安装后可直接启动 GUI：

```bash
bili-music-list-gui
```

界面包含两个页签：

- `导出收藏夹 / Export`：输入 `media_id` 并导出歌曲列表
- `CSV AI 清洗 / AI Refine`：对现有 CSV 逐批做 AI 清洗

日志窗口会实时显示运行输出与进度，适合不熟悉命令行的用户。

### 1. 仅规则解析

```bash
bili-music-list --media-id 123456789 --parser heuristic --output output/music_list.csv
```

### 2. 规则 + LLM 混合解析

```bash
bili-music-list \
  --media-id 123456789 \
  --parser hybrid \
  --llm-base-url https://api.openai.com/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model gpt-4.1-mini \
  --output output/music_list.csv

如果没有提供 LLM 参数，`hybrid` 会自动退回 `heuristic`。
```

### 3. 私有收藏夹

先把浏览器里的 Cookie 保存到 `cookie.txt`，再执行：

```bash
bili-music-list --media-id 123456789 --cookie-file cookie.txt
```

### 4. 可选详情抓取

默认不会逐个请求视频详情接口，因为该接口容易触发 `412` 风控。

如果你确认网络环境稳定、并且希望补充详情描述，可手动开启：

```bash
bili-music-list --media-id 123456789 --with-detail --output output/music_list.csv
```

### 5. 对现有 CSV 做 AI 二次解析

如果已经生成了 `output/music_list.csv`，可以继续用 AI 清洗歌曲名和歌手，并输出到另一个文件：

```bash
bili-music-list ^
  --input-csv output/music_list.csv ^
  --ai-output output/music_list_ai.csv ^
  --llm-base-url https://api.openai.com/v1 ^
  --llm-api-key YOUR_API_KEY ^
  --llm-model gpt-4.1-mini ^
  --llm-batch-size 10 ^
  --llm-retries 5 ^
  --llm-delay-ms 1500 ^
  --llm-max-tokens 800
```

输出仍然是一首歌一行，新增 `reason` 字段表示 AI 的清洗依据。
如果你的 LLM 接口容易出现 `503`、断连或限流，建议适当减小 `--llm-batch-size`，并增大 `--llm-retries` 和 `--llm-delay-ms`。

## 输出字段

默认 CSV 为“每首歌一行”的简洁格式：

- `song_title`: 歌曲名
- `artist`: 歌手
- `confidence`: 置信度
- `source_bvid`: 来源视频 BV 号
- `source_video_title`: 来源视频标题
- `source_video_url`: 来源视频链接
- `uploader`: 来源 UP 主

另外还会额外生成一个精简版文件，默认文件名类似 `music_list_simple.csv`：

- 每行一首歌
- 格式为 `歌曲名 歌手`
- 如果歌手未知，则只写 `歌曲名`
- 如果提供了 LLM 参数，精简版会使用更严格的 AI 清洗结果
- 如果没有提供 LLM 参数，精简版会退回原始解析结果

## 已验证的接口

- 收藏夹列表：`https://api.bilibili.com/x/v3/fav/resource/list`
- 视频详情：`https://api.bilibili.com/x/web-interface/view`

## 后续可扩展

- 接入 Bilibili 官方登录态管理
- 增加批量收藏夹处理
- 增加歌名标准化和第三方音乐平台匹配
- 为 LLM 增加本地缓存，减少重复请求
