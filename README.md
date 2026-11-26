# mediasum

本地音视频 → 完整逐字稿 + 详细总结（HTML）+ 摘要。全程离线，音频不出本机。

## 安装

只在 macOS（Apple Silicon）上验证过。MLX 依赖 Metal，Linux 与 Intel Mac 无法运行。

```bash
git clone git@github.com:baiywenjun/mediasum.git ~/tools/mediasum
cd ~/tools/mediasum

# 1. Python 3.12 隔离环境
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. 模型权重（约 105 MB，下载后完全离线运行）
./ms models download-all
# 国内网络慢：HF_ENDPOINT=https://hf-mirror.com ./ms models download-all

# 3. 自检
./ms config
```

系统层前置依赖：`ffmpeg`（需含 `ffprobe`）、`uv`。

`.venv/` 与 `models/` 已列入 `.gitignore`，克隆后必须按上面第 1、2 步重建，否则会报模型缺失。

## 三条命令

```bash
cd ~/tools/mediasum

./ms ~/Videos/课.mp4                       # 完整处理
./ms ~/Videos/会.mp4 --diar auto           # 会议：自动判断有几个说话人
./ms ~/Videos/课.mp4 --from summary        # 只重跑归纳（调 prompt 时用）
```

产物在 `~/Videos/notes/<文件名>/`：

| 文件 | 内容 |
|---|---|
| `01_逐字稿.txt` | `[HH:MM:SS] 文本`，段落按句号对齐，多人时带「说话人 N：」 |
| `02_详细总结.html` | 摘要 + 章节脉络 + 术语 + 关键画面与讲解对照 + 可展开逐字稿，图片内嵌，单文件可离线打开 |
| `03_摘要.md` | 450–580 字连续段落 |

中间文件在 `~/Videos/notes/.work/<文件名>/`，`--from` 重跑靠它复用缓存，别随手删。

## 终端界面

不想敲参数时用 TUI。它和命令行是同一套引擎，界面上填的东西等价于拼出一条 `ms` 命令：

```bash
cd ~/tools/mediasum
./ms-tui                              # 打开界面
./ms-tui ~/Videos/课.mp4              # 打开并预填文件
./ms tui                              # 等价写法
```

| 区域 | 说明 |
|---|---|
| 路径 + 扫描 | 可直接粘贴路径、把文件拖进终端窗口；「扫描」按修改时间列出 `~/Videos/media` 下的音视频 |
| 产物 / 详细程度 / 说话人 / 裁剪 | 分别对应 `--no-html` 一类开关、`--detail`、`--diar`、`--crop` |
| 阶段行 | `✓` 已完成，高亮的是进行中，`−` 表示本次路径不需要（纯音频跳过抽帧、说话人关闭） |
| 进度行 | 抽帧和归纳显示 `x/y`；转写阶段只能显示计时，因为 mlx-whisper 的进度不对外输出 |

| 快捷键 | 作用 |
|---|---|
| `Ctrl+R` | 开始处理 |
| `Ctrl+K` | 取消（终止整个进程组，不会留下 mlx-whisper 孤儿进程） |
| `Ctrl+O` | 打开产物目录 |
| `q` | 退出（有任务在跑时会拦住，要求先取消） |

两点设计上的取舍：

- **界面上改的配置只在本次生效**，不写回 `config.toml`。要改默认值仍改配置文件，避免「界面上点过一次，以后每次都被改掉」。
- **同一时刻只跑一个任务**。M1 的 16 GB 内存跑 Whisper 基本占满，并发两个会直接 OOM，所以开始按钮在运行期间是禁用的。

## 两个必须知道的操作

**录屏类视频要裁掉聊天区。** 不裁的话 OCR 会把弹幕、二维码、礼物提示全读进来，噪声能占到识别文字的一半。

```bash
./ms probe ~/Videos/课.mp4                 # 会给出建议裁剪值
./ms ~/Videos/课.mp4 --crop 0.02,0.07,0.78,0.95   # 左,上,右,下（0–1 比例）
```

腾讯课堂/小鹅通这类右侧带聊天栏的，`0.78` 这个右边界基本通用；纯 PPT 全屏录制不用裁。

**多人场景要显式开分离。** 默认是关的，单人讲解能省 10 分钟。`auto` 会先探测，只有检出 2 人以上才真正分离。

```bash
./ms models download-diar                  # 首次用，约 33 MB
./ms ~/Videos/会.mp4 --diar auto
```

## 常用参数

```bash
--detail brief|standard|detailed   详细程度：只要摘要 / 常规 / 加关键原话与待决问题
--no-html --no-transcript --no-abstract    关掉不要的产物
--format md                        逐字稿输出 Markdown（丢进 Obsidian）
--srt                              额外导出字幕
--set llm.provider=openai          临时切云端模型（需 OPENAI_API_KEY）
--out <目录>                       改产物位置
--dry-run                          只看配置和路径，不执行
```

所有参数都能落在 `config.toml` 里，配置文件带逐项注释；`./ms config` 查当前生效值。
命令行 `--set a.b=c` 只对当次生效，适合临时试验。

## 处理耗时（M1 / 16 GB 实测）

| 环节 | 12 分钟样本 |
|---|---|
| 转写 | 1 分 30 秒 |
| 标点与分段 | 1 秒 |
| 抽帧 + OCR（15 帧） | 34 秒 |
| 说话人分离（可选） | 98 秒 |
| 归纳（本地 7B，6 块） | 4 分 24 秒 |
| **合计** | **6 分 33 秒** |

两小时视频按线性外推：转写约 25–30 分钟，归纳 15–25 分钟，总计约 45 分钟。

## 已知限制

- **同音字**：Whisper 会把「拖堂」听成「脱下膛」。用 `asr.initial_prompt` 里的术语表缓解，但无法根除，关键内容仍需人工校对。
- **本地 7B 的摘要口吻**：偏「这门课将带你掌握…」这种宣传腔，干货密度不如云端模型。这是模型能力上限，与 prompt 无关——需要交付级质量就切 `llm.provider=openai`。
- **OCR 错字**：艺术字体和低对比度 PPT 标题会识别出错（样本里「程序员进阶必备」被读成「桯序员进阶必宝」）。
- **画面是购物页/直播界面时**，OCR 文字价值很低，靠 `frames.ocr_blocklist` 过滤明显噪声，但仍会有残留。
