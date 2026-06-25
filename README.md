# Qwen3-ASR 本地语音识别服务

本地语音识别服务，使用 Qwen3-ASR 模型进行语音转文字，提供 OpenAI 兼容的 API 接口。

本项目使用 `mlx-audio` 在 Apple Silicon 上原生运行，无需 CUDA。

## 支持的模型

| 模型 | 量化 | 说明 |
|------|------|------|
| Qwen3-ASR-1.7B | 8-bit | 推荐，精度更高 |

## 项目结构

```
qwen3-asr-server/
├── asr_server.py          # ASR 服务主程序（FastAPI + Uvicorn）
├── requirements.txt       # Python 依赖
├── run.sh                 # 后台服务管理脚本
├── server.json            # 服务配置（自动生成）
├── README.md              # 本文件
├── .gitignore             # Git 忽略规则
├── doc/                   # 文档
│   └── mlx-performance-optimization.md  # MLX 性能优化方案
├── models/                # 模型文件（符号链接）
│   └── Qwen3-ASR-1.7B -> ~/llm/mlx-community/Qwen3-ASR-1.7B-8bit/
└── logs/                  # 服务日志（按日分割）
    └── asr_server_20260624.log
```

## 快速开始

### 1. 激活虚拟环境

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

**前台运行：**
```bash
python asr_server.py
```

**后台运行（推荐）：**
```bash
./run.sh start      # 启动
./run.sh stop       # 停止
./run.sh restart    # 重启
./run.sh status     # 查看状态
```

服务将在 `http://127.0.0.1:8000` 启动。

### 4. 测试转录

```bash
# 使用 Whisper 兼容 API（文件上传）
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio/test.mp3" \
  -F "model=Qwen3-ASR-1.7B"

# 使用 OpenAI 兼容 API（URL 方式）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-ASR-1.7B",
    "messages": [{
      "role": "user",
      "content": [{
        "type": "audio_url",
        "audio_url": {"url": "file:///path/to/audio.wav"}
      }]
    }]
  }'
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 列出可用模型 |
| `/api/v1/models` | GET | 列出可用模型（LM Studio 兼容） |
| `/v1/chat/completions` | POST | OpenAI 兼容的语音识别 |
| `/api/v1/chat` | POST | OpenAI 兼容（LM Studio 兼容） |
| `/v1/audio/transcriptions` | POST | Whisper 兼容的语音识别 |
| `/api/v1/models/load` | POST | 模型加载（LM Studio 兼容） |
| `/health` | GET | 健康检查（显示已加载模型） |

## Python 调用示例

```python
import requests

# Whisper 风格 API
url = "http://localhost:8000/v1/audio/transcriptions"
files = {"file": open("audio/test.mp3", "rb")}
data = {"model": "Qwen3-ASR-1.7B"}
response = requests.post(url, files=files, data=data)
print(response.json()["text"])
```

## 配置选项

```bash
# 指定端口
python asr_server.py --port 8080

# 启动时预加载模型并执行 warmup（推荐，消除首次请求延迟）
python asr_server.py --preload

# 后台运行时指定端口（修改 run.sh 中的 PORT 变量）
```

> `--preload` 会在启动时加载模型并运行一次 warmup 推理，预编译 Metal shader。
> 不使用 `--preload` 时，warmup 会在首次请求时自动执行（首次请求会慢 2-5 秒）。

## 支持的语言

中文、英语、粤语、阿拉伯语、德语、法语、西班牙语、葡萄牙语、印尼语、意大利语、韩语、俄语、泰语、越南语、日语、土耳其语、Hindi、马来语、荷兰语、瑞典语、丹麦语、芬兰语、波兰语、捷克语、菲律宾语、波斯语、希腊语、罗马尼亚语、匈牙利语、马其顿语

## 故障排除

### 模型加载失败
- 确保已安装 `mlx-audio`：`pip install mlx-audio`
- 首次运行需要下载模型，请耐心等待
- 检查 `models/` 目录下的符号链接是否正确

### 内存不足
- 1.7B 模型（8-bit 量化）需要约 1.7GB 内存
- 2 小时音频的 KV cache 约 5.1GB，总内存占用约 6.8GB
- 如需更小内存，可考虑使用 0.6B 版本

### Apple Silicon 加速
- 本项目使用 MLX 框架，在 Apple Silicon 上原生加速
- 无需 CUDA，M1/M2/M3/M4/M5 芯片均可使用
- 启动时自动配置 wired limit 并执行 warmup 预编译 Metal shader
- 根据音频时长动态调整 prefill_step_size，长音频处理更高效
- 详见 [MLX 性能优化方案](doc/mlx-performance-optimization.md)

### 关于 mlx_audio 模型层优化

本项目对 `mlx_audio` 的 Qwen3-ASR 模型代码有一项额外优化：消除 `_generate_single_chunk` 中对 `_preprocess_audio` 的重复调用（每次推理节省一次 feature extraction + encoder forward，约 15-25% 提速）。

该改动在 `mlx_audio` 包内部（`.venv/lib/.../mlx_audio/stt/models/qwen3_asr/qwen3_asr.py`），不在本仓库中。**clone 本仓库后自动获得 wired limit / warmup / 动态 prefill 优化，但不包含此项。** 如需应用，手动修改 `mlx_audio` 包中的 `qwen3_asr.py`：

```python
# stream_generate() 新增两个参数：
def stream_generate(self, audio, *, ..., precomputed_audio_features=None, precomputed_num_audio_tokens=None):
    # 如果传入预计算特征，跳过重复编码
    if precomputed_audio_features is not None:
        audio_features = precomputed_audio_features
        input_ids = self._build_prompt(precomputed_num_audio_tokens, language, system_prompt)
    else:
        # 原始路径 ...

# _generate_single_chunk() 预计算一次后传入：
input_features, feature_attention_mask, num_audio_tokens = self._preprocess_audio(audio_chunk)
audio_features = self.get_audio_features(input_features, feature_attention_mask)
# ... 传给 stream_generate(precomputed_audio_features=audio_features, ...)
```

详见 [MLX 性能优化方案 - P0](doc/mlx-performance-optimization.md#p0--消除重复计算-x)。

### 查看日志
```bash
# 查看今天的日志
tail -f logs/asr_server_$(date +%Y%m%d).log

# 查看所有日志
ls -la logs/
```
