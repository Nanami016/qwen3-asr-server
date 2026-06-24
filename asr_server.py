"""
Qwen3-ASR 本地语音识别服务
提供 OpenAI 兼容的 API 接口，替代 LM Studio 无法加载 qwen3_asr 模型的问题。

支持模型:
    - Qwen3-ASR-1.7B (8-bit)

用法:
    python asr_server.py
    python asr_server.py --port 1234
    # 或使用 run.sh 后台运行
    ./run.sh start
"""

import os
import sys
import json
import base64
import tempfile
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
import torch

# ─── Logging (daily log files) ───────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"asr_server_{time.strftime('%Y%m%d')}.log"

_log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
_log_datefmt = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    datefmt=_log_datefmt,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("qwen3-asr-server")

# ============================================================
# 全局配置
# ============================================================

SERVER_CONFIG_PATH = Path(__file__).parent / "server.json"
MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL = "Qwen3-ASR-1.7B"  # 默认模型

# 支持的模型列表（本地目录名 -> HuggingFace 模型名）
SUPPORTED_MODELS = {
    "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
}

# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="Qwen3-ASR Server", version="2.0.0")

# 全局模型实例缓存（模型名 -> 模型实例）
_model_cache = {}


def get_device():
    """自动选择最佳计算设备"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("检测到 Apple Silicon MPS，将使用 CPU 运行（qwen-asr 需要 CUDA）")
        return "cpu"
    else:
        return "cpu"


def resolve_model_path(model_name: str) -> str:
    """解析模型路径：优先本地，否则使用 HuggingFace 模型名"""
    local_path = MODELS_DIR / model_name
    if local_path.exists():
        return str(local_path)
    # 回退到 HuggingFace 模型名
    return SUPPORTED_MODELS.get(model_name, model_name)


def load_model(model_name: str = None):
    """加载 Qwen3-ASR 模型（带缓存）"""
    global _model_cache

    if model_name is None:
        model_name = DEFAULT_MODEL

    # 已缓存则直接返回
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = resolve_model_path(model_name)
    hf_name = SUPPORTED_MODELS.get(model_name, model_name)
    logger.info(f"正在加载模型: {model_name} ({model_path}) ...")

    if not Path(model_path).exists():
        logger.info("本地模型不存在，将从 HuggingFace 下载...")

    try:
        from qwen_asr import Qwen3ASRModel

        device = get_device()
        logger.info(f"计算设备: {device}")

        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            device_map=device,
            max_inference_batch_size=4,
            max_new_tokens=32768,
        )
        _model_cache[model_name] = model
        logger.info(f"✅ 模型 {model_name} 加载成功!")
        return model

    except ImportError:
        logger.error("❌ 未安装 qwen-asr 包，请运行: pip install -U qwen-asr")
        raise
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


# ============================================================
# API 端点
# ============================================================

@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models():
    """列出可用模型（兼容 OpenAI / LM Studio API）"""
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": 1700000000,
                "owned_by": "qwen",
                "permission": [],
            }
            for name in SUPPORTED_MODELS
        ],
    }


@app.post("/v1/chat/completions")
@app.post("/api/v1/chat")
async def chat_completions(request: Request):
    """
    兼容 OpenAI Chat Completions API 的语音识别端点。
    
    接受格式:
    {
        "model": "Qwen/Qwen3-ASR-0.6B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "file:///path/to/audio.wav"}}
                ]
            }
        ]
    }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON 请求体")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="缺少 messages 字段")

    # 支持模型选择
    model_name = body.get("model", DEFAULT_MODEL)
    # 兼容 HuggingFace 格式的模型名（如 Qwen/Qwen3-ASR-0.6B）
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    # 从消息中提取音频 URL
    audio_url = None
    language = None
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "audio_url":
                    audio_url = part["audio_url"].get("url")
                elif part.get("type") == "text":
                    # 可能包含语言提示
                    pass
        elif isinstance(msg.get("content"), str):
            # 纯文本消息，跳过
            pass

    if not audio_url:
        raise HTTPException(status_code=400, detail="未找到音频 URL（需要 audio_url 类型的内容）")

    # 加载模型并转录
    try:
        model = load_model(model_name)

        # 处理音频 URL
        audio_input = audio_url
        if audio_url.startswith("file://"):
            audio_input = audio_url[7:]  # 去掉 file:// 前缀
        elif audio_url.startswith("data:"):
            # base64 编码的音频
            # data:audio/wav;base64,xxxxx
            header, data = audio_url.split(",", 1)
            audio_bytes = base64.b64decode(data)
            # 保存到临时文件
            suffix = ".wav"
            if "mp3" in header:
                suffix = ".mp3"
            elif "ogg" in header:
                suffix = ".ogg"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            audio_input = tmp.name

        logger.info(f"正在转录音频: {audio_url[:80]}...")
        results = model.transcribe(
            audio=audio_input,
            language=language,
        )

        # 清理临时文件
        if audio_url.startswith("data:") and os.path.exists(audio_input):
            os.unlink(audio_input)

        text = results[0].text if results else ""
        detected_lang = results[0].language if results else "unknown"

        logger.info(f"识别完成 [{detected_lang}]: {text[:100]}...")

        return {
            "id": "asr-001",
            "object": "chat.completion",
            "created": 1700000000,
            "model": SUPPORTED_MODELS.get(model_name, model_name),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[{detected_lang}] {text}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "asr_result": {
                "language": detected_lang,
                "text": text,
            },
        }

    except Exception as e:
        logger.error(f"转录失败: {e}")
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
):
    """
    兼容 OpenAI Audio Transcriptions API（Whisper 风格）。

    用法:
        curl -X POST http://localhost:1234/v1/audio/transcriptions \
            -F "file=@audio.wav" \
            -F "model=Qwen3-ASR-0.6B"
    """
    try:
        # 解析模型名（兼容 HuggingFace 格式）
        model_name = model
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        # 保存上传的音频文件
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        content = await file.read()
        tmp.write(content)
        tmp.close()

        # 加载模型并转录
        asr_model = load_model(model_name)

        logger.info(f"正在转录: {file.filename} ({len(content)} bytes)...")
        results = asr_model.transcribe(
            audio=tmp.name,
            language=language,
        )

        # 清理临时文件
        os.unlink(tmp.name)

        text = results[0].text if results else ""
        detected_lang = results[0].language if results else "unknown"

        logger.info(f"识别完成 [{detected_lang}]: {text[:100]}...")

        # 兼容 OpenAI Whisper API 返回格式
        return {
            "text": text,
            "language": detected_lang,
        }

    except Exception as e:
        logger.error(f"转录失败: {e}")
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")


@app.post("/api/v1/models/load")
async def load_model_endpoint(request: Request):
    """兼容 LM Studio API 的模型加载端点"""
    try:
        body = await request.json()
        model_name = body.get("model", DEFAULT_MODEL)
    except Exception:
        model_name = DEFAULT_MODEL

    try:
        load_model(model_name)
        return {"status": "ok", "message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "models": {
            name: name in _model_cache
            for name in SUPPORTED_MODELS
        },
        "device": get_device(),
    }


# ============================================================
# 入口
# ============================================================

def update_server_json(host: str, port: int, model_name: str = DEFAULT_MODEL):
    """更新 server.json 配置"""
    config = {
        "software": "Qwen3-ASR Server (Python)",
        "host": f"http://127.0.0.1:{port}",
        "api_key": "sk-qwen3-asr-local",
        "model": SUPPORTED_MODELS.get(model_name, model_name),
        "models": list(SUPPORTED_MODELS.values()),
    }
    SERVER_CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"已更新 server.json: {config['host']}")


def main():
    global DEFAULT_MODEL
    parser = argparse.ArgumentParser(description="Qwen3-ASR 本地语音识别服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"默认模型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--preload", action="store_true", help="启动时预加载默认模型")
    args = parser.parse_args()

    DEFAULT_MODEL = args.model

    # 更新 server.json
    update_server_json(args.host, args.port, args.model)

    # 预加载模型
    if args.preload:
        try:
            load_model(args.model)
        except Exception as e:
            logger.warning(f"预加载模型失败: {e}")
            logger.info("服务仍将启动，模型将在首次请求时加载")

    logger.info(f"🚀 Qwen3-ASR Server 启动中...")
    logger.info(f"   地址: http://{args.host}:{args.port}")
    logger.info(f"   默认模型: {args.model}")
    logger.info(f"   可用模型: {', '.join(SUPPORTED_MODELS.keys())}")
    logger.info(f"   模型目录: {MODELS_DIR}")
    logger.info(f"   日志文件: {LOG_FILE}")
    logger.info(f"   API 端点:")
    logger.info(f"     GET  /v1/models")
    logger.info(f"     POST /v1/chat/completions  (OpenAI 兼容)")
    logger.info(f"     POST /v1/audio/transcriptions  (Whisper 兼容)")
    logger.info(f"     GET  /health")
    logger.info(f"")
    logger.info(f"💡 提示: 将 server.json 中的 software 改为你的客户端配置")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
