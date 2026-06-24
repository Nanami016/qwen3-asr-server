"""
Qwen3-ASR 本地语音识别服务
提供 OpenAI 兼容的 API 接口，使用 MLX 加速（Apple Silicon 原生）。

支持模型:
    - Qwen3-ASR-1.7B (8-bit, MLX 量化)

用法:
    python asr_server.py
    python asr_server.py --port 8000
    # 或使用 run.sh 后台运行
    ./run.sh start
"""

import os
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
DEFAULT_MODEL = "Qwen3-ASR-1.7B"

# 支持的模型列表（本地目录名 -> HuggingFace 模型名）
SUPPORTED_MODELS = {
    "Qwen3-ASR-1.7B": "mlx-community/Qwen3-ASR-1.7B-8bit",
}

# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="Qwen3-ASR Server", version="3.0.0")

# 全局模型实例缓存（模型名 -> (model, generate_fn)）
_model_cache = {}


def resolve_model_path(model_name: str) -> str:
    """解析模型路径：优先本地，否则使用 HuggingFace 模型名"""
    local_path = MODELS_DIR / model_name
    if local_path.exists():
        return str(local_path)
    return SUPPORTED_MODELS.get(model_name, model_name)


def load_model(model_name: str = None):
    """加载 Qwen3-ASR 模型（MLX 加速，带缓存）"""
    global _model_cache

    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = resolve_model_path(model_name)
    logger.info(f"正在加载模型: {model_name} ({model_path}) ...")

    try:
        from mlx_audio.stt.utils import load_model as mlx_load_model
        from mlx_audio.stt.generate import generate_transcription

        model = mlx_load_model(model_path)
        _model_cache[model_name] = (model, generate_transcription)
        logger.info(f"✅ 模型 {model_name} 加载成功 (MLX 加速)!")
        return (model, generate_transcription)

    except ImportError:
        logger.error("❌ 未安装 mlx-audio，请运行: pip install mlx-audio")
        raise
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


def transcribe_audio(model_name: str, audio_path: str) -> dict:
    """使用 MLX 转录音频"""
    model, generate_fn = load_model(model_name)
    logger.info(f"正在转录: {audio_path}")

    start_time = time.time()
    result = generate_fn(model, audio=audio_path)
    elapsed = time.time() - start_time

    # 提取文本和语言
    text = result.text if hasattr(result, 'text') else str(result)
    language = result.language if hasattr(result, 'language') else "unknown"

    logger.info(f"识别完成 [{language}] ({elapsed:.2f}s): {text[:100]}...")
    return {"text": text, "language": language, "duration": elapsed}


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
        "model": "Qwen3-ASR-1.7B",
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

    model_name = body.get("model", DEFAULT_MODEL)
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    audio_url = None
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "audio_url":
                    audio_url = part["audio_url"].get("url")

    if not audio_url:
        raise HTTPException(status_code=400, detail="未找到音频 URL")

    try:
        audio_input = audio_url
        tmp_file = None

        if audio_url.startswith("file://"):
            audio_input = audio_url[7:]
        elif audio_url.startswith("data:"):
            header, data = audio_url.split(",", 1)
            audio_bytes = base64.b64decode(data)
            suffix = ".wav"
            if "mp3" in header:
                suffix = ".mp3"
            elif "ogg" in header:
                suffix = ".ogg"
            tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_file.write(audio_bytes)
            tmp_file.close()
            audio_input = tmp_file.name

        result = transcribe_audio(model_name, audio_input)

        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)

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
                        "content": f"[{result['language']}] {result['text']}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "asr_result": result,
        }

    except Exception as e:
        logger.error(f"转录失败: {e}")
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,  # API 兼容参数，暂未使用
):
    """
    兼容 OpenAI Audio Transcriptions API（Whisper 风格）。

    用法:
        curl -X POST http://localhost:8000/v1/audio/transcriptions \\
            -F "file=@audio.wav" \\
            -F "model=Qwen3-ASR-1.7B"
    """
    try:
        model_name = model
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        content = await file.read()
        tmp.write(content)
        tmp.close()

        result = transcribe_audio(model_name, tmp.name)
        os.unlink(tmp.name)

        return {"text": result["text"], "language": result["language"]}

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
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "engine": "mlx",
        "models": {name: name in _model_cache for name in SUPPORTED_MODELS},
    }


# ============================================================
# 入口
# ============================================================

def update_server_json(host: str, port: int, model_name: str = DEFAULT_MODEL):
    """更新 server.json 配置"""
    config = {
        "software": "Qwen3-ASR Server (MLX)",
        "host": f"http://127.0.0.1:{port}",
        "api_key": "sk-qwen3-asr-local",
        "model": SUPPORTED_MODELS.get(model_name, model_name),
        "models": list(SUPPORTED_MODELS.values()),
    }
    SERVER_CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"已更新 server.json: {config['host']}")


def main():
    global DEFAULT_MODEL
    parser = argparse.ArgumentParser(description="Qwen3-ASR 本地语音识别服务 (MLX)")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"默认模型 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--preload", action="store_true", help="启动时预加载默认模型")
    args = parser.parse_args()

    DEFAULT_MODEL = args.model
    update_server_json(args.host, args.port, args.model)

    if args.preload:
        try:
            load_model(args.model)
        except Exception as e:
            logger.warning(f"预加载模型失败: {e}")
            logger.info("服务仍将启动，模型将在首次请求时加载")

    logger.info(f"🚀 Qwen3-ASR Server 启动中 (MLX 加速)...")
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
