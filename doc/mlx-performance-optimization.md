# MLX 性能优化方案

> 硬件环境：Apple M5 / 34GB 统一内存 / 10 核 / Metal 4
> 模型：Qwen3-ASR-1.7B（8-bit 量化，group_size=64）
> MLX 版本：0.31.2

---

## 优化总览

| 优先级 | 优化项 | 预估收益 | 状态 |
|--------|--------|----------|------|
| P0 | 消除重复 `_preprocess_audio` + `get_audio_features` | 15-25% | [x] 已完成 |
| P1 | 减少 `mx.clear_cache()` 调用 | 5-10% | [x] 已完成（随 P0 一并解决） |
| P2 | 设置 MLX wired limit | 5-15% | [x] 已完成 |
| P3 | Metal shader 预编译（warmup） | 首次请求 -2~5s | [x] 已完成 |
| P4 | 动态 `prefill_step_size`（按音频时长自适应） | 中 | [x] 已完成 |
| P5 | `mx.compile` encoder forward | 高 | [ ] 待评估 |
| P6 | KV cache 量化 | 中 | [ ] 待评估 |
| P7 | 异步文件 I/O | 低 | [ ] 待评估 |

---

## P0 — 消除重复计算 [x]

**问题**：`_generate_single_chunk` 在 `stream_generate` 已完成 feature extraction + encoder forward 之后，又调用了一次 `_preprocess_audio`。同一段音频被完全相同的处理流程执行了两遍。

**调用链（优化前）**：

```
generate()
  └→ _generate_single_chunk()
       └→ stream_generate()
       │    └→ _preprocess_audio()        ← 第 1 次（feature extraction + encoder forward）
       │    └→ get_audio_features()
       └→ _preprocess_audio()             ← 第 2 次（完全重复）
```

**调用链（优化后）**：

```
generate()
  └→ _generate_single_chunk()
       ├→ _preprocess_audio()             ← 仅 1 次
       ├→ get_audio_features()
       └→ stream_generate(precomputed_audio_features=..., precomputed_num_audio_tokens=...)
            └→ 跳过重复编码，直接使用预计算结果
```

**改动文件**：`mlx_audio/stt/models/qwen3_asr/qwen3_asr.py`

- `stream_generate`：新增 `precomputed_audio_features` 和 `precomputed_num_audio_tokens` 参数
- `_generate_single_chunk`：预先调用一次编码，将结果传给 `stream_generate`
- `stream_transcribe`：同上，每个 chunk 只编码一次

---

## P1 — 减少 `mx.clear_cache()` [x]

**问题**：`stream_generate` 内部调用了 3 次 `mx.clear_cache()`，而 `mlx_lm.generate_step` 每 256 tokens 已自带清理。频繁清理导致 GPU 内存反复分配释放，增加开销。

**优化**：随 P0 一并解决。预计算路径跳过了 `_preprocess_audio` 和 `get_audio_features` 阶段的 `clear_cache`，从 3 次减少到 2 次（仅在 phase 边界保留必要的清理）。

---

## P2 — 设置 Wired Limit [x]

**问题**：M5 34GB 统一内存，`max_recommended_working_set_size` 约 25.5GB，但默认未设置 wired limit。Metal 可能将 GPU buffer 换出到系统内存，导致不必要的迁移开销。

**改动文件**：`asr_server.py` — `_configure_mlx_memory()`

**实现**：模型加载时调用 `mx.set_wired_limit()`，将推荐工作集的 90%（约 23GB）锁定为 GPU 缓存。

```
MLX wired limit: 23000 MB (max recommended 25559 MB)
```

---

## P3 — Metal Shader 预编译 [x]

**问题**：首次 MLX 推理会触发 Metal shader 编译，额外增加 2-5 秒延迟。

**改动文件**：`asr_server.py` — `_warmup_model()`

**实现**：模型加载后自动生成 0.5 秒静音音频并跑一次推理，预编译所有 Metal shader。临时文件使用 `finally` 块确保异常时也能清理。

```
🔥 Warmup 完成 (3.21s) — Metal shader 已预编译
```

---

## P4 — 动态 prefill_step_size [x]

**问题**：默认 `prefill_step_size=2048`。对于长音频，prefill 循环迭代次数过多（2h 音频需要 46 次迭代），每次迭代有 kernel launch 开销。

**改动文件**：`asr_server.py` — `_compute_prefill_step_size()` + `transcribe_audio()`

**实现**：根据音频时长动态计算 `prefill_step_size`，公式为 `max(4096, min(estimated_tokens // 2, 16384))`。

**Prefill 迭代次数对比**：

| 音频时长 | Audio Tokens | 优化前 (2048) | 优化后 (动态) | 减少 |
|----------|-------------|---------------|---------------|------|
| 10s | 130 | 1 iter | 1 iter | — |
| 1min | 780 | 1 iter | 1 iter | — |
| 5min | 3,900 | 2 iters | 1 iter | -50% |
| 10min | 7,800 | 4 iters | 2 iters | -50% |
| 30min | 23,400 | 12 iters | 2 iters | -83% |
| 1h | 46,800 | 23 iters | 3 iters | -87% |
| 2h | 93,600 | 46 iters | 6 iters | -87% |

---

## KV Cache 内存占用分析

基于 Qwen3-ASR-1.7B 模型参数：28 层、8 KV heads、head_dim=128、8-bit 量化。

**每 token KV cache**：`28 × 2 × 8 × 128 × 1 = 57,344 bytes (56 KB)`

**音频 token 率**：~13 tokens/sec（100 frames/sec，chunk_size=100，Conv2d 压缩约 8x）

| 音频时长 | Audio Tokens | KV Cache | + 模型权重 | 总内存 | 状态 |
|----------|-------------|----------|-----------|--------|------|
| 10s | 130 | 7.1 MB | 1.7 GB | 1.71 GB | ✅ |
| 1min | 780 | 42.7 MB | 1.7 GB | 1.74 GB | ✅ |
| 5min | 3,900 | 213 MB | 1.7 GB | 1.91 GB | ✅ |
| 10min | 7,800 | 427 MB | 1.7 GB | 2.12 GB | ✅ |
| 30min | 23,400 | 1.28 GB | 1.7 GB | 2.98 GB | ✅ |
| 1h | 46,800 | 2.56 GB | 1.7 GB | 4.26 GB | ✅ |
| 2h | 93,600 | 5.12 GB | 1.7 GB | 6.82 GB | ✅ |

> 结论：即使处理 2 小时音频，总内存占用仅 6.8 GB，远低于 25.5 GB 推荐上限。KV cache 量化（P6）对本场景非必要。

---

## P5 — mx.compile encoder forward [ ]

**问题**：encoder 的 24 层 Transformer forward 每次调用都重新执行计算图。`mx.compile` 可以将计算图固化，减少 kernel dispatch 开销。

**难点**：encoder 内部有动态形状（根据音频长度变化的 chunk 数量），需要确认 `mx.compile` 对动态形状的处理。

**建议**：先对固定形状的子模块（如单层 encoder layer）做 compile，再逐步扩展。

---

## P6 — KV Cache 量化 [ ]（非必要）

**分析结论**：经计算，2 小时音频的 KV cache 仅 5.1 GB，加上模型权重共 6.8 GB，远低于 25.5 GB 推荐上限。**KV cache 量化对当前使用场景无实际收益**。

**保留选项**：若未来需要处理更长音频或同时加载多个模型，`mlx_lm.generate_step` 原生支持 `kv_bits=4, kv_group_size=64` 参数，可将 KV cache 压缩到 1/8。

---

## P7 — 异步文件 I/O [ ]

**问题**：音频文件的读取（`load_audio`）是同步阻塞的，在 FastAPI 的 async 端点中会短暂阻塞事件循环。

**建议**：使用 `asyncio.to_thread` 包装 `transcribe_audio`，或使用 `aiofiles` 异步读取文件后传入推理函数。

**注意**：由于 Python GIL 的存在，`to_thread` 对 CPU 密集型推理代码无并发收益，但对 I/O 等待有帮助。

---

## 预估收益

| 场景 | 优化前 | 优化后（P0-P4） | 说明 |
|------|--------|-----------------|------|
| 首次请求 | ~7-8s（含 shader 编译） | ~4-5s | P3 warmup 消除编译延迟 |
| 短音频（<10s） | ~4.8s | ~3.5-4s | P0 消除重复编码 |
| 中音频（1-5min） | ~8-15s | ~6-12s | P0 + P4 减少 prefill 迭代 |
| 长音频（30min-2h） | prefill 迭代过多 | prefill 减少 83-87% | P4 动态 step size |

---

## 参考资料

- [MLX Large Models Guide](https://github.com/ml-explore/mlx-lm/tree/main#large-models)
- [mlx-audio STT 源码](https://github.com/Blaizzy/mlx-audio)
- [MLX Metal Performance](https://ml-explore.github.io/mlx/build/html/usage/metal.html)
