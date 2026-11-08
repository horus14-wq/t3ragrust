# Future ideas

Ideas worth doing later, with enough context to pick them up cold. Each entry
says why it matters, what's known, and what's still a guess.

---

## Dedicated GPU embedding path ("embedding server")

**Status:** idea. Related bead: `t3-aj3` (the incremental version inside t3rag).

### Why

GPU indexing today uses only a small fraction of the hardware. Measured on
vast.ai RTX 4090s (2026-11-08) with the fp32 Qwen3-Embedding-0.6B ONNX export,
`--pool 2`, and default batching (`embed_batch=16`, `embed_token_budget=4096`):

| Corpus | Avg passage | Rate | ≈ tokens/s | ≈ compute |
|---|---|---|---|---|
| struct | ~600 tokens | 33 docs/s | ~20K | ~24 TFLOPS |
| reflect | ~1,150 tokens | ~10.6 docs/s | ~12K | ~14 TFLOPS |
| raw-chunked (`--chunk-chars 4000`, `--pool 1`) | ~1,000 tokens | ~9.5 chunks/s | ~9.5K | ~11 TFLOPS |

A 4090's peak is ~82 TFLOPS in fp32 and ~165 TFLOPS in fp16 on its tensor
cores. `nvidia-smi` utilization swung between 34% and 100%, so the GPU often
sat waiting. Indexing the four fable5-v2 corpora took ~2.5 hours across two
boxes. A purpose-built path could plausibly do the same work 5–10× faster.
That figure is an estimate, not a measurement.

### Where the time goes

1. **fp32 instead of fp16.** The tensor cores sit idle. The fix is fp16 export
   support: feed f16 `past_key_values` and read back an f16 `last_hidden_state`.
   Expected ~2–3×.
2. **An LLM-style export, not one built for embedding.** The onnx-community
   export is a merged-KV decoder graph. ONNX Runtime's generic
   `MultiHeadAttention` builds the full attention matrix (no flash attention),
   so cost grows with the square of passage length. That's why reflect/raw ran
   ~3× slower than struct.
3. **Small, padded batches.** Batches cap at 4,096 tokens and 16 docs and are
   padded to their longest member. Inputs aren't length-sorted, so a lot of
   the compute goes into padding tokens.
4. **CPU work runs inline.** Tokenizing, BM25 (Tantivy), HNSW inserts, and
   disk writes run in the same loop between GPU batches instead of alongside
   them.

### What the server would look like

A separate GPU-side embedding worker that t3rag's indexer feeds, rather than
inline ONNX Runtime sessions:

- **fp16/bf16 by default** on CUDA, with flash attention (or another fused
  attention kernel).
- **Length-bucketed dynamic batching:** sort a window of pending passages by
  token count and batch by a token budget large enough to fill the GPU
  (e.g. 32K+ tokens), not a fixed doc count.
- **Pipelined:** a tokenizer thread feeds the GPU, and a writer thread handles
  BM25, HNSW, and passage-store writes, so the GPU never waits on the CPU.
- **One worker per GPU,** so multi-GPU boxes scale linearly (today: one
  `CUDA_VISIBLE_DEVICES`-pinned `t3rag index` per GPU, by hand).
- **The same model and output vectors** as the CPU path, so corpora built on
  the GPU stay query-compatible with the CPU uint8 query embedder.

### Options to evaluate

- **Use an existing server.** Hugging Face Text Embeddings Inference (TEI) or
  vLLM's embedding mode already do fp16, fused attention, and dynamic
  batching. t3rag would call it over HTTP during `index` (e.g. `--embed-url`).
  Least code, but adds a non-Rust runtime dependency for GPU indexing only;
  CPU users are unaffected.
- **Stay in-process with ONNX Runtime.** Do items 1, 3, and 4 inside t3rag
  (bead `t3-aj3`). No new dependency, but attention stays unfused unless a
  better export or ORT's fused attention kicks in for this graph.
- **Candle or another Rust-native runtime** with CUDA kernels and flash
  attention. Keeps everything in Rust; the most work.

### Things learned the hard way (keep these for any GPU path)

- The ORT CUDA plugin (`libonnxruntime_providers_cuda.so`) is built for
  **CUDA 13 + cuDNN 9** and needs **glibc ≥ 2.38**. Use
  `nvidia/cuda:13.0.x-cudnn-runtime-ubuntu24.04`. The ep-cuda binary also
  needs `libonnxruntime_providers_{shared,cuda}.so` next to it (bead `t3-2v5`).
- On CUDA, the default **uint8 export runs int8 ops on the CPU**: 3.75 docs/s
  vs ~33 docs/s for fp32 on the same 4090. Pin the export with
  `T3RAG_ONNX_FILE`.
- **Two sessions on one 24 GB GPU run out of memory** on 8,192-token
  passages: each ORT memory arena keeps its peak size. Use one session per
  GPU for long inputs.
- **Prebuild everything before renting.** Ship the binary and plugin libs,
  and use a startup script that downloads data and starts indexing on its
  own. That way the paid box starts useful work within ~3 minutes of launch.
