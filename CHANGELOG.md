# Changelog

## 0.1.0 — 2026-11-26

First public release. Windows + NVIDIA CUDA is the only configuration verified
below; CPU/Vulkan/Metal paths come from upstream and are untested here.

### Performance

- Stateful streaming codec (`stream_begin` / `stream_decode` / `stream_end`)
  instead of re-decoding the growing prefix per chunk: RTF 2.97 → 0.30 on a 35 s
  line, flat instead of growing with utterance length.
- The first streamed chunk is one frame, not a full stride: TTFA with a prepared
  profile 2.9–6.9 s → 0.064–0.099 s.
- Flash attention and CUDA-graph capture for the decode steps.
- Fast-AR with a KV cache; one codebook-embedding gather per frame; sampling
  window and fast embeddings kept on the GPU.
- Partial top-k instead of a full sort; pre-cast norm weights with fused
  RMS_NORM+MUL; KV length rounded to 256 for the GQA attention path.
- Codec decode on a worker thread, overlapped with the AR.
- VRAM with a prepared profile 6.54 GB → 4.35 GB.

### Added

- `--prepare-voice`: encode the reference once and persist a `.s2voice` profile.
- `--seed` for reproducible generation.
- `--stream-file`, `--stream-decode-stride` for file output through the streaming
  path.
- HTTP `params` object exposing `top_k`, `top_p`, `temperature`, `seed`,
  `max_tokens` and the stream buffer.
- `S2_CUDA_GRAPHS` build option, `build.ps1` GPU-SM detection, and DLL
  deployment so the build directory is self-contained.
- `DECISIONS.md`, `PERFORMANCE.md`, OpenAPI description for the HTTP server.

### Changed

- Streaming is the only file path; the offline single-pass path was removed.
- Concurrent HTTP requests are queued instead of answered with 503.
- Reference-prompt KV and the KV buffer are reused across server requests.

### Fixed

- A failed GPU codec backend, an unreadable reference, a failed prefill, a
  tokenizer without `<|im_end|>` and a second bind on the same port are now
  errors (CLI exit 1, HTTP 400) instead of silent CPU fallbacks or empty audio.
