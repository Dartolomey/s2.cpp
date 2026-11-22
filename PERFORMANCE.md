# Upstream vs this fork: RTF, TTFA, memory

Same machine, same model, same voice profile, same sampling parameters. Only the
engine binary differs.

- GPU: RTX 4080 SUPER, 16 GB, CUDA 13.2, Windows 11
- Model: `s2-pro-q4_k_m.gguf` for RTF/memory, `s2-pro-q8_0.gguf` for TTFA (the harness default)
- Reference: `voice-reference.wav` (40.8 s) + transcript, saved profile
  `runs/v020-line8-baseline/voice/bench.s2voice`
- Params: temp 0.8, top_p 0.8, top_k 30, max_tokens 1024, stride 4
- Upstream: `forks/s2-upstream/build/s2.exe`, source `2c33261` (no seed support)
- Ours: `Main/build/s2.exe` at `4fadca4` (engine sha `11f04e23…`)

Upstream has no `--seed`, so its audio is a different draw. RTF and TTFA are
normalised by audio duration / wall time, so the comparison remains valid.

## RTF, 8 tests, CLI streaming (q4_k_m)

| test | upstream | ours | speed-up |
|---:|---:|---:|---:|
| 1 (35 s) | 2.971 | 0.309 | 9.6x |
| 2 (36 s) | 3.046 | 0.310 | 9.8x |
| 3 (5.1 s) | 0.833 | 0.360 | 2.3x |
| 4 (5.3 s) | 0.841 | 0.351 | 2.4x |
| 5 (3.8 s) | 0.872 | 0.375 | 2.3x |
| 6 (4.4 s) | 0.861 | 0.367 | 2.3x |
| 7 (3.1 s) | 0.895 | 0.397 | 2.3x |
| 8 (5.0 s) | 0.857 | 0.382 | 2.2x |

Upstream's cost grows with the length of the utterance: its codec re-decodes the
growing prefix for every chunk (quadratic), which is why the 35 s tests are ~3.0
while the 3-5 s tests are ~0.86. This fork keeps the codec state, so its RTF is
flat.

Runs: upstream `runs/v077-upstream-full`, ours `runs/v074-release-final`.

## TTFA, 8 tests, server (q8_0)

| mode / config | upstream | ours | speed-up |
|---|---:|---:|---:|
| profile, `buffer0` | 2.943-6.915 | **0.122-0.138** | 21-57x |
| profile, `low_latency` | 0.385-0.470 | **0.120-0.139** | 2.8-3.9x |
| profile, `buffer3000` | 3.389-15.927 | **1.153-1.370** | 2.5-13.8x |
| raw, `buffer0` (reference encoded in the request) | 4.097-7.836 | **1.030-1.146** | 3.6-7.6x |

Upstream's `buffer0` waits for its own decode cadence before emitting the first
chunk; `low_latency` forces a faster first byte but at the cost of more decode
work. Our streaming path emits the first PCM after the first 4-frame chunk
because the codec is stateful and the graph is already warm.

Runs: upstream `runs/v080-upstream-ttfa`, ours `runs/v066-ttfa-nomemset`.

## Memory (q4_k_m, one 35 s synthesis)

| metric | upstream | ours |
|---|---:|---:|
| VRAM delta, voice profile (no reference encode) | 6.54 GB | **4.34 GB** (-34%) |
| VRAM delta, raw reference (40.8 s encoded in-request) | **7.02 GB** | 8.66 GB |
| Peak working set | 1.24-1.25 GB | **1.15-1.16 GB** |
| KV cache for this request | 0.325 GB | 0.360 GB |

The profile path is the intended deployment: the reference is encoded once
(`--prepare-voice`), and every request then runs at ~4.3 GB. The raw path pays
for the encoder peak, and there our chunked encoder uses ~1.6 GB more than
upstream's single-pass one; that is the one memory metric where upstream is
better, and it only applies when the client sends the reference audio itself.

Runs: `bench/mem_breakdown.py`, tags `ours-voice`, `ours-rawref`,
`upstream-voice`, `upstream-rawref` in `runs/mem-breakdown/`.

## Why the difference

- Stateful codec (`stream_begin/stream_decode/stream_end`) instead of re-decoding
  the prefix per chunk — removes the quadratic growth and most of the TTFA wait.
- Flash attention and CUDA graphs for the decode steps.
- Fast-AR with a KV cache instead of rebuilding the whole prefix per codebook.
- One codebook-embedding gather per frame instead of ten.
- Sampling-window and fast embeddings kept on the GPU.
- Codec on a worker thread, overlapped with the AR.
- Norm weights pre-cast to F32 and RMS_NORM+MUL fused.
- KV length rounded to 256 to enable the GQA attention path.
- KV cache sized to the request instead of a fixed 32k context.
