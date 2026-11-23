# Upstream vs this fork: RTF, TTFA, memory

Same machine, same model, same voice profile, same sampling parameters. Only the
engine binary differs.

- GPU: RTX 4080 SUPER, 16 GB, CUDA 13.2, Windows 11
- Model: `s2-pro-q4_k_m.gguf` for RTF/memory, `s2-pro-q8_0.gguf` for TTFA (the harness default)
- Reference: `voice-reference.wav` (40.8 s) + transcript, saved profile
  `runs/v020-line8-baseline/voice/bench.s2voice`
- Params: temp 0.8, top_p 0.8, top_k 30, max_tokens 1024, stride 4
- Upstream: `forks/s2-upstream/build/s2.exe`, source `2c33261` (no seed support)
- Ours: `Main/build/s2.exe` at `49c67ba` (encoder restored, first frame streamed, request queue)

Upstream has no `--seed`, so its audio is a different draw. RTF and TTFA are
normalised by audio duration / wall time, so the comparison remains valid.

## RTF, 8 tests, CLI streaming (q4_k_m)

| test | upstream | ours | speed-up |
|---:|---:|---:|---:|
| 1 (35 s) | 2.971 | 0.298 | 10.0x |
| 2 (36 s) | 3.046 | 0.297 | 10.2x |
| 3 (5.1 s) | 0.833 | 0.347 | 2.4x |
| 4 (5.3 s) | 0.841 | 0.345 | 2.4x |
| 5 (3.8 s) | 0.872 | 0.374 | 2.3x |
| 6 (4.4 s) | 0.861 | 0.348 | 2.5x |
| 7 (3.1 s) | 0.895 | 0.378 | 2.4x |
| 8 (5.0 s) | 0.857 | 0.364 | 2.4x |

Upstream's cost grows with the length of the utterance: its codec re-decodes the
growing prefix for every chunk (quadratic), which is why the 35 s tests are ~3.0
while the 3-5 s tests are ~0.86. This fork keeps the codec state, so its RTF is
flat. (Our RTF is slightly lower than in the first comparison because the
reference encoder was restored to the pre-port implementation; see below.)

Runs: upstream `runs/v077-upstream-full`, ours `runs/v084-merged-full`.

## TTFA, 8 tests, server (q8_0)

| mode / config | upstream | ours | speed-up |
|---|---:|---:|---:|
| profile, `buffer0` | 2.943-6.915 | **0.064-0.099** | 30-108x |
| profile, `low_latency` | 0.385-0.470 | **0.063-0.165** | 2.3-7.5x |
| profile, `buffer3000` | 3.389-15.927 | **1.191-1.340** | 2.5-13.4x |
| raw, `buffer0` (reference encoded in the request) | 4.097-7.836 | **0.890-1.027** | 4.0-8.8x |

Upstream's `buffer0` waits for its own decode cadence before emitting the first
chunk; `low_latency` forces a faster first byte but costs more decode work. Our
streaming path emits the first PCM after the first 4-frame chunk because the
codec is stateful and the graph is already warm. Our raw path is faster now that
the reference encoder is the pre-port one.

Runs: upstream `runs/v080-upstream-ttfa`, ours `runs/v085-merged-ttfa`, `runs/v089-*` and `runs/v092-ttfa-final` (first-frame streaming). Ranges span the runs; the machine was shared, so single runs vary by ~20 ms.

## Memory (q4_k_m, one 35 s synthesis)

| metric | upstream | ours |
|---|---:|---:|
| VRAM delta, voice profile (no reference encode) | 6.54 GB | **4.35 GB** (-33%) |
| VRAM delta, raw reference (40.8 s encoded in-request) | 7.02 GB | **7.00 GB** |
| Peak working set | 1.24-1.25 GB | **1.14-1.17 GB** |
| KV cache for this request | 0.325 GB | 0.360 GB |

With the profile prepared once (`--prepare-voice`), every request runs at ~4.3 GB.
The raw-reference peak now matches upstream; the earlier 8.66 GB figure was the
newer encoder, which was reverted (see below).

Runs: `bench/mem_breakdown.py`, tags `ours-voice`/`ours-rawref` (first
comparison), `merged-voice`/`merged-rawref` (after the merge), `upstream-voice`/
`upstream-rawref`.

## Why the difference

- Stateful codec decoder (`stream_begin/stream_decode/stream_end`) instead of
  re-decoding the prefix per chunk: removes the quadratic growth and most of the
  TTFA wait. The codec *encoder* is the pre-port implementation (see below).
- The first streamed chunk is one frame, not a full stride: the client gets the
  first PCM as soon as one frame is decoded instead of four.
- Flash attention and CUDA graphs for the decode steps.
- Fast-AR with a KV cache instead of rebuilding the whole prefix per codebook.
- One codebook-embedding gather per frame instead of ten.
- Sampling-window and fast embeddings kept on the GPU.
- Codec on a worker thread, overlapped with the AR.
- Norm weights pre-cast to F32 and RMS_NORM+MUL fused.
- KV length rounded to 256 to enable the GQA attention path.
- KV cache sized to the request instead of a fixed 32k context.

## The encoder was restored

The port also rewrote the reference encoder, and its numerically equivalent but
different ops (snake division vs reciprocal multiply, `ggml_conv_1d` vs a
gather+matmul) flipped 1,322 of 8,790 reference codes (15.0%). Listening
preferred the old codes, so the encoder path was reverted while the decoder
stayed stateful. The resulting profile is byte-identical to the pre-port one, the
full 8-test WER is 0.0015 (was 0.0374), the raw-reference peak is 7.00 GB (was
8.66), and `ref_encode` is ~813 ms (was ~950). The branch and verification are
recorded in [DECISIONS.md](DECISIONS.md).
