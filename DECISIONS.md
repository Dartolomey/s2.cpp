# Decision log

Why the engine looks the way it does. Format: date — decision — reason — how it was verified — who insisted.

Numbers in this file come from the local `runs/<tag>` directories. Those are not
part of the repository: they hold model-specific audio, logs and timings. A run tag
is given for every measurement so it can be reproduced with the harness in `bench/`.

## 2026-11-22

### Engine baseline
Decision: the baseline is a clean `rodrigomatta/s2.cpp` @ `2c33261` in `Main`.
Reason: controlled measurements; no mixed states.
Verified: `git log` = 2c33261, no int4 files in `src/`, Release/CUDA SM89 build.
Insisted: user.

### Rolling back the "upstream" tree
Decision: the uncommitted PR #50/#51 changes in the copied `engines/s2-repro` tree (semantic window, CUDA graphs, single-pass offline codec) were removed and the tree rebuilt.
Reason: the tree was presented as upstream but contained modifications; the previous RTF numbers 2.337/4.825 were not measured on a clean base.
Verified: `git status` clean, `s2.exe` size 945,664 instead of 1,777,152.
Insisted: agent, user agreed.

### Model used for measurements
Decision: `s2-pro-q8_0.gguf` is used instead of `s2-pro-bf16` (safetensors).
Reason: s2.cpp only reads GGUF; the BF16 directory has no `codec.pth`, so it cannot be exported without downloading more files.
Verified: engine `--help`, directory contents, availability of the exporter.
Insisted: user.

### Reference voice
Decision: the reference is encoded from scratch for every version, the `.s2voice` profile is saved, its sha256 is recorded and compared across versions; within one version the tests share a single profile.
Reason: earlier iterations did not refresh the reference, so it was impossible to tell when the voice broke.
Verified: the profile is created with `--save-voice`, the hash is written to `runs/<tag>/voice/sha256.txt`.
Insisted: user.

### Seed
Decision: added `--seed` (CLI and `seed` in the server params). The seed is applied at the start of every generation; the value is stored in `runs/seed.txt` and reused.
Reason: upstream had no seed and the sampler used `std::random_device`, so the same input produced different audio and WAV comparisons between versions were impossible.
Verified: two runs with `--seed 42` produced the same WAV sha256; `--seed 43` produced a different one.
Commit: `50426b5`.
Insisted: user.

### Temperature and generation parameters
Decision: `temperature=0.8`, `top_p=0.8`, `top_k=30`, `max_tokens=1024`, `stride=4`; the values are stored in `runs/params.json` and only changed by explicit flags.
Reason: temperature affects expressiveness and emotion, not just repeatability, so it must not silently change between versions.
Note: the old `bench/bench.py` used 0.7/0.7/30, so its numbers are not comparable with these.
Verified: the values are written to each run's `meta.json`.
Insisted: user.

### TTFA
Decision: TTFA is measured from sending the HTTP request to the first PCM byte, in two modes: `profile` (no reference encoding) and `raw` (with `reference` + transcript).
Reason: the earlier 4.83 s figure was CLI streaming RTF, not TTFA; an agent needs the delay until the first sound.
Verified: `http_generate` in `bench/main_bench.py`.
Insisted: user.

### Fast measurement mode
Decision: by default, offline RTF on all 6 tests, stream RTF on tests 1 and 3, and TTFA profile/raw on all 6 with the request aborted after the first chunk.
Reason: a full run takes about 20 minutes, which is too long for iteration.
Verified: the TTFA request closes after the first chunk and the server cancels generation.
Insisted: user (complaint about duration).

### Known limitations of the current harness
- `wave` cannot read the engine's float32 WAV, so duration is measured with `ffprobe`.
- Server access to 127.0.0.1 was blocked by the environment; the probe hung in `SynSent`. The user opened access.
- The old `bench/out` and `bench/results.json` belong to earlier parameters and are not a baseline.

### PR 50 + 51 + 52
Decision: merged #50 (flash attention + CUDA graphs), #52 (CUDA grid fixes) and #51 (single-pass offline codec) into `exp/pr50-52-57`; #57 was postponed.
Reason: #50+#52 gave 1.75x on the AR, but offline was slow in decode; #51 removed a 15x re-decode of the prefix. #57 conflicts with #50 (both cache fast-AR decode).
Verified (test 1, q8_0, seed 42):
- offline RTF 2.599 → 0.791; decode 33.4 s → 2.45 s (13.6x); kv_init 7.9 ms;
- stream RTF 5.466 → 4.046; stream_decode 117.5 s dominated and was not touched;
- the voice profile sha256 did not change.
Run: `runs/v002-pr50-52-51`.
Status: waiting for the quality gate (ASR/listening); the WAV differs from v001 because flash attention/graphs changed the numerics and the sampling diverged.

### Stateful codec from s2-ilaria
Decision: ported `src/s2_codec.cpp`/`include/s2_codec.h` from ilaria into `exp/stateful-codec` and switched the streaming loop to `stream_begin/stream_decode/stream_end`; each frame is decoded once, holdback and decode_context are gone.
Reason: in v002, 82% of the stream was `stream_decode` (117.5 s of 143.5).
Verified (test 1): stream RTF 4.046 → 0.749; stream_decode 117.5 → 1.2 s (decode_avg 1.56 ms/frame); TTFA profile 24.78 → 2.50 s, raw 25.87 → 3.67 s; offline was unaffected (0.774).
Side effect: the voice profile sha256 changed (7f6bc9… → 0a226b…) because the codec encoder changed. Cloning quality still needs to be checked.
Runs: `runs/v003-stateful-smoke`, `runs/v004-stateful-full`.
Status: waiting for the quality gate (listening to 1/3/6, voice comparison).

### TTFA: server start buffer
Decision: measure the primary TTFA with `stream_start_buffer_ms=0`; the harness reads the first 4 KB chunk instead of 64 KB.
Reason: the server buffers 3000 ms of audio by default before the first byte, so the old 24.8/2.5 s numbers reflected that buffer, not latency.
Verified (7 tests, v005): profile buffer3000 2.43–2.55 s; buffer0 0.28–0.33 s; low_latency 0.28–0.33 s; raw buffer0 1.37–1.49 s (encoding ~1.1 s).
Consequence: `low_latency` is unnecessary for TTFA; buffer0 at stride 4 is enough for an agent.
Run: `runs/v005-ttfa`.

### GPU copies of embedding tables (part of PR57)
Decision: tried separately (`exp/gpu-embed`, commits `ffree5b`+`fa57f9d`), then reverted.
Reason: no gain and memory-bound. Test 1: stream RTF 0.792 vs 0.749; fast_decode 18.8 ms vs 16.8; the offline run was killed by the watchdog with 514 MB VRAM free (the copy took +424 MB).
Verified: `runs/v007-gpu-embed`, voice profile unchanged (0a226b…).
Consequence: do not enable without fallback logic and memory headroom. The branch is kept for reference.

### Fast-AR KV: diagnosis
- A full merge of PR57 into our stack built fine, but the long offline test climbed to 15.7 GB VRAM and hung. Both the PR50 fast-graph cache and the PR57 KV cache were alive at the same time.
- Disabling only the PR50 cache (no KV): the offline test was killed by the watchdog at 656 MB free VRAM; fast_decode 18.9 ms (worse). The graph is rebuilt for every codebook, the scheduler is not reset and memory grows.
- PR57 alone (`exp/pr57-alone`, test 3): works without OOM, AR 47–51 ms/frame, offline/stream RTF 1.18. So PR57 itself is fine; the problem is the two mechanisms coexisting.
Conclusion: do it as one step — "remove the PR50 cache, add the PR57 KV" — with no intermediate state and no GPU embedding copies.

### Fast-AR KV: port (branch exp/fastar-kv2)
Decision: removed the PR50 fast-graph cache and added the PR57 KV path with CPU embeddings; slow-AR graphs, PR51, the stateful codec and the seed were kept.
Verified (stream-only, 7 tests, q8_0):
- RTF 0.716–0.829; test 1: 0.749 → 0.716, test 3: 0.841 → 0.772;
- fast_decode 16.8 → 15.1 ms/frame; AR 25.2 → 24.5 s;
- TTFA profile 0.28–0.34 s, raw 1.38–1.54 s (buffer0);
- voice profile unchanged.
Side effect: the offline path is memory-bound (single-pass codec + KV), so offline is no longer used — files are written through the streaming path.
Status: waiting for listening; the gain is modest (~4–8%).

### Publication goal
Decision: the project is positioned as a standalone engine to hand over and publish, not a one-off bench. Recorded in `AGENTS.md` section 10.
Consequences: generation parameters (`top_k`, `top_p`, `temperature`, `seed`, `max_tokens`, `stride`, buffer) are set from outside at every level; public interfaces are a contract; the build is reproducible; documentation ships in the same commit.
Verified: `top_k` already flows through the CLI, HTTP `params` and the C API; partial top-k uses `params.top_k`, without a constant.

### Partial top-k in the sampler
Decision: replaced the full logit sort with a `partial_sort` over `top_k` (q4, test 1).
Reason: 10 samplers per frame sorted 4096 logits each; sorting all of them is unnecessary.
Verified: RTF 0.539 → **0.503**; fast_decode 11.84 → 10.63 ms; slow-AR 11.19 → 10.87 (noise); the audio was **bit-identical** to v013 (sha256 `4F8249…`).
Run: `runs/v014-q4-partial`. Commit: `sampler: partial top-k...`.

### Q4_0 + MSE: phases 0/1
Decision: built a mixed `s2-q4m.gguf` (120 tensors `wo/w1/w3` → Q4_0 with MSE scale, everything else unchanged) and checked the forced MMQ path for Q4_0.
Verified (test 1, stream): q4_k_m 0.503; q4m on stock Q4_0 0.521; q4m with forced MMQ 0.512. Slower in both cases; slow-AR 11.2–11.6 ms vs 10.9 for q4_k.
Conclusion: ggml's Q4_0 kernels are not faster than Q4_K on Ada; a custom Q4_0 kernel would have to beat ggml by more than 5–8% just to break even — an unjustified risk. The experiment was reverted; `models/s2-q4m.gguf` is kept for a possible quality check.

### Semantic embedding slice in VRAM
Decision: the 4096 rows of the semantic window from the tied embeddings are copied to the GPU once (F16, +21 MB); logits are computed with a direct `mul_mat` over the slice, without `get_rows`.
Reason: with GPU `get_rows`, Nsight showed the scheduler still moved the slice to the CPU and moved 42 MB/frame D2H plus H2D.
Verified (q4, 7 tests): RTF 0.539–0.673 → **0.495–0.609**; test 1 slow-AR 10.87 → 10.28 ms; D2H 4170 MB → 18 MB.
Runs: `runs/v019-embedwin-full`. Commit: `model: keep sampling-window embeddings on the GPU`.

### Tier 0 speedups (byte-identical)
- One `get_rows` for all 10 codebooks + views (was 10 gathers): slow-AR 10.28 → 8.68 ms, full RTF 0.495–0.609 → 0.395–0.500. Commit `760d315`.
- F32 copy of `fast_embeddings` in VRAM (42 MB), removed 9 CPU gathers/frame: line 8 0.473 → 0.465. Commit `ae7140a`.
- Codec decoding on its own thread with a chunk queue (same stride-4 boundaries): full RTF 0.395–0.500 → **0.382–0.476**, line 8 0.465 → 0.450. Commit `pipeline: decode codec chunks on a worker thread`.
- All three WAVs on test 8 were byte-identical to the `B356BE41…` baseline.
- The KV buffer is reused between requests (`init_kv_cache` used to allocate a new one and never free the old one — a 170–325 MB leak per request in the server). Byte-identical, TTFA unchanged (~0.24 s profile / ~1.15 s raw).

### Re-prefill fast-AR (A/B, rejected)
Decision: added an experimental `S2_FASTAR=refill` path — one cached graph with fixed length 10, padded prefix, causal mask, no KV.
Result (line 8): RTF 0.492 vs 0.450 with KV; AR 19.0 vs 16.8 ms; the WAV **did not match** (101 frames vs 87). A full prefix through attention gives different rounding and more work; dropping KV does not pay off here, unlike Python/paged KV in Qwen3.
Conclusion: the path was kept behind the flag for reference; the default is KV. (Later removed during release cleanup.)

### New test line 8
`[angry]Ну всё, собирайся, мы идем гулять, погода просто невероятная!` — baseline `runs/v020-line8-baseline` (hash `B356BE41…`), after Tier 0 — `runs/v024-line8-codec-thread.wav` (same hash).

### Prefix-KV per voice (TTFA; not byte-identical on a cache hit)
Decision: cache the KV of the fixed prompt head (system prompt + reference codes) per voice. A hit imports the prefix rows and prefills only the user text; a miss prefills the whole prompt in one graph (byte-identical) and exports the prefix after generation.
Interface: `params.prefix_kv_cache` (default **off**), HTTP field `prefix_kv_cache`, CLI `--prefix-kv`.
Reason for off by default: the split prefill uses different GEMM shapes, so a hit draws a different but valid sample.
Measurements (q8_0, profile, buffer0, 8 tests): TTFA 0.24–0.29 → **0.133–0.151 s** (≈45%), buffer3000 1.57–1.73 → 1.41–1.51 s; raw (no cache) unchanged.
Quality: WER (parakeet ultra) 0.047 without the cache vs 0.0485 with it (one word in test 2); `runs/v037-server-prefixkv` vs `runs/v038-server-noprefix`; WER in `runs/wer-prefixkv.json`. A repeat run `runs/v039-server-prefixkv-2` matched v037 on every WAV (determinism).
The miss path (first request per voice) is byte-identical: `runs/v040-line8-prefixkv-miss.wav` = `B356BE41…`.
Also fixed: a prefill now invalidates the cached decode graph; after KV reuse was introduced, the old graph survived the request and wrote into freed buffers (`GGML_ASSERT(buf != NULL)` in the server).
TTFA runs: `runs/v036-ttfa-prefixkv`. Commit `083faf2`.
Control full run after the change: `runs/v041-prefixkv-full` — RTF 0.342–0.427, all 8 WAVs byte-identical to `runs/v025-codec-thread-full` (the CLI path was unchanged; part of the RTF difference is GPU state).

### "Whole 9-codebook loop in one graph" — measured, not done
Line 8 profile (q4_k_m, temporary timers in `run_fast_graph`/`eval_cached`, then removed):
- fast-AR 7.6–7.9 ms/frame: upload 0.14–0.17, compute **7.1–7.5**, download 0.43–0.50.
- slow-AR 7.9–8.1 ms/frame: upload 0.16, compute **7.54**, download 0.07.
94–95% of the time is compute itself; syncs and launches account for ~0.5 ms/frame, so a fused loop with device-side sampling would buy at most 3%, not 10–20%.
Bandwidth: a pure read benchmark on this 4080S (`tmp/bw.cu`) reaches 700 GB/s; the engine reads ~2.38 GB (slow) + ~2.4 GB (fast) per frame and sustains 317–330 GB/s.
Batch-1 matvec already uses `mul_mat_vec_q` (dispatch picks it before MMQ); forcing MMQ is not faster: RTF 0.413 vs 0.391 within ±5% noise, WAV byte-identical.
Conclusion: RTF is now limited by bytes per frame (quantization) or by a dedicated matvec kernel; the loop graph does not pay off.

### Refactor guided by Nsight (Tier 0.5)
Profiling with `--cuda-graph-trace=node` showed ~1840 kernels per frame before the changes, ~4.1 ms of them small elementwise/copy kernels, and that matvec already sustains ~570 GB/s (2.82 GB/frame across slow+fast).
- **Norm weights cast to F32 once + passed directly to `ggml_mul`** (removed the per-replay cast+repeat) enables the CUDA backend's `RMS_NORM+MUL` fusion. In the trace: `op_repeat` 20,159 → 176 calls, `cpy half→float` 21,053 → 1,334, standalone `op_mul` 22,249 → 2,442; ~42 ms of GPU time on a 4 s line. WAV byte-identical (`B356BE41…`). Commit `953f964`.
- **KV length rounded up to 256** enables flash attention's GQA path (four query heads per KV head, 4x fewer K/V reads). Long test: slow 7.6 → 6.7 ms/frame, RTF 0.321–0.326 → **0.304**; all 8 tests RTF 0.342–0.427 → **0.305–0.393**. The tensor-core GQA path rounds differently, so samples change: WER 0.047 → **0.0374**, the run is deterministic (`runs/v059-gqa-full`, `runs/wer-gqa.json`). VRAM +up to 37 MB. Commit `857df43`.
Remaining: decode attention is now ~0.8 ms/frame; prefill (190 ms per request) is unaffected by the padding — there K/V has the exact length n_past+n_tokens and attention is not the bottleneck.

### How close to the hardware limit
One frame reads 4.63 GB of weights (2.21 GB slow AR + 2.42 GB across the nine fast-AR steps). A pure streaming read on this 4080S reaches ~700 GB/s (measured with `tmp/bw.cu`); the engine's matvec kernels sustain ~570 GB/s, 81% of that ceiling. The matvec floor is therefore ~6.6 ms/frame and the engine spends ~8.1.
The rest of the frame: attention ~0.8 ms, norms/elementwise ~1.9 ms, sampling ~0.4 ms, codec (worker thread) ~0.3 ms exposed. Prefill runs at ~138 TFLOPS effective (0.1 ms/token), the practical ceiling of ggml's quantized matmul here.
Measured launch overhead: removing every host sync from the fast-AR loop would save at most ~0.5 ms/frame (~3%). The remaining headroom is bytes per frame, not scheduling.

### Upstream vs this fork (measured)
Same machine (RTX 4080 SUPER, Windows 11, CUDA 13.2), same voice profile, same parameters. RTF and memory on q4_k_m, TTFA on q8_0 (the harness default).
- RTF, 8 tests: upstream 2.971/3.046 on the 35 s tests and ~0.86 on the 3–5 s tests; ours 0.309/0.310 and 0.351–0.397. Speed-up 9.6–9.8x on long utterances, 2.2–2.4x on short ones. Upstream's cost grows with the utterance because its codec re-decodes the growing prefix every chunk; ours is flat.
- TTFA (8 tests): profile `buffer0` 2.943–6.915 s → 0.122–0.138 s (21–57x); `low_latency` 0.385–0.470 → 0.120–0.139 (2.8–3.9x); `buffer3000` 3.389–15.927 → 1.153–1.370 (2.5–13.8x); raw reference 4.097–7.836 → 1.030–1.146 (3.6–7.6x).
- Memory: voice-profile path 6.54 GB → 4.34 GB VRAM (-34%), peak WS 1.24 GB → 1.15 GB. The raw-reference path is the one metric where upstream is better: 7.02 GB vs our 8.66 GB, because our chunked encoder peaks higher; the intended deployment encodes the reference once (`--prepare-voice`) and serves every request at ~4.3 GB.
Runs: upstream `runs/v077-upstream-full`, `runs/v080-upstream-ttfa`; ours `runs/v074-release-final`, `runs/v066-ttfa-nomemset`; harness `bench/mem_breakdown.py`. Full report: `PERFORMANCE.md` (repo) and `runs/COMPARE-upstream-vs-ours.md`.

### What the stateful codec port did to the voice
The port replaced the whole codec file (encoder and decoder). Comparing the saved reference profiles (stored codes, no synthesis):
- v001 (old codec) vs v002 (old codec): 0 of 8,790 codes differ — the old path was deterministic.
- v002 (old) vs v003 (stateful): 1,322 codes differ (15.0%), 40.2% of frames; per codebook `[29, 45, 98, 105, 105, 129, 169, 198, 211, 233]`.
- v003 vs v020: 0 differences — the encoder has not changed since, so every later optimization ran on the same reference codes.
Reading: RVQ codebook 0 carries the coarse timbre and flips in only 29/879 frames (3.3%); differences grow towards the fine residual books. So the port shifted fine texture and touched the coarse envelope occasionally — the same speaker, a slightly different rendering — rather than replacing the voice. The cause is the encoder's new numerics (conv/weight layout, float order): a nearest-neighbour VQ is discontinuous, so a tiny perturbation flips indices even though the encoder logic did not change.
The decoder itself cannot change the voice: it renders codes. Stateful decode is mathematically the same causal computation as the offline one (carried conv tails and attention KV), differing only in floating-point order; a real bug would show as clicks repeating every chunk (46 ms), which listening and ASR did not show.

Where the gain came from: the **decoder**, not the encoder. v002 → v003 on the 35 s test: `stream_decode` 117.5 s → 1.20 s (98x, 153.8 → 1.56 ms/frame), total RTF 4.05 → 0.74 (5.4x), with the AR unchanged at ~25.2 s. The old path held back 144 frames (6.7 s of audio) and re-decoded the prefix. The encoder is not in the streaming loop at all: `ref_encode` is ~0.95 s per reference (old path measured 1.16–1.35 s, upstream 0.89 s in one run — the same order, no gain), and it is the whole difference between raw TTFA (~1.03 s) and profile TTFA (~0.12 s).

### Encoder restored (branch `exp/old-encoder`)
Listening preferred the old codes, so the encoder path was reverted while keeping the stateful decoder. The numeric cause: the port turned snake's division into a reciprocal multiply and the conv into gather+matmul; a nearest-neighbour VQ amplifies that into 15% different codes.
Verified after the revert:
- reference profile byte-identical to v002 (`7F6BC963…`), streaming audio with it byte-identical to the old-profile A/B;
- full 8 tests: WER 0.0015 (was 0.0374), RTF 0.295–0.377 (no regression);
- raw-reference peak 6.99 GB (was 8.66), `ref_encode` 814–828 ms (was 946–980).
The legacy ops are used only by the encoder (the decoder keeps the fast path). Commit `c362001`.
Merged into the working branch as `7c97c5b` and re-measured: full run `runs/v084-merged-full` (RTF 0.298–0.378, WER 0.0015), TTFA `runs/v085-merged-ttfa` (profile 0.123–0.140 s, raw 0.957–1.027 s), memory (profile path 4.35 GB, raw 7.00 GB, WS 1.14–1.17 GB). `PERFORMANCE.md` and README updated with the new peak (7.0 GB instead of 8.7).

### Streaming: first frame as soon as it exists; queue instead of 503
- The first streamed chunk used to be a full stride (4 frames), so a client waited for four decode steps before any PCM. The server now streams the first frame immediately (`stream_first_chunk_frames=1`, `--stream-first-chunk`): TTFA (q8, prefix-KV, 8 tests) 0.123–0.140 → **0.064–0.099 s** across two runs, low_latency 0.120–0.140 → 0.063–0.165 (the machine was shared; buffer0 is the stable one). File output keeps the stride, so files stay byte-identical.
- Audio effect: the codes are unchanged (same AR), the sample count is identical; only chunk-boundary rounding differs — max |Δ| 32–54 LSB, **SNR 66.9–67.2 dB**, ASR transcripts identical.
- Concurrent `/generate` requests now wait for the single synthesis slot instead of getting an immediate 503. Bounds: `--max-queued` (default 4 waiters) and `--queue-timeout-ms` (default 60 s); queue-full and timeout still return 503 with distinct messages. Verified: two concurrent requests both 200 (second waits for the first), `--max-queued 0` → immediate 503, `--queue-timeout-ms 1000` → 503 after 1.0 s.
- Still no batch inference: queued requests cost separate full passes (documented in README limitations).






### Prefill: measurement and conclusion
Breakdown (current build, q4_k_m):
- full prompt 1136 tokens: build+mask 1.4, alloc 1.0–9.1, upload 0.3–3.6, **compute 116–138 ms**;
- prefix-KV tail of 193 tokens (n_past=1093): 0.0 / 0.5 / 0.2 / **25.5 ms**; 207 tokens — 26.9 ms; 43 tokens — 13.2 ms.
Inside a full prefill: MMQ 66 ms, flash attention 12 ms, elementwise ~21 ms, gaps between kernels 12 ms, CPU dispatch ~14 ms. Efficiency ~138 TFLOPS, 0.1 ms/token — that is the MMQ ceiling.
Only the CPU dispatch (~10–12 ms for a repeated shape) can be cached, which would give ~9% TTFA but needs a padded graph and changes the numerics. Not done: the gain does not justify the risk.
Allocation on repeated requests is cheap (0.5 ms) — the scheduler reuses a larger graph's buffer; the 8–9 ms in the first request is `cudaMalloc`, not memset (memset of 300 MB ≈ 0.7 ms).

### Seed and "the fork sounds better"
Comparing against the fork (`forks/s2-upstream`, 2c33261) on the same texts: the difference is the **seed**, not the pipeline. The fork has no `--seed`, so every run draws a different sample; our fixed seed 42 gives an even, repeatable voice. User decision (23 Nov): keep the fixed seed — it gives a stable voice, but repeated phrases can feel monotonous; use `--seed -1` for variety.

### Release preparation
- Offline was removed from the launch options: the CLI always writes through the streaming sink and `Pipeline::synthesize` was deleted. The `--stream-file` flag was removed (a breaking interface change); the default stride is 4.
- The in-memory entry points (`synthesize_to_memory`, `synthesize_raw`, `synthesize_with_prompt_codes`) stay: the server uses them for non-streaming responses and so does the C API.
- Dead code removed: `SlowARModel::prefill` (the old chunked prefill), the `backend_requires_single_token_semantic_prefill` helper, the `S2_FASTAR=refill` experiment (padded graph, cache, environment variable) and its fields.
- Optimal variants are the defaults: streaming, codec follows the backend, stride 4; the non-optimal or numerically different ones are parameters (`--codec-auto`, `--codec-cpu`, `--prefix-kv`, `--stream-decode-stride`).
- Verification: all 8 tests after the cleanup were byte-identical to `runs/v062-recheck` (`runs/v071-release`). Commit `bad3e62`.

### Release: build, DLLs, silent failures
Found and fixed during the build audit:
- **The codec crashed on CPU** (`--codec-cpu`, CPU-only build): ggml's `conv_2d_dw` assert rejected the `[tap][channel]` kernel layout that `ggml_conv_2d_dw_direct` builds. Added `patches/ggml-cpu-conv2d-dw-kernel-layout.patch`; the CPU codec now produces the same duration (2.74 s) and a clean transcript. The GPU path is untouched.
- **Missing runtime DLLs** on a clean machine: `build.ps1` copied only `ggml*.dll`. It now copies `cublas64_*`/`cublasLt64_*` and the MSVC runtime, then runs `s2.exe --version` with a minimal `PATH`, so a build with a missing DLL fails here instead of producing a binary that dies silently with `0xC0000135`.
- `build.ps1` was in the local `.git/info/exclude`, so a clean clone could not build. It is now in the repository.
- Unknown CLI flags were ignored silently — they now print a warning.
- ggml wrote its DEBUG logs (the `CUDA Graph id N reused` spam) past `--log-level`; added a `ggml_log_set` bridge to our logger. `--log-level error` is now silent.
- Version `0.1.0` in CMake, `--version`, and a version line in `--help`; license headers in every source file; README: build from a clean clone, runtime dependency table, third-party licenses.
Verification: line 8 byte-identical after all changes (`C80B5F1B…`). Commit `4fadca4`.

Two more silent failures:
- **Two servers on one port.** Windows with `SO_REUSEADDR` lets a second process bind an already occupied port, so a server restart silently left two instances and requests went to a random one. `SO_EXCLUSIVEADDRUSE` is now set and the error prints the address. Commit `f916993`.
- **CUDA architecture defaulted to 89.** A binary built for the wrong SM only fails when CUDA is first used ("no kernel image is available"). `build.ps1` now detects the SM with `nvidia-smi`; `-Architectures` remains for manual selection. Commit `f916993`.

Full run after all changes: `runs/v074-release-final`, RTF 0.309–0.397, all 8 WAVs byte-identical to `runs/v062-recheck`.

### Verified configuration and reference preparation
- README now states which configuration is actually verified: Windows + CUDA (Ada, SM 89) only. Linux, CPU-only, Vulkan and Metal are marked unverified rather than advertised; the CPU codec path is fixed and checked, but no full CPU-only run was done.
- Added `--prepare-voice`: encode the reference and save the `.s2voice` profile, then exit without synthesizing. It is the separate reference-preparation step the server needs — a request that loads a profile skips the encode entirely (TTFA profile 0.13 s vs raw 1.0–1.1 s). Verified: 879 frames and codes identical to the baseline profile; only a trailing newline in the transcript passed with `-Raw` differed, so the file hash differed.
- Tokenizer note: `tokenizer.json` is a separate 13 MB file (the GGUF has only `vocab_size` and the audio pad id). A pruned model must ship a tokenizer whose ids match its vocabulary; the engine takes the path explicitly (`-t`) and looks up the special tokens by string, but the semantic range comes from the GGUF metadata, so both files must belong to the same model. A tokenizer without `<|im_end|>` now fails at load instead of generating to `max_tokens` (a pruned tokenizer that drops control tokens would never stop).
Commit `da925a8`, `b093af3`.

### Which quants are actually verified
The README model table used to present seven quants as if all had been tried. Only two have:
- `q4_k_m` — the working model: full 8-test CLI streaming runs, ASR round-trip, determinism; ~4.3 GB VRAM steady, ~1.0 GB RSS, peak ~8.7 GB while encoding a 40 s reference;
- `q8_0` — verified on the server path: full 8-test server suite (profile + raw), TTFA matrix, ASR round-trip.
The rest (`f16`, `q6_k`, `q5_k_m`, `q3_k`, `q2_k`) are upstream's published files and are marked unverified; the usage examples now use `q4_k_m`. The old VRAM table (7.2 GB for q4_k_m, 2.5 GB KV pre-allocated for 32k) was upstream's and contradicted our measurements: the KV cache is sized to `prompt + max_new_tokens` (0.17–0.33 GB) and the steady total is ~4.3 GB. Commit `ad8446e`.


