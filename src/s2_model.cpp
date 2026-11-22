#include "../include/s2_model.h"
#include "../include/s2_log.h"
#include "s2_ggml_utils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <thread>
#ifdef __linux__
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace s2 {

static int32_t resolve_n_threads(int32_t n_threads) {
    if (n_threads > 0) return n_threads;
    const auto hw = std::thread::hardware_concurrency();
    return static_cast<int32_t>(hw > 0 ? hw : 4);
}

static const char * backend_type_name(BackendType backend_type) {
    switch (backend_type) {
        case BackendType::CPU:    return "CPU";
        case BackendType::Vulkan: return "Vulkan";
        case BackendType::CUDA:   return "CUDA";
        case BackendType::Metal:  return "Metal";
    }
    return "Unknown";
}

static bool backend_requires_single_token_semantic_prefill(ggml_backend_t backend) {
#ifdef GGML_USE_CUDA
    if (backend != nullptr && ggml_backend_is_cuda(backend)) {
        return true;
    }
#endif
    return false;
}

ggml_tensor * SlowARModel::norm_weight_f32(ggml_tensor * w) {
    if (!w) return nullptr;
    if (w->type == GGML_TYPE_F32) return w;
    if (w->type != GGML_TYPE_F16) return nullptr;

    auto it = norm_f32_.find(w);
    if (it != norm_f32_.end()) return it->second;

    ggml_backend_t backend = backend_gpu_ ? backend_gpu_ : backend_cpu_;
    if (!backend) return nullptr;

    if (!ctx_norm_f32_) {
        ggml_init_params p = { 1024 * ggml_tensor_overhead(), nullptr, true };
        ctx_norm_f32_ = ggml_init(p);
        if (!ctx_norm_f32_) return nullptr;
    }

    const int64_t n = ggml_nelements(w);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx_norm_f32_, GGML_TYPE_F32, n);
    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(backend, ggml_nbytes(dst));
    if (!buf) return nullptr;
    ggml_backend_tensor_alloc(buf, dst, ggml_backend_buffer_get_base(buf));
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    norm_f32_bufs_.push_back(buf);

    std::vector<ggml_fp16_t> h(static_cast<size_t>(n));
    ggml_backend_tensor_get(w, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); ++i) f[i] = ggml_fp16_to_fp32(h[i]);
    ggml_backend_tensor_set(dst, f.data(), 0, f.size() * sizeof(float));

    norm_f32_[w] = dst;
    return dst;
}

void SlowARModel::free_norm_f32() {
    for (ggml_backend_buffer_t b : norm_f32_bufs_) ggml_backend_buffer_free(b);
    norm_f32_bufs_.clear();
    norm_f32_.clear();
    if (ctx_norm_f32_) ggml_free(ctx_norm_f32_);
    ctx_norm_f32_ = nullptr;
}

// ggml_mul broadcasts a 1-D weight over the rows, and the CUDA backend folds the
// multiply into the RMS_NORM kernel when the weight is F32 and applied directly.
ggml_tensor * SlowARModel::rms_norm_weighted(ggml_context * ctx, ggml_tensor * x,
                                             ggml_tensor * weight, float eps) {
    ggml_tensor * cur = ggml_rms_norm(ctx, x, eps);
    ggml_tensor * w = norm_weight_f32(weight);
    if (!w) {
        w = (weight->type == cur->type) ? weight : ggml_cast(ctx, weight, cur->type);
        w = repeat_checked(ctx, w, cur, "repeat:rms_norm");
    }
    return ggml_mul(ctx, cur, w);
}

static ggml_tensor * repeat_interleave_heads(ggml_context * ctx, ggml_tensor * x,
                                              int32_t repeat_factor) {
    if (repeat_factor == 1) return x;
    ggml_tensor * xf = (x->type != GGML_TYPE_F32) ? ggml_cast(ctx, x, GGML_TYPE_F32) : x;
    ggml_tensor * x4 = ggml_reshape_4d(ctx, ggml_cont(ctx, xf),
                                        xf->ne[0], 1, xf->ne[1], xf->ne[2]);
    ggml_tensor * target = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                               xf->ne[0], repeat_factor, xf->ne[1], xf->ne[2]);
    ggml_tensor * repeated = ggml_repeat(ctx, x4, target);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, repeated),
                           xf->ne[0], xf->ne[1] * repeat_factor, xf->ne[2]);
}

static ggml_tensor * last_token_view(ggml_context * ctx, ggml_tensor * x, int32_t n_tokens) {
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], (n_tokens - 1) * x->nb[1]);
}

static size_t align_up_size(size_t size, size_t alignment) {
    if (alignment == 0) {
        return size;
    }
    const size_t rem = size % alignment;
    return rem == 0 ? size : size + (alignment - rem);
}

static void free_backend_buffers(std::vector<ggml_backend_buffer_t> & buffers) {
    for (ggml_backend_buffer_t buffer : buffers) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
    buffers.clear();
}

static size_t total_backend_buffer_bytes(const std::vector<ggml_backend_buffer_t> & buffers) {
    size_t total = 0;
    for (ggml_backend_buffer_t buffer : buffers) {
        if (buffer != nullptr) {
            total += ggml_backend_buffer_get_size(buffer);
        }
    }
    return total;
}

static bool allocate_weight_buffers(ggml_backend_t backend,
                                    const std::vector<ggml_tensor *> & tensors,
                                    std::vector<ggml_backend_buffer_t> & out_buffers,
                                    size_t & total_bytes,
                                    size_t & max_buffer_bytes,
                                    std::string & error_message) {
    free_backend_buffers(out_buffers);
    total_bytes = 0;
    max_buffer_bytes = 0;
    error_message.clear();

    if (backend == nullptr || tensors.empty()) {
        return true;
    }

    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    max_buffer_bytes = ggml_backend_buft_get_max_size(buft);
    if (max_buffer_bytes == 0) {
        max_buffer_bytes = static_cast<size_t>(-1);
    }

    struct BufferChunk {
        size_t bytes = 0;
        std::vector<ggml_tensor *> tensors;
    };

    std::vector<BufferChunk> chunks;
    BufferChunk current_chunk;

    for (ggml_tensor * tensor : tensors) {
        const size_t alloc_size = align_up_size(
            ggml_backend_buft_get_alloc_size(buft, tensor),
            alignment
        );
        total_bytes += alloc_size;

        const bool exceeds_chunk_limit =
            current_chunk.bytes > 0 &&
            max_buffer_bytes != static_cast<size_t>(-1) &&
            (current_chunk.bytes + alloc_size) > max_buffer_bytes;

        if (exceeds_chunk_limit) {
            chunks.push_back(std::move(current_chunk));
            current_chunk = BufferChunk();
        }

        current_chunk.bytes += alloc_size;
        current_chunk.tensors.push_back(tensor);
    }

    if (!current_chunk.tensors.empty()) {
        chunks.push_back(std::move(current_chunk));
    }

    out_buffers.reserve(chunks.size());

    for (const BufferChunk & chunk : chunks) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, chunk.bytes);
        if (buffer == nullptr) {
            error_message =
                std::string("failed to allocate ") + ggml_backend_buft_name(buft) +
                " buffer chunk of size " + std::to_string(chunk.bytes);
            free_backend_buffers(out_buffers);
            return false;
        }

        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        out_buffers.push_back(buffer);

        auto alloc = ggml_tallocr_new(buffer);
        for (ggml_tensor * tensor : chunk.tensors) {
            if (ggml_tallocr_alloc(&alloc, tensor) != GGML_STATUS_SUCCESS) {
                error_message =
                    std::string("failed to place tensor '") + tensor->name +
                    "' in backend buffer chunk";
                free_backend_buffers(out_buffers);
                return false;
            }
        }
    }

    return true;
}

SlowARModel::SlowARModel() {}

SlowARModel::~SlowARModel() {
    // Free the cached decode-graph contexts before the schedulers (which hold
    // tensor pointers into them) and before ctx_buf_/fast_ctx_buf_ are destroyed.
    invalidate_cached_graphs();
    free_fast_cache();
    free_embeddings_window();
    free_fast_embeddings_gpu();
    free_refill_graph();
    free_norm_f32();

    if (fast_sched_)     ggml_backend_sched_free(fast_sched_);
    if (sched_)          ggml_backend_sched_free(sched_);

    if (kv_buf_) ggml_backend_buffer_free(kv_buf_);
    free_backend_buffers(weights_.model_bufs_gpu);
    free_backend_buffers(weights_.model_bufs_cpu);

    if (backend_gpu_)    ggml_backend_free(backend_gpu_);
    if (backend_cpu_)    ggml_backend_free(backend_cpu_);

    if (ctx_kv_)         ggml_free(ctx_kv_);

    weights_.ctx_w = nullptr;
}

bool SlowARModel::load_shared(gguf_context * ctx_gguf, const std::string & gguf_path, int32_t gpu_device, BackendType backend_type, int32_t n_gpu_layers) {

    backend_cpu_ = ggml_backend_cpu_init();
    if (!backend_cpu_) {
        std::cerr << "[Model] Failed to init CPU backend." << std::endl;
        return false;
    }

    struct gguf_init_params params = { true, &weights_.ctx_w };
    gguf_context * local_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!local_gguf) {
        std::cerr << "[Model] Failed to load GGUF from " << gguf_path << std::endl;
        return false;
    }

    gguf_free(local_gguf);

    S2_LOG_INFO_STREAM("[Model] Reading metadata from " << gguf_path << std::endl);

    auto get_u32 = [&](const char * key, uint32_t def) -> uint32_t {
        int id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { S2_LOG_WARN_STREAM("[GGUF] missing key: " << key << " (using default " << def << ")\n"); return def; }
        uint32_t v = gguf_get_val_u32(ctx_gguf, id);
        S2_LOG_INFO_STREAM("[GGUF] " << key << " = " << v << "\n");
        return v;
    };
    auto get_f32 = [&](const char * key, float def) -> float {
        int id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { S2_LOG_WARN_STREAM("[GGUF] missing key: " << key << " (using default " << def << ")\n"); return def; }
        float v = gguf_get_val_f32(ctx_gguf, id);
        S2_LOG_INFO_STREAM("[GGUF] " << key << " = " << v << "\n");
        return v;
    };
    auto get_bool = [&](const char * key, bool def) -> bool {
        int id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { S2_LOG_WARN_STREAM("[GGUF] missing key: " << key << " (using default " << (def?"true":"false") << ")\n"); return def; }
        bool v = gguf_get_val_bool(ctx_gguf, id);
        S2_LOG_INFO_STREAM("[GGUF] " << key << " = " << (v?"true":"false") << "\n");
        return v;
    };

    hparams_ = ModelHParams();

    std::string arch_prefix = "fish-speech.";
    {
        int arch_id = gguf_find_key(ctx_gguf, "general.architecture");
        if (arch_id >= 0) {
            std::string arch = gguf_get_val_str(ctx_gguf, arch_id);
            arch_prefix = arch + ".";
            hparams_.has_fast_decoder = (arch == "fish-speech");
            S2_LOG_INFO_STREAM("[Model] Architecture: " << arch << std::endl);
        }
    }

    hparams_.context_length      = (int32_t)get_u32((arch_prefix + "context_length").c_str(), 32768);
    hparams_.vocab_size          = (int32_t)get_u32((arch_prefix + "vocab_size").c_str(), 155776);
    hparams_.embedding_length    = (int32_t)get_u32((arch_prefix + "embedding_length").c_str(), 2560);
    hparams_.feed_forward_length = (int32_t)get_u32((arch_prefix + "feed_forward_length").c_str(), 9728);
    hparams_.block_count         = (int32_t)get_u32((arch_prefix + "block_count").c_str(), 36);
    hparams_.head_count          = (int32_t)get_u32((arch_prefix + "attention.head_count").c_str(), 32);
    hparams_.head_count_kv       = (int32_t)get_u32((arch_prefix + "attention.head_count_kv").c_str(), 8);
    hparams_.rope_freq_base      = get_f32((arch_prefix + "rope.freq_base").c_str(), 1e6f);
    hparams_.rms_norm_eps        = get_f32((arch_prefix + "attention.layer_norm_rms_epsilon").c_str(), 1e-6f);

    hparams_.codebook_size            = (int32_t)get_u32("fish_speech.codebook_size", 4096);
    hparams_.num_codebooks            = (int32_t)get_u32("fish_speech.num_codebooks", 10);
    hparams_.semantic_begin_id        = (int32_t)get_u32("fish_speech.semantic_begin_id", 151678);
    hparams_.semantic_end_id          = (int32_t)get_u32("fish_speech.semantic_end_id", 155773);
    hparams_.tie_word_embeddings      = get_bool("fish_speech.tie_word_embeddings", true);
    hparams_.attention_qk_norm        = get_bool("fish_speech.attention_qk_norm", false);
    hparams_.scale_codebook_embeddings = get_bool("fish_speech.scale_codebook_embeddings", false);

    if (hparams_.has_fast_decoder) {
        hparams_.fast_context_length   = (int32_t)get_u32("fish_speech.fast_context_length", 11);
        hparams_.fast_embedding_length = (int32_t)get_u32("fish_speech.fast_embedding_length", 2560);
        hparams_.fast_feed_forward_length = (int32_t)get_u32("fish_speech.fast_feed_forward_length", 9728);
        hparams_.fast_block_count      = (int32_t)get_u32("fish_speech.fast_block_count", 4);
        hparams_.fast_head_count       = (int32_t)get_u32("fish_speech.fast_head_count", 32);
        hparams_.fast_head_count_kv    = (int32_t)get_u32("fish_speech.fast_head_count_kv", 8);
        hparams_.fast_head_dim         = (int32_t)get_u32("fish_speech.fast_head_dim", 128);
        hparams_.fast_rope_freq_base   = get_f32("fish_speech.fast_rope_freq_base", 1e6f);
        hparams_.fast_rms_norm_eps     = get_f32("fish_speech.fast_layer_norm_rms_eps", 1e-6f);
        hparams_.fast_attention_qk_norm = get_bool("fish_speech.fast_attention_qk_norm", false);
        hparams_.fast_has_project_in   = get_bool("fish_speech.fast_project_in", false);
    }

    S2_LOG_INFO_STREAM("[Model] Layers: " << hparams_.block_count
              << ", Dim: " << hparams_.embedding_length
              << ", Vocab: " << hparams_.vocab_size
              << ", head_count: " << hparams_.head_count
              << ", has_fast_decoder: " << hparams_.has_fast_decoder << std::endl);

    const bool wants_gpu_backend =
        backend_type != BackendType::CPU &&
        (gpu_device >= 0 || backend_type == BackendType::Metal);

    if (n_gpu_layers == -1) {
        n_gpu_layers = wants_gpu_backend ? hparams_.block_count : 0;
    }

    if (n_gpu_layers < 0) {
        S2_LOG_WARN_STREAM("[Model] Warning: --gpu-layers " << n_gpu_layers << " is negative, clamping to 0." << std::endl);
        n_gpu_layers = 0;
    }
    if (n_gpu_layers > hparams_.block_count) {
        S2_LOG_WARN_STREAM("[Model] Warning: --gpu-layers " << n_gpu_layers << " exceeds block_count (" << hparams_.block_count << "), clamping." << std::endl);
        n_gpu_layers = hparams_.block_count;
    }
    n_gpu_layers_ = n_gpu_layers;
    S2_LOG_INFO_STREAM("[Model] GPU layers: " << n_gpu_layers_ << " / " << hparams_.block_count << std::endl);

    if (n_gpu_layers_ > 0 && wants_gpu_backend) {
#ifdef GGML_USE_CUDA
        if (!backend_gpu_ && backend_type == BackendType::CUDA) {
            const int32_t device_count = ggml_backend_cuda_get_device_count();
            if (device_count <= 0) {
                std::cerr << "[Model] CUDA requested, but no CUDA devices were found." << std::endl;
            } else if (gpu_device < 0 || gpu_device >= device_count) {
                std::cerr << "[Model] CUDA device index " << gpu_device
                          << " is out of range (available: 0-" << (device_count - 1)
                          << ")." << std::endl;
            } else {
                backend_gpu_ = ggml_backend_cuda_init(static_cast<size_t>(gpu_device));
                if (!backend_gpu_) {
                    std::cerr << "[Model] CUDA init failed for device " << gpu_device << "." << std::endl;
                }
            }
        }
#endif
#ifdef GGML_USE_VULKAN
        if (!backend_gpu_ && backend_type == BackendType::Vulkan) {
            const int32_t device_count = ggml_backend_vk_get_device_count();
            if (device_count <= 0) {
                std::cerr << "[Model] Vulkan requested, but no Vulkan devices were found." << std::endl;
            } else if (gpu_device < 0 || gpu_device >= device_count) {
                std::cerr << "[Model] Vulkan device index " << gpu_device
                          << " is out of range (available: 0-" << (device_count - 1)
                          << ")." << std::endl;
            } else {
                backend_gpu_ = ggml_backend_vk_init(static_cast<size_t>(gpu_device));
                if (!backend_gpu_) {
                    std::cerr << "[Model] Vulkan init failed for device " << gpu_device << "." << std::endl;
                }
            }
        }
#endif
#ifdef GGML_USE_METAL
        if (!backend_gpu_ && backend_type == BackendType::Metal) {
            backend_gpu_ = ggml_backend_metal_init();
            if (!backend_gpu_) {
                std::cerr << "[Model] Metal init failed." << std::endl;
            }
        }
#endif
    }

    if (n_gpu_layers_ > 0 && wants_gpu_backend && !backend_gpu_) {
        std::cerr << "[Model] Requested " << backend_type_name(backend_type)
                  << " backend is unavailable; aborting GPU offload." << std::endl;
        return false;
    }

    auto req_t = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(weights_.ctx_w, name.c_str());
        if (!t) {
            throw std::runtime_error("missing tensor: " + name);
        }
        return t;
    };

    try {
        weights_.embeddings          = req_t("embeddings.weight");
        weights_.codebook_embeddings = req_t("codebook_embeddings.weight");
        weights_.norm                = req_t("norm.weight");

        weights_.layers.resize(hparams_.block_count);
        for (int32_t i = 0; i < hparams_.block_count; ++i) {
            auto & layer = weights_.layers[i];
            std::string stem = "layers." + std::to_string(i) + ".";

            layer.attention_norm = req_t(stem + "attention_norm.weight");
            layer.ffn_norm       = req_t(stem + "ffn_norm.weight");
            layer.wqkv           = req_t(stem + "attention.wqkv.weight");
            layer.wo             = req_t(stem + "attention.wo.weight");
            layer.w1             = req_t(stem + "feed_forward.w1.weight");
            layer.w2             = req_t(stem + "feed_forward.w2.weight");
            layer.w3             = req_t(stem + "feed_forward.w3.weight");

            if (hparams_.attention_qk_norm) {
                layer.q_norm = req_t(stem + "attention.q_norm.weight");
                layer.k_norm = req_t(stem + "attention.k_norm.weight");
            }
        }

        if (hparams_.has_fast_decoder) {
            if (hparams_.fast_has_project_in) {
                weights_.fast_project_in = req_t("fast_project_in.weight");
            }
            weights_.fast_embeddings = req_t("fast_embeddings.weight");
            weights_.fast_norm       = req_t("fast_norm.weight");
            weights_.fast_output     = req_t("fast_output.weight");

            weights_.fast_layers.resize(hparams_.fast_block_count);
            for (int32_t i = 0; i < hparams_.fast_block_count; ++i) {
                auto & layer = weights_.fast_layers[i];
                std::string stem = "fast_layers." + std::to_string(i) + ".";

                layer.attention_norm = req_t(stem + "attention_norm.weight");
                layer.ffn_norm       = req_t(stem + "ffn_norm.weight");
                layer.wqkv           = req_t(stem + "attention.wqkv.weight");
                layer.wo             = req_t(stem + "attention.wo.weight");
                layer.w1             = req_t(stem + "feed_forward.w1.weight");
                layer.w2             = req_t(stem + "feed_forward.w2.weight");
                layer.w3             = req_t(stem + "feed_forward.w3.weight");

                if (hparams_.fast_attention_qk_norm) {
                    layer.q_norm = req_t(stem + "attention.q_norm.weight");
                    layer.k_norm = req_t(stem + "attention.k_norm.weight");
                }
            }
        }
    } catch (const std::exception & e) {
        std::cerr << "[Model] " << e.what() << std::endl;
        return false;
    }

    std::vector<ggml_tensor *> weight_tensors;
    weight_tensors.reserve(1 + 1 + 1 + 1 + 1 + 1 +
                           weights_.layers.size() * 7 +
                           weights_.fast_layers.size() * 7);

    auto push_unique_tensor = [&](ggml_tensor * tensor) {
        if (!tensor) {
            return;
        }
        if (std::find(weight_tensors.begin(), weight_tensors.end(), tensor) == weight_tensors.end()) {
            weight_tensors.push_back(tensor);
        }
    };

    weight_tensor_set_.clear();
    weight_tensor_set_.reserve(weight_tensors.capacity());

    push_unique_tensor(weights_.embeddings);
    push_unique_tensor(weights_.codebook_embeddings);
    push_unique_tensor(weights_.norm);
    push_unique_tensor(weights_.fast_project_in);
    push_unique_tensor(weights_.fast_embeddings);
    push_unique_tensor(weights_.fast_norm);
    push_unique_tensor(weights_.fast_output);

    for (const auto & layer : weights_.layers) {
        push_unique_tensor(layer.attention_norm);
        push_unique_tensor(layer.ffn_norm);
        push_unique_tensor(layer.q_norm);
        push_unique_tensor(layer.k_norm);
        push_unique_tensor(layer.wqkv);
        push_unique_tensor(layer.wo);
        push_unique_tensor(layer.w1);
        push_unique_tensor(layer.w2);
        push_unique_tensor(layer.w3);
    }

    for (const auto & layer : weights_.fast_layers) {
        push_unique_tensor(layer.attention_norm);
        push_unique_tensor(layer.ffn_norm);
        push_unique_tensor(layer.q_norm);
        push_unique_tensor(layer.k_norm);
        push_unique_tensor(layer.wqkv);
        push_unique_tensor(layer.wo);
        push_unique_tensor(layer.w1);
        push_unique_tensor(layer.w2);
        push_unique_tensor(layer.w3);
    }

    weight_tensor_set_.insert(weight_tensors.begin(), weight_tensors.end());

    const bool full_model_offload =
        backend_gpu_ != nullptr &&
        backend_type != BackendType::CUDA &&
        n_gpu_layers_ == hparams_.block_count;

    auto get_weight_backend = [&](const std::string & name) -> ggml_backend_t {

        if (full_model_offload) {
            return backend_gpu_;
        }

        if (name == "embeddings.weight" ||
            name == "codebook_embeddings.weight" ||
            name == "fast_embeddings.weight" ||
            name == "norm.weight") {
            return backend_cpu_;
        }

        if (name.rfind("fast_layers.", 0) == 0 ||
            name.rfind("fast_", 0) == 0) {
            return (n_gpu_layers_ > 0 && backend_gpu_) ? backend_gpu_ : backend_cpu_;
        }

        if (name.rfind("layers.", 0) == 0) {
            size_t dot_pos = name.find('.', 7);
            int layer = std::stoi(name.substr(7, dot_pos - 7));
            return (layer < n_gpu_layers_ && backend_gpu_) ? backend_gpu_ : backend_cpu_;
        }
        return backend_cpu_;
    };

    std::vector<ggml_tensor *> gpu_weight_tensors;
    std::vector<ggml_tensor *> cpu_weight_tensors;
    for (ggml_tensor * t : weight_tensors) {
        if (!t) continue;
        ggml_backend_t target = get_weight_backend(t->name);
        if (target == backend_gpu_) {
            gpu_weight_tensors.push_back(t);
        } else {
            cpu_weight_tensors.push_back(t);
        }
    }

    std::vector<ggml_backend_buffer_t> gpu_buffers;
    std::vector<ggml_backend_buffer_t> cpu_buffers;

    if (!gpu_weight_tensors.empty() && backend_gpu_) {
        size_t gpu_bytes = 0;
        size_t gpu_max_buffer = 0;
        std::string gpu_alloc_error;
        if (!allocate_weight_buffers(
                backend_gpu_,
                gpu_weight_tensors,
                gpu_buffers,
                gpu_bytes,
                gpu_max_buffer,
                gpu_alloc_error)) {
            const double requested_mb = gpu_bytes / (1024.0 * 1024.0);
            const double max_chunk_mb =
                gpu_max_buffer == static_cast<size_t>(-1)
                    ? 0.0
                    : gpu_max_buffer / (1024.0 * 1024.0);
            std::cerr << "[Model] GPU weight buffer allocation failed." << std::endl;
            std::cerr << "[Model] Requested GPU memory: " << requested_mb << " MB for "
                      << gpu_weight_tensors.size() << " tensors (" << n_gpu_layers_
                      << " layers)" << std::endl;
            if (gpu_max_buffer != static_cast<size_t>(-1)) {
                std::cerr << "[Model] Backend max buffer size: " << max_chunk_mb
                          << " MB per allocation." << std::endl;
            }
            if (!gpu_alloc_error.empty()) {
                std::cerr << "[Model] " << gpu_alloc_error << std::endl;
            }
            std::cerr << "[Model] Suggest using a lower --gpu-layers value (e.g., --gpu-layers "
                      << std::max(1, n_gpu_layers_ / 2) << ")" << std::endl;
            return false;
        }
    }

    if (!cpu_weight_tensors.empty()) {
        size_t cpu_bytes = 0;
        size_t cpu_max_buffer = 0;
        std::string cpu_alloc_error;
        if (!allocate_weight_buffers(
                backend_cpu_,
                cpu_weight_tensors,
                cpu_buffers,
                cpu_bytes,
                cpu_max_buffer,
                cpu_alloc_error)) {
            free_backend_buffers(gpu_buffers);
            std::cerr << "[Model] Failed to allocate CPU weight buffer." << std::endl;
            if (!cpu_alloc_error.empty()) {
                std::cerr << "[Model] " << cpu_alloc_error << std::endl;
            }
            return false;
        }
    }

    weights_.model_bufs_gpu = std::move(gpu_buffers);
    weights_.model_bufs_cpu = std::move(cpu_buffers);

    {
        ggml_backend_t backends[2];
        int n_backends;
        if (backend_gpu_) {
            backends[0] = backend_gpu_;
            backends[1] = backend_cpu_;
            n_backends = 2;
        } else {
            backends[0] = backend_cpu_;
            n_backends = 1;
        }

        sched_ = ggml_backend_sched_new(backends, NULL, n_backends, 32768, false, true);
        if (!sched_) {
            std::cerr << "[Model] Failed to create Slow-AR scheduler." << std::endl;
            return false;
        }

        if (hparams_.has_fast_decoder) {
            fast_sched_ = ggml_backend_sched_new(backends, NULL, n_backends, 16384, false, true);
            if (!fast_sched_) {
                std::cerr << "[Model] Failed to create Fast-AR scheduler." << std::endl;
                return false;
            }
        }
    }

    if (n_gpu_layers_ > 0 && backend_gpu_) {
        const int32_t first_cpu_layer = n_gpu_layers_;
        const int32_t last_gpu_layer = n_gpu_layers_ - 1;
        if (first_cpu_layer < hparams_.block_count) {
            S2_LOG_INFO_STREAM("[Model] Layers 0-" << last_gpu_layer << " on "
                      << ggml_backend_name(backend_gpu_)
                      << ", " << first_cpu_layer << "-" << (hparams_.block_count - 1) << " on CPU"
                      << std::endl);
        } else {
            S2_LOG_INFO_STREAM("[Model] Layers 0-" << (hparams_.block_count - 1) << " on "
                      << ggml_backend_name(backend_gpu_) << " (all)" << std::endl);
        }
    } else {
        S2_LOG_INFO_STREAM("[Model] All " << hparams_.block_count << " layers on CPU" << std::endl);
    }

    const size_t gpu_weight_bytes = total_backend_buffer_bytes(weights_.model_bufs_gpu);
    if (!weights_.model_bufs_gpu.empty()) {
        S2_LOG_INFO_STREAM("[Model] GPU weight buffers: " << weights_.model_bufs_gpu.size()
                  << " chunk(s), " << (gpu_weight_bytes / 1024.0 / 1024.0)
                  << " MB total" << std::endl);
    }
    const size_t cpu_weight_bytes = total_backend_buffer_bytes(weights_.model_bufs_cpu);
    if (!weights_.model_bufs_cpu.empty()) {
        S2_LOG_INFO_STREAM("[Model] CPU weight buffers: " << weights_.model_bufs_cpu.size()
                  << " chunk(s), " << (cpu_weight_bytes / 1024.0 / 1024.0)
                  << " MB total" << std::endl);
    }
    const size_t total_bytes = gpu_weight_bytes + cpu_weight_bytes;
    S2_LOG_INFO_STREAM("[Model] Total model size: "
              << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl);

    S2_LOG_INFO_STREAM("[Model] KV cache: " << (n_gpu_layers_ > 0 && backend_gpu_ ? "GPU" : "CPU")
              << ", n_gpu_layers=" << n_gpu_layers_ << std::endl);

    if (backend_type == BackendType::CUDA &&
        backend_gpu_ &&
        ggml_is_quantized(weights_.embeddings->type)) {
        S2_LOG_INFO_STREAM("[Model] Keeping quantized embedding tables on CPU for CUDA stability."
                  << std::endl);
    }

    return true;
}

bool SlowARModel::read_tensor_data(const std::string & gguf_path, gguf_context * ctx_gguf) {
    const size_t data_offset = gguf_get_data_offset(ctx_gguf);
    const int64_t n_tensors  = gguf_get_n_tensors(ctx_gguf);

    std::FILE * f = std::fopen(gguf_path.c_str(), "rb");
    if (!f) {
        std::cerr << "[Model] Cannot reopen " << gguf_path << " for data loading." << std::endl;
        return false;
    }
    std::vector<uint8_t> tmp;
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * tname = gguf_get_tensor_name(ctx_gguf, ti);
        ggml_tensor * t = ggml_get_tensor(weights_.ctx_w, tname);

        if (!t || weight_tensor_set_.find(t) == weight_tensor_set_.end()) continue;

        const size_t toff  = data_offset + gguf_get_tensor_offset(ctx_gguf, ti);
        const size_t tsize = ggml_nbytes(t);
        if (tmp.size() < tsize) tmp.resize(tsize);
#ifdef _WIN32
        _fseeki64(f, (int64_t)toff, SEEK_SET);
#else
        fseeko(f, (off_t)toff, SEEK_SET);
#endif
        if (std::fread(tmp.data(), 1, tsize, f) != tsize) {
            std::cerr << "[Model] Failed to read tensor: " << tname << std::endl;
            std::fclose(f);
            return false;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, tsize);
    }
    tmp.clear();
    tmp.shrink_to_fit();
    std::fclose(f);

    S2_LOG_INFO_STREAM("[Model] Weights loaded. Total tensors: " << n_tensors << std::endl);
    return true;
}

bool SlowARModel::load(const std::string & gguf_path, int32_t gpu_device, BackendType backend_type, int32_t n_gpu_layers) {

    struct gguf_init_params params = { true, nullptr };
    gguf_context * ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf) {
        std::cerr << "[Model] Failed to load GGUF from " << gguf_path << std::endl;
        return false;
    }

    if (!load_shared(ctx_gguf, gguf_path, gpu_device, backend_type, n_gpu_layers)) {
        gguf_free(ctx_gguf);
        return false;
    }

    if (!read_tensor_data(gguf_path, ctx_gguf)) {
        gguf_free(ctx_gguf);
        return false;
    }

    gguf_free(ctx_gguf);
    return true;
}

bool SlowARModel::init_kv_cache(int32_t max_seq_len) {
    // Round the cache length up to the flash-attention key stride (256). That
    // kernel's GQA path, where four query heads share one KV head's reads, only
    // engages when the key count is a multiple of the stride. The extra rows are
    // always masked out, so this only changes which kernel is used.
    max_seq_len = (max_seq_len + 255) / 256 * 256;
    // Reuse a KV buffer that is large enough: a new allocation per request leaks
    // VRAM in the server and adds a large cudaMalloc to every request.
    if (kv_buf_ && ctx_kv_ && max_seq_len_ >= max_seq_len) {
        n_past_ = 0;
        ggml_backend_tensor_memset(memory_k_, 0, 0, ggml_nbytes(memory_k_));
        ggml_backend_tensor_memset(memory_v_, 0, 0, ggml_nbytes(memory_v_));
        return true;
    }

    // The KV cache tensors (memory_k_/memory_v_) are re-created here, so any cached
    // decode graph that references them is stale.
    invalidate_cached_graphs();
    if (kv_buf_) { ggml_backend_buffer_free(kv_buf_); kv_buf_ = nullptr; }
    if (ctx_kv_) { ggml_free(ctx_kv_); ctx_kv_ = nullptr; }
    memory_k_ = nullptr;
    memory_v_ = nullptr;

    max_seq_len_ = max_seq_len;
    n_past_      = 0;

    const int32_t dim = hparams_.embedding_length;
    if (dim == 0) return true;

    int32_t head_dim = 0;
    if (hparams_.attention_qk_norm && !weights_.layers.empty() && weights_.layers[0].q_norm) {
        head_dim = static_cast<int32_t>(weights_.layers[0].q_norm->ne[0]);
    } else {
        head_dim = hparams_.embedding_length / hparams_.head_count;
    }

    const int32_t n_head_kv = hparams_.head_count_kv;
    const int32_t n_layer   = hparams_.block_count;

    const size_t ctx_kv_size = 2ull * ggml_tensor_overhead() + (1ull << 20);
    ggml_init_params p = {
         ctx_kv_size,
         nullptr,
         true,
    };
    ctx_kv_ = ggml_init(p);
    if (!ctx_kv_) {
        std::cerr << "[Model] Failed to init KV context." << std::endl;
        return false;
    }

    // Flash-attention-friendly layout: {head_dim, max_seq_len, n_head_kv, n_layer}.
    // (seq is the middle dim so a per-layer K/V view is contiguous for ggml_flash_attn_ext.)
    memory_k_ = ggml_new_tensor_4d(ctx_kv_, GGML_TYPE_F16, head_dim, max_seq_len, n_head_kv, n_layer);
    memory_v_ = ggml_new_tensor_4d(ctx_kv_, GGML_TYPE_F16, head_dim, max_seq_len, n_head_kv, n_layer);

    ggml_backend_t kv_backend = (n_gpu_layers_ > 0 && backend_gpu_) ? backend_gpu_ : backend_cpu_;
    kv_buf_ = ggml_backend_alloc_ctx_tensors(ctx_kv_, kv_backend);
    if (!kv_buf_) {
        std::cerr << "[Model] Failed to allocate KV cache buffer." << std::endl;
        return false;
    }

    ggml_backend_tensor_memset(memory_k_, 0, 0, ggml_nbytes(memory_k_));
    ggml_backend_tensor_memset(memory_v_, 0, 0, ggml_nbytes(memory_v_));

    S2_LOG_INFO_STREAM("[Model] KV cache allocated on "
              << (n_gpu_layers_ > 0 && backend_gpu_ ? ggml_backend_name(backend_gpu_) : "CPU")
              << ", size: " << (ggml_backend_buffer_get_size(kv_buf_) / 1024.0 / 1024.0)
              << " MB" << std::endl);

    return true;
}

void SlowARModel::reset() {
    n_past_ = 0;
}

void SlowARModel::invalidate_cached_graphs() {
    if (slow_ctx_) { ggml_free(slow_ctx_); slow_ctx_ = nullptr; }
    slow_gf_           = nullptr;
    slow_semantic_ids_ = nullptr;
    slow_positions_    = nullptr;
    slow_semantic_mask_ = nullptr;
    slow_n_past_idx_   = nullptr;
    slow_token_scale_  = nullptr;
    slow_attn_mask_    = nullptr;
    slow_hidden_last_  = nullptr;
    slow_logits_       = nullptr;
    slow_window_ids_t_ = nullptr;
    slow_cb_ids_ = nullptr;
    if (sched_) ggml_backend_sched_reset(sched_);

    if (fast_kv_buf_ || fast_kv_ctx_) free_fast_cache();
    if (fast_sched_) ggml_backend_sched_reset(fast_sched_);
}

void SlowARModel::clear_kv_cache() {
    // Keep the buffer: the next request reuses it (init_kv_cache only reallocates
    // when a larger sequence is requested). Stale rows past n_past_ are never read.
    n_past_ = 0;
}

// The KV cache is laid out as {head_dim, seq, n_head_kv, n_layer}: the prefix is
// the first n_prefix rows of every (layer, kv-head) block, so it is exported as
// [layer][kv_head][n_prefix][head_dim] for K and then for V.
bool SlowARModel::export_kv_prefix(int32_t n_prefix, std::vector<uint8_t> & out) const {
    if (!memory_k_ || !memory_v_ || !kv_buf_) return false;
    if (n_prefix <= 0 || n_prefix > max_seq_len_) return false;

    const int32_t n_layer = static_cast<int32_t>(memory_k_->ne[3]);
    const int32_t n_kv    = static_cast<int32_t>(memory_k_->ne[2]);
    const size_t  chunk   = static_cast<size_t>(n_prefix) * memory_k_->nb[1];
    const size_t  per_t   = static_cast<size_t>(n_layer) * n_kv * chunk;

    out.resize(2 * per_t);
    size_t pos = 0;
    for (const ggml_tensor * t : { memory_k_, memory_v_ }) {
        for (int32_t l = 0; l < n_layer; ++l) {
            for (int32_t h = 0; h < n_kv; ++h) {
                const size_t off = static_cast<size_t>(l) * t->nb[3] +
                                   static_cast<size_t>(h) * t->nb[2];
                ggml_backend_tensor_get(t, out.data() + pos, off, chunk);
                pos += chunk;
            }
        }
    }
    return true;
}

bool SlowARModel::import_kv_prefix(const std::vector<uint8_t> & in, int32_t n_prefix) {
    if (!memory_k_ || !memory_v_ || !kv_buf_) return false;
    if (n_prefix <= 0 || n_prefix > max_seq_len_) return false;

    const int32_t n_layer = static_cast<int32_t>(memory_k_->ne[3]);
    const int32_t n_kv    = static_cast<int32_t>(memory_k_->ne[2]);
    const size_t  chunk   = static_cast<size_t>(n_prefix) * memory_k_->nb[1];
    const size_t  per_t   = static_cast<size_t>(n_layer) * n_kv * chunk;
    if (in.size() != 2 * per_t) return false;

    size_t pos = 0;
    for (ggml_tensor * t : { memory_k_, memory_v_ }) {
        for (int32_t l = 0; l < n_layer; ++l) {
            for (int32_t h = 0; h < n_kv; ++h) {
                const size_t off = static_cast<size_t>(l) * t->nb[3] +
                                   static_cast<size_t>(h) * t->nb[2];
                ggml_backend_tensor_set(t, in.data() + pos, off, chunk);
                pos += chunk;
            }
        }
    }
    return true;
}

bool SlowARModel::prefill_fast(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                          int32_t n_threads, StepResult & result) {
    return eval_cached(flat_tokens, n_tokens, n_threads, result);
}

bool SlowARModel::prefill(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                          int32_t n_threads, StepResult & result) {
    if (n_tokens <= 0) {
        return false;
    }

    const int32_t codebook_dim = hparams_.num_codebooks + 1;
    int32_t chunk_tokens = n_tokens;
    if (backend_gpu_ != nullptr && n_gpu_layers_ > 0) {
        int32_t semantic_prompt_tokens = 0;
        for (int32_t t = 0; t < n_tokens; ++t) {
            const int32_t semantic = flat_tokens[static_cast<size_t>(t) * codebook_dim];
            if (semantic >= hparams_.semantic_begin_id &&
                semantic <= hparams_.semantic_end_id) {
                semantic_prompt_tokens++;
                if (semantic_prompt_tokens > 1) {
                    break;
                }
            }
        }

        if (semantic_prompt_tokens > 1 &&
            backend_requires_single_token_semantic_prefill(backend_gpu_)) {
            chunk_tokens = 1;
        } else {

            const int32_t adaptive_gpu_chunk = std::clamp(
                128 / std::max(1, n_gpu_layers_),
                8,
                64
            );
            if (n_tokens > adaptive_gpu_chunk) {
                chunk_tokens = adaptive_gpu_chunk;
            }
        }
    }

    if (chunk_tokens >= n_tokens) {
        return eval_cached(flat_tokens, n_tokens, n_threads, result);
    }

    S2_LOG_INFO_STREAM("[Model] Chunked prefill: " << n_tokens
              << " tokens in blocks of " << chunk_tokens << std::endl);

    StepResult chunk_result;
    std::vector<int32_t> chunk;
    chunk.reserve(static_cast<size_t>(chunk_tokens) * codebook_dim);

    for (int32_t start = 0; start < n_tokens; start += chunk_tokens) {
        const int32_t cur_tokens = std::min(chunk_tokens, n_tokens - start);
        const auto begin = flat_tokens.begin() + static_cast<ptrdiff_t>(start) * codebook_dim;
        const auto end = begin + static_cast<ptrdiff_t>(cur_tokens) * codebook_dim;
        chunk.assign(begin, end);
        if (!eval_cached(chunk, cur_tokens, n_threads, chunk_result)) {
            return false;
        }
    }

    result = std::move(chunk_result);
    return true;
}

bool SlowARModel::step(const std::vector<int32_t> & flat_tokens, int32_t n_threads,
                       StepResult & result) {
    return eval_cached(flat_tokens, 1, n_threads, result);
}

void SlowARModel::free_embeddings_window() {
    if (embed_buf_) ggml_backend_buffer_free(embed_buf_);
    if (ctx_embed_) ggml_free(ctx_embed_);
    embed_buf_ = nullptr;
    ctx_embed_ = nullptr;
    embeddings_window_ = nullptr;
    embeddings_window_ids_ = nullptr;
    embeddings_window_n_ = 0;
}

// Copy only the sampling-window rows of the tied embedding table to the GPU once.
// The full table stays on the CPU for text prefill; the logits head then reads the
// small GPU slice instead of uploading ~40 MB of rows every frame.
bool SlowARModel::ensure_embeddings_window(const std::vector<int32_t> & ids, int32_t n_threads) {
    const int32_t n = static_cast<int32_t>(ids.size());
    if (embeddings_window_ && embeddings_window_n_ == n) return true;
    free_embeddings_window();
    if (!backend_gpu_ || !weights_.embeddings || n <= 0) return false;

    const int32_t dim = hparams_.embedding_length;
    const size_t out_bytes = static_cast<size_t>(dim) * n * sizeof(float);
    ggml_init_params p = { 16ull * 1024 * 1024 + out_bytes, nullptr, false };
    ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;

    ggml_tensor * idt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    std::memcpy(idt->data, ids.data(), static_cast<size_t>(n) * sizeof(int32_t));
    ggml_tensor * rows = ggml_get_rows(ctx, weights_.embeddings, idt);
    if (!rows) { ggml_free(ctx); return false; }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);
    ggml_graph_compute_with_ctx(ctx, gf, resolve_n_threads(n_threads));

    ggml_init_params ep = { 4ull * ggml_tensor_overhead(), nullptr, true };
    ctx_embed_ = ggml_init(ep);
    if (!ctx_embed_) { ggml_free(ctx); return false; }
    embeddings_window_ = ggml_new_tensor_2d(ctx_embed_, GGML_TYPE_F16, dim, n);
    embeddings_window_ids_ = ggml_new_tensor_1d(ctx_embed_, GGML_TYPE_I32, n);
    embed_buf_ = ggml_backend_alloc_ctx_tensors(ctx_embed_, backend_gpu_);
    if (!embed_buf_) {
        free_embeddings_window();
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(embed_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<ggml_fp16_t> h(static_cast<size_t>(dim) * n);
    ggml_fp32_to_fp16_row(static_cast<const float *>(rows->data), h.data(),
                          static_cast<int64_t>(dim) * n);
    ggml_backend_tensor_set(embeddings_window_, h.data(), 0,
                            h.size() * sizeof(ggml_fp16_t));

    std::vector<int32_t> local(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) local[static_cast<size_t>(i)] = i;
    ggml_backend_tensor_set(embeddings_window_ids_, local.data(), 0,
                            local.size() * sizeof(int32_t));
    embeddings_window_n_ = n;
    ggml_free(ctx);
    return true;
}

void SlowARModel::free_fast_embeddings_gpu() {
    if (fast_emb_buf_) ggml_backend_buffer_free(fast_emb_buf_);
    if (ctx_fast_emb_) ggml_free(ctx_fast_emb_);
    fast_emb_buf_ = nullptr;
    ctx_fast_emb_ = nullptr;
    fast_embeddings_gpu_ = nullptr;
}

// The Fast-AR embedding table is read once per codebook step (nine times a
// frame). Keeping an F32 copy on the GPU removes those CPU gathers and their
// host/device round-trips; the CPU table stays for the fallback path.
bool SlowARModel::ensure_fast_embeddings_gpu(int32_t n_threads) {
    if (fast_embeddings_gpu_) return true;
    if (!backend_gpu_ || !weights_.fast_embeddings) return false;

    const int32_t dim = hparams_.embedding_length;
    const int64_t nelem = ggml_nelements(weights_.fast_embeddings);
    const int32_t n = static_cast<int32_t>(nelem / dim);
    if (n <= 0) return false;

    ggml_init_params p = { 16ull * 1024 * 1024 + static_cast<size_t>(nelem) * sizeof(float),
                           nullptr, false };
    ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;
    ggml_tensor * idt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    for (int32_t i = 0; i < n; ++i) static_cast<int32_t *>(idt->data)[i] = i;
    ggml_tensor * rows = ggml_get_rows(ctx, weights_.fast_embeddings, idt);
    if (!rows) { ggml_free(ctx); return false; }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);
    ggml_graph_compute_with_ctx(ctx, gf, resolve_n_threads(n_threads));

    ggml_init_params ep = { 4ull * ggml_tensor_overhead(), nullptr, true };
    ctx_fast_emb_ = ggml_init(ep);
    if (!ctx_fast_emb_) { ggml_free(ctx); return false; }
    fast_embeddings_gpu_ = ggml_new_tensor_2d(ctx_fast_emb_, GGML_TYPE_F32, dim, n);
    fast_emb_buf_ = ggml_backend_alloc_ctx_tensors(ctx_fast_emb_, backend_gpu_);
    if (!fast_emb_buf_) {
        free_fast_embeddings_gpu();
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(fast_emb_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(fast_embeddings_gpu_, rows->data, 0,
                            static_cast<size_t>(nelem) * sizeof(float));
    ggml_free(ctx);
    return true;
}

bool SlowARModel::eval_cached(const std::vector<int32_t> & flat_tokens,
                               int32_t n_tokens, int32_t n_threads,
                               StepResult & result) {
    if (n_tokens <= 0) return false;
    const int32_t codebook_dim = hparams_.num_codebooks + 1;
    if (static_cast<int32_t>(flat_tokens.size()) != n_tokens * codebook_dim) {
        std::fprintf(stderr, "[eval_cached] expected %d ints for %d tokens, got %zu\n",
            n_tokens * codebook_dim, n_tokens, flat_tokens.size());
        return false;
    }
    if (n_past_ + n_tokens > max_seq_len_) {
        std::fprintf(stderr, "[eval_cached] KV cache overflow (%d + %d > %d)\n",
            n_past_, n_tokens, max_seq_len_);
        return false;
    }

    // A prefill rebuilds (and resets) the scheduler below, which releases the
    // buffers of any cached decode graph. Drop that graph so the next decode
    // rebuilds it instead of writing into freed buffers.
    if (n_tokens > 1 && slow_gf_ != nullptr) {
        invalidate_cached_graphs();
    }

    const int32_t dim       = hparams_.embedding_length;
    const int32_t n_head    = hparams_.head_count;
    const int32_t n_head_kv = hparams_.head_count_kv;

    int32_t head_dim = 0;
    if (hparams_.attention_qk_norm && !weights_.layers.empty() && weights_.layers[0].q_norm) {
        head_dim = static_cast<int32_t>(weights_.layers[0].q_norm->ne[0]);
    } else {
        head_dim = static_cast<int32_t>(weights_.layers[0].wo->ne[0] / n_head);
    }

    const int32_t q_size   = n_head * head_dim;
    const int32_t kv_size  = n_head_kv * head_dim;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float sem_scale  = 1.0f / std::sqrt(static_cast<float>(codebook_dim));

    std::vector<int32_t> semantic_vals(n_tokens);
    std::vector<int32_t> pos_vals(n_tokens);
    std::vector<float>   semantic_mask_vals(n_tokens, 0.0f);
    std::vector<float>   token_scale_vals;
    std::vector<std::vector<int32_t>> cb_vals(hparams_.num_codebooks, std::vector<int32_t>(n_tokens, 0));

    if (hparams_.scale_codebook_embeddings) {
        token_scale_vals.resize(n_tokens);
    }

    for (int32_t t = 0; t < n_tokens; ++t) {
        const int32_t semantic = flat_tokens[t * codebook_dim];
        const bool is_semantic = (semantic >= hparams_.semantic_begin_id &&
                                  semantic <= hparams_.semantic_end_id);

        semantic_vals[t]      = semantic;
        pos_vals[t]           = n_past_ + t;
        semantic_mask_vals[t] = is_semantic ? 1.0f : 0.0f;

        if (!token_scale_vals.empty()) {
            token_scale_vals[t] = is_semantic ? sem_scale : 1.0f;
        }

        for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
            if (!is_semantic) continue;
            const int32_t v = flat_tokens[t * codebook_dim + cb + 1];
            cb_vals[cb][t] = v + cb * hparams_.codebook_size;
        }
    }

    // Sampling window is constant for the model's lifetime (semantic range + optional
    // im_end), so compute it once here for both the cached and the build path.
    const int32_t sem_begin = hparams_.semantic_begin_id;
    const int32_t sem_end   = hparams_.semantic_end_id;
    const int32_t sem_count = (sem_end >= sem_begin) ? (sem_end - sem_begin + 1) : 0;

    std::vector<int32_t> window_ids;
    window_ids.reserve(static_cast<size_t>(sem_count) + 1);
    const bool include_im_end = (im_end_id_ >= 0 && im_end_id_ < hparams_.vocab_size &&
                                 (im_end_id_ < sem_begin || im_end_id_ > sem_end));
    if (include_im_end) {
        window_ids.push_back(im_end_id_);
    }
    for (int32_t s = sem_begin; s <= sem_end; ++s) {
        window_ids.push_back(s);
    }
    const int32_t n_window = static_cast<int32_t>(window_ids.size());
    if (n_window <= 0) {
        std::fprintf(stderr, "[eval_cached] empty sampling window\n");
        return false;
    }

    std::vector<int32_t> window_local_ids(static_cast<size_t>(n_window));
    for (int32_t i = 0; i < n_window; ++i) window_local_ids[i] = i;
    const bool gpu_window = backend_gpu_ != nullptr &&
        ensure_embeddings_window(window_ids, n_threads);

    // Decode (n_tokens == 1) causal-mask values. Shape is fixed [max_seq_len_, 1]; the
    // values change each frame as n_past_ advances, so they are re-uploaded each frame.
    std::vector<ggml_fp16_t> attn_mask_vals;
    if (n_tokens == 1) {
        attn_mask_vals.assign(static_cast<size_t>(max_seq_len_), 0);
        for (int64_t i = n_past_ + 1; i < max_seq_len_; ++i) {
            attn_mask_vals[static_cast<size_t>(i)] = ggml_fp32_to_fp16(-INFINITY);
        }
    }

    // Cached decode fast path: reuse the shape-stable graph built on an earlier frame.
    // Only input values are re-uploaded; graph build + scheduler split/alloc are skipped.
    if (n_tokens == 1 && slow_gf_ != nullptr) {
        ggml_backend_cpu_set_n_threads(backend_cpu_, resolve_n_threads(n_threads));
        ggml_backend_tensor_set(slow_semantic_ids_,  semantic_vals.data(),      0, sizeof(int32_t));
        ggml_backend_tensor_set(slow_positions_,     pos_vals.data(),           0, sizeof(int32_t));
        ggml_backend_tensor_set(slow_semantic_mask_, semantic_mask_vals.data(), 0, sizeof(float));
        ggml_backend_tensor_set(slow_n_past_idx_,    &n_past_,                  0, sizeof(int32_t));
        if (slow_token_scale_) {
            ggml_backend_tensor_set(slow_token_scale_, token_scale_vals.data(), 0, sizeof(float));
        }
        ggml_backend_tensor_set(slow_attn_mask_, attn_mask_vals.data(), 0,
                                attn_mask_vals.size() * sizeof(ggml_fp16_t));
        if (slow_window_ids_t_) {
            ggml_backend_tensor_set(slow_window_ids_t_,
                                    gpu_window ? window_local_ids.data() : window_ids.data(), 0,
                                    n_window * sizeof(int32_t));
        }
        std::vector<int32_t> cb_all(static_cast<size_t>(hparams_.num_codebooks) * n_tokens);
        for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
            std::copy(cb_vals[cb].begin(), cb_vals[cb].end(),
                      cb_all.begin() + static_cast<size_t>(cb) * n_tokens);
        }
        ggml_backend_tensor_set(slow_cb_ids_, cb_all.data(), 0,
                                cb_all.size() * sizeof(int32_t));

        if (ggml_backend_sched_graph_compute(sched_, slow_gf_) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[eval_cached] cached sched compute failed\n");
            invalidate_cached_graphs();
            return false;
        }

        result.hidden.resize(dim);
        result.window_ids = window_ids;
        result.logits.resize(static_cast<size_t>(n_window));
        ggml_backend_tensor_get(slow_hidden_last_, result.hidden.data(), 0, dim * sizeof(float));
        ggml_backend_tensor_get(slow_logits_, result.logits.data(), 0, n_window * sizeof(float));
        n_past_ += n_tokens;
        return true;
    }

    if (ctx_size_ == 0) {
        ctx_size_ = 10u * 1024u * 1024u;
        ctx_buf_.resize(ctx_size_);
    }
    ggml_init_params p = { ctx_size_, ctx_buf_.data(), true };
    ggml_context * ctx0 = ggml_init(p);
    if (!ctx0) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor * semantic_ids   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_tensor * positions      = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_tensor * semantic_mask  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    ggml_tensor * n_past_idx     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_tensor * token_scale    = nullptr;
    if (hparams_.scale_codebook_embeddings) {
        token_scale = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    }

    // Causal mask for flash attention.
    // Prefill (n_tokens > 1): [n_past_ + n_tokens, n_tokens], built once per prompt.
    // Decode (n_tokens == 1): fixed [max_seq_len_, 1] so the graph is shape-stable for CUDA graphs.
    ggml_tensor * attn_mask = nullptr;
    if (n_tokens > 1) {
        attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past_ + n_tokens, n_tokens);
        attn_mask_vals.assign(static_cast<size_t>(n_past_ + n_tokens) * n_tokens, 0);
        for (int32_t j = 0; j < n_tokens; ++j) {
            for (int64_t i = static_cast<int64_t>(j) + n_past_ + 1; i < n_past_ + n_tokens; ++i) {
                attn_mask_vals[static_cast<size_t>(i + j * (n_past_ + n_tokens))] =
                    ggml_fp32_to_fp16(-INFINITY);
            }
        }
    } else {
        // Decode: attn_mask_vals was already filled above; shape is fixed [max_seq_len_, 1].
        attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, max_seq_len_, 1);
    }

    ggml_tensor * x = ggml_get_rows(ctx0, weights_.embeddings, semantic_ids);
    if (x->type != GGML_TYPE_F32) x = ggml_cast(ctx0, x, GGML_TYPE_F32);

    ggml_tensor * codebook_ids = ggml_new_tensor_1d(
        ctx0, GGML_TYPE_I32, static_cast<int64_t>(hparams_.num_codebooks) * n_tokens);
    ggml_tensor * emb_all = ggml_get_rows(ctx0, weights_.codebook_embeddings, codebook_ids);
    if (emb_all->type != GGML_TYPE_F32) emb_all = ggml_cast(ctx0, emb_all, GGML_TYPE_F32);

    ggml_tensor * codebook_sum = nullptr;
    for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
        ggml_tensor * emb = ggml_view_2d(
            ctx0, emb_all, dim, n_tokens, emb_all->nb[1],
            static_cast<size_t>(cb) * n_tokens * emb_all->nb[1]);
        codebook_sum = (codebook_sum == nullptr) ? emb : ggml_add(ctx0, codebook_sum, emb);
    }

    if (codebook_sum != nullptr) {

        codebook_sum = ggml_mul(ctx0, codebook_sum, semantic_mask);
        x = ggml_add(ctx0, x, codebook_sum);
    }
    if (token_scale != nullptr) {
        x = ggml_mul(ctx0, x, token_scale);
    }

    for (int32_t il = 0; il < hparams_.block_count; ++il) {
        const auto & layer = weights_.layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(ctx0, x, layer.attention_norm, hparams_.rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(ctx0, layer.wqkv, attn_in, "mul_mat:wqkv");
        const size_t elem_size = ggml_element_size(qkv);

        ggml_tensor * q2d = ggml_view_2d(ctx0, qkv, q_size, n_tokens, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], q_size * elem_size);
        ggml_tensor * v2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], (q_size + kv_size) * elem_size);

        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q2d), head_dim, n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k2d), head_dim, n_head_kv, n_tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_cont(ctx0, v2d), head_dim, n_head_kv, n_tokens);

        if (hparams_.attention_qk_norm) {
            q = rms_norm_weighted(ctx0, q, layer.q_norm, hparams_.rms_norm_eps);
            k = rms_norm_weighted(ctx0, k, layer.k_norm, hparams_.rms_norm_eps);
        }

        q = ggml_rope_ext(ctx0, q, positions, nullptr, head_dim, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(ctx0, k, positions, nullptr, head_dim, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        const size_t layer_off_k = static_cast<size_t>(il) * memory_k_->nb[3];
        const size_t layer_off_v = static_cast<size_t>(il) * memory_v_->nb[3];
        const size_t token_off_k = static_cast<size_t>(n_past_) * memory_k_->nb[1];
        const size_t token_off_v = static_cast<size_t>(n_past_) * memory_v_->nb[1];

        // Write RoPE'd k/v into the cache, transposed to the flash layout {head_dim, seq, n_head_kv}.
        // Decode (n_tokens == 1): ggml_set_rows with a runtime row index keeps the write location
        // out of the view offset, so the graph is shape-stable for CUDA graphs. K/V are the
        // set_rows results (views of the full-length cache), so flash attention depends on the
        // in-place write and the scheduler cannot reorder it after the read.
        ggml_tensor * K;
        ggml_tensor * V;
        if (n_tokens == 1) {
            ggml_tensor * mem_k_layer = ggml_view_3d(ctx0, memory_k_,
                head_dim, max_seq_len_, n_head_kv,
                memory_k_->nb[1], memory_k_->nb[2], layer_off_k);
            ggml_tensor * mem_v_layer = ggml_view_3d(ctx0, memory_v_,
                head_dim, max_seq_len_, n_head_kv,
                memory_v_->nb[1], memory_v_->nb[2], layer_off_v);
            K = ggml_set_rows(ctx0, mem_k_layer, ggml_permute(ctx0, k, 0, 2, 1, 3), n_past_idx);
            V = ggml_set_rows(ctx0, mem_v_layer, ggml_permute(ctx0, v, 0, 2, 1, 3), n_past_idx);
        } else {
            ggml_tensor * k_slot = ggml_view_3d(ctx0, memory_k_,
                head_dim, n_tokens, n_head_kv,
                memory_k_->nb[1], memory_k_->nb[2],
                layer_off_k + token_off_k);
            ggml_tensor * v_slot = ggml_view_3d(ctx0, memory_v_,
                head_dim, n_tokens, n_head_kv,
                memory_v_->nb[1], memory_v_->nb[2],
                layer_off_v + token_off_v);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3), k_slot));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3), v_slot));
            K = ggml_view_3d(ctx0, memory_k_,
                head_dim, n_past_ + n_tokens, n_head_kv,
                memory_k_->nb[1], memory_k_->nb[2], layer_off_k);
            V = ggml_view_3d(ctx0, memory_v_,
                head_dim, n_past_ + n_tokens, n_head_kv,
                memory_v_->nb[1], memory_v_->nb[2], layer_off_v);
        }

        // Flash attention: Q permuted, K/V as contiguous cache views (native GQA, un-expanded).
        ggml_tensor * Q = ggml_permute(ctx0, q, 0, 2, 1, 3);

        ggml_tensor * attn     = ggml_flash_attn_ext(ctx0, Q, K, V, attn_mask, attn_scale, 0.0f, 0.0f);
        ggml_tensor * attn_cur = ggml_cont(ctx0, ggml_reshape_2d(ctx0, attn, q_size, n_tokens));
        ggml_tensor * attn_out = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:wo");

        ggml_tensor * h     = ggml_add(ctx0, x, attn_out);
        ggml_tensor * ff_in = rms_norm_weighted(ctx0, h, layer.ffn_norm, hparams_.rms_norm_eps);
        ggml_tensor * gate  = mul_mat_checked(ctx0, layer.w1, ff_in, "mul_mat:w1");
        ggml_tensor * up    = mul_mat_checked(ctx0, layer.w3, ff_in, "mul_mat:w3");
        ggml_tensor * ff_h  = ggml_swiglu_split(ctx0, gate, up);
        ggml_tensor * ff_out = mul_mat_checked(ctx0, layer.w2, ff_h, "mul_mat:w2");

        x = ggml_add(ctx0, h, ff_out);
    }

    ggml_tensor * slow_out  = rms_norm_weighted(ctx0, x, weights_.norm, hparams_.rms_norm_eps);
    ggml_tensor * slow_cont = ggml_cont(ctx0, slow_out);
    ggml_tensor * hidden_last = ggml_cpy(ctx0,
        last_token_view(ctx0, slow_cont, n_tokens),
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, 1));

    // Sampling window rows: on the GPU they are a constant table plus constant
    // local indices, so the scheduler keeps get_rows on the GPU. Otherwise the
    // old CPU-table path is used.
    ggml_tensor * window_ids_t = nullptr;
    ggml_tensor * logits      = nullptr;
    if (gpu_window) {
        // The slice is constant, so no gather is needed: one matmul over the
        // GPU-resident rows keeps everything on the device.
        logits = mul_mat_checked(ctx0, embeddings_window_, hidden_last, "mul_mat:logits_window");
    } else {
        window_ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_window);
        ggml_tensor * logits_emb = ggml_get_rows(ctx0, weights_.embeddings, window_ids_t);
        if (logits_emb->type != GGML_TYPE_F32) {
            logits_emb = ggml_cast(ctx0, logits_emb, GGML_TYPE_F32);
        }
        logits = mul_mat_checked(ctx0, logits_emb, hidden_last, "mul_mat:logits_window");
    }
    ggml_build_forward_expand(gf, logits);
    if (gpu_window) {
        // Both inputs of get_rows live on the GPU; without this the scheduler can
        // still place the op on the CPU and copy the whole slice back and forth.
        ggml_backend_sched_set_tensor_backend(sched_, embeddings_window_, backend_gpu_);
        ggml_backend_sched_set_tensor_backend(sched_, embeddings_window_ids_, backend_gpu_);
    }

    ggml_backend_cpu_set_n_threads(backend_cpu_, resolve_n_threads(n_threads));
    ggml_backend_sched_reset(sched_);

    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        std::fprintf(stderr, "[eval_cached] sched alloc failed\n");
        ggml_backend_sched_reset(sched_);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(semantic_ids,  semantic_vals.data(), 0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(positions,     pos_vals.data(),       0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(semantic_mask, semantic_mask_vals.data(), 0, n_tokens * sizeof(float));
    if (n_tokens == 1) {
        ggml_backend_tensor_set(n_past_idx, &n_past_, 0, sizeof(int32_t));
    }
    if (window_ids_t) {
        ggml_backend_tensor_set(window_ids_t,
                                gpu_window ? window_local_ids.data() : window_ids.data(),
                                0, n_window * sizeof(int32_t));
    }
    if (token_scale) {
        ggml_backend_tensor_set(token_scale, token_scale_vals.data(), 0, n_tokens * sizeof(float));
    }
    if (attn_mask) {
        ggml_backend_tensor_set(attn_mask, attn_mask_vals.data(), 0,
                                attn_mask_vals.size() * sizeof(ggml_fp16_t));
    }
    {
        std::vector<int32_t> cb_all(static_cast<size_t>(hparams_.num_codebooks) * n_tokens);
        for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
            std::copy(cb_vals[cb].begin(), cb_vals[cb].end(),
                      cb_all.begin() + static_cast<size_t>(cb) * n_tokens);
        }
        ggml_backend_tensor_set(codebook_ids, cb_all.data(), 0,
                                cb_all.size() * sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[eval_cached] sched compute failed\n");
        ggml_backend_sched_reset(sched_);
        ggml_free(ctx0);
        return false;
    }

    result.hidden.resize(dim);
    result.window_ids = window_ids;
    result.logits.resize(static_cast<size_t>(n_window));
    ggml_backend_tensor_get(hidden_last, result.hidden.data(), 0, dim * sizeof(float));
    ggml_backend_tensor_get(logits, result.logits.data(), 0, n_window * sizeof(float));

    if (n_tokens == 1) {
        // Cache the shape-stable decode graph for reuse on subsequent frames. The
        // scheduler stays allocated (is_alloc) so tensor addresses remain stable for
        // CUDA-graph replay; the context is freed later by invalidate_cached_graphs().
        slow_ctx_           = ctx0;
        slow_gf_            = gf;
        slow_semantic_ids_  = semantic_ids;
        slow_positions_     = positions;
        slow_semantic_mask_ = semantic_mask;
        slow_n_past_idx_    = n_past_idx;
        slow_token_scale_   = token_scale;
        slow_attn_mask_     = attn_mask;
        slow_hidden_last_   = hidden_last;
        slow_logits_        = logits;
        slow_window_ids_t_  = window_ids_t;
        slow_cb_ids_ = codebook_ids;
    } else {
        ggml_backend_sched_reset(sched_);
        ggml_free(ctx0);
    }
    n_past_ += n_tokens;
    return true;
}

void SlowARModel::free_refill_graph() {
    if (refill_ctx_) ggml_free(refill_ctx_);
    refill_ctx_ = nullptr;
    refill_gf_ = nullptr;
    refill_hidden_ = nullptr;
    refill_prefix_ids_ = nullptr;
    refill_positions_ = nullptr;
    refill_out_idx_ = nullptr;
    refill_logits_ = nullptr;
}

bool SlowARModel::fast_decode_uncached(const std::vector<float> & hidden_in,
                               const std::vector<int32_t> & prefix_tokens,
                               int32_t n_threads,
                               std::vector<float> & logits_out) {
    if (!hparams_.has_fast_decoder) {
        std::fprintf(stderr, "[fast_decode] model has no fast decoder\n");
        return false;
    }
    if (static_cast<int32_t>(hidden_in.size()) != hparams_.embedding_length) {
        std::fprintf(stderr, "[fast_decode] expected hidden size %d, got %zu\n",
            hparams_.embedding_length, hidden_in.size());
        return false;
    }
    if (static_cast<int32_t>(prefix_tokens.size()) >= hparams_.num_codebooks) {
        std::fprintf(stderr, "[fast_decode] prefix too long (%zu >= %d)\n",
            prefix_tokens.size(), hparams_.num_codebooks);
        return false;
    }

    const int32_t fast_dim  = hparams_.fast_embedding_length;
    const int32_t n_head    = hparams_.fast_head_count;
    const int32_t n_head_kv = hparams_.fast_head_count_kv;
    const int32_t head_dim  = (hparams_.fast_head_dim > 0)
                                ? hparams_.fast_head_dim
                                : fast_dim / n_head;
    const int32_t q_size    = n_head * head_dim;
    const int32_t kv_size   = n_head_kv * head_dim;
    const float attn_scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    // Fixed sequence length (num_codebooks) so the graph is shape-stable for CUDA
    // graph capture. The prefix is right-padded with masked-out dummy tokens and the
    // output token is gathered by a runtime index (see out_idx below).
    const int32_t n_tokens  = hparams_.num_codebooks;

    // Per-call input values (positions are fixed; prefix is right-padded, out_idx is runtime).
    std::vector<int32_t> pos_vals(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) pos_vals[i] = i;
    std::vector<int32_t> prefix_id_vals(static_cast<size_t>(n_tokens - 1), 0);
    for (size_t i = 0; i < prefix_tokens.size(); ++i) {
        prefix_id_vals[i] = prefix_tokens[i];
    }
    const int32_t out_pos = static_cast<int32_t>(prefix_tokens.size());

    if (refill_gf_) {
        ggml_backend_cpu_set_n_threads(backend_cpu_, resolve_n_threads(n_threads));
        ggml_backend_tensor_set(refill_hidden_, hidden_in.data(), 0,
                                hidden_in.size() * sizeof(float));
        ggml_backend_tensor_set(refill_prefix_ids_, prefix_id_vals.data(), 0,
                                prefix_id_vals.size() * sizeof(int32_t));
        ggml_backend_tensor_set(refill_out_idx_, &out_pos, 0, sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(fast_sched_, refill_gf_) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[fast_decode] refill compute failed\n");
            free_refill_graph();
            return false;
        }
        logits_out.resize(hparams_.codebook_size);
        ggml_backend_tensor_get(refill_logits_, logits_out.data(), 0,
                                hparams_.codebook_size * sizeof(float));
        return true;
    }

    if (fast_ctx_size_ == 0) {
        fast_ctx_size_ = 8u * 1024u * 1024u;
        fast_ctx_buf_.resize(fast_ctx_size_);
    }
    ggml_init_params p = { fast_ctx_size_, fast_ctx_buf_.data(), true };
    ggml_context * ctx0 = ggml_init(p);
    if (!ctx0) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor * hidden0 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams_.embedding_length, 1);

    ggml_tensor * projected = (weights_.fast_project_in != nullptr)
        ? mul_mat_checked(ctx0, weights_.fast_project_in, hidden0, "mul_mat:fast_project_in")
        : hidden0;
    if (projected->type != GGML_TYPE_F32) {
        projected = ggml_cast(ctx0, projected, GGML_TYPE_F32);
    }

    ggml_tensor * x = projected;
    // Prefix is always length (n_tokens - 1): real prefix tokens first, zero-padded dummy
    // tokens after. Dummies sit beyond the causal diagonal / are discarded by the runtime
    // output index, so only the real prefix affects the result.
    ggml_tensor * prefix_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens - 1);
    ggml_tensor * prefix_emb = ggml_get_rows(ctx0,
        fast_embeddings_gpu_ ? fast_embeddings_gpu_ : weights_.fast_embeddings, prefix_ids);
    if (prefix_emb->type != GGML_TYPE_F32) {
        prefix_emb = ggml_cast(ctx0, prefix_emb, GGML_TYPE_F32);
    }
    x = ggml_concat(ctx0, x, prefix_emb, 1);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_tensor * out_idx   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);

    for (int32_t il = 0; il < hparams_.fast_block_count; ++il) {
        const auto & layer = weights_.fast_layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(ctx0, x, layer.attention_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(ctx0, layer.wqkv, attn_in, "mul_mat:fast_wqkv");
        const size_t elem_size = ggml_element_size(qkv);

        ggml_tensor * q2d = ggml_view_2d(ctx0, qkv, q_size, n_tokens, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], q_size * elem_size);
        ggml_tensor * v2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], (q_size + kv_size) * elem_size);

        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q2d), head_dim, n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k2d), head_dim, n_head_kv, n_tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_cont(ctx0, v2d), head_dim, n_head_kv, n_tokens);

        if (hparams_.fast_attention_qk_norm) {
            q = rms_norm_weighted(ctx0, q, layer.q_norm, hparams_.fast_rms_norm_eps);
            k = rms_norm_weighted(ctx0, k, layer.k_norm, hparams_.fast_rms_norm_eps);
        }

        q = ggml_rope_ext(ctx0, q, positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(ctx0, k, positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        ggml_tensor * k_rep = repeat_interleave_heads(ctx0, k, n_head / n_head_kv);
        ggml_tensor * v_rep = repeat_interleave_heads(ctx0, v, n_head / n_head_kv);

        ggml_tensor * Q   = ggml_permute(ctx0, q,     0, 2, 1, 3);
        ggml_tensor * K   = ggml_permute(ctx0, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ  = mul_mat_checked(ctx0, K, Q, "mul_mat:fast_kq");
        ggml_tensor * KQs = ggml_scale(ctx0, KQ, attn_scale);
        ggml_tensor * KQm = ggml_diag_mask_inf(ctx0, KQs, 0);
        ggml_tensor * KQf = ggml_soft_max(ctx0, KQm);

        ggml_tensor * V       = ggml_cont(ctx0, ggml_permute(ctx0, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(ctx0, V, KQf, "mul_mat:fast_kqv");
        ggml_tensor * KQVm    = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = ggml_cpy(ctx0, KQVm,
                                          ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tokens));
        ggml_tensor * attn_out = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:fast_wo");

        ggml_tensor * h     = ggml_add(ctx0, x, attn_out);
        ggml_tensor * ff_in = rms_norm_weighted(ctx0, h, layer.ffn_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * gate  = mul_mat_checked(ctx0, layer.w1, ff_in, "mul_mat:fast_w1");
        ggml_tensor * up    = mul_mat_checked(ctx0, layer.w3, ff_in, "mul_mat:fast_w3");
        ggml_tensor * ff_h  = ggml_swiglu_split(ctx0, gate, up);
        ggml_tensor * ff_out = mul_mat_checked(ctx0, layer.w2, ff_h, "mul_mat:fast_w2");

        x = ggml_add(ctx0, h, ff_out);
    }

    ggml_tensor * fast_out  = rms_norm_weighted(ctx0, x, weights_.fast_norm, hparams_.fast_rms_norm_eps);
    ggml_tensor * fast_cont = ggml_cont(ctx0, fast_out);
    // Gather the last *real* token's hidden (out_idx == prefix_tokens.size()) instead of the
    // fixed last column, so the output position is a graph input rather than a shape.
    ggml_tensor * fast_last = ggml_get_rows(ctx0, fast_cont, out_idx);
    ggml_tensor * logits = mul_mat_checked(ctx0, weights_.fast_output, fast_last, "mul_mat:fast_logits");
    ggml_build_forward_expand(gf, logits);

    ggml_backend_cpu_set_n_threads(backend_cpu_, resolve_n_threads(n_threads));
    ggml_backend_sched_reset(fast_sched_);

    if (!ggml_backend_sched_alloc_graph(fast_sched_, gf)) {
        std::fprintf(stderr, "[fast_decode] sched alloc failed\n");
        ggml_backend_sched_reset(fast_sched_);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(hidden0,    hidden_in.data(),      0, hidden_in.size() * sizeof(float));
    ggml_backend_tensor_set(positions,  pos_vals.data(),       0, pos_vals.size() * sizeof(int32_t));
    ggml_backend_tensor_set(prefix_ids, prefix_id_vals.data(), 0, prefix_id_vals.size() * sizeof(int32_t));
    ggml_backend_tensor_set(out_idx,    &out_pos,              0, sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(fast_sched_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[fast_decode] sched compute failed\n");
        ggml_backend_sched_reset(fast_sched_);
        ggml_free(ctx0);
        return false;
    }

    logits_out.resize(hparams_.codebook_size);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, hparams_.codebook_size * sizeof(float));

    refill_ctx_ = ctx0;
    refill_gf_ = gf;
    refill_hidden_ = hidden0;
    refill_prefix_ids_ = prefix_ids;
    refill_positions_ = positions;
    refill_out_idx_ = out_idx;
    refill_logits_ = logits;
    return true;
}

// --- Fast-AR with a KV cache -------------------------------------------------------------------
//
// One frame decodes num_codebooks-1 codes with the 4-layer Fast-AR transformer. Its input at step i is
// [hidden, code_0 .. code_(i-1)], and the old path re-ran the whole prefix through the transformer for
// every code: nine graph builds, allocations and launch sequences per frame, each larger than the last.
// Here every position is evaluated once: its K/V rows go to a tiny cache, and the graph for "one new token"
// (or "hidden + first code" at the start of a frame) has a fixed shape, is allocated once and just replayed
// with new inputs. A fixed graph also lets the CUDA backend capture and replay it as a CUDA graph.

bool SlowARModel::init_fast_cache() {
    if (fast_kv_buf_) return true;
    if (fast_cache_failed_ || !hparams_.has_fast_decoder) return false;

    const int32_t n_layer = hparams_.fast_block_count;
    const int32_t kv_size = hparams_.fast_head_count_kv *
        (hparams_.fast_head_dim > 0 ? hparams_.fast_head_dim : hparams_.fast_embedding_length / hparams_.fast_head_count);
    // positions 0 .. num_codebooks-1 (hidden + up to num_codebooks-1 codes), padded to a multiple of 8
    fast_cache_len_ = (hparams_.num_codebooks + 7) / 8 * 8;

    ggml_init_params p = { (size_t)(2 * n_layer + 2) * ggml_tensor_overhead(), nullptr, true };
    fast_kv_ctx_ = ggml_init(p);
    if (!fast_kv_ctx_) { fast_cache_failed_ = true; return false; }
    auto fail = [this](const char * msg) {
        std::cerr << msg << std::endl;
        free_fast_cache();
        fast_cache_failed_ = true;
        return false;
    };
    for (int32_t il = 0; il < n_layer; ++il) {
        fast_k_.push_back(ggml_new_tensor_2d(fast_kv_ctx_, GGML_TYPE_F32, kv_size, fast_cache_len_));
        fast_v_.push_back(ggml_new_tensor_2d(fast_kv_ctx_, GGML_TYPE_F32, kv_size, fast_cache_len_));
    }
    ggml_backend_t kv_backend = (n_gpu_layers_ > 0 && backend_gpu_) ? backend_gpu_ : backend_cpu_;
    fast_kv_buf_ = ggml_backend_alloc_ctx_tensors(fast_kv_ctx_, kv_backend);
    if (!fast_kv_buf_) {
        return fail("[Model] Fast-AR KV cache allocation failed; using the uncached path.");
    }
    ggml_backend_buffer_clear(fast_kv_buf_, 0);

    if (!build_fast_graph(fast_graph_[0], 1) || !build_fast_graph(fast_graph_[1], 2)) {
        return fail("[Model] Fast-AR graph setup failed; using the uncached path.");
    }
    return true;
}

void SlowARModel::free_fast_cache() {
    for (auto & g : fast_graph_) {
        if (g.sched) ggml_backend_sched_free(g.sched);
        if (g.ctx)   ggml_free(g.ctx);
        g = FastGraph();
    }
    if (fast_kv_buf_) ggml_backend_buffer_free(fast_kv_buf_);
    if (fast_kv_ctx_) ggml_free(fast_kv_ctx_);
    fast_kv_buf_ = nullptr;
    fast_kv_ctx_ = nullptr;
    fast_k_.clear();
    fast_v_.clear();
    fast_cached_hidden_.clear();
    fast_cached_prefix_.clear();
}

bool SlowARModel::build_fast_graph(FastGraph & g, int32_t n_tok) {
    const int32_t fast_dim  = hparams_.fast_embedding_length;
    const int32_t n_head    = hparams_.fast_head_count;
    const int32_t n_head_kv = hparams_.fast_head_count_kv;
    const int32_t head_dim  = (hparams_.fast_head_dim > 0) ? hparams_.fast_head_dim : fast_dim / n_head;
    const int32_t q_size    = n_head * head_dim;
    const int32_t kv_size   = n_head_kv * head_dim;
    const int32_t L         = fast_cache_len_;
    const float attn_scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    g.n_tok = n_tok;
    g.ctx_buf.resize(4u * 1024u * 1024u);
    ggml_init_params p = { g.ctx_buf.size(), g.ctx_buf.data(), true };
    g.ctx = ggml_init(p);
    if (!g.ctx) return false;
    ggml_context * ctx0 = g.ctx;
    g.gf = ggml_new_graph_custom(ctx0, 2048, false);

    auto input = [&](ggml_tensor * t, const char * name) { ggml_set_name(t, name); ggml_set_input(t); return t; };

    g.token_ids = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1), "fast_token_ids");
    g.positions = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok), "fast_positions");
    g.rows      = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tok), "fast_rows");
    g.mask      = input(ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, L, n_tok), "fast_mask");

    ggml_tensor * x = ggml_get_rows(ctx0,
        fast_embeddings_gpu_ ? fast_embeddings_gpu_ : weights_.fast_embeddings, g.token_ids);
    if (x->type != GGML_TYPE_F32) x = ggml_cast(ctx0, x, GGML_TYPE_F32);
    if (n_tok == 2) {
        g.hidden = input(ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams_.embedding_length, 1), "fast_hidden");
        ggml_tensor * projected = (weights_.fast_project_in != nullptr)
            ? mul_mat_checked(ctx0, weights_.fast_project_in, g.hidden, "mul_mat:fast_project_in")
            : g.hidden;
        if (projected->type != GGML_TYPE_F32) projected = ggml_cast(ctx0, projected, GGML_TYPE_F32);
        x = ggml_concat(ctx0, projected, x, 1);
    }

    for (int32_t il = 0; il < hparams_.fast_block_count; ++il) {
        const auto & layer = weights_.fast_layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(ctx0, x, layer.attention_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(ctx0, layer.wqkv, attn_in, "mul_mat:fast_wqkv");
        const size_t elem_size = ggml_element_size(qkv);

        ggml_tensor * q2d = ggml_view_2d(ctx0, qkv, q_size, n_tok, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(ctx0, qkv, kv_size, n_tok, qkv->nb[1], q_size * elem_size);
        ggml_tensor * v2d = ggml_view_2d(ctx0, qkv, kv_size, n_tok, qkv->nb[1], (q_size + kv_size) * elem_size);

        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q2d), head_dim, n_head, n_tok);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k2d), head_dim, n_head_kv, n_tok);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_cont(ctx0, v2d), head_dim, n_head_kv, n_tok);

        if (hparams_.fast_attention_qk_norm) {
            q = rms_norm_weighted(ctx0, q, layer.q_norm, hparams_.fast_rms_norm_eps);
            k = rms_norm_weighted(ctx0, k, layer.k_norm, hparams_.fast_rms_norm_eps);
        }

        q = ggml_rope_ext(ctx0, q, g.positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(ctx0, k, g.positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        // write this layer's new K/V rows into the cache, then attend over the whole (masked) cache
        ggml_tensor * k_cache = ggml_set_rows(ctx0, fast_k_[il], ggml_reshape_2d(ctx0, k, kv_size, n_tok), g.rows);
        ggml_tensor * v_cache = ggml_set_rows(ctx0, fast_v_[il], ggml_reshape_2d(ctx0, v, kv_size, n_tok), g.rows);
        const size_t f32 = sizeof(float);
        ggml_tensor * K = ggml_view_3d(ctx0, k_cache, head_dim, L, n_head_kv, kv_size * f32, head_dim * f32, 0);
        ggml_tensor * V = ggml_view_3d(ctx0, v_cache, head_dim, L, n_head_kv, kv_size * f32, head_dim * f32, 0);

        ggml_tensor * Q   = ggml_permute(ctx0, q, 0, 2, 1, 3);
        ggml_tensor * KQ  = mul_mat_checked(ctx0, K, Q, "mul_mat:fast_kq");           // [L, n_tok, n_head]
        ggml_tensor * KQf = ggml_soft_max_ext(ctx0, KQ, g.mask, attn_scale, 0.0f);

        ggml_tensor * Vt      = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));   // [L, head_dim, n_head_kv]
        ggml_tensor * KQV     = mul_mat_checked(ctx0, Vt, KQf, "mul_mat:fast_kqv");   // [head_dim, n_tok, n_head]
        ggml_tensor * KQVm    = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = ggml_cpy(ctx0, KQVm, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tok));
        ggml_tensor * attn_out = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:fast_wo");

        ggml_tensor * h     = ggml_add(ctx0, x, attn_out);
        ggml_tensor * ff_in = rms_norm_weighted(ctx0, h, layer.ffn_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * gate  = mul_mat_checked(ctx0, layer.w1, ff_in, "mul_mat:fast_w1");
        ggml_tensor * up    = mul_mat_checked(ctx0, layer.w3, ff_in, "mul_mat:fast_w3");
        ggml_tensor * ff_h  = ggml_swiglu_split(ctx0, gate, up);
        ggml_tensor * ff_out = mul_mat_checked(ctx0, layer.w2, ff_h, "mul_mat:fast_w2");

        x = ggml_add(ctx0, h, ff_out);
    }

    ggml_tensor * fast_out  = rms_norm_weighted(ctx0, x, weights_.fast_norm, hparams_.fast_rms_norm_eps);
    ggml_tensor * fast_cont = ggml_cont(ctx0, fast_out);
    ggml_tensor * fast_last = ggml_cpy(ctx0,
        last_token_view(ctx0, fast_cont, n_tok),
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, fast_dim, 1));
    g.logits = mul_mat_checked(ctx0, weights_.fast_output, fast_last, "mul_mat:fast_logits");
    ggml_set_name(g.logits, "fast_logits");
    ggml_set_output(g.logits);
    ggml_build_forward_expand(g.gf, g.logits);

    ggml_backend_t backends[2];
    int n_backends;
    if (backend_gpu_) {
        backends[0] = backend_gpu_;
        backends[1] = backend_cpu_;
        n_backends = 2;
    } else {
        backends[0] = backend_cpu_;
        n_backends = 1;
    }
    g.sched = ggml_backend_sched_new(backends, NULL, n_backends, 2048, false, true);
    if (!g.sched) return false;
    return ggml_backend_sched_alloc_graph(g.sched, g.gf);
}

// Evaluate g.n_tok new positions starting at first_pos. The 2-token graph takes the frame's hidden state
// and then `token`; the 1-token graph takes just `token`.
bool SlowARModel::run_fast_graph(FastGraph & g, const float * hidden, int32_t token,
                                 int32_t first_pos, int32_t n_threads, std::vector<float> & logits_out) {
    const int32_t L = fast_cache_len_;
    std::vector<int32_t> pos(g.n_tok);
    std::vector<int64_t> rows(g.n_tok);
    std::vector<float>   mask(static_cast<size_t>(L) * g.n_tok, 0.0f);
    for (int32_t t = 0; t < g.n_tok; ++t) {
        pos[t]  = first_pos + t;
        rows[t] = first_pos + t;
        for (int32_t j = first_pos + t + 1; j < L; ++j) {
            mask[static_cast<size_t>(t) * L + j] = -INFINITY;
        }
    }
    if (g.hidden) {
        ggml_backend_tensor_set(g.hidden, hidden, 0, ggml_nbytes(g.hidden));
    }
    ggml_backend_tensor_set(g.token_ids, &token, 0, sizeof(int32_t));
    ggml_backend_tensor_set(g.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.rows, rows.data(), 0, rows.size() * sizeof(int64_t));
    ggml_backend_tensor_set(g.mask, mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_cpu_set_n_threads(backend_cpu_, resolve_n_threads(n_threads));
    if (ggml_backend_sched_graph_compute(g.sched, g.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[fast_decode] sched compute failed\n");
        return false;
    }
    logits_out.resize(hparams_.codebook_size);
    ggml_backend_tensor_get(g.logits, logits_out.data(), 0, hparams_.codebook_size * sizeof(float));
    return true;
}

bool SlowARModel::fast_decode(const std::vector<float> & hidden_in,
                              const std::vector<int32_t> & prefix_tokens,
                              int32_t n_threads,
                              std::vector<float> & logits_out) {
    if (prefix_tokens.empty() || !hparams_.has_fast_decoder ||
        static_cast<int32_t>(hidden_in.size()) != hparams_.embedding_length ||
        static_cast<int32_t>(prefix_tokens.size()) >= hparams_.num_codebooks) {
        return fast_decode_uncached(hidden_in, prefix_tokens, n_threads, logits_out);
    }
    if (const char * mode = std::getenv("S2_FASTAR")) {
        if (std::strcmp(mode, "refill") == 0) {
            ensure_fast_embeddings_gpu(n_threads);
            return fast_decode_uncached(hidden_in, prefix_tokens, n_threads, logits_out);
        }
    }
    ensure_fast_embeddings_gpu(n_threads);
    if (!init_fast_cache()) {
        return fast_decode_uncached(hidden_in, prefix_tokens, n_threads, logits_out);
    }

    // The caller asks for the logits after [hidden, code_0 .. code_(n-1)]. Within one frame each call
    // extends the previous one by a single code; anything else restarts the cache from the hidden state.
    const size_t n = prefix_tokens.size();
    const bool extends =
        n >= 2 && fast_cached_prefix_.size() == n - 1 &&
        fast_cached_hidden_ == hidden_in &&
        std::equal(fast_cached_prefix_.begin(), fast_cached_prefix_.end(), prefix_tokens.begin());

    bool ok;
    if (extends) {
        ok = run_fast_graph(fast_graph_[0], nullptr, prefix_tokens[n - 1], static_cast<int32_t>(n), n_threads, logits_out);
    } else {
        // A new frame (or a prefix that does not continue the last one). The graphs mask out the rows that
        // lie ahead, but a masked row is still read: if an earlier decode left non-finite K/V values there,
        // 0 * NaN poisons the attention output for every later frame. Start from a zeroed cache.
        ggml_backend_buffer_clear(fast_kv_buf_, 0);
        ok = run_fast_graph(fast_graph_[1], hidden_in.data(), prefix_tokens[0], 0, n_threads, logits_out);
        for (size_t i = 1; ok && i < n; ++i) {
            ok = run_fast_graph(fast_graph_[0], nullptr, prefix_tokens[i], static_cast<int32_t>(i) + 1, n_threads, logits_out);
        }
    }
    if (!ok) {
        fast_cached_prefix_.clear();
        return false;
    }
    if (!extends) fast_cached_hidden_ = hidden_in;
    fast_cached_prefix_ = prefix_tokens;
    return true;
}


}
