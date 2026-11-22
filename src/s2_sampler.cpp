// s2.cpp - Fish Audio S2 Pro local engine.
// Derived from https://github.com/rodrigomatt/s2.cpp and distributed under the
// Fish Audio Research License (see LICENSE.md). Commercial use requires a
// separate license from Fish Audio.

#include "../include/s2_sampler.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace s2 {

static thread_local std::mt19937 sampler_gen(std::random_device{}());

void set_sampler_seed(int64_t seed) {
    if (seed < 0) {
        sampler_gen.seed(std::random_device{}());
        return;
    }
    const uint32_t lo = static_cast<uint32_t>(static_cast<uint64_t>(seed));
    const uint32_t hi = static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32);
    std::seed_seq seq{lo, hi};
    sampler_gen.seed(seq);
}

static void apply_softmax(std::vector<float> & probs, float temp = 1.0f) {
    if (probs.empty()) return;
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (float & p : probs) {
        if (temp > 0.0f) {
            p = std::exp((p - max_val) / temp);
        } else {
            p = (p == max_val) ? 1.0f : 0.0f;
        }
        sum += p;
    }
    if (sum > 0) {
        for (float & p : probs) p /= sum;
    }
}

static std::vector<float> softmax_from_sorted_logits(const std::vector<std::pair<float, int32_t>> & items) {
    std::vector<float> probs(items.size(), 0.0f);
    if (items.empty()) return probs;

    const float max_val = items.front().first;
    float sum = 0.0f;
    for (size_t i = 0; i < items.size(); ++i) {
        probs[i] = std::exp(items[i].first - max_val);
        sum += probs[i];
    }

    if (sum > 0.0f) {
        for (float & p : probs) p /= sum;
    }
    return probs;
}

int32_t sample_token(const float * logits, int32_t vocab_size, const SamplerParams & params) {
    if (vocab_size <= 0) return 0;

    std::vector<std::pair<float, int32_t>> items;
    items.reserve(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) {
        items.push_back({logits[i], i});
    }

    const int32_t k = params.top_k > 0 ? std::min(params.top_k, vocab_size) : vocab_size;
    if (k < static_cast<int32_t>(items.size())) {
        std::partial_sort(items.begin(), items.begin() + k, items.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
    } else {
        std::sort(items.begin(), items.end(), [](const auto & a, const auto & b) {
            return a.first > b.first;
        });
    }
    const float top_p = std::clamp(params.top_p, 0.0f, 1.0f);
    const std::vector<float> sorted_probs = softmax_from_sorted_logits(items);

    std::vector<std::pair<float, int32_t>> filtered;
    filtered.reserve(k);

    float cumsum = 0.0f;
    for (int32_t i = 0; i < (int32_t)items.size(); ++i) {
        cumsum += sorted_probs[i];
        const bool remove_for_top_k = (i >= k);
        const bool remove_for_top_p = (i > 0 && cumsum > top_p);
        if (remove_for_top_k || remove_for_top_p) {
            continue;
        }
        filtered.push_back(items[i]);
    }

    if (filtered.empty()) {
        filtered.push_back(items.front());
    }

    if (params.temperature <= 0.0f) {
        return filtered[0].second;
    }

    std::vector<float> probs(filtered.size());
    for (size_t i = 0; i < filtered.size(); ++i) {
        probs[i] = filtered[i].first;
    }
    apply_softmax(probs, params.temperature);

    float sum_p = 0.0f;
    for (float p : probs) sum_p += p;
    if (sum_p <= 0.0f) {
        return filtered[0].second;
    }
    for (float & p : probs) p /= sum_p;

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());

    const int32_t sampled_idx = dist(sampler_gen);
    return filtered[sampled_idx].second;
}

}
