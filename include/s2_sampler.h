// s2.cpp - Fish Audio S2 Pro local engine.
// Derived from https://github.com/rodrigomatt/s2.cpp and distributed under the
// Fish Audio Research License (see LICENSE.md). Commercial use requires a
// separate license from Fish Audio.

#pragma once

#include <cstdint>
#include <vector>

namespace s2 {

struct SamplerParams {
    float   temperature     = 0.8f;
    float   top_p           = 0.8f;
    int32_t top_k           = 30;
};

int32_t sample_token(const float * logits, int32_t vocab_size, const SamplerParams & params);

// Seed the sampler for reproducible generation. Pass a negative value to go
// back to a random seed. Applies to the calling thread.
void set_sampler_seed(int64_t seed);


}
