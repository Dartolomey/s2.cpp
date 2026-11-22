// s2.cpp - Fish Audio S2 Pro local engine.
// Derived from https://github.com/rodrigomatt/s2.cpp and distributed under the
// Fish Audio Research License (see LICENSE.md). Commercial use requires a
// separate license from Fish Audio.

#pragma once

#include <cstdint>

namespace s2 {

enum class BackendType : int32_t {
    CPU = -1,
    Vulkan = 0,
    CUDA = 1,
    Metal = 2,
};

}
