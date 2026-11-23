// s2.cpp - Fish Audio S2 Pro local engine.
// Derived from https://github.com/rodrigomatt/s2.cpp and distributed under the
// Fish Audio Research License (see LICENSE.md). Commercial use requires a
// separate license from Fish Audio.

#pragma once

#include "httplib.h"
#include "s2_audio.h"
#include "s2_pipeline.h"

#include <cstdint>
#include <string>

namespace s2
{

  struct ServerParams
  {
    std::string host = "127.0.0.1";
    int32_t port = 3030;
    // Concurrent /generate requests: one runs, the rest wait for the slot instead
    // of getting a 503. Bounded so a burst cannot park unbounded handler threads.
    int32_t max_queued_requests = 4;
    int32_t queue_timeout_ms = 60000;

    PipelineParams pipeline;
  };

  class Server
  {
  public:
    Server();
    ~Server();

    bool serve(const ServerParams& params);
  };

}
