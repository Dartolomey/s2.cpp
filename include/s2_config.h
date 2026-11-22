// s2.cpp - Fish Audio S2 Pro local engine.
// Derived from https://github.com/rodrigomatt/s2.cpp and distributed under the
// Fish Audio Research License (see LICENSE.md). Commercial use requires a
// separate license from Fish Audio.

#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(S2_STATIC)
#    define S2_Export
#  elif defined(S2_LIBRARY)
#    define S2_Export __declspec(dllexport)
#  else
#    define S2_Export __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define S2_Export __attribute__((visibility("default")))
#else
#  define S2_Export
#endif
