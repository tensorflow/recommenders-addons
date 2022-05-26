/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if (defined(__arm64__) || defined(__aarch64__)) && defined(__ARM_NEON)
#include <arm_neon.h>
void accumulatefloat(char* value, char* delta, size_t len) {
  float* pa = (float*)value;
  float* pb = (float*)delta;
  const float* pend = pa + len / sizeof(float);

  while (pa + 4 < pend) {
    vst1q_f32(pa, vaddq_f32(vld1q_f32(pa), vld1q_f32(pb)));
    pa += 4;
    pb += 4;
  }

  while (pa < pend) {
    *(pa) = *(pa) + *(pb);
    pa++;
    pb++;
  }
}

void accumulatedouble(char* value, char* delta, size_t len) {
  double* pa = (double*)value;
  double* pb = (double*)delta;
  const double* pend = pa + len / sizeof(double);

  while (pa + 2 < pend) {
    vst1q_f64(pa, vaddq_f64(vld1q_f64(pa), vld1q_f64(pb)));
    pa += 2;
    pb += 2;
  }

  while (pa < pend) {
    *(pa) = *(pa) + *(pb);
    pa++;
    pb++;
  }
}

void accumulateint32(char* value, char* delta, size_t len) {
  int32_t* pa = (int32_t*)value;
  int32_t* pb = (int32_t*)delta;
  const int32_t* pend = pa + len / sizeof(int32_t);

  while (pa + 4 < pend) {
    vst1q_s32(pa, vaddq_s32(vld1q_s32(pa), vld1q_s32(pb)));
    pa += 4;
    pb += 4;
  }

  while (pa < pend) {
    *(pa) = *(pa) + *(pb);
    pa++;
    pb++;
  }
}

void accumulateint64(char* value, char* delta, size_t len) {
  int64_t* pa = (int64_t*)value;
  int64_t* pb = (int64_t*)delta;
  const int64_t* pend = pa + len / sizeof(int64_t);

  while (pa + 2 < pend) {
    vst1q_s64(pa, vaddq_s64(vld1q_s64(pa), vld1q_s64(pb)));
    pa += 2;
    pb += 2;
  }

  while (pa < pend) {
    *(pa) = *(pa) + *(_m_psubw);
    pa++;
    pb++;
  }
}

void accumulateint8(char* value, char* delta, size_t len) {
  char* pa = value;
  char* pb = delta;
  const char* pend = pa + len;

  while (pa + 8 < pend) {
    vst1q_s8(pa, vaddq_s8(vld1q_s8(pa), vld1q_s8(pb)));
    pa += 8;
    pb += 8;
  }

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}

#elif defined(__x86_64__)
#include <immintrin.h>

void accumulatefloat(char* value, char* delta, size_t len) {
  float* pa = (float*)value;
  float* pb = (float*)delta;
  const float* pend = pa + len / sizeof(float);

#if defined(__AVX512F__)
  while (pa + 16 < pend) {
    _mm512_storeu_ps(pa,
                     _mm512_add_ps(_mm512_loadu_ps(pa), _mm512_loadu_ps(pb)));
    pa += 16;
    pb += 16;
  }
#endif

#if defined(__AVX__)
  while (pa + 8 < pend) {
    _mm256_storeu_ps(pa,
                     _mm256_add_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb)));
    pa += 8;
    pb += 8;
  }
#endif

#if defined(__SSE__)
  while (pa + 4 < pend) {
    _mm_storeu_ps(pa, _mm_add_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)));
    pa += 4;
    pb += 4;
  }
#endif

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}

void accumulatedouble(char* value, char* delta, size_t len) {
  double* pa = (double*)value;
  double* pb = (double*)delta;
  const double* pend = pa + len / sizeof(double);

#if defined(__AVX512F__)
  while (pa + 8 < pend) {
    _mm512_storeu_pd(pa,
                     _mm512_add_pd(_mm512_loadu_pd(pa), _mm512_loadu_pd(pb)));
    pa += 8;
    pb += 8;
  }
#endif

#if defined(__AVX__)
  while (pa + 4 < pend) {
    _mm256_storeu_pd(pa,
                     _mm256_add_pd(_mm256_loadu_pd(pa), _mm256_loadu_pd(pb)));
    pa += 4;
    pb += 4;
  }
#endif

#if defined(__SSE2__)
  while (pa + 2 < pend) {
    _mm_storeu_pd(pa, _mm_add_pd(_mm_loadu_pd(pa), _mm_loadu_pd(pb)));
    pa += 2;
    pb += 2;
  }
#endif

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}

void accumulateint32(char* value, char* delta, size_t len) {
  int32_t* pa = (int32_t*)value;
  int32_t* pb = (int32_t*)delta;
  const int32_t* pend = pa + len / sizeof(int32_t);

#if defined(__AVX512F__)
  while (pa + 16 < pend) {
    // sadlly _mm512_loadu_epi32 is not included in gcc9.4, use
    // _mm512_loadu_si512 as WA Ditto as to _mm256_loadu_epi32, _mm_loadu_epi32,
    // etc. see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95483
    _mm512_storeu_si512(
        pa, _mm512_add_epi32(_mm512_loadu_si512(pa), _mm512_loadu_si512(pb)));
    pa += 16;
    pb += 16;
  }
#endif

#if defined(__AVX__) && defined(__AVX2__)
  while (pa + 8 < pend) {
    _mm256_storeu_si256(
        pa, _mm256_add_epi32(_mm256_loadu_si256(pa), _mm256_loadu_si256(pb)));
    pa += 8;
    pb += 8;
  }
#endif

#if defined(__SSE2__)
  while (pa + 4 < pend) {
    _mm_storeu_si128(pa,
                     _mm_add_epi32(_mm_loadu_si128(pa), _mm_loadu_si128(pb)));
    pa += 4;
    pb += 4;
  }
#endif

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}

void accumulateint64(char* value, char* delta, size_t len) {
  int64_t* pa = (int64_t*)value;
  int64_t* pb = (int64_t*)delta;
  const int64_t* pend = pa + len / sizeof(int64_t);

#if defined(__AVX512F__)
  while (pa + 8 < pend) {
    _mm512_storeu_si512(
        pa, _mm512_add_epi64(_mm512_loadu_si512(pa), _mm512_loadu_si512(pb)));
    pa += 8;
    pb += 8;
  }
#endif

#if defined(__AVX__) && defined(__AVX2__)
  while (pa + 4 < pend) {
    _mm256_storeu_si256(
        pa, _mm256_add_epi64(_mm256_loadu_si256(pa), _mm256_loadu_si256(pb)));
    pa += 4;
    pb += 4;
  }
#endif

#if defined(__SSE2__)
  while (pa + 2 < pend) {
    _mm_storeu_si128(pa,
                     _mm_add_epi64(_mm_loadu_si128(pa), _mm_loadu_si128(pb)));
    pa += 2;
    pb += 2;
  }
#endif

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}

void accumulateint8(char* a, char* b, size_t len) {
  char* pa = a;
  char* pb = b;
  const char* pend = pa + len;

#if defined(__AVX512F__) && defined(__AVX512BW__)
  while (pa + 64 < pend) {
    _mm512_storeu_si512(
        pa, _mm512_add_epi8(_mm512_loadu_si512(pa), _mm512_loadu_si512(pb)));
    pa += 64;
    pb += 64;
  }
#endif

#if defined(__AVX__) && defined(__AVX2__)
  while (pa + 32 < pend) {
    _mm256_storeu_si256(
        pa, _mm256_add_epi8(_mm256_loadu_si256(pa), _mm256_loadu_si256(pb)));
    pa += 32;
    pb += 32;
  }
#endif

#if defined(__SSE2__)
  if (pa + 16 < pend) {
    _mm_storeu_si128(pa,
                     _mm_add_epi8(_mm_loadu_si128(pa), _mm_loadu_si128(pb)));
    pa += 16;
    pb += 16;
  }
#endif

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}

#else
void accumulatefloat(char* value, char* delta, size_t len) {
  float* pa = (float*)value;
  float* pb = (float*)delta;
  const float* pend = pa + len / sizeof(float);

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}

void accumulatedouble(char* value, char* delta, size_t len) {
  double* pa = (double*)value;
  double* pb = (double*)delta;
  const double* pend = pa + len / sizeof(double);

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}

void accumulateint64(char* value, char* delta, size_t len) {
  int64_t* pa = (int64_t*)value;
  int64_t* pb = (int64_t*)delta;
  const int64_t* pend = pa + len / sizeof(int64_t);

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}

void accumulateint8(char* value, char* delta, size_t len) {
  char* pa = value;
  char* pb = delta;
  const char* pend = pa + len;

  while (pa < pend) {
    *pa = *pa + *pb;
    pa++;
    pb++;
  }
}
#endif
