#ifndef EDGEVECTOR_QUANTIZE_MATH_HPP
#define EDGEVECTOR_QUANTIZE_MATH_HPP

// ============================================================================
// EdgeVector :: quantize_math.hpp
//
// Binary quantization and the SIMD-friendly Hamming distance kernel.
//
// A float32 vector of dimension `dim` is compressed to one bit per component
// (sign bit: src[i] > 0), giving a 32x memory reduction over float32 storage.
// Similarity is then recovered from the Hamming distance via the SimHash
// angle estimator, theta ~= pi * hamming / dim.
//
// Bit layout
// ----------
// The quantized buffer is a whole number of 64-bit words. Component `i` lives
// in bit (i % 64) of word (i / 64), LSB-first within the word. Words are
// stored to the byte buffer with std::memcpy, so on a little-endian target
// (x86/ARM) component 0 is the least significant bit of dst[0]. Any trailing
// padding bits beyond `dim` are zeroed by quantize(), which keeps the Hamming
// distance exact for dimensions that are not a multiple of 64.
//
// Critical-path contract (see CLAUDE.md section 3)
// ------------------------------------------------
// quantize(), hamming_distance() and approx_cosine_from_hamming() perform zero
// heap allocation, throw nothing, and are marked noexcept. All bitwise work is
// done on std::uint64_t; bytes are moved into words with std::memcpy, which is
// strict-aliasing safe and lowers to a single load at -O3.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>

// The Hamming kernel has a hand-written AVX2 path (nibble-LUT popcount via
// vpshufb + vpsadbw) that is EXACTLY equivalent to the portable path - the
// test suite gates bit-identical results against a naive per-bit reference.
// Define EDGEVECTOR_NO_AVX2 to force the portable kernel on AVX2 hardware.
#if defined(__AVX2__) && !defined(EDGEVECTOR_NO_AVX2)
#define EDGEVECTOR_USE_AVX2 1
#include <immintrin.h>
#endif

namespace edgevector {

namespace detail {

// pi as a float32 literal; std::acos(-1) is not usable in a constexpr context.
constexpr float kPiF = 3.14159265358979323846f;

// Number of 64-bit words needed to hold `dim` bits.
constexpr std::size_t word_count(std::size_t dim) noexcept {
    return (dim + 63u) / 64u;
}

// Number of meaningful bits in word `w` (64 for every full word, the remainder
// for the final partial word). Bits beyond this count are padding.
constexpr std::size_t bits_in_word(std::size_t dim, std::size_t w) noexcept {
    return (dim - w * 64u) < 64u ? (dim - w * 64u) : 64u;
}

} // namespace detail

// Bytes required to store one quantized vector of dimension `dim`, padded so
// the buffer is always a whole number of 64-bit words.
constexpr std::size_t padded_bytes(std::size_t dim) noexcept {
    return detail::word_count(dim) * 8u;
}

// Quantize `dim` float32 values into a binary vector.
// Rule: bit i = 1 if src[i] > 0.0f, else 0. Padding bits are zeroed.
// `dst` must have capacity padded_bytes(dim) and be 8-byte aligned.
inline void quantize(const float* src, std::size_t dim, std::uint8_t* dst) noexcept {
    const std::size_t words = detail::word_count(dim);

    for (std::size_t w = 0; w < words; ++w) {
        const std::size_t base = w * 64u;
        const std::size_t bits = detail::bits_in_word(dim, w);

        // Accumulator starts at zero, so every bit we do not set - including
        // the padding tail of the final word - is left at zero.
        std::uint64_t acc = 0u;
        for (std::size_t b = 0; b < bits; ++b) {
            const std::uint64_t bit = (src[base + b] > 0.0f) ? 1u : 0u;
            acc |= (bit << b);
        }

        std::memcpy(dst + w * 8u, &acc, sizeof(std::uint64_t));
    }
}

// Hamming distance between two quantized vectors of dimension `dim`.
// Both buffers are 8-byte aligned and padded_bytes(dim) long.
//
// Two implementations with bit-identical results (test-gated):
//   - AVX2: 32 bytes per step via the classic nibble-LUT popcount
//     (vpshufb twice + vpsadbw), horizontally summed at the end.
//   - portable: four independent scalar POPCNT accumulators so the adds do
//     not serialize on one dependency chain.
// Padding bits are zero in both operands, so they never contribute either
// way.
inline std::uint32_t hamming_distance(const std::uint8_t* a,
                                      const std::uint8_t* b,
                                      std::size_t dim) noexcept {
    const std::size_t words = detail::word_count(dim);
    std::size_t w = 0;
    std::uint64_t distance = 0u;

#if defined(EDGEVECTOR_USE_AVX2)
    if (words >= 4u) {
        const __m256i low_mask = _mm256_set1_epi8(0x0f);
        const __m256i lut = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        __m256i acc = _mm256_setzero_si256();
        for (; w + 4u <= words; w += 4u) {
            const __m256i va = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(a + w * 8u));
            const __m256i vb = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(b + w * 8u));
            const __m256i x = _mm256_xor_si256(va, vb);
            const __m256i lo = _mm256_and_si256(x, low_mask);
            const __m256i hi =
                _mm256_and_si256(_mm256_srli_epi16(x, 4), low_mask);
            const __m256i cnt = _mm256_add_epi8(
                _mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
            acc = _mm256_add_epi64(acc,
                                   _mm256_sad_epu8(cnt, _mm256_setzero_si256()));
        }
        std::uint64_t lanes[4]; // storeu: old MinGW cannot 32-align the stack
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), acc);
        distance = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    }
    for (; w < words; ++w) {
        std::uint64_t wa;
        std::uint64_t wb;
        std::memcpy(&wa, a + w * 8u, sizeof(std::uint64_t));
        std::memcpy(&wb, b + w * 8u, sizeof(std::uint64_t));
        distance += static_cast<std::uint64_t>(__builtin_popcountll(wa ^ wb));
    }
#else
    std::uint64_t d0 = 0u;
    std::uint64_t d1 = 0u;
    std::uint64_t d2 = 0u;
    std::uint64_t d3 = 0u;
    for (; w + 4u <= words; w += 4u) {
        std::uint64_t wa[4];
        std::uint64_t wb[4];
        std::memcpy(wa, a + w * 8u, 4u * sizeof(std::uint64_t));
        std::memcpy(wb, b + w * 8u, 4u * sizeof(std::uint64_t));
        d0 += static_cast<std::uint64_t>(__builtin_popcountll(wa[0] ^ wb[0]));
        d1 += static_cast<std::uint64_t>(__builtin_popcountll(wa[1] ^ wb[1]));
        d2 += static_cast<std::uint64_t>(__builtin_popcountll(wa[2] ^ wb[2]));
        d3 += static_cast<std::uint64_t>(__builtin_popcountll(wa[3] ^ wb[3]));
    }
    distance = (d0 + d1) + (d2 + d3);
    for (; w < words; ++w) {
        std::uint64_t wa;
        std::uint64_t wb;
        std::memcpy(&wa, a + w * 8u, sizeof(std::uint64_t));
        std::memcpy(&wb, b + w * 8u, sizeof(std::uint64_t));
        distance += static_cast<std::uint64_t>(__builtin_popcountll(wa ^ wb));
    }
#endif

    return static_cast<std::uint32_t>(distance);
}

// Estimated cosine similarity recovered from a Hamming distance:
//   theta = pi * hamming / dim;  cos(theta)
inline float approx_cosine_from_hamming(std::uint32_t hamming, std::size_t dim) noexcept {
    if (dim == 0u) {
        return 1.0f; // degenerate input; no angle is defined
    }

    const float theta = detail::kPiF
                      * static_cast<float>(hamming)
                      / static_cast<float>(dim);
    return std::cos(theta);
}

// Reference float32 cosine similarity. Baseline for parity testing and future
// re-ranking only - never called on the query hot path.
inline float cosine_similarity_f32(const float* a, const float* b, std::size_t dim) noexcept {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (std::size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom == 0.0f) {
        return 0.0f; // a zero vector has no direction
    }
    return dot / denom;
}

// ============================================================================
// ASYMMETRIC (query-side) SCORING
//
// Hamming distance throws away the query's magnitudes: both sides are +-1.
// When the full-precision float query is still in hand at search time - it
// always is - a much sharper estimate costs no extra index memory:
//
//     score(x) = dot(q_float, sign(x)),  sign(x) in {-1,+1}^dim
//
// computed ADC-style: one 256-entry table of partial sums per code byte,
// built once per query in O(padded_bytes * 256) adds, then each candidate is
// scored with padded_bytes table lookups (64 for dim = 512). This resolves
// the massive integer ties of Hamming ranking with float-grade resolution and
// is the re-ranking stage of HNSWGraph::search_reranked.
//
// Padding is inert by construction: padding components contribute q = 0 to
// their table entries, so the stored padding bits cannot affect the score.
// ============================================================================

// Number of floats a caller must allocate for one query's score table.
constexpr std::size_t asymmetric_table_floats(std::size_t dim) noexcept {
    return padded_bytes(dim) * 256u;
}

// Fills `table` (asymmetric_table_floats(dim) floats) for query `q`.
// table[b * 256 + v] = sum over the 8 components of code byte b, adding
// +q[j] where byte value v has the bit set and -q[j] where it does not.
// Built with a subset-sum recurrence: each entry is one add away from the
// entry with its lowest set bit cleared. No allocation.
inline void build_asymmetric_table(const float* q, std::size_t dim,
                                   float* table) noexcept {
    const std::size_t nbytes = padded_bytes(dim);
    for (std::size_t b = 0; b < nbytes; ++b) {
        float qv[8];
        float all_negative = 0.0f;
        for (std::size_t i = 0; i < 8u; ++i) {
            const std::size_t j = b * 8u + i;
            qv[i] = (j < dim) ? q[j] : 0.0f;
            all_negative -= qv[i];
        }

        float* t = table + b * 256u;
        t[0] = all_negative;
        for (std::uint32_t v = 1u; v < 256u; ++v) {
            const std::uint32_t low = v & (0u - v);
            const std::uint32_t bit =
                static_cast<std::uint32_t>(__builtin_ctz(low));
            t[v] = t[v ^ low] + 2.0f * qv[bit];
        }
    }
}

// Scores one quantized vector against a table built by
// build_asymmetric_table. Higher is more similar. Zero allocation; on the
// re-ranking path of search_reranked. Four independent accumulators keep the
// dependent-load chain from serializing (summation order differs from a
// single accumulator by ordinary float reassociation; the exactness test
// gates the result against a double-precision reference).
inline float asymmetric_score(const float* table, const std::uint8_t* x,
                              std::size_t dim) noexcept {
    const std::size_t nbytes = padded_bytes(dim);
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    std::size_t b = 0;
    for (; b + 4u <= nbytes; b += 4u) {
        s0 += table[(b + 0u) * 256u + static_cast<std::size_t>(x[b + 0u])];
        s1 += table[(b + 1u) * 256u + static_cast<std::size_t>(x[b + 1u])];
        s2 += table[(b + 2u) * 256u + static_cast<std::size_t>(x[b + 2u])];
        s3 += table[(b + 3u) * 256u + static_cast<std::size_t>(x[b + 3u])];
    }
    for (; b < nbytes; ++b) {
        s0 += table[b * 256u + static_cast<std::size_t>(x[b])];
    }
    return (s0 + s1) + (s2 + s3);
}

} // namespace edgevector

#endif // EDGEVECTOR_QUANTIZE_MATH_HPP
