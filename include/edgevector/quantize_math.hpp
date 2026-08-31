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
inline std::uint32_t hamming_distance(const std::uint8_t* a,
                                      const std::uint8_t* b,
                                      std::size_t dim) noexcept {
    const std::size_t words = detail::word_count(dim);
    std::uint32_t distance = 0u;

    for (std::size_t w = 0; w < words; ++w) {
        std::uint64_t wa;
        std::uint64_t wb;
        std::memcpy(&wa, a + w * 8u, sizeof(std::uint64_t));
        std::memcpy(&wb, b + w * 8u, sizeof(std::uint64_t));

        // Padding bits are zero in both operands, so they never contribute.
        distance += static_cast<std::uint32_t>(__builtin_popcountll(wa ^ wb));
    }

    return distance;
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

} // namespace edgevector

#endif // EDGEVECTOR_QUANTIZE_MATH_HPP
