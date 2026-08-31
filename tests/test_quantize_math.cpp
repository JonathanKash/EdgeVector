// ============================================================================
// Isolated unit test for edgevector/quantize_math.hpp
//
// Plain main() + hand-rolled checks, no test framework. Returns non-zero if
// any case fails. Test scaffolding is allowed to allocate; the zero-allocation
// rule binds the library functions, not this harness.
// ============================================================================

#include "edgevector/quantize_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* case_name) {
    if (ok) {
        std::printf("  PASS  %s\n", case_name);
    } else {
        std::printf("  FAIL  %s\n", case_name);
        ++g_failures;
    }
}

// Quantized-buffer holder. Backing it with an std::uint64_t vector guarantees
// the 8-byte alignment the library API requires of its callers.
class QuantizedBuffer {
public:
    explicit QuantizedBuffer(std::size_t dim)
        : words_(edgevector::padded_bytes(dim) / 8u, 0u) {}

    std::uint8_t* data() noexcept {
        return reinterpret_cast<std::uint8_t*>(words_.data());
    }
    const std::uint8_t* data() const noexcept {
        return reinterpret_cast<const std::uint8_t*>(words_.data());
    }
    std::size_t size() const noexcept { return words_.size() * 8u; }

private:
    std::vector<std::uint64_t> words_;
};

double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = x.size();
    double mx = 0.0;
    double my = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mx += x[i];
        my += y[i];
    }
    mx /= static_cast<double>(n);
    my /= static_cast<double>(n);

    double cov = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = x[i] - mx;
        const double dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    const double denom = std::sqrt(vx) * std::sqrt(vy);
    return (denom == 0.0) ? 0.0 : (cov / denom);
}

// ---------------------------------------------------------------------------
// Case 1: buffer layout
// ---------------------------------------------------------------------------
void test_layout() {
    std::printf("[1] Layout\n");
    check(edgevector::padded_bytes(64) == 8u, "padded_bytes(64) == 8");
    check(edgevector::padded_bytes(65) == 16u, "padded_bytes(65) == 16");
    check(edgevector::padded_bytes(100) == 16u, "padded_bytes(100) == 16");
}

// ---------------------------------------------------------------------------
// Case 2: quantize correctness on a hand-built 8-dim vector
// ---------------------------------------------------------------------------
void test_quantize_bits() {
    std::printf("[2] Quantize bit pattern\n");

    // signs:      +      -      +      -      +      +      -      0
    // bit i:      1      0      1      0      1      1      0      0
    // LSB-first byte = 0b00110101 = 0x35
    const float src[8] = { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, 6.0f, -7.0f, 0.0f };

    QuantizedBuffer q(8);
    edgevector::quantize(src, 8, q.data());

    check(q.data()[0] == 0x35u, "8-dim sign pattern encodes to 0x35");

    bool padding_zero = true;
    for (std::size_t i = 1; i < q.size(); ++i) {
        if (q.data()[i] != 0u) {
            padding_zero = false;
        }
    }
    check(padding_zero, "padding bytes 1..7 are zero");

    // Exactly 0.0f is not > 0.0f, so component 7 must be a zero bit.
    check((q.data()[0] & 0x80u) == 0u, "zero-valued component quantizes to 0");
}

// ---------------------------------------------------------------------------
// Case 3: Hamming ground truth, including the padded tail (dim = 100)
// ---------------------------------------------------------------------------
void test_hamming_ground_truth(std::size_t dim) {
    std::printf("[3] Hamming ground truth (dim = %zu)\n", dim);

    std::vector<float> v(dim);
    std::vector<float> flipped(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        // Alternating and never zero, so every sign flips cleanly.
        v[i] = ((i % 2u) == 0u) ? static_cast<float>(i + 1u)
                                : -static_cast<float>(i + 1u);
        flipped[i] = -v[i];
    }

    QuantizedBuffer qa(dim);
    QuantizedBuffer qb(dim);
    QuantizedBuffer qf(dim);
    edgevector::quantize(v.data(), dim, qa.data());
    edgevector::quantize(v.data(), dim, qb.data());
    edgevector::quantize(flipped.data(), dim, qf.data());

    const std::uint32_t self_dist =
        edgevector::hamming_distance(qa.data(), qb.data(), dim);
    const std::uint32_t flip_dist =
        edgevector::hamming_distance(qa.data(), qf.data(), dim);

    char name[128];
    std::snprintf(name, sizeof(name),
                  "dim %zu: vector vs itself == 0 (got %u)", dim, self_dist);
    check(self_dist == 0u, name);

    std::snprintf(name, sizeof(name),
                  "dim %zu: vector vs sign-flipped copy == %zu (got %u)",
                  dim, dim, flip_dist);
    check(flip_dist == static_cast<std::uint32_t>(dim), name);
}

// ---------------------------------------------------------------------------
// Case 4: cosine parity over controlled similarities spanning [-1, 1]
// (CLAUDE.md validation requirement)
// ---------------------------------------------------------------------------
void test_cosine_parity() {
    std::printf("[4] Cosine parity vs float32 baseline\n");

    const std::size_t dim = 512;
    const std::size_t pairs = 1000;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> similarity_dist(-1.0f, 1.0f);

    std::vector<float> a(dim);
    std::vector<float> b(dim);
    std::vector<double> exact;
    std::vector<double> approx;
    exact.reserve(pairs);
    approx.reserve(pairs);

    double abs_error_sum = 0.0;

    for (std::size_t p = 0; p < pairs; ++p) {
        double dot_aa = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            a[i] = dist(rng);
            dot_aa += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        }

        const float alpha = similarity_dist(rng);

        double dot_ga = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            b[i] = dist(rng);
            dot_ga += static_cast<double>(b[i]) * static_cast<double>(a[i]);
        }

        const double projection_scale = dot_ga / dot_aa;
        double g_perp_norm_sq = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            b[i] -= static_cast<float>(projection_scale) * a[i];
            g_perp_norm_sq +=
                static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        const float inv_a_norm =
            static_cast<float>(1.0 / std::sqrt(dot_aa));
        const float inv_g_perp_norm =
            static_cast<float>(1.0 / std::sqrt(g_perp_norm_sq));
        const float orthogonal_scale =
            std::sqrt(1.0f - alpha * alpha);
        for (std::size_t i = 0; i < dim; ++i) {
            a[i] *= inv_a_norm;
            b[i] = alpha * a[i] +
                   orthogonal_scale * b[i] * inv_g_perp_norm;
        }

        QuantizedBuffer qa(dim);
        QuantizedBuffer qb(dim);
        edgevector::quantize(a.data(), dim, qa.data());
        edgevector::quantize(b.data(), dim, qb.data());

        const std::uint32_t h =
            edgevector::hamming_distance(qa.data(), qb.data(), dim);
        const double exact_cos = static_cast<double>(
            edgevector::cosine_similarity_f32(a.data(), b.data(), dim));
        const double approx_cos = static_cast<double>(
            edgevector::approx_cosine_from_hamming(h, dim));

        exact.push_back(exact_cos);
        approx.push_back(approx_cos);
        abs_error_sum += std::fabs(exact_cos - approx_cos);
    }

    const double mae = abs_error_sum / static_cast<double>(pairs);
    const double corr = pearson(exact, approx);

    std::printf("      mean absolute error = %.6f  (gate < 0.05)\n", mae);
    std::printf("      pearson correlation = %.6f  (gate > 0.98)\n", corr);

    check(mae < 0.05, "mean absolute error < 0.05");
    check(corr > 0.98, "pearson correlation > 0.98");
}

// ---------------------------------------------------------------------------
// Case 5: ranking preservation with ten planted high-similarity candidates
// separated from a low-similarity background
// ---------------------------------------------------------------------------
void test_ranking_preservation() {
    std::printf("[5] Ranking preservation (top-10 of 200 candidates)\n");

    const std::size_t dim = 512;
    const std::size_t candidates = 200;
    const std::size_t top_k = 10;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> background_similarity(0.0f, 0.30f);

    std::vector<float> query(dim);
    double query_norm_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        query[i] = dist(rng);
        query_norm_sq +=
            static_cast<double>(query[i]) * static_cast<double>(query[i]);
    }
    const float inv_query_norm =
        static_cast<float>(1.0 / std::sqrt(query_norm_sq));
    for (std::size_t i = 0; i < dim; ++i) {
        query[i] *= inv_query_norm;
    }
    QuantizedBuffer q_query(dim);
    edgevector::quantize(query.data(), dim, q_query.data());

    std::vector<float> cand(dim);
    std::vector<std::pair<float, std::size_t> > by_cosine;
    std::vector<std::pair<std::uint32_t, std::size_t> > by_hamming;
    by_cosine.reserve(candidates);
    by_hamming.reserve(candidates);

    for (std::size_t c = 0; c < candidates; ++c) {
        const float alpha = (c < top_k)
            ? 0.70f + (0.20f * static_cast<float>(c) /
                       static_cast<float>(top_k - 1u))
            : background_similarity(rng);

        double dot_gq = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            cand[i] = dist(rng);
            dot_gq +=
                static_cast<double>(cand[i]) * static_cast<double>(query[i]);
        }

        double dot_qq = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            dot_qq +=
                static_cast<double>(query[i]) * static_cast<double>(query[i]);
        }
        const double projection_scale = dot_gq / dot_qq;
        double g_perp_norm_sq = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            cand[i] -= static_cast<float>(projection_scale) * query[i];
            g_perp_norm_sq +=
                static_cast<double>(cand[i]) * static_cast<double>(cand[i]);
        }

        const float inv_g_perp_norm =
            static_cast<float>(1.0 / std::sqrt(g_perp_norm_sq));
        const float orthogonal_scale =
            std::sqrt(1.0f - alpha * alpha);
        for (std::size_t i = 0; i < dim; ++i) {
            cand[i] = alpha * query[i] +
                      orthogonal_scale * cand[i] * inv_g_perp_norm;
        }

        QuantizedBuffer q_cand(dim);
        edgevector::quantize(cand.data(), dim, q_cand.data());

        by_cosine.emplace_back(
            edgevector::cosine_similarity_f32(query.data(), cand.data(), dim), c);
        by_hamming.emplace_back(
            edgevector::hamming_distance(q_query.data(), q_cand.data(), dim), c);
    }

    // Best float32 cosine is the highest; best Hamming distance is the lowest.
    std::sort(by_cosine.begin(), by_cosine.end(),
              [](const std::pair<float, std::size_t>& l,
                 const std::pair<float, std::size_t>& r) {
                  return l.first > r.first;
              });
    std::sort(by_hamming.begin(), by_hamming.end(),
              [](const std::pair<std::uint32_t, std::size_t>& l,
                 const std::pair<std::uint32_t, std::size_t>& r) {
                  return l.first < r.first;
              });

    std::size_t overlap = 0;
    for (std::size_t i = 0; i < top_k; ++i) {
        for (std::size_t j = 0; j < top_k; ++j) {
            if (by_cosine[i].second == by_hamming[j].second) {
                ++overlap;
                break;
            }
        }
    }

    std::printf("      top-10 overlap = %zu / 10  (gate >= 9)\n", overlap);
    check(overlap >= 9u, "top-10 sets overlap in at least 9 of 10");
}

} // namespace

int main() {
    std::printf("=== EdgeVector :: quantize_math unit tests ===\n\n");

    test_layout();
    test_quantize_bits();
    test_hamming_ground_truth(64);
    test_hamming_ground_truth(100);
    test_cosine_parity();
    test_ranking_preservation();

    std::printf("\n=== %s ===\n",
                (g_failures == 0) ? "ALL CASES PASSED" : "FAILURES DETECTED");
    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
    }
    return (g_failures == 0) ? 0 : 1;
}
