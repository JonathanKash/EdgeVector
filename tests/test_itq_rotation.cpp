// ============================================================================
// Isolated unit test for edgevector/itq_rotation.hpp
//
// Verifies the mathematical invariants ITQ must hold (orthogonality, exact
// cosine preservation, monotone objective improvement, determinism), the
// persistence format's hostility to bad files, and — end to end — that the
// learned rotation actually improves retrieval accuracy on anisotropic data
// with a decaying spectrum, which is the shape real embedding data has.
// ============================================================================

#include "edgevector/hnsw_graph.hpp"
#include "edgevector/itq_rotation.hpp"
#include "edgevector/quantize_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// Anisotropic clustered data: cluster centers + noise, then dimension d
// scaled by exp(-4d/dim) — an exponentially decaying spectrum, the regime
// where one-bit-per-dimension quantization wastes most of its bits and ITQ
// is designed to help.
void fill_anisotropic(std::vector<float>& out, std::size_t n, std::size_t dim,
                      std::size_t n_clusters, std::mt19937& rng) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_int_distribution<std::size_t> pick(0, n_clusters - 1u);

    std::vector<float> centers(n_clusters * dim);
    for (float& c : centers) {
        c = g(rng);
    }
    std::vector<float> scale(dim);
    for (std::size_t d = 0; d < dim; ++d) {
        scale[d] = std::exp(-4.0f * static_cast<float>(d) /
                            static_cast<float>(dim));
    }

    out.assign(n * dim, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const float* c = centers.data() + pick(rng) * dim;
        for (std::size_t d = 0; d < dim; ++d) {
            out[i * dim + d] = (c[d] + 0.5f * g(rng)) * scale[d];
        }
    }
}

// ---------------------------------------------------------------------------
// Case 1: identity before training
// ---------------------------------------------------------------------------
void test_identity() {
    std::printf("[1] Identity rotation before training\n");

    const std::size_t dim = 64;
    edgevector::ItqRotation rot(dim);

    std::mt19937 rng(42);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> x(dim);
    std::vector<float> y(dim);
    for (std::size_t d = 0; d < dim; ++d) {
        x[d] = g(rng);
    }
    rot.rotate(x.data(), y.data());

    bool same = true;
    for (std::size_t d = 0; d < dim; ++d) {
        if (std::fabs(x[d] - y[d]) > 1e-6f) {
            same = false;
        }
    }
    check(same, "rotate() with an untrained rotation is the identity");
}

// ---------------------------------------------------------------------------
// Case 2: training invariants — orthogonality, cosine preservation,
// objective improvement, determinism, rotate_quantize equivalence
// ---------------------------------------------------------------------------
void test_training_invariants() {
    std::printf("[2] Training invariants (dim = 64, n = 2000)\n");

    const std::size_t dim = 64;
    const std::size_t n = 2000;
    std::mt19937 rng(42);
    std::vector<float> data;
    fill_anisotropic(data, n, dim, 50, rng);

    edgevector::ItqRotation rot(dim);
    const edgevector::ItqStatus st =
        rot.train(data.data(), n, dim, /*iterations=*/20, /*seed=*/42u);
    check(st == edgevector::ItqStatus::ok, "train() returns ok");

    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "learned matrix is orthogonal (max |G^T G - I| = %.2e, "
                  "gate < 1e-3)", rot.orthogonality_error());
    check(rot.orthogonality_error() < 1e-3, msg);

    std::snprintf(msg, sizeof(msg),
                  "quantization error drops: %.4f -> %.4f (gate: > 5%% drop)",
                  rot.error_before_training(), rot.error_after_training());
    check(rot.error_after_training() <
              0.95 * rot.error_before_training(), msg);

    // Cosine preservation: an orthogonal transform must not move cosines.
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> a(dim);
    std::vector<float> b(dim);
    std::vector<float> ra(dim);
    std::vector<float> rb(dim);
    double worst = 0.0;
    for (int t = 0; t < 20; ++t) {
        for (std::size_t d = 0; d < dim; ++d) {
            a[d] = g(rng);
            b[d] = g(rng);
        }
        rot.rotate(a.data(), ra.data());
        rot.rotate(b.data(), rb.data());
        const double before = static_cast<double>(
            edgevector::cosine_similarity_f32(a.data(), b.data(), dim));
        const double after = static_cast<double>(
            edgevector::cosine_similarity_f32(ra.data(), rb.data(), dim));
        const double err = std::fabs(before - after);
        if (err > worst) {
            worst = err;
        }
    }
    std::snprintf(msg, sizeof(msg),
                  "cosine similarity preserved exactly (worst |delta| = %.2e, "
                  "gate < 1e-4)", worst);
    check(worst < 1e-4, msg);

    // Determinism: same data + seed => bit-identical matrix.
    edgevector::ItqRotation rot2(dim);
    rot2.train(data.data(), n, dim, 20, 42u);
    check(std::memcmp(rot.matrix(), rot2.matrix(),
                      dim * dim * sizeof(float)) == 0,
          "training is deterministic for a fixed seed");

    // rotate_quantize == rotate then quantize.
    std::vector<float> scratch(dim);
    std::vector<std::uint64_t> qa(edgevector::padded_bytes(dim) / 8u, 0u);
    std::vector<std::uint64_t> qb(edgevector::padded_bytes(dim) / 8u, 0u);
    rot.rotate(a.data(), ra.data());
    edgevector::quantize(ra.data(), dim, reinterpret_cast<std::uint8_t*>(qa.data()));
    rot.rotate_quantize(a.data(), scratch.data(),
                        reinterpret_cast<std::uint8_t*>(qb.data()));
    check(std::memcmp(qa.data(), qb.data(),
                      edgevector::padded_bytes(dim)) == 0,
          "rotate_quantize matches rotate-then-quantize bit for bit");
}

// ---------------------------------------------------------------------------
// Case 3: input validation and persistence
// ---------------------------------------------------------------------------
void test_validation_and_persistence() {
    std::printf("[3] Input validation and persistence\n");

    const std::size_t dim = 64;
    const std::size_t n = 2000;
    std::mt19937 rng(42);
    std::vector<float> data;
    fill_anisotropic(data, n, dim, 50, rng);

    edgevector::ItqRotation rot(dim);
    check(rot.train(nullptr, n, dim) == edgevector::ItqStatus::bad_input,
          "null data -> bad_input");
    check(rot.train(data.data(), dim - 1u, dim) ==
              edgevector::ItqStatus::bad_input,
          "n < dim -> bad_input");
    check(rot.train(data.data(), n, dim + 1u) ==
              edgevector::ItqStatus::bad_input,
          "dimension mismatch -> bad_input");

    check(rot.train(data.data(), n, dim, 20, 42u) == edgevector::ItqStatus::ok,
          "train() on valid input returns ok");

    const char* const kFile = "ev_test_rotation.evrt";
    const char* const kBadFile = "ev_test_rotation_bad.evrt";
    check(rot.save(kFile) == edgevector::ItqStatus::ok, "save() returns ok");

    edgevector::ItqRotation loaded(dim);
    check(loaded.load(kFile) == edgevector::ItqStatus::ok, "load() returns ok");
    check(std::memcmp(rot.matrix(), loaded.matrix(),
                      dim * dim * sizeof(float)) == 0,
          "matrix round-trips bit for bit");

    // Wrong dimension object.
    edgevector::ItqRotation wrong_dim(dim / 2u);
    check(wrong_dim.load(kFile) == edgevector::ItqStatus::incompatible,
          "dimension mismatch on load -> incompatible");

    // Corruptions.
    {
        std::FILE* in = std::fopen(kFile, "rb");
        std::fseek(in, 0, SEEK_END);
        const long len = std::ftell(in);
        std::fseek(in, 0, SEEK_SET);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(len));
        const bool read_ok =
            std::fread(bytes.data(), 1u, bytes.size(), in) == bytes.size();
        std::fclose(in);
        check(read_ok, "reference rotation file read back");

        std::vector<std::uint8_t> bad = bytes;
        bad[0] = static_cast<std::uint8_t>('X');
        std::FILE* out = std::fopen(kBadFile, "wb");
        std::fwrite(bad.data(), 1u, bad.size(), out);
        std::fclose(out);
        check(loaded.load(kBadFile) == edgevector::ItqStatus::bad_magic,
              "flipped magic -> bad_magic");

        out = std::fopen(kBadFile, "wb");
        std::fwrite(bytes.data(), 1u, bytes.size() - 7u, out);
        std::fclose(out);
        check(loaded.load(kBadFile) == edgevector::ItqStatus::io_error,
              "truncated file -> io_error");

        // Valid header, garbage (non-orthogonal) matrix.
        std::vector<std::uint8_t> garbage = bytes;
        for (std::size_t i = 16u; i < garbage.size(); ++i) {
            garbage[i] = static_cast<std::uint8_t>(i * 37u);
        }
        out = std::fopen(kBadFile, "wb");
        std::fwrite(garbage.data(), 1u, garbage.size(), out);
        std::fclose(out);
        const edgevector::ItqStatus st = loaded.load(kBadFile);
        check(st == edgevector::ItqStatus::corrupt,
              "non-orthogonal matrix -> corrupt");

        // A failed load leaves the identity, not the garbage.
        std::vector<float> x(dim, 1.0f);
        std::vector<float> y(dim);
        loaded.rotate(x.data(), y.data());
        bool identity = true;
        for (std::size_t d = 0; d < dim; ++d) {
            if (std::fabs(y[d] - 1.0f) > 1e-6f) {
                identity = false;
            }
        }
        check(identity, "failed load resets to the identity rotation");
    }

    std::remove(kFile);
    std::remove(kBadFile);
}

// ---------------------------------------------------------------------------
// Case 4: end to end — on anisotropic data, ITQ-rotated codes must beat raw
// codes at retrieving the true float32 neighbors.
// ---------------------------------------------------------------------------
void test_end_to_end_recall() {
    std::printf("[4] End-to-end recall on anisotropic data "
                "(dim = 128, n = 1500)\n");

    const std::size_t dim = 128;
    const std::size_t n = 1500;
    const std::size_t n_queries = 25;
    const std::uint32_t k = 10u;
    const std::size_t rb = edgevector::padded_bytes(dim);

    std::mt19937 rng(42);
    std::vector<float> floats;
    fill_anisotropic(floats, n + n_queries, dim, 40, rng);
    // Last n_queries rows are held-out queries from the same distribution.
    const float* qfloats = floats.data() + n * dim;

    edgevector::ItqRotation rot(dim);
    check(rot.train(floats.data(), n, dim, /*iterations=*/15, /*seed=*/42u) ==
              edgevector::ItqStatus::ok,
          "rotation trained on the corpus");

    // Two code blocks over the same floats: raw and rotated.
    std::vector<std::uint64_t> raw_words((rb / 8u) * n, 0u);
    std::vector<std::uint64_t> itq_words((rb / 8u) * n, 0u);
    std::uint8_t* raw_base = reinterpret_cast<std::uint8_t*>(raw_words.data());
    std::uint8_t* itq_base = reinterpret_cast<std::uint8_t*>(itq_words.data());
    std::vector<float> scratch(dim);
    for (std::size_t i = 0; i < n; ++i) {
        edgevector::quantize(floats.data() + i * dim, dim, raw_base + i * rb);
        rot.rotate_quantize(floats.data() + i * dim, scratch.data(),
                            itq_base + i * rb);
    }

    edgevector::HNSWGraph raw_graph(raw_base, rb, dim,
                                    static_cast<std::uint32_t>(n));
    edgevector::HNSWGraph itq_graph(itq_base, rb, dim,
                                    static_cast<std::uint32_t>(n));
    for (std::uint32_t i = 0; i < n; ++i) {
        raw_graph.insert(i);
        itq_graph.insert(i);
    }

    std::vector<std::pair<float, std::uint32_t>> gt(n);
    std::vector<std::uint64_t> qw(rb / 8u, 0u);
    std::uint8_t* qbits = reinterpret_cast<std::uint8_t*>(qw.data());
    std::vector<float> qrot(dim);
    edgevector::ScoredResult res[10];

    std::size_t raw_hits = 0;
    std::size_t itq_hits = 0;
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        const float* q = qfloats + qi * dim;

        // Float32 cosine ground truth — identical for both variants because
        // an orthogonal rotation preserves cosine exactly.
        for (std::size_t i = 0; i < n; ++i) {
            gt[i] = std::make_pair(
                -edgevector::cosine_similarity_f32(q, floats.data() + i * dim,
                                                   dim),
                static_cast<std::uint32_t>(i));
        }
        std::partial_sort(gt.begin(), gt.begin() + k, gt.end());

        // Raw codes: quantized query, asymmetric re-rank.
        edgevector::quantize(q, dim, qbits);
        std::uint32_t found =
            raw_graph.search_reranked(qbits, q, k, 100u, res);
        for (std::uint32_t i = 0; i < found; ++i) {
            for (std::size_t t = 0; t < k; ++t) {
                if (res[i].id == gt[t].second) {
                    ++raw_hits;
                    break;
                }
            }
        }

        // ITQ codes: rotate the query, then the same pipeline.
        rot.rotate(q, qrot.data());
        edgevector::quantize(qrot.data(), dim, qbits);
        found = itq_graph.search_reranked(qbits, qrot.data(), k, 100u, res);
        for (std::uint32_t i = 0; i < found; ++i) {
            for (std::size_t t = 0; t < k; ++t) {
                if (res[i].id == gt[t].second) {
                    ++itq_hits;
                    break;
                }
            }
        }
    }

    const double denom = static_cast<double>(n_queries * k);
    const double raw_recall = static_cast<double>(raw_hits) / denom;
    const double itq_recall = static_cast<double>(itq_hits) / denom;
    std::printf("      recall@10 vs float32 truth (asym re-rank): "
                "raw codes %.3f, ITQ codes %.3f\n", raw_recall, itq_recall);
    check(itq_recall >= raw_recall,
          "ITQ codes never lose to raw codes");
    check(itq_recall > raw_recall + 0.05,
          "ITQ codes beat raw codes by more than 5 points");
}

} // namespace

int main() {
    std::printf("=== EdgeVector :: itq_rotation unit tests ===\n\n");

    test_identity();
    test_training_invariants();
    test_validation_and_persistence();
    test_end_to_end_recall();

    std::printf("\n=== %s ===\n",
                (g_failures == 0) ? "ALL CASES PASSED" : "FAILURES DETECTED");
    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
    }
    return (g_failures == 0) ? 0 : 1;
}
