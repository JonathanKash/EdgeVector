// ============================================================================
// EdgeVector ITQ pipeline example: when your embeddings have a decaying
// spectrum (real embeddings always do), a learned rotation makes every bit
// of the binary code informative.
//
// Build and run (from examples/):   make run        (runs quickstart too)
// or via CMake (from the repo root):
//     cmake -B build -DEDGEVECTOR_BUILD_EXAMPLES=ON && cmake --build build
//     ./build/itq_pipeline
//
// The program synthesizes anisotropic data (energy concentrated in the
// leading dimensions), trains an ItqRotation, builds two indexes over the
// SAME floats - raw sign codes vs rotated codes - and measures recall
// against exact float32 ground truth for both. It checks its own results
// (the rotated index must win) and exits non-zero otherwise; CI runs it.
//
// The two properties that make ITQ practical here:
//   1. Rotation-only (no centering/PCA), so cosine is preserved EXACTLY -
//      ground truth and exact re-ranking are unchanged.
//   2. Deterministic per seed - the same training data gives bit-identical
//      rotations on every platform.
// ============================================================================

#include <edgevector/edgevector.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace edgevector;

static int g_failures = 0;
static void expect(bool ok, const char* what) {
    std::printf("  %s  %s\n", ok ? "ok " : "FAIL", what);
    if (!ok) ++g_failures;
}

int main() {
    std::printf("EdgeVector %s ITQ pipeline example\n\n", EDGEVECTOR_VERSION);

    // ------------------------------------------------------------------
    // 0. Anisotropic embedding-like data: clustered, with dimension d
    //    scaled by exp(-4d/dim) - a decaying spectrum. Sign quantization
    //    alone wastes most of its bits on the low-energy tail.
    // ------------------------------------------------------------------
    const std::size_t dim = 256;
    const std::uint32_t n = 5000;
    const std::size_t n_queries = 40;
    const std::uint32_t k = 10;
    const std::size_t rb = padded_bytes(dim);

    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> centers(60 * dim);
    for (float& c : centers) c = gauss(rng);
    std::vector<float> scale(dim);
    for (std::size_t d = 0; d < dim; ++d)
        scale[d] = std::exp(-4.0f * static_cast<float>(d) /
                            static_cast<float>(dim));

    const std::size_t total = n + n_queries;
    std::vector<float> floats(total * dim);
    for (std::size_t i = 0; i < total; ++i) {
        const float* c = centers.data() + (i % 60) * dim;
        for (std::size_t d = 0; d < dim; ++d)
            floats[i * dim + d] = (c[d] + 0.5f * gauss(rng)) * scale[d];
    }
    const float* qfloats = floats.data() + static_cast<std::size_t>(n) * dim;

    // ------------------------------------------------------------------
    // 1. Train the rotation on the corpus (offline; deterministic).
    //    Persist it next to your index files - queries need the same G.
    // ------------------------------------------------------------------
    // A 2k-row subsample and 10 iterations train in seconds and capture the
    // spectrum fine; production rotations can afford the full corpus.
    ItqRotation rot(dim);
    expect(rot.train(floats.data(), 2000, dim, /*iterations=*/10) ==
               ItqStatus::ok,
           "rotation trained");
    std::printf("      quantization error %.3f -> %.3f, orthogonality %.1e\n",
                rot.error_before_training(), rot.error_after_training(),
                rot.orthogonality_error());
    expect(rot.save("itq_example.evrt") == ItqStatus::ok, "rotation saved");
    ItqRotation loaded(dim);
    expect(loaded.load("itq_example.evrt") == ItqStatus::ok,
           "rotation reloaded (validated: magic, length, orthogonality)");

    // ------------------------------------------------------------------
    // 2. Two code blocks over the SAME floats: raw and rotated.
    // ------------------------------------------------------------------
    std::vector<std::uint64_t> raw_w((rb / 8) * n), itq_w((rb / 8) * n);
    auto* raw_codes = reinterpret_cast<std::uint8_t*>(raw_w.data());
    auto* itq_codes = reinterpret_cast<std::uint8_t*>(itq_w.data());
    std::vector<float> scratch(dim);
    for (std::uint32_t i = 0; i < n; ++i) {
        quantize(floats.data() + i * dim, dim, raw_codes + i * rb);
        loaded.rotate_quantize(floats.data() + i * dim, scratch.data(),
                               itq_codes + i * rb);
    }

    HNSWGraph raw_graph(raw_codes, rb, dim, n);
    HNSWGraph itq_graph(itq_codes, rb, dim, n);
    std::vector<std::uint32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0u);
    expect(raw_graph.insert_batch(ids.data(), n, 0) == n, "raw index built");
    expect(itq_graph.insert_batch(ids.data(), n, 0) == n, "ITQ index built");

    // ------------------------------------------------------------------
    // 3. Query both with asymmetric re-ranking; measure recall against
    //    exact float32 cosine ground truth. Cosine is rotation-invariant,
    //    so ONE ground truth serves both indexes.
    // ------------------------------------------------------------------
    std::vector<float> qrot(dim);
    std::vector<std::uint64_t> qw(rb / 8);
    auto* qbits = reinterpret_cast<std::uint8_t*>(qw.data());
    ScoredResult res[10];
    std::vector<std::pair<float, std::uint32_t>> gt(n);

    std::size_t raw_hits = 0, itq_hits = 0;
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        const float* q = qfloats + qi * dim;
        for (std::uint32_t i = 0; i < n; ++i)
            gt[i] = {-cosine_similarity_f32(q, floats.data() + i * dim, dim),
                     i};
        std::partial_sort(gt.begin(), gt.begin() + k, gt.end());

        quantize(q, dim, qbits); // raw pipeline: quantize the query as-is
        std::uint32_t found = raw_graph.search_reranked(qbits, q, k, 100, res);
        for (std::uint32_t i = 0; i < found; ++i)
            for (std::size_t t = 0; t < k; ++t)
                if (res[i].id == gt[t].second) { ++raw_hits; break; }

        loaded.rotate(q, qrot.data()); // ITQ pipeline: rotate, then quantize
        quantize(qrot.data(), dim, qbits);
        found = itq_graph.search_reranked(qbits, qrot.data(), k, 100, res);
        for (std::uint32_t i = 0; i < found; ++i)
            for (std::size_t t = 0; t < k; ++t)
                if (res[i].id == gt[t].second) { ++itq_hits; break; }
    }
    const double denom = static_cast<double>(n_queries * k);
    const double raw_recall = raw_hits / denom;
    const double itq_recall = itq_hits / denom;
    std::printf("      recall@10 vs float32 truth: raw %.3f, ITQ %.3f\n",
                raw_recall, itq_recall);
    expect(itq_recall > raw_recall, "rotated codes beat raw codes");

    // ------------------------------------------------------------------
    // 4. Exact re-ranking over the ROTATED index uses the ORIGINAL floats
    //    and the ORIGINAL query: orthogonal rotations preserve cosine.
    // ------------------------------------------------------------------
    const float* q = qfloats; // first query again
    loaded.rotate(q, qrot.data());
    quantize(qrot.data(), dim, qbits);
    const std::uint32_t found = itq_graph.search_exact_reranked(
        qbits, q, floats.data(), dim, k, 100, res);
    expect(found == k, "exact re-rank over the rotated index works "
                       "with unrotated floats");

    std::remove("itq_example.evrt");
    std::printf("\n%s\n", g_failures == 0 ? "All ITQ example checks passed."
                                          : "ITQ EXAMPLE FAILURES - see above.");
    return g_failures == 0 ? 0 : 1;
}
