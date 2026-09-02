// ============================================================================
// EdgeVector benchmark: 100,000 x 512-d vectors, single thread.
//
// Not a pass/fail test — it measures and prints, in markdown, the numbers the
// README reports. Build with the release (-DNDEBUG) configuration; numbers
// from an assert-enabled build are not comparable.
//
// TWO DATASETS, reported separately and honestly:
//
//  1. "clustered" — 1,000 Gaussian cluster centers, vectors = center + 0.5 *
//     noise. This mimics the manifold structure of real embedding data
//     (model embeddings are never iid), and is the scenario the recall
//     numbers should be read from.
//  2. "iid random" — structureless N(0,1) vectors. This is the known
//     adversarial case for EVERY approximate-nearest-neighbor index: with no
//     manifold to exploit, pairwise Hamming distances concentrate (sigma/mu
//     ~= 4% at 512 bits) and graph navigation loses its gradient. Recall is
//     low here BY THE NATURE OF THE DATA, not by a defect; it is included so
//     nobody mistakes the clustered numbers for a universal promise.
//
// TWO ACCURACY METRICS per sweep row:
//
//  - recall vs the BINARY ground truth (exact Hamming top-10): measures the
//    graph in isolation.
//  - recall vs the FLOAT32 ground truth (exact cosine top-10 on the original
//    floats): the end-to-end number an application actually experiences —
//    reported for plain Hamming ranking and for search_reranked(), whose
//    asymmetric dot(q_float, sign(x)) re-ranking is the accuracy feature.
// ============================================================================

#include "edgevector/hnsw_graph.hpp"
#include "edgevector/itq_rotation.hpp"
#include "edgevector/quantize_math.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

constexpr std::size_t kDim = 512;
constexpr std::size_t kN = 100000;
constexpr std::size_t kQueries = 1000;
constexpr std::uint32_t kK = 10u;
constexpr std::size_t kClusters = 1000;

class RecordBlock {
public:
    RecordBlock(std::size_t dim, std::size_t count)
        : record_bytes_(edgevector::padded_bytes(dim)),
          words_((record_bytes_ / 8u) * count, 0u) {}

    std::uint8_t* record(std::size_t i) noexcept {
        return reinterpret_cast<std::uint8_t*>(words_.data()) + i * record_bytes_;
    }
    const std::uint8_t* record(std::size_t i) const noexcept {
        return reinterpret_cast<const std::uint8_t*>(words_.data()) +
               i * record_bytes_;
    }
    const std::uint8_t* base() const noexcept {
        return reinterpret_cast<const std::uint8_t*>(words_.data());
    }
    std::size_t record_bytes() const noexcept { return record_bytes_; }

private:
    std::size_t record_bytes_;
    std::vector<std::uint64_t> words_;
};

// Draws one raw float vector: either iid N(0,1), or center + 0.5 * noise.
struct Sampler {
    bool clustered;
    std::mt19937 rng;
    std::normal_distribution<float> g{0.0f, 1.0f};
    std::uniform_int_distribution<std::size_t> pick{0, kClusters - 1u};
    std::vector<float> centers;

    explicit Sampler(bool is_clustered)
        : clustered(is_clustered), rng(42) {
        if (clustered) {
            centers.resize(kClusters * kDim);
            for (float& c : centers) {
                c = g(rng);
            }
        }
    }

    void draw(float* out) {
        if (clustered) {
            const float* c = centers.data() + pick(rng) * kDim;
            for (std::size_t d = 0; d < kDim; ++d) {
                out[d] = c[d] + 0.5f * g(rng);
            }
        } else {
            for (std::size_t d = 0; d < kDim; ++d) {
                out[d] = g(rng);
            }
        }
    }
};

std::size_t overlap10(const std::uint32_t* got, std::uint32_t n_got,
                      const std::uint32_t* truth) {
    std::size_t hits = 0;
    for (std::uint32_t i = 0; i < n_got; ++i) {
        for (std::size_t t = 0; t < kK; ++t) {
            if (got[i] == truth[t]) {
                ++hits;
                break;
            }
        }
    }
    return hits;
}

void run_scenario(const char* name, bool clustered, bool full_report) {
    std::printf("## Scenario: %s\n\n", name);

    Sampler sampler(clustered);

    // Dataset: keep the floats (for float32 ground truth) and the codes.
    std::vector<float> floats(kN * kDim);
    RecordBlock data(kDim, kN);
    for (std::size_t i = 0; i < kN; ++i) {
        sampler.draw(floats.data() + i * kDim);
        edgevector::quantize(floats.data() + i * kDim, kDim, data.record(i));
    }

    edgevector::HNSWGraph graph(data.base(), data.record_bytes(), kDim,
                                static_cast<std::uint32_t>(kN),
                                /*m=*/16u, /*ef_construction=*/200u,
                                /*max_ef_search=*/256u, /*seed=*/42u);
    Clock::time_point t0 = Clock::now();
    for (std::uint32_t i = 0; i < kN; ++i) {
        graph.insert(i);
    }
    const double build_s = seconds_since(t0);

    if (full_report) {
        const char* const kGraphFile = "ev_bench_graph.evhg";
        t0 = Clock::now();
        if (graph.save_graph(kGraphFile) != edgevector::GraphIoStatus::ok) {
            std::printf("FATAL: save_graph failed\n");
            std::exit(1);
        }
        const double save_s = seconds_since(t0);

        edgevector::HNSWGraph loaded(data.base(), data.record_bytes(), kDim,
                                     static_cast<std::uint32_t>(kN),
                                     16u, 200u, 256u, 42u);
        t0 = Clock::now();
        if (loaded.load_graph(kGraphFile) != edgevector::GraphIoStatus::ok) {
            std::printf("FATAL: load_graph failed\n");
            std::exit(1);
        }
        const double load_s = seconds_since(t0);
        std::remove(kGraphFile);

        // Parallel build on a throwaway graph: the serial graph stays the
        // one measured below so recall/latency numbers remain comparable.
        double par_build_s = 0.0;
        double reclaim_ms = 0.0;
        unsigned hw_threads = 1u;
        bool par_ok = false;
        {
#if defined(EDGEVECTOR_HAS_THREADS)
            hw_threads = std::thread::hardware_concurrency();
            if (hw_threads == 0u) {
                hw_threads = 1u;
            }
#endif
            std::vector<std::uint32_t> ids(kN);
            for (std::size_t i = 0; i < kN; ++i) {
                ids[i] = static_cast<std::uint32_t>(i);
            }
            edgevector::HNSWGraph par(data.base(), data.record_bytes(), kDim,
                                      static_cast<std::uint32_t>(kN),
                                      16u, 200u, 256u, 42u);
            t0 = Clock::now();
            const std::size_t inserted =
                par.insert_batch(ids.data(), kN, hw_threads);
            par_build_s = seconds_since(t0);
            par_ok = (inserted == kN) && par.validate_integrity();

            // Slot-reclamation latency on the throwaway graph (bytes are
            // shared with the measured graph, so they stay unchanged: the
            // remove -> reinsert cycle costs the same either way).
            t0 = Clock::now();
            for (std::uint32_t r = 0u; r < 100u; ++r) {
                const std::uint32_t id = r * 997u;
                par.remove(id);
                par.reinsert(id);
            }
            reclaim_ms = 1000.0 * seconds_since(t0) / 100.0;
            par_ok = par_ok && par.validate_integrity();
        }

        std::printf("| Stage | Time |\n|---|---|\n");
        std::printf("| Build (insert %zu vectors, 1 thread) | %.1f s |\n",
                    kN, build_s);
        std::printf("| Build, insert_batch (%u threads) | %.1f s (%.1fx; "
                    "integrity %s) |\n",
                    hw_threads, par_build_s, build_s / par_build_s,
                    par_ok ? "validated" : "FAILED");
        std::printf("| Reclaim one slot (remove + reinsert), avg of 100 | %.2f ms |\n",
                    reclaim_ms);
        std::printf("| Save graph | %.3f s |\n", save_s);
        std::printf("| Load graph (startup cost with persistence) | %.3f s |\n",
                    load_s);
        std::printf("| Load vs rebuild | %.0fx faster |\n\n", build_s / load_s);

        const std::size_t quant_payload = kN * edgevector::padded_bytes(kDim);
        const std::size_t float_payload = kN * kDim * sizeof(float);
        std::printf("| Memory | Size |\n|---|---|\n");
        std::printf("| Quantized vectors | %.2f MB |\n",
                    static_cast<double>(quant_payload) / 1.0e6);
        std::printf("| float32 equivalent | %.2f MB (%.1fx larger) |\n",
                    static_cast<double>(float_payload) / 1.0e6,
                    static_cast<double>(float_payload) /
                        static_cast<double>(quant_payload));
        std::printf("| Graph (links + scratch) | %.2f MB |\n\n",
                    static_cast<double>(graph.graph_memory_bytes()) / 1.0e6);
    } else {
        std::printf("Build: %.1f s (1 thread).\n\n", build_s);
    }
    std::fflush(stdout);

    // Queries (same distribution as the data), in both representations.
    std::vector<float> qfloats(kQueries * kDim);
    std::vector<std::uint64_t> qwords(
        (edgevector::padded_bytes(kDim) / 8u) * kQueries, 0u);
    std::uint8_t* qbase = reinterpret_cast<std::uint8_t*>(qwords.data());
    const std::size_t qstride = edgevector::padded_bytes(kDim);
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        sampler.draw(qfloats.data() + qi * kDim);
        edgevector::quantize(qfloats.data() + qi * kDim, kDim,
                             qbase + qi * qstride);
    }

    // Exact binary (Hamming) ground truth.
    std::vector<std::uint32_t> bin_truth(kQueries * kK);
    t0 = Clock::now();
    {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> all(kN);
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const std::uint8_t* q = qbase + qi * qstride;
            for (std::size_t i = 0; i < kN; ++i) {
                all[i] = std::make_pair(
                    edgevector::hamming_distance(q, data.record(i), kDim),
                    static_cast<std::uint32_t>(i));
            }
            std::partial_sort(all.begin(), all.begin() + kK, all.end());
            for (std::size_t i = 0; i < kK; ++i) {
                bin_truth[qi * kK + i] = all[i].second;
            }
        }
    }
    const double brute_s = seconds_since(t0);
    const double brute_qps = static_cast<double>(kQueries) / brute_s;

    // Exact float32 cosine ground truth (rank by dot / ||x||; ||q|| is a
    // per-query constant that cannot change the order).
    std::vector<std::uint32_t> f32_truth(kQueries * kK);
    {
        std::vector<float> inv_norm(kN);
        for (std::size_t i = 0; i < kN; ++i) {
            double s = 0.0;
            const float* x = floats.data() + i * kDim;
            for (std::size_t d = 0; d < kDim; ++d) {
                s += static_cast<double>(x[d]) * static_cast<double>(x[d]);
            }
            inv_norm[i] =
                (s > 0.0) ? static_cast<float>(1.0 / std::sqrt(s)) : 0.0f;
        }
        std::vector<std::pair<float, std::uint32_t>> all(kN);
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const float* q = qfloats.data() + qi * kDim;
            for (std::size_t i = 0; i < kN; ++i) {
                const float* x = floats.data() + i * kDim;
                float dot = 0.0f;
                for (std::size_t d = 0; d < kDim; ++d) {
                    dot += q[d] * x[d];
                }
                all[i] = std::make_pair(-dot * inv_norm[i],
                                        static_cast<std::uint32_t>(i));
            }
            std::partial_sort(all.begin(), all.begin() + kK, all.end());
            for (std::size_t i = 0; i < kK; ++i) {
                f32_truth[qi * kK + i] = all[i].second;
            }
        }
    }

    std::printf("Exact binary scan baseline: %.2f ms/query (%.0f QPS).\n\n",
                1000.0 * brute_s / static_cast<double>(kQueries), brute_qps);

    std::printf("| ef | recall@10 (binary GT) | float GT, Hamming rank | float GT, asym re-rank | float GT, **exact re-rank** | lat Hamming | lat asym | lat exact |\n");
    std::printf("|---|---|---|---|---|---|---|---|\n");
    const std::uint32_t ef_sweep[] = {10u, 25u, 50u, 100u, 200u};
    std::vector<edgevector::SearchResult> results(kK);
    std::vector<edgevector::ScoredResult> scored(kK);
    std::uint32_t got_ids[kK];

    for (const std::uint32_t ef : ef_sweep) {
        std::size_t bin_hits = 0;
        std::size_t f32_plain_hits = 0;
        std::size_t f32_rerank_hits = 0;
        std::size_t f32_exact_hits = 0;

        t0 = Clock::now();
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const std::uint32_t found = graph.search(
                qbase + qi * qstride, kK, ef, results.data());
            for (std::uint32_t i = 0; i < found; ++i) {
                got_ids[i] = results[i].id;
            }
            bin_hits += overlap10(got_ids, found, &bin_truth[qi * kK]);
            f32_plain_hits += overlap10(got_ids, found, &f32_truth[qi * kK]);
        }
        const double plain_s = seconds_since(t0);

        t0 = Clock::now();
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const std::uint32_t found = graph.search_reranked(
                qbase + qi * qstride, qfloats.data() + qi * kDim, kK, ef,
                scored.data());
            for (std::uint32_t i = 0; i < found; ++i) {
                got_ids[i] = scored[i].id;
            }
            f32_rerank_hits += overlap10(got_ids, found, &f32_truth[qi * kK]);
        }
        const double rerank_s = seconds_since(t0);

        t0 = Clock::now();
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const std::uint32_t found = graph.search_exact_reranked(
                qbase + qi * qstride, qfloats.data() + qi * kDim,
                floats.data(), kDim, kK, ef, scored.data());
            for (std::uint32_t i = 0; i < found; ++i) {
                got_ids[i] = scored[i].id;
            }
            f32_exact_hits += overlap10(got_ids, found, &f32_truth[qi * kK]);
        }
        const double exact_s = seconds_since(t0);

        const double denom = static_cast<double>(kQueries * kK);
        std::printf("| %u | %.3f | %.3f | %.3f | **%.3f** | %.0f us | %.0f us | %.0f us |\n",
                    ef,
                    static_cast<double>(bin_hits) / denom,
                    static_cast<double>(f32_plain_hits) / denom,
                    static_cast<double>(f32_rerank_hits) / denom,
                    static_cast<double>(f32_exact_hits) / denom,
                    1.0e6 * plain_s / static_cast<double>(kQueries),
                    1.0e6 * rerank_s / static_cast<double>(kQueries),
                    1.0e6 * exact_s / static_cast<double>(kQueries));
        std::fflush(stdout); // survive being killed mid-run when redirected
    }
    std::printf("\n");
    std::fflush(stdout);

    // Filtered search across selectivities (allow every stride-th id).
    // Recall is measured against the exact brute-force top-10 over the
    // allowed set; sparse filters take the adaptive exact-scan path.
    if (full_report) {
        std::printf("Filtered search (ef = 100, k = 10, 300 queries; recall "
                    "vs exact filtered ground truth):\n\n");
        std::printf("| selectivity | allowed ids | recall@10 | mean latency | QPS (1 thread) |\n");
        std::printf("|---|---|---|---|---|\n");
        const std::size_t strides[] = {1000u, 100u, 10u}; // 0.1%, 1%, 10%
        const std::size_t nq = 300;
        edgevector::SearchContext fctx = graph.make_context();
        for (const std::size_t stride : strides) {
            std::vector<std::uint64_t> af((kN + 63u) / 64u, 0u);
            std::vector<std::uint32_t> allowed;
            for (std::size_t id = 0; id < kN; id += stride) {
                af[id >> 6u] |= (1ull << (id & 63u));
                allowed.push_back(static_cast<std::uint32_t>(id));
            }

            std::size_t fhits = 0;
            std::vector<std::pair<std::uint32_t, std::uint32_t>> brute;
            brute.reserve(allowed.size());
            double total_s = 0.0;
            for (std::size_t qi = 0; qi < nq; ++qi) {
                const std::uint8_t* q = qbase + qi * qstride;
                t0 = Clock::now();
                const std::uint32_t found = graph.search(
                    fctx, q, kK, 100u, results.data(), af.data());
                total_s += seconds_since(t0);

                brute.clear();
                for (const std::uint32_t id : allowed) {
                    brute.emplace_back(
                        edgevector::hamming_distance(q, data.record(id), kDim),
                        id);
                }
                std::partial_sort(brute.begin(),
                                  brute.begin() +
                                      static_cast<std::ptrdiff_t>(kK),
                                  brute.end());
                for (std::uint32_t i = 0; i < found; ++i) {
                    got_ids[i] = results[i].id;
                }
                std::uint32_t truth10[kK];
                for (std::size_t t = 0; t < kK; ++t) {
                    truth10[t] = brute[t].second;
                }
                fhits += overlap10(got_ids, found, truth10);
            }
            std::printf("| %.1f%% | %zu | %.3f | %.0f us | %.0f |\n",
                        100.0 / static_cast<double>(stride), allowed.size(),
                        static_cast<double>(fhits) /
                            static_cast<double>(nq * kK),
                        1.0e6 * total_s / static_cast<double>(nq),
                        static_cast<double>(nq) / total_s);
            std::fflush(stdout);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// ITQ scenario: anisotropic clustered data (dimension d scaled by
// exp(-4d/dim), the decaying-spectrum shape real embedding data has), with
// two indexes over the SAME floats — raw sign codes vs ITQ-rotated codes —
// compared on recall against float32 cosine ground truth. N is 50k here (the
// scenario builds two graphs and trains a rotation).
//
// Note the exact-re-rank columns for the ITQ variant deliberately re-rank
// with the ORIGINAL floats and the ORIGINAL query: an orthogonal rotation
// preserves cosine exactly, so the rotated index needs no rotated float
// corpus.
// ---------------------------------------------------------------------------
void run_itq_scenario() {
    const std::size_t n = 50000;
    const std::size_t n_queries = 500;
    const std::size_t n_train = 5000; // subsample for rotation training
    std::printf("## Scenario: ITQ rotation on anisotropic data "
                "(N = %zu, %zu queries, spectrum exp(-4d/%zu))\n\n",
                n, n_queries, kDim);
    std::fflush(stdout);

    // Anisotropic clustered floats (data rows, then query rows).
    std::vector<float> floats((n + n_queries) * kDim);
    {
        Sampler sampler(true);
        std::vector<float> scale(kDim);
        for (std::size_t d = 0; d < kDim; ++d) {
            scale[d] = std::exp(-4.0f * static_cast<float>(d) /
                                static_cast<float>(kDim));
        }
        for (std::size_t i = 0; i < n + n_queries; ++i) {
            sampler.draw(floats.data() + i * kDim);
            for (std::size_t d = 0; d < kDim; ++d) {
                floats[i * kDim + d] *= scale[d];
            }
        }
    }
    const float* qfloats = floats.data() + n * kDim;

    // Train the rotation on a corpus subsample.
    edgevector::ItqRotation rot(kDim);
    Clock::time_point t0 = Clock::now();
    if (rot.train(floats.data(), n_train, kDim, /*iterations=*/10,
                  /*seed=*/42u) != edgevector::ItqStatus::ok) {
        std::printf("FATAL: ITQ training failed\n");
        std::exit(1);
    }
    std::printf("Rotation trained on %zu samples in %.1f s "
                "(quantization error %.4f -> %.4f, orthogonality %.1e).\n\n",
                n_train, seconds_since(t0), rot.error_before_training(),
                rot.error_after_training(), rot.orthogonality_error());
    std::fflush(stdout);

    // Two code blocks over the same floats.
    RecordBlock raw_codes(kDim, n);
    RecordBlock itq_codes(kDim, n);
    {
        std::vector<float> scratch(kDim);
        for (std::size_t i = 0; i < n; ++i) {
            edgevector::quantize(floats.data() + i * kDim, kDim,
                                 raw_codes.record(i));
            rot.rotate_quantize(floats.data() + i * kDim, scratch.data(),
                                itq_codes.record(i));
        }
    }

    edgevector::HNSWGraph raw_graph(raw_codes.base(), raw_codes.record_bytes(),
                                    kDim, static_cast<std::uint32_t>(n),
                                    16u, 200u, 256u, 42u);
    edgevector::HNSWGraph itq_graph(itq_codes.base(), itq_codes.record_bytes(),
                                    kDim, static_cast<std::uint32_t>(n),
                                    16u, 200u, 256u, 42u);
    t0 = Clock::now();
    for (std::uint32_t i = 0; i < n; ++i) {
        raw_graph.insert(i);
        itq_graph.insert(i);
    }
    std::printf("Both graphs built in %.1f s total (1 thread).\n\n",
                seconds_since(t0));
    std::fflush(stdout);

    // Float32 cosine ground truth (identical for both variants: orthogonal
    // rotations preserve cosine).
    std::vector<std::uint32_t> f32_truth(n_queries * kK);
    {
        std::vector<float> inv_norm(n);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            const float* x = floats.data() + i * kDim;
            for (std::size_t d = 0; d < kDim; ++d) {
                s += static_cast<double>(x[d]) * static_cast<double>(x[d]);
            }
            inv_norm[i] =
                (s > 0.0) ? static_cast<float>(1.0 / std::sqrt(s)) : 0.0f;
        }
        std::vector<std::pair<float, std::uint32_t>> all(n);
        for (std::size_t qi = 0; qi < n_queries; ++qi) {
            const float* q = qfloats + qi * kDim;
            for (std::size_t i = 0; i < n; ++i) {
                const float* x = floats.data() + i * kDim;
                float dot = 0.0f;
                for (std::size_t d = 0; d < kDim; ++d) {
                    dot += q[d] * x[d];
                }
                all[i] = std::make_pair(-dot * inv_norm[i],
                                        static_cast<std::uint32_t>(i));
            }
            std::partial_sort(all.begin(), all.begin() + kK, all.end());
            for (std::size_t i = 0; i < kK; ++i) {
                f32_truth[qi * kK + i] = all[i].second;
            }
        }
    }

    std::printf("| ef | raw Hamming | ITQ Hamming | raw asym | ITQ asym | raw exact | ITQ exact |\n");
    std::printf("|---|---|---|---|---|---|---|\n");
    std::fflush(stdout);

    std::vector<std::uint64_t> qw(edgevector::padded_bytes(kDim) / 8u, 0u);
    std::uint8_t* qbits = reinterpret_cast<std::uint8_t*>(qw.data());
    std::vector<float> qrot(kDim);
    std::vector<edgevector::SearchResult> hres(kK);
    std::vector<edgevector::ScoredResult> sres(kK);
    std::uint32_t got_ids[kK];

    const std::uint32_t ef_sweep[] = {25u, 100u};
    for (const std::uint32_t ef : ef_sweep) {
        std::size_t hits[6] = {0, 0, 0, 0, 0, 0};
        for (std::size_t qi = 0; qi < n_queries; ++qi) {
            const float* q = qfloats + qi * kDim;
            const std::uint32_t* truth = &f32_truth[qi * kK];

            // Variant 0: raw codes, quantized query.
            edgevector::quantize(q, kDim, qbits);
            std::uint32_t found = raw_graph.search(qbits, kK, ef, hres.data());
            for (std::uint32_t i = 0; i < found; ++i) got_ids[i] = hres[i].id;
            hits[0] += overlap10(got_ids, found, truth);
            found = raw_graph.search_reranked(qbits, q, kK, ef, sres.data());
            for (std::uint32_t i = 0; i < found; ++i) got_ids[i] = sres[i].id;
            hits[2] += overlap10(got_ids, found, truth);
            found = raw_graph.search_exact_reranked(qbits, q, floats.data(),
                                                    kDim, kK, ef, sres.data());
            for (std::uint32_t i = 0; i < found; ++i) got_ids[i] = sres[i].id;
            hits[4] += overlap10(got_ids, found, truth);

            // Variant 1: ITQ codes, rotated query (exact re-rank still uses
            // the ORIGINAL floats and query - cosine is preserved).
            rot.rotate(q, qrot.data());
            edgevector::quantize(qrot.data(), kDim, qbits);
            found = itq_graph.search(qbits, kK, ef, hres.data());
            for (std::uint32_t i = 0; i < found; ++i) got_ids[i] = hres[i].id;
            hits[1] += overlap10(got_ids, found, truth);
            found = itq_graph.search_reranked(qbits, qrot.data(), kK, ef,
                                              sres.data());
            for (std::uint32_t i = 0; i < found; ++i) got_ids[i] = sres[i].id;
            hits[3] += overlap10(got_ids, found, truth);
            found = itq_graph.search_exact_reranked(qbits, q, floats.data(),
                                                    kDim, kK, ef, sres.data());
            for (std::uint32_t i = 0; i < found; ++i) got_ids[i] = sres[i].id;
            hits[5] += overlap10(got_ids, found, truth);
        }
        const double denom = static_cast<double>(n_queries * kK);
        std::printf("| %u | %.3f | **%.3f** | %.3f | **%.3f** | %.3f | **%.3f** |\n",
                    ef,
                    static_cast<double>(hits[0]) / denom,
                    static_cast<double>(hits[1]) / denom,
                    static_cast<double>(hits[2]) / denom,
                    static_cast<double>(hits[3]) / denom,
                    static_cast<double>(hits[4]) / denom,
                    static_cast<double>(hits[5]) / denom);
        std::fflush(stdout);
    }
    std::printf("\n");
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 1M-vector scale scenario: clustered data, multi-threaded build, and the
// recall/latency sweep against exact binary ground truth. Float32 ground
// truth is omitted at this scale (it would need 2 GB of float residents);
// the 100k scenario above carries the float-truth accuracy story, this one
// carries the scale story.
// ---------------------------------------------------------------------------
void run_million_scenario() {
    const std::size_t n = 1000000;
    const std::size_t n_queries = 1000;
    std::printf("## Scenario: 1M vectors (clustered, %zu queries)\n\n",
                n_queries);
    std::fflush(stdout);

    Sampler sampler(true);
    std::vector<float> raw(kDim);
    RecordBlock data(kDim, n);
    Clock::time_point t0 = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        sampler.draw(raw.data());
        edgevector::quantize(raw.data(), kDim, data.record(i));
    }
    std::printf("Dataset generated and quantized in %.1f s.\n\n",
                seconds_since(t0));
    std::fflush(stdout);

    edgevector::HNSWGraph graph(data.base(), data.record_bytes(), kDim,
                                static_cast<std::uint32_t>(n),
                                /*m=*/16u, /*ef_construction=*/200u,
                                /*max_ef_search=*/256u, /*seed=*/42u);
    unsigned hw_threads = 1u;
#if defined(EDGEVECTOR_HAS_THREADS)
    hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0u) {
        hw_threads = 1u;
    }
#endif
    {
        std::vector<std::uint32_t> ids(n);
        for (std::size_t i = 0; i < n; ++i) {
            ids[i] = static_cast<std::uint32_t>(i);
        }
        t0 = Clock::now();
        const std::size_t inserted =
            graph.insert_batch(ids.data(), n, hw_threads);
        const double build_s = seconds_since(t0);
        const bool ok = (inserted == n) && graph.validate_integrity();
        std::printf("| Stage | Value |\n|---|---|\n");
        std::printf("| Build, insert_batch (%u threads) | %.1f s (integrity %s) |\n",
                    hw_threads, build_s, ok ? "validated" : "FAILED");
    }

    const char* const kGraphFile = "ev_bench_graph_1m.evhg";
    t0 = Clock::now();
    if (graph.save_graph(kGraphFile) != edgevector::GraphIoStatus::ok) {
        std::printf("FATAL: save_graph failed\n");
        std::exit(1);
    }
    const double save_s = seconds_since(t0);
    edgevector::HNSWGraph loaded(data.base(), data.record_bytes(), kDim,
                                 static_cast<std::uint32_t>(n),
                                 16u, 200u, 256u, 42u);
    t0 = Clock::now();
    if (loaded.load_graph(kGraphFile) != edgevector::GraphIoStatus::ok) {
        std::printf("FATAL: load_graph failed\n");
        std::exit(1);
    }
    const double load_s = seconds_since(t0);
    std::remove(kGraphFile);
    std::printf("| Save / load graph | %.2f s / %.2f s |\n", save_s, load_s);
    std::printf("| Quantized vectors (RAM) | %.1f MB (float32 equivalent: %.1f MB) |\n",
                static_cast<double>(n * edgevector::padded_bytes(kDim)) / 1.0e6,
                static_cast<double>(n * kDim * sizeof(float)) / 1.0e6);
    std::printf("| Graph (links + scratch) | %.1f MB |\n\n",
                static_cast<double>(loaded.graph_memory_bytes()) / 1.0e6);
    std::fflush(stdout);

    // Queries + exact binary ground truth.
    std::vector<std::uint64_t> qwords(
        (edgevector::padded_bytes(kDim) / 8u) * n_queries, 0u);
    std::uint8_t* qbase = reinterpret_cast<std::uint8_t*>(qwords.data());
    const std::size_t qstride = edgevector::padded_bytes(kDim);
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        sampler.draw(raw.data());
        edgevector::quantize(raw.data(), kDim, qbase + qi * qstride);
    }
    std::vector<std::uint32_t> truth(n_queries * kK);
    t0 = Clock::now();
    {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> all(n);
        for (std::size_t qi = 0; qi < n_queries; ++qi) {
            const std::uint8_t* q = qbase + qi * qstride;
            for (std::size_t i = 0; i < n; ++i) {
                all[i] = std::make_pair(
                    edgevector::hamming_distance(q, data.record(i), kDim),
                    static_cast<std::uint32_t>(i));
            }
            std::partial_sort(all.begin(), all.begin() + kK, all.end());
            for (std::size_t i = 0; i < kK; ++i) {
                truth[qi * kK + i] = all[i].second;
            }
        }
    }
    const double brute_s = seconds_since(t0);
    std::printf("Exact binary scan baseline: %.2f ms/query (%.0f QPS).\n\n",
                1000.0 * brute_s / static_cast<double>(n_queries),
                static_cast<double>(n_queries) / brute_s);

    std::printf("| ef | recall@10 (binary GT) | mean latency | QPS (1 thread) |\n");
    std::printf("|---|---|---|---|\n");
    const std::uint32_t ef_sweep[] = {10u, 50u, 100u};
    std::vector<edgevector::SearchResult> results(kK);
    std::uint32_t got_ids[kK];
    for (const std::uint32_t ef : ef_sweep) {
        std::size_t hits = 0;
        t0 = Clock::now();
        for (std::size_t qi = 0; qi < n_queries; ++qi) {
            const std::uint32_t found = loaded.search(
                qbase + qi * qstride, kK, ef, results.data());
            for (std::uint32_t i = 0; i < found; ++i) {
                got_ids[i] = results[i].id;
            }
            hits += overlap10(got_ids, found, &truth[qi * kK]);
        }
        const double sweep_s = seconds_since(t0);
        std::printf("| %u | %.3f | %.0f us | %.0f |\n",
                    ef,
                    static_cast<double>(hits) /
                        static_cast<double>(n_queries * kK),
                    1.0e6 * sweep_s / static_cast<double>(n_queries),
                    static_cast<double>(n_queries) / sweep_s);
        std::fflush(stdout);
    }
    std::printf("\n");
    std::fflush(stdout);
}

} // namespace

// Optional argv[1]: "clustered", "random", "itq", or "million" runs just
// that scenario, so each fits inside a CI/automation time budget; no
// argument runs all.
int main(int argc, char** argv) {
    const bool want_clustered =
        (argc < 2) || (std::strcmp(argv[1], "clustered") == 0);
    const bool want_random =
        (argc < 2) || (std::strcmp(argv[1], "random") == 0);
    const bool want_itq =
        (argc < 2) || (std::strcmp(argv[1], "itq") == 0);
    const bool want_million =
        (argc < 2) || (std::strcmp(argv[1], "million") == 0);

    std::printf("# EdgeVector benchmark\n\n");
    std::printf("N = %zu vectors, dim = %zu (%zu quantized bytes each), "
                "M = 16, ef_construction = 200, k = %u, %zu queries, "
                "single thread.\n\n",
                kN, kDim, edgevector::padded_bytes(kDim), kK, kQueries);
    std::fflush(stdout);

    if (want_clustered) {
        run_scenario("clustered (embedding-like, 1000 clusters)", true, true);
    }
    if (want_random) {
        run_scenario("iid random (adversarial worst case for any ANN index)",
                     false, false);
    }
    if (want_itq) {
        run_itq_scenario();
    }
    if (want_million) {
        run_million_scenario();
    }

    std::printf("Done.\n");
    return 0;
}
