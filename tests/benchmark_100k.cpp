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
// ============================================================================

#include "edgevector/hnsw_graph.hpp"
#include "edgevector/quantize_math.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
          count_(count),
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
    std::size_t count_;
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

void run_scenario(const char* name, bool clustered, bool full_report) {
    std::printf("## Scenario: %s\n\n", name);

    Sampler sampler(clustered);
    std::vector<float> raw(kDim);

    RecordBlock data(kDim, kN);
    for (std::size_t i = 0; i < kN; ++i) {
        sampler.draw(raw.data());
        edgevector::quantize(raw.data(), kDim, data.record(i));
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

        std::printf("| Stage | Time |\n|---|---|\n");
        std::printf("| Build (insert %zu vectors, 1 thread) | %.1f s |\n",
                    kN, build_s);
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

    // Queries (same distribution as the data) and exact ground truth.
    std::vector<std::uint64_t> qwords(
        (edgevector::padded_bytes(kDim) / 8u) * kQueries, 0u);
    std::uint8_t* qbase = reinterpret_cast<std::uint8_t*>(qwords.data());
    const std::size_t qstride = edgevector::padded_bytes(kDim);
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        sampler.draw(raw.data());
        edgevector::quantize(raw.data(), kDim, qbase + qi * qstride);
    }

    std::vector<std::uint32_t> truth(kQueries * kK);
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
                truth[qi * kK + i] = all[i].second;
            }
        }
    }
    const double brute_s = seconds_since(t0);
    const double brute_qps = static_cast<double>(kQueries) / brute_s;
    std::printf("Exact brute-force baseline: %.2f ms/query (%.0f QPS).\n\n",
                1000.0 * brute_s / static_cast<double>(kQueries), brute_qps);

    std::printf("| ef | recall@10 | mean latency | QPS (1 thread) | vs exact |\n");
    std::printf("|---|---|---|---|---|\n");
    const std::uint32_t ef_sweep[] = {10u, 25u, 50u, 100u, 200u};
    std::vector<edgevector::SearchResult> results(kK);
    for (const std::uint32_t ef : ef_sweep) {
        std::size_t hits = 0;
        t0 = Clock::now();
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const std::uint32_t found = graph.search(
                qbase + qi * qstride, kK, ef, results.data());
            for (std::uint32_t i = 0; i < found; ++i) {
                for (std::size_t t = 0; t < kK; ++t) {
                    if (results[i].id == truth[qi * kK + t]) {
                        ++hits;
                        break;
                    }
                }
            }
        }
        const double sweep_s = seconds_since(t0);
        const double recall = static_cast<double>(hits) /
                              static_cast<double>(kQueries * kK);
        const double qps = static_cast<double>(kQueries) / sweep_s;
        std::printf("| %u | %.3f | %.0f us | %.0f | %.1fx |\n",
                    ef, recall,
                    1.0e6 * sweep_s / static_cast<double>(kQueries), qps,
                    qps / brute_qps);
    }
    std::printf("\n");
}

} // namespace

int main() {
    std::printf("# EdgeVector benchmark\n\n");
    std::printf("N = %zu vectors, dim = %zu (%zu quantized bytes each), "
                "M = 16, ef_construction = 200, k = %u, %zu queries, "
                "single thread.\n\n",
                kN, kDim, edgevector::padded_bytes(kDim), kK, kQueries);

    run_scenario("clustered (embedding-like, 1000 clusters)", true, true);
    run_scenario("iid random (adversarial worst case for any ANN index)",
                 false, false);

    std::printf("Done.\n");
    return 0;
}
