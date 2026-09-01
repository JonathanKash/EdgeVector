// ============================================================================
// Isolated unit test for edgevector/hnsw_graph.hpp
//
// Plain main() + hand-rolled checks, no test framework. Returns non-zero if
// any case fails. Test scaffolding may allocate; the zero-allocation rule
// binds the library query path — and case 5 PROVES it holds by replacing the
// global operator new with a counting shim and asserting the counter does not
// move across searches.
// ============================================================================

#include "edgevector/hnsw_graph.hpp"
#include "edgevector/quantize_math.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <utility>
#include <vector>

#if defined(_GLIBCXX_HAS_GTHREADS)
#include <thread>
#endif

// ---------------------------------------------------------------------------
// Global allocation counter. Every path into the C++ free store goes through
// these replacements, so a zero delta across search() calls is hard evidence
// of a zero-allocation query path, not a code-reading claim.
// ---------------------------------------------------------------------------
static std::size_t g_new_calls = 0;

// The malloc/free pairing below is the canonical way to replace the global
// allocator; GCC 13+'s -Wmismatched-new-delete cannot see that these are the
// matched replacements and misfires, so it is silenced for the shim only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t n) {
    ++g_new_calls;
    void* p = std::malloc(n == 0u ? 1u : n);
    if (p == nullptr) {
        std::abort();
    }
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

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

// Contiguous, 8-byte-aligned block of quantized records (same pattern as the
// mmap test: uint64_t backing guarantees the alignment the library requires).
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
    std::size_t count() const noexcept { return count_; }

private:
    std::size_t record_bytes_;
    std::size_t count_;
    std::vector<std::uint64_t> words_;
};

void fill_random_block(RecordBlock& block, std::size_t dim, std::mt19937& rng,
                       std::vector<float>* keep_floats = nullptr) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(dim);
    if (keep_floats != nullptr) {
        keep_floats->assign(block.count() * dim, 0.0f);
    }
    for (std::size_t i = 0; i < block.count(); ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(rng);
            if (keep_floats != nullptr) {
                (*keep_floats)[i * dim + d] = raw[d];
            }
        }
        edgevector::quantize(raw.data(), dim, block.record(i));
    }
}

// Brute-force top-k by Hamming distance with the same (distance, id) tie
// order the graph uses, so recall comparisons are exact.
void brute_force_topk(const RecordBlock& data, std::size_t dim,
                      const std::uint8_t* query, std::size_t k,
                      std::vector<std::uint32_t>& out_ids) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> all; // (dist, id)
    all.reserve(data.count());
    for (std::size_t i = 0; i < data.count(); ++i) {
        all.emplace_back(
            edgevector::hamming_distance(query, data.record(i), dim),
            static_cast<std::uint32_t>(i));
    }
    std::sort(all.begin(), all.end());
    out_ids.clear();
    for (std::size_t i = 0; i < k && i < all.size(); ++i) {
        out_ids.push_back(all[i].second);
    }
}

double measure_recall(edgevector::HNSWGraph& graph, const RecordBlock& data,
                      std::size_t dim, std::size_t n_queries,
                      std::uint32_t k, std::uint32_t ef, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(dim);
    std::vector<std::uint64_t> qwords(edgevector::padded_bytes(dim) / 8u, 0u);
    std::uint8_t* qbytes = reinterpret_cast<std::uint8_t*>(qwords.data());
    std::vector<std::uint32_t> truth;
    std::vector<edgevector::SearchResult> results(k);

    std::size_t hits = 0;
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(rng);
        }
        edgevector::quantize(raw.data(), dim, qbytes);

        brute_force_topk(data, dim, qbytes, k, truth);
        const std::uint32_t found =
            graph.search(qbytes, k, ef, results.data());

        for (std::uint32_t i = 0; i < found; ++i) {
            for (std::size_t t = 0; t < truth.size(); ++t) {
                if (results[i].id == truth[t]) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) /
           static_cast<double>(n_queries * static_cast<std::size_t>(k));
}

// ---------------------------------------------------------------------------
// Case 1: edge behaviour (empty graph, single node, insert contract)
// ---------------------------------------------------------------------------
void test_edges() {
    std::printf("[1] Edge behaviour and insert contract\n");

    const std::size_t dim = 512;
    std::mt19937 rng(42);
    RecordBlock data(dim, 4);
    fill_random_block(data, dim, rng);

    edgevector::HNSWGraph graph(data.base(), data.record_bytes(), dim,
                                /*capacity=*/4u);

    edgevector::SearchResult out[8];
    check(graph.search(data.record(0), 4u, 16u, out) == 0u,
          "empty graph: search returns 0 results");

    check(graph.insert(0u), "insert(0) succeeds");
    check(!graph.insert(0u), "duplicate insert(0) is rejected");
    check(!graph.insert(4u), "out-of-range insert(4) is rejected");
    check(graph.size() == 1u, "size() == 1 after one insert");

    std::uint32_t found = graph.search(data.record(0), 4u, 16u, out);
    check(found == 1u, "single-node graph: k=4 returns exactly 1 result");
    check(found == 1u && out[0].id == 0u && out[0].distance == 0u,
          "single-node graph: the node finds itself at distance 0");

    check(graph.insert(1u) && graph.insert(2u) && graph.insert(3u),
          "remaining inserts succeed");
    found = graph.search(data.record(2), 10u, 16u, out);
    check(found == 4u, "k > size clamps to size (4 results)");
    check(found == 4u && out[0].id == 2u && out[0].distance == 0u,
          "exact query vector ranks itself first at distance 0");
}

// ---------------------------------------------------------------------------
// Case 2: near-exhaustive recall on a small set
// ---------------------------------------------------------------------------
void test_small_recall() {
    std::printf("[2] Small-set recall (N = 200, ef = 200)\n");

    const std::size_t dim = 512;
    const std::size_t n = 200;
    std::mt19937 rng(42);
    RecordBlock data(dim, n);
    fill_random_block(data, dim, rng);

    edgevector::HNSWGraph graph(data.base(), data.record_bytes(), dim,
                                static_cast<std::uint32_t>(n));
    for (std::uint32_t i = 0; i < n; ++i) {
        graph.insert(i);
    }

    const double recall = measure_recall(graph, data, dim, 20u, 10u, 200u, rng);
    std::printf("      recall@10 = %.4f  (gate >= 0.95)\n", recall);
    check(recall >= 0.95, "recall@10 >= 0.95 with a near-exhaustive beam");
}

// ---------------------------------------------------------------------------
// Case 3: recall at scale (N = 2000, ef = 100) — the graph must find what
// brute force finds without visiting everything.
// ---------------------------------------------------------------------------
void test_scale_recall(edgevector::HNSWGraph& graph, const RecordBlock& data,
                       std::size_t dim) {
    std::printf("[3] Recall at scale (N = 2000, 50 queries, ef = 100)\n");

    std::mt19937 rng(7);
    const double recall = measure_recall(graph, data, dim, 50u, 10u, 100u, rng);
    std::printf("      recall@10 = %.4f  (gate >= 0.90)\n", recall);
    check(recall >= 0.90, "recall@10 >= 0.90 at ef = 100");
}

// ---------------------------------------------------------------------------
// Case 4: results are sorted ascending by (distance, id)
// ---------------------------------------------------------------------------
void test_sorted_output(edgevector::HNSWGraph& graph, const RecordBlock& data,
                        std::size_t dim) {
    std::printf("[4] Result ordering\n");

    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(dim);
    std::vector<std::uint64_t> qwords(edgevector::padded_bytes(dim) / 8u, 0u);
    std::uint8_t* qbytes = reinterpret_cast<std::uint8_t*>(qwords.data());
    std::vector<edgevector::SearchResult> results(50);

    bool all_sorted = true;
    for (int t = 0; t < 10; ++t) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(rng);
        }
        edgevector::quantize(raw.data(), dim, qbytes);
        const std::uint32_t found = graph.search(qbytes, 50u, 100u, results.data());
        for (std::uint32_t i = 1; i < found; ++i) {
            const bool ok =
                (results[i - 1u].distance < results[i].distance) ||
                (results[i - 1u].distance == results[i].distance &&
                 results[i - 1u].id < results[i].id);
            if (!ok) {
                all_sorted = false;
            }
        }
    }
    check(all_sorted, "10 searches x 50 results ascending by (distance, id)");
    (void)data;
}

// ---------------------------------------------------------------------------
// Case 5: the zero-allocation proof — legacy search, context search, and
// asymmetric re-ranked search all count zero allocations.
// ---------------------------------------------------------------------------
void test_zero_allocation(edgevector::HNSWGraph& graph, const RecordBlock& data,
                          std::size_t dim) {
    std::printf("[5] Zero-allocation query path (instrumented operator new)\n");

    std::mt19937 rng(13);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(dim);
    std::vector<std::uint64_t> qwords(edgevector::padded_bytes(dim) / 8u, 0u);
    std::uint8_t* qbytes = reinterpret_cast<std::uint8_t*>(qwords.data());
    edgevector::SearchResult results[10];
    edgevector::ScoredResult scored[10];

    // Context creation is allowed to allocate; do it before the snapshot.
    edgevector::SearchContext ctx = graph.make_context();

    // Warm-up search, then snapshot the counter.
    edgevector::quantize(raw.data(), dim, qbytes);
    graph.search(qbytes, 10u, 100u, results);

    const std::size_t before = g_new_calls;
    std::uint32_t total_found = 0u;
    for (int t = 0; t < 100; ++t) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(rng);
        }
        edgevector::quantize(raw.data(), dim, qbytes);
        total_found += graph.search(qbytes, 10u, 100u, results);
        total_found += graph.search(ctx, qbytes, 10u, 100u, results, nullptr);
        total_found += graph.search_reranked(ctx, qbytes, raw.data(), 10u,
                                             100u, scored, nullptr);
    }
    const std::size_t delta = g_new_calls - before;

    std::printf("      operator new calls across 300 searches "
                "(plain + context + reranked): %zu  (gate == 0)\n", delta);
    check(delta == 0u, "zero heap allocations across all three search modes");
    check(total_found == 3000u, "every search returned a full k = 10");
    (void)data;
}

// ---------------------------------------------------------------------------
// Case 7: SearchContext isolation and (where std::thread exists) genuine
// concurrent queries — every context must reproduce the default context's
// results exactly.
// ---------------------------------------------------------------------------
void test_contexts_and_threads(const edgevector::HNSWGraph& graph,
                               edgevector::HNSWGraph& mutable_graph,
                               std::size_t dim) {
    std::printf("[7] Search contexts and concurrency\n");

    const std::size_t n_queries = 100;
    std::mt19937 rng(23);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(dim);

    const std::size_t qstride = edgevector::padded_bytes(dim);
    std::vector<std::uint64_t> qwords((qstride / 8u) * n_queries, 0u);
    std::uint8_t* qbase = reinterpret_cast<std::uint8_t*>(qwords.data());
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(rng);
        }
        edgevector::quantize(raw.data(), dim, qbase + qi * qstride);
    }

    // Reference answers through the legacy (default-context) API.
    std::vector<edgevector::SearchResult> expected(n_queries * 10u);
    std::vector<std::uint32_t> expected_found(n_queries);
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        expected_found[qi] = mutable_graph.search(
            qbase + qi * qstride, 10u, 100u, expected.data() + qi * 10u);
    }

    // A separate context on a const graph must reproduce them exactly.
    edgevector::SearchContext ctx = graph.make_context();
    bool identical = true;
    edgevector::SearchResult got[10];
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        const std::uint32_t found =
            graph.search(ctx, qbase + qi * qstride, 10u, 100u, got, nullptr);
        if (found != expected_found[qi]) {
            identical = false;
            continue;
        }
        for (std::uint32_t i = 0; i < found; ++i) {
            if (got[i].id != expected[qi * 10u + i].id ||
                got[i].distance != expected[qi * 10u + i].distance) {
                identical = false;
            }
        }
    }
    check(identical, "independent context reproduces default-context results");

#if defined(_GLIBCXX_HAS_GTHREADS)
    {
        const int n_threads = 4;
        std::vector<std::thread> threads;
        std::vector<int> bad(n_threads, 0);
        for (int t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, t]() {
                edgevector::SearchContext tctx = graph.make_context();
                edgevector::SearchResult tres[10];
                for (std::size_t qi = static_cast<std::size_t>(t);
                     qi < n_queries; qi += n_threads) {
                    const std::uint32_t found = graph.search(
                        tctx, qbase + qi * qstride, 10u, 100u, tres, nullptr);
                    if (found != expected_found[qi]) {
                        bad[t] = 1;
                        continue;
                    }
                    for (std::uint32_t i = 0; i < found; ++i) {
                        if (tres[i].id != expected[qi * 10u + i].id) {
                            bad[t] = 1;
                        }
                    }
                }
            });
        }
        int any_bad = 0;
        for (auto& th : threads) {
            th.join();
        }
        for (int b : bad) {
            any_bad += b;
        }
        check(any_bad == 0,
              "4 threads with private contexts reproduce serial results");
    }
#else
    std::printf("      (std::thread unavailable on this toolchain; "
                "context isolation verified sequentially)\n");
#endif
}

// ---------------------------------------------------------------------------
// Case 8: soft deletion, restore, and allow-bitmap filtering
// ---------------------------------------------------------------------------
void test_delete_and_filter(std::size_t dim) {
    std::printf("[8] Soft delete and filtered search\n");

    const std::size_t n = 300;
    std::mt19937 rng(29);
    RecordBlock data(dim, n);
    fill_random_block(data, dim, rng);

    edgevector::HNSWGraph graph(data.base(), data.record_bytes(), dim,
                                static_cast<std::uint32_t>(n));
    for (std::uint32_t i = 0; i < n; ++i) {
        graph.insert(i);
    }

    edgevector::SearchResult res[20];
    const std::uint8_t* q = data.record(7); // query an indexed vector

    std::uint32_t found = graph.search(q, 10u, 100u, res);
    check(found == 10u && res[0].id == 7u && res[0].distance == 0u,
          "before deletion: the vector finds itself first");

    check(graph.remove(7u), "remove(7) succeeds");
    check(!graph.remove(7u), "double remove(7) is rejected");
    check(graph.is_deleted(7u), "is_deleted(7) == true");
    check(graph.live_size() == n - 1u, "live_size() dropped by one");

    found = graph.search(q, 10u, 100u, res);
    bool absent = true;
    for (std::uint32_t i = 0; i < found; ++i) {
        if (res[i].id == 7u) {
            absent = false;
        }
    }
    check(found == 10u && absent,
          "deleted id never returned; k still filled from live nodes");

    check(graph.restore(7u), "restore(7) succeeds");
    found = graph.search(q, 10u, 100u, res);
    check(found == 10u && res[0].id == 7u,
          "restored id is returned again, ranked first");

    // Allow-bitmap: permit only ids < 100.
    std::vector<std::uint64_t> allow((n + 63u) / 64u, 0u);
    for (std::uint32_t id = 0; id < 100u; ++id) {
        allow[id >> 6u] |= (1ull << (id & 63u));
    }
    edgevector::SearchContext ctx = graph.make_context();
    found = graph.search(ctx, q, 10u, 100u, res, allow.data());
    bool all_allowed = (found == 10u);
    for (std::uint32_t i = 0; i < found; ++i) {
        if (res[i].id >= 100u) {
            all_allowed = false;
        }
    }
    check(all_allowed, "allow-bitmap search returns only permitted ids");

    // Filter + delete compose.
    graph.remove(res[0].id);
    const std::uint32_t excluded = res[0].id;
    found = graph.search(ctx, q, 10u, 100u, res, allow.data());
    bool composed = (found == 10u);
    for (std::uint32_t i = 0; i < found; ++i) {
        if (res[i].id >= 100u || res[i].id == excluded) {
            composed = false;
        }
    }
    check(composed, "deletion and allow-bitmap compose correctly");
}

// ---------------------------------------------------------------------------
// Case 9: asymmetric re-ranking must beat plain Hamming ranking against the
// TRUE (float32 cosine) ground truth — the end-to-end accuracy that matters.
// ---------------------------------------------------------------------------
void test_reranked_accuracy(edgevector::HNSWGraph& graph,
                            const std::vector<float>& floats,
                            std::size_t n, std::size_t dim) {
    std::printf("[9] Asymmetric re-ranking vs float32 ground truth\n");

    const std::size_t n_queries = 30;
    const std::uint32_t k = 10u;
    std::mt19937 rng(31);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(dim);
    std::vector<std::uint64_t> qwords(edgevector::padded_bytes(dim) / 8u, 0u);
    std::uint8_t* qbytes = reinterpret_cast<std::uint8_t*>(qwords.data());

    edgevector::SearchResult plain[10];
    edgevector::ScoredResult reranked[10];
    edgevector::ScoredResult exact[10];
    std::vector<std::pair<float, std::uint32_t>> gt(n);

    std::size_t plain_hits = 0;
    std::size_t rerank_hits = 0;
    std::size_t exact_hits = 0;
    bool ordered = true;
    std::size_t allocs_during_exact = 0;

    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(rng);
        }
        edgevector::quantize(raw.data(), dim, qbytes);

        // Float32 cosine ground truth (descending similarity, id ascending).
        for (std::size_t i = 0; i < n; ++i) {
            gt[i] = std::make_pair(
                -edgevector::cosine_similarity_f32(
                    raw.data(), floats.data() + i * dim, dim),
                static_cast<std::uint32_t>(i));
        }
        std::partial_sort(gt.begin(), gt.begin() + k, gt.end());

        const std::uint32_t pf = graph.search(qbytes, k, 100u, plain);
        const std::uint32_t rf =
            graph.search_reranked(qbytes, raw.data(), k, 100u, reranked);
        const std::size_t before = g_new_calls;
        const std::uint32_t xf = graph.search_exact_reranked(
            qbytes, raw.data(), floats.data(), dim, k, 100u, exact);
        allocs_during_exact += g_new_calls - before;

        for (std::uint32_t i = 1; i < rf; ++i) {
            if (reranked[i - 1u].score < reranked[i].score) {
                ordered = false;
            }
        }
        for (std::uint32_t i = 0; i < pf; ++i) {
            for (std::size_t t = 0; t < k; ++t) {
                if (plain[i].id == gt[t].second) {
                    ++plain_hits;
                    break;
                }
            }
        }
        for (std::uint32_t i = 0; i < rf; ++i) {
            for (std::size_t t = 0; t < k; ++t) {
                if (reranked[i].id == gt[t].second) {
                    ++rerank_hits;
                    break;
                }
            }
        }
        for (std::uint32_t i = 0; i < xf; ++i) {
            for (std::size_t t = 0; t < k; ++t) {
                if (exact[i].id == gt[t].second) {
                    ++exact_hits;
                    break;
                }
            }
        }
    }

    const double denom = static_cast<double>(n_queries * k);
    const double plain_recall = static_cast<double>(plain_hits) / denom;
    const double rerank_recall = static_cast<double>(rerank_hits) / denom;
    const double exact_recall = static_cast<double>(exact_hits) / denom;
    std::printf("      recall@10 vs float32 truth: hamming ranking %.3f, "
                "asymmetric re-rank %.3f, exact re-rank %.3f\n",
                plain_recall, rerank_recall, exact_recall);
    check(ordered, "re-ranked results are ordered by descending score");
    check(rerank_recall >= plain_recall,
          "asymmetric re-ranking never loses to plain Hamming ranking");
    check(rerank_recall > plain_recall + 0.05,
          "asymmetric re-ranking beats plain Hamming by more than 5 points");
    check(exact_recall >= rerank_recall,
          "exact re-ranking never loses to asymmetric re-ranking");
    check(allocs_during_exact == 0u,
          "exact re-ranking allocates nothing");
}

// ---------------------------------------------------------------------------
// Case 6: graph persistence — a loaded graph must be indistinguishable from
// the one that was saved, and a damaged or mismatched file must be rejected
// leaving the target graph empty.
// ---------------------------------------------------------------------------
void test_persistence(edgevector::HNSWGraph& graph, const RecordBlock& data,
                      std::size_t dim) {
    std::printf("[6] Graph persistence (save/load round trip)\n");

    const char* const kGraphFile = "ev_test_graph.evhg";
    const char* const kBadFile = "ev_test_graph_bad.evhg";

    // Tombstones must survive the round trip: delete two ids before saving.
    check(graph.remove(11u) && graph.remove(23u),
          "two ids soft-deleted before save");
    check(graph.save_graph(kGraphFile) == edgevector::GraphIoStatus::ok,
          "save_graph returns ok");

    edgevector::HNSWGraph loaded(data.base(), data.record_bytes(), dim,
                                 graph.capacity(), /*m=*/16u,
                                 /*ef_construction=*/200u,
                                 /*max_ef_search=*/256u, /*seed=*/42u);
    check(loaded.load_graph(kGraphFile) == edgevector::GraphIoStatus::ok,
          "load_graph returns ok");
    check(loaded.size() == graph.size(), "loaded size matches");
    check(loaded.is_deleted(11u) && loaded.is_deleted(23u) &&
              loaded.deleted_count() == 2u,
          "tombstones round-trip through the graph file (format v2)");

    // Identical results — ids AND distances — for 20 fresh queries.
    std::mt19937 rng(17);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(dim);
    std::vector<std::uint64_t> qwords(edgevector::padded_bytes(dim) / 8u, 0u);
    std::uint8_t* qbytes = reinterpret_cast<std::uint8_t*>(qwords.data());
    edgevector::SearchResult a[10];
    edgevector::SearchResult b[10];

    bool identical = true;
    for (int t = 0; t < 20; ++t) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(rng);
        }
        edgevector::quantize(raw.data(), dim, qbytes);
        const std::uint32_t fa = graph.search(qbytes, 10u, 100u, a);
        const std::uint32_t fb = loaded.search(qbytes, 10u, 100u, b);
        if (fa != fb) {
            identical = false;
            continue;
        }
        for (std::uint32_t i = 0; i < fa; ++i) {
            if (a[i].id != b[i].id || a[i].distance != b[i].distance) {
                identical = false;
            }
        }
    }
    check(identical, "20 queries: loaded graph returns identical (id, dist)");

    // Rejection: flipped magic byte.
    {
        std::FILE* in = std::fopen(kGraphFile, "rb");
        std::fseek(in, 0, SEEK_END);
        const long len = std::ftell(in);
        std::fseek(in, 0, SEEK_SET);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(len));
        const bool read_ok =
            std::fread(bytes.data(), 1u, bytes.size(), in) == bytes.size();
        std::fclose(in);
        check(read_ok, "reference graph file read back");

        bytes[0] = static_cast<std::uint8_t>('X');
        std::FILE* out = std::fopen(kBadFile, "wb");
        std::fwrite(bytes.data(), 1u, bytes.size(), out);
        std::fclose(out);
        check(loaded.load_graph(kBadFile) == edgevector::GraphIoStatus::bad_magic,
              "flipped magic -> bad_magic");
        check(loaded.size() == 0u, "failed load leaves the graph empty");

        // Rejection: truncated body (magic restored first, so the failure is
        // attributable to the truncation alone).
        bytes[0] = static_cast<std::uint8_t>('E');
        out = std::fopen(kBadFile, "wb");
        std::fwrite(bytes.data(), 1u, bytes.size() - 5u, out);
        std::fclose(out);
        const edgevector::GraphIoStatus st = loaded.load_graph(kBadFile);
        check(st == edgevector::GraphIoStatus::io_error ||
                  st == edgevector::GraphIoStatus::corrupt,
              "truncated file rejected");
    }

    // Rejection: parameter mismatch (different M).
    {
        edgevector::HNSWGraph wrong_m(data.base(), data.record_bytes(), dim,
                                      graph.capacity(), /*m=*/8u);
        check(wrong_m.load_graph(kGraphFile) ==
                  edgevector::GraphIoStatus::incompatible,
              "M mismatch -> incompatible");
    }

    // A successfully re-loaded graph keeps working and accepts new inserts
    // when spare capacity exists (none here: capacity is full, so the insert
    // must be cleanly refused, not corrupt anything).
    check(loaded.load_graph(kGraphFile) == edgevector::GraphIoStatus::ok,
          "re-load after failed loads returns ok");
    check(!loaded.insert(0u), "insert of an existing id is still rejected");

    // Leave the shared graph as we found it for the cases that follow.
    check(graph.restore(11u) && graph.restore(23u),
          "shared graph's tombstones restored after the round trip");

    std::remove(kGraphFile);
    std::remove(kBadFile);
}

} // namespace

int main() {
    std::printf("=== EdgeVector :: hnsw_graph unit tests ===\n\n");

    test_edges();
    test_small_recall();

    // Cases 3-9 share one 2000-node graph so the build cost is paid once.
    const std::size_t dim = 512;
    const std::size_t n = 2000;
    std::mt19937 rng(42);
    RecordBlock data(dim, n);
    std::vector<float> floats;
    fill_random_block(data, dim, rng, &floats);

    edgevector::HNSWGraph graph(data.base(), data.record_bytes(), dim,
                                static_cast<std::uint32_t>(n),
                                /*m=*/16u, /*ef_construction=*/200u,
                                /*max_ef_search=*/256u, /*seed=*/42u);
    for (std::uint32_t i = 0; i < n; ++i) {
        graph.insert(i);
    }

    test_scale_recall(graph, data, dim);
    test_sorted_output(graph, data, dim);
    test_zero_allocation(graph, data, dim);
    test_persistence(graph, data, dim);
    test_contexts_and_threads(graph, graph, dim);
    test_delete_and_filter(dim);
    test_reranked_accuracy(graph, floats, n, dim);

    std::printf("\n=== %s ===\n",
                (g_failures == 0) ? "ALL CASES PASSED" : "FAILURES DETECTED");
    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
    }
    return (g_failures == 0) ? 0 : 1;
}
