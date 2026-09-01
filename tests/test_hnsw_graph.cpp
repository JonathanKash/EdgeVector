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

// ---------------------------------------------------------------------------
// Global allocation counter. Every path into the C++ free store goes through
// these replacements, so a zero delta across search() calls is hard evidence
// of a zero-allocation query path, not a code-reading claim.
// ---------------------------------------------------------------------------
static std::size_t g_new_calls = 0;

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

void fill_random_block(RecordBlock& block, std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(dim);
    for (std::size_t i = 0; i < block.count(); ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(rng);
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
// Case 5: the zero-allocation proof
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
    }
    const std::size_t delta = g_new_calls - before;

    std::printf("      operator new calls across 100 searches: %zu  (gate == 0)\n",
                delta);
    check(delta == 0u, "zero heap allocations across 100 searches");
    check(total_found == 1000u, "every search returned a full k = 10");
    (void)data;
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

    check(graph.save_graph(kGraphFile) == edgevector::GraphIoStatus::ok,
          "save_graph returns ok");

    edgevector::HNSWGraph loaded(data.base(), data.record_bytes(), dim,
                                 graph.capacity(), /*m=*/16u,
                                 /*ef_construction=*/200u,
                                 /*max_ef_search=*/256u, /*seed=*/42u);
    check(loaded.load_graph(kGraphFile) == edgevector::GraphIoStatus::ok,
          "load_graph returns ok");
    check(loaded.size() == graph.size(), "loaded size matches");

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

    std::remove(kGraphFile);
    std::remove(kBadFile);
}

} // namespace

int main() {
    std::printf("=== EdgeVector :: hnsw_graph unit tests ===\n\n");

    test_edges();
    test_small_recall();

    // Cases 3-5 share one 2000-node graph so the build cost is paid once.
    const std::size_t dim = 512;
    const std::size_t n = 2000;
    std::mt19937 rng(42);
    RecordBlock data(dim, n);
    fill_random_block(data, dim, rng);

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

    std::printf("\n=== %s ===\n",
                (g_failures == 0) ? "ALL CASES PASSED" : "FAILURES DETECTED");
    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
    }
    return (g_failures == 0) ? 0 : 1;
}
