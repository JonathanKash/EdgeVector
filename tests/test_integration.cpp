// ============================================================================
// End-to-end integration test: quantize -> write index file -> mmap it ->
// build an HNSW graph directly over the mapping (zero-copy) -> search.
//
// Also enforces the CLAUDE.md memory target. INTERPRETATION (Overseer
// decision): the >= 30x gate is applied to the vector payload — quantized
// record bytes vs the float32 equivalent (dim * 4 bytes per vector) — because
// that is the quantity binary quantization controls. Graph link memory is
// identical for a float32 index and a quantized index, so it is reported for
// transparency but not counted against the quantizer on either side.
// ============================================================================

#include "edgevector/hnsw_graph.hpp"
#include "edgevector/mmap_storage.hpp"
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

// Counting operator new (same shim as test_hnsw_graph.cpp): proves the search
// path stays allocation-free when reading vectors straight out of the mapping.
static std::size_t g_new_calls = 0;

// See test_hnsw_graph.cpp: GCC 13+'s -Wmismatched-new-delete misfires on the
// canonical malloc/free allocator replacement; silenced for the shim only.
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

const char* const kIndexFile = "ev_test_integration.evec";

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

} // namespace

int main() {
    std::printf("=== EdgeVector :: integration test (mmap + HNSW) ===\n\n");

    const std::size_t dim = 512;
    const std::size_t n = 2000;
    const std::uint32_t k = 10u;
    const std::uint32_t ef = 100u;

    // -- Stage 1: quantize a dataset and write the index file ----------------
    std::printf("[1] Build and persist the quantized index file\n");

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    RecordBlock original(dim, n);
    {
        std::vector<float> raw(dim);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t d = 0; d < dim; ++d) {
                raw[d] = dist(rng);
            }
            edgevector::quantize(raw.data(), dim, original.record(i));
        }
    }

    const edgevector::StorageStatus wrote = edgevector::write_storage_file(
        kIndexFile, static_cast<std::uint64_t>(dim),
        static_cast<std::uint64_t>(n), original.base());
    check(wrote == edgevector::StorageStatus::ok, "index file written");

    // -- Stage 2: map it back and hand the mapping to the graph --------------
    std::printf("[2] Zero-copy graph build over the mapping\n");

    edgevector::MMapStorage storage;
    check(storage.open(kIndexFile) == edgevector::StorageStatus::ok,
          "index file mapped");
    check(storage.count() == n && storage.dim() == dim,
          "mapped header matches (dim = 512, count = 2000)");
    check(storage.record_bytes() == edgevector::padded_bytes(dim),
          "mapped record stride == padded_bytes(512)");

    edgevector::HNSWGraph graph(storage.vector(0), storage.record_bytes(),
                                dim, static_cast<std::uint32_t>(n),
                                /*m=*/16u, /*ef_construction=*/200u,
                                /*max_ef_search=*/256u, /*seed=*/42u);
    bool all_inserted = true;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (!graph.insert(i)) {
            all_inserted = false;
        }
    }
    check(all_inserted, "all 2000 mapped vectors inserted");

    // -- Stage 3: recall against brute force over the SAME mapping -----------
    std::printf("[3] Recall through the mapping (50 queries, ef = 100)\n");

    std::vector<float> raw(dim);
    std::vector<std::uint64_t> qwords(edgevector::padded_bytes(dim) / 8u, 0u);
    std::uint8_t* qbytes = reinterpret_cast<std::uint8_t*>(qwords.data());
    std::vector<edgevector::SearchResult> results(k);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> all(n); // (dist, id)

    std::mt19937 qrng(7);
    std::size_t hits = 0;
    const std::size_t n_queries = 50;
    for (std::size_t qi = 0; qi < n_queries; ++qi) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(qrng);
        }
        edgevector::quantize(raw.data(), dim, qbytes);

        for (std::size_t i = 0; i < n; ++i) {
            all[i] = std::make_pair(
                edgevector::hamming_distance(qbytes, storage.vector(i), dim),
                static_cast<std::uint32_t>(i));
        }
        std::sort(all.begin(), all.end());

        const std::uint32_t found = graph.search(qbytes, k, ef, results.data());
        for (std::uint32_t i = 0; i < found; ++i) {
            for (std::size_t t = 0; t < k; ++t) {
                if (results[i].id == all[t].second) {
                    ++hits;
                    break;
                }
            }
        }
    }
    const double recall = static_cast<double>(hits) /
                          static_cast<double>(n_queries * k);
    std::printf("      recall@10 = %.4f  (gate >= 0.90)\n", recall);
    check(recall >= 0.90, "recall@10 >= 0.90 through the mapping");

    // -- Stage 4: zero allocation while searching mapped data ----------------
    std::printf("[4] Zero-allocation search over the mapping\n");

    const std::size_t before = g_new_calls;
    for (int t = 0; t < 50; ++t) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(qrng);
        }
        edgevector::quantize(raw.data(), dim, qbytes);
        graph.search(qbytes, k, ef, results.data());
    }
    const std::size_t delta = g_new_calls - before;
    std::printf("      operator new calls across 50 searches: %zu  (gate == 0)\n",
                delta);
    check(delta == 0u, "zero heap allocations across 50 mapped searches");

    // -- Stage 5: persist the graph, reload it, search without rebuilding ----
    // The full edge-device startup story: map the vector file, load the graph
    // file, serve queries — no construction cost at boot.
    std::printf("[5] Graph persistence over the mapping\n");

    const char* const kGraphFile = "ev_test_integration.evhg";
    check(graph.save_graph(kGraphFile) == edgevector::GraphIoStatus::ok,
          "graph saved");

    edgevector::HNSWGraph reloaded(storage.vector(0), storage.record_bytes(),
                                   dim, static_cast<std::uint32_t>(n),
                                   /*m=*/16u, /*ef_construction=*/200u,
                                   /*max_ef_search=*/256u, /*seed=*/42u);
    check(reloaded.load_graph(kGraphFile) == edgevector::GraphIoStatus::ok,
          "graph loaded into a fresh instance");

    std::vector<edgevector::SearchResult> results2(k);
    bool identical = true;
    for (int t = 0; t < 20; ++t) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(qrng);
        }
        edgevector::quantize(raw.data(), dim, qbytes);
        const std::uint32_t fa = graph.search(qbytes, k, ef, results.data());
        const std::uint32_t fb = reloaded.search(qbytes, k, ef, results2.data());
        if (fa != fb) {
            identical = false;
            continue;
        }
        for (std::uint32_t i = 0; i < fa; ++i) {
            if (results[i].id != results2[i].id ||
                results[i].distance != results2[i].distance) {
                identical = false;
            }
        }
    }
    check(identical, "reloaded graph returns identical (id, dist) for 20 queries");

    const std::size_t before2 = g_new_calls;
    for (int t = 0; t < 20; ++t) {
        reloaded.search(qbytes, k, ef, results2.data());
    }
    check(g_new_calls - before2 == 0u,
          "zero heap allocations searching the reloaded graph");

    // -- Stage 6: memory footprint target ------------------------------------
    std::printf("[6] Memory footprint (CLAUDE.md >= 30x target)\n");

    const std::size_t quant_payload = n * edgevector::padded_bytes(dim);
    const std::size_t float_payload = n * dim * sizeof(float);
    const double payload_ratio = static_cast<double>(float_payload) /
                                 static_cast<double>(quant_payload);
    const std::size_t graph_bytes = graph.graph_memory_bytes();
    const double total_ratio =
        static_cast<double>(float_payload + graph_bytes) /
        static_cast<double>(quant_payload + graph_bytes);

    std::printf("      quantized payload : %zu bytes\n", quant_payload);
    std::printf("      float32 payload   : %zu bytes\n", float_payload);
    std::printf("      payload ratio     : %.2fx  (gate >= 30x)\n", payload_ratio);
    std::printf("      graph overhead    : %zu bytes (identical either way; "
                "whole-index ratio %.2fx, informational)\n",
                graph_bytes, total_ratio);
    check(payload_ratio >= 30.0, "quantized payload >= 30x smaller than float32");

    // -- Cleanup --------------------------------------------------------------
    std::remove(kGraphFile);
    storage.close(); // must precede deletion: Windows cannot delete mapped files
    std::remove(kIndexFile);

    std::printf("\n=== %s ===\n",
                (g_failures == 0) ? "ALL CASES PASSED" : "FAILURES DETECTED");
    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
    }
    return (g_failures == 0) ? 0 : 1;
}
