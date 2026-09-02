// ============================================================================
// EdgeVector quickstart: a complete, runnable tour of the library.
//
// Build and run (from examples/):
//     make run
// or directly:
//     g++ -std=c++17 -O2 -march=native -I../include quickstart.cpp -o quickstart
//     ./quickstart
// or via CMake (from the repo root):
//     cmake -B build -DEDGEVECTOR_BUILD_EXAMPLES=ON && cmake --build build
//     ./build/quickstart
//
// The program synthesizes an embedding-like dataset, then walks the whole
// production pipeline: quantize -> build (multi-threaded) -> persist ->
// reload with zero rebuild -> search three ways (Hamming, asymmetric
// re-rank, float-exact re-rank) -> filter -> delete -> reclaim a slot.
// Every step checks its own results and the program exits non-zero if
// anything is off, so this file doubles as a smoke test (CI runs it).
// ============================================================================

#include <edgevector/edgevector.hpp>

#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

using namespace edgevector;

static int g_failures = 0;
static void expect(bool ok, const char* what) {
    std::printf("  %s  %s\n", ok ? "ok " : "FAIL", what);
    if (!ok) ++g_failures;
}

int main() {
    std::printf("EdgeVector %s quickstart\n\n", EDGEVECTOR_VERSION);

    // ------------------------------------------------------------------
    // 0. A dataset. In real life these are your embeddings; here we make
    //    10,000 clustered 512-d vectors so the search has structure to find.
    // ------------------------------------------------------------------
    const std::size_t dim = 512;
    const std::uint32_t n = 10000;
    const std::size_t rb = padded_bytes(dim); // 64 bytes per vector at 512-d

    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> centers(100 * dim);
    for (float& c : centers) c = gauss(rng);

    std::vector<float> floats(static_cast<std::size_t>(n) * dim);
    for (std::uint32_t i = 0; i < n; ++i) {
        const float* c = centers.data() + (i % 100) * dim;
        for (std::size_t d = 0; d < dim; ++d)
            floats[i * dim + d] = c[d] + 0.5f * gauss(rng);
    }

    // ------------------------------------------------------------------
    // 1. Quantize: 2,048 float bytes -> 64 code bytes per vector (32x).
    //    Codes must live in an 8-byte-aligned block; backing the buffer
    //    with uint64_t guarantees that.
    // ------------------------------------------------------------------
    std::vector<std::uint64_t> code_words((rb / 8) * n);
    auto* codes = reinterpret_cast<std::uint8_t*>(code_words.data());
    for (std::uint32_t i = 0; i < n; ++i)
        quantize(floats.data() + i * dim, dim, codes + i * rb);
    std::printf("[1] quantized %u vectors: %.1f KB of codes (float32: %.1f KB)\n",
                n, n * rb / 1024.0, n * dim * 4 / 1024.0);

    // ------------------------------------------------------------------
    // 2. Build the index using every hardware thread, then persist BOTH
    //    artifacts: the vector file (EVEC) and the graph file (EVHG).
    // ------------------------------------------------------------------
    HNSWGraph builder(codes, rb, dim, n); // M=16, ef_construction=200 defaults
    std::vector<std::uint32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0u);
    expect(builder.insert_batch(ids.data(), n, 0) == n, "built the index");
    expect(builder.validate_integrity(), "index passes integrity validation");
    expect(write_storage_file("quickstart.evec", dim, n, codes) ==
               StorageStatus::ok, "vector file written");
    expect(builder.save_graph("quickstart.evhg") == GraphIoStatus::ok,
           "graph file written");

    // ------------------------------------------------------------------
    // 3. Device startup: mmap the vectors, load the graph. No rebuild.
    // ------------------------------------------------------------------
    MMapStorage store;
    expect(store.open("quickstart.evec") == StorageStatus::ok,
           "vector file mapped (zero-copy)");
    HNSWGraph graph(store.vector(0), store.record_bytes(),
                    static_cast<std::size_t>(store.dim()),
                    static_cast<std::uint32_t>(store.count()));
    expect(graph.load_graph("quickstart.evhg") == GraphIoStatus::ok,
           "graph loaded (no rebuild)");

    // ------------------------------------------------------------------
    // 4. Search, three ways. The query arrives as floats; quantize it.
    //    Mode 1: Hamming - fastest, binary-metric ranking.
    //    Mode 2: asymmetric re-rank - float-grade ranking, zero extra memory.
    //    Mode 3: exact re-rank - float32-exact results; the float corpus can
    //            stay on flash (here it is just our in-RAM array).
    // ------------------------------------------------------------------
    const float* qf = floats.data() + 7 * dim; // query with vector #7 itself
    alignas(8) std::uint8_t qbits[64];
    quantize(qf, dim, qbits);

    SearchResult hits[10];
    std::uint32_t found = graph.search(qbits, 10, /*ef=*/50, hits);
    expect(found == 10 && hits[0].id == 7 && hits[0].distance == 0,
           "Hamming search: the query's own vector ranks first");

    ScoredResult best[10];
    found = graph.search_reranked(qbits, qf, 10, 50, best);
    expect(found == 10 && best[0].id == 7,
           "asymmetric re-rank agrees (dot(q, sign(x)) scoring)");

    found = graph.search_exact_reranked(qbits, qf, floats.data(), dim,
                                        10, 100, best);
    expect(found == 10 && best[0].id == 7,
           "exact re-rank agrees (true float32 cosine over the pool)");

    // ------------------------------------------------------------------
    // 5. Concurrency: one SearchContext per querying thread. The graph
    //    itself is shared and const during queries.
    // ------------------------------------------------------------------
    SearchContext ctx = graph.make_context();
    found = graph.search(ctx, qbits, 10, 50, hits, nullptr);
    expect(found == 10 && hits[0].id == 7,
           "context-based search (thread-safe form) agrees");

    // ------------------------------------------------------------------
    // 6. Filtering: an allow-bitmap restricts results (bit per id).
    // ------------------------------------------------------------------
    std::vector<std::uint64_t> allow((n + 63) / 64, 0);
    for (std::uint32_t id = 0; id < n; id += 2)      // permit even ids only
        allow[id >> 6] |= (1ull << (id & 63));
    found = graph.search(ctx, qbits, 10, 100, hits, allow.data());
    bool only_even = (found == 10);
    for (std::uint32_t i = 0; i < found; ++i)
        if (hits[i].id % 2 != 0) only_even = false;
    expect(only_even, "filtered search returns only permitted ids");

    // ------------------------------------------------------------------
    // 7. Dynamics: soft-delete, then reclaim the slot for a NEW vector
    //    (remove -> overwrite the bytes at that index -> reinsert).
    // ------------------------------------------------------------------
    expect(graph.remove(7), "vector 7 soft-deleted");
    found = graph.search(qbits, 10, 50, hits);
    bool absent = true;
    for (std::uint32_t i = 0; i < found; ++i)
        if (hits[i].id == 7) absent = false;
    expect(absent, "deleted vector no longer returned");

    // NOTE: reclaiming rewrites vector bytes, so it needs a writable block -
    // here we rebuild the RAM copy's slot and reinsert in `builder`, which
    // owns the writable codes. (A mapped file is read-only by design.)
    std::vector<float> fresh(dim);
    for (std::size_t d = 0; d < dim; ++d) fresh[d] = gauss(rng);
    builder.remove(7);
    quantize(fresh.data(), dim, codes + 7 * rb);
    expect(builder.reinsert(7), "slot 7 reclaimed for a brand-new vector");
    alignas(8) std::uint8_t fresh_bits[64];
    quantize(fresh.data(), dim, fresh_bits);
    found = builder.search(fresh_bits, 1, 50, hits);
    expect(found == 1 && hits[0].id == 7 && hits[0].distance == 0,
           "reclaimed slot serves the new vector");

    // ------------------------------------------------------------------
    // Cleanup.
    // ------------------------------------------------------------------
    store.close();
    std::remove("quickstart.evec");
    std::remove("quickstart.evhg");

    std::printf("\n%s\n", g_failures == 0 ? "All quickstart checks passed."
                                          : "QUICKSTART FAILURES - see above.");
    return g_failures == 0 ? 0 : 1;
}
