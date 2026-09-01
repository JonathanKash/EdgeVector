#ifndef EDGEVECTOR_HNSW_GRAPH_HPP
#define EDGEVECTOR_HNSW_GRAPH_HPP

// ============================================================================
// EdgeVector :: hnsw_graph.hpp
//
// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016) over
// binary-quantized vectors, using the Hamming kernel and the asymmetric
// (query-side) scorer from quantize_math.hpp.
//
// MEMORY MODEL
// ------------
// The graph does NOT own vector data. It is constructed over an external,
// contiguous, 8-byte-aligned block of quantized records (an MMapStorage
// mapping or an in-memory buffer): record i lives at
// `vectors + i * record_bytes`, laid out per quantize_math.hpp. Node ids are
// indices into that block.
//
// Capacity can GROW: grow(new_capacity, new_vectors) extends every internal
// structure (existing links are untouched) and rebinds the vector base -
// the caller enlarges or relocates its block first, then hands over the new
// pointer (equal capacity = pure rebind after a realloc/remap). Growth is
// serial-only and O(capacity). SearchContexts are sized for the capacity at
// their creation: after grow(), searches through an outdated context safely
// return 0 results - recreate contexts via make_context(). The internal
// default context is recreated automatically.
//
// Per-node links are stored flat: one std::uint32_t array per node holding
// M0 = 2*M slots for layer 0 followed by M slots for each upper layer the
// node participates in. Node levels are drawn from the standard geometric
// distribution with mL = 1 / ln(M), capped at 31.
//
// SEARCH MODES
// ------------
//   search()          - k-NN by Hamming distance.
//   search_reranked() - retrieves an ef-wide candidate pool by Hamming, then
//                       re-ranks it with the asymmetric score
//                       dot(q_float, sign(x)) - float-grade resolution over
//                       Hamming's coarse integer ties, at zero extra index
//                       memory. Requires the float query alongside the
//                       quantized one.
// Both accept an optional caller-owned allow-bitmap (bit id set = id may be
// returned) and always exclude soft-deleted nodes; filtered-out nodes still
// route traversal. NOTE: a highly selective filter degrades toward a scan of
// the reachable graph, as in every filtered-HNSW implementation.
//
// DELETION AND SLOT RECLAMATION
// -----------------------------
// remove(id) soft-deletes: the node stays in the graph as a routing waypoint
// (links intact) but is never returned by queries. restore(id) undoes it.
// Tombstones persist in the graph file (format v2).
//
// reinsert(id) RECLAIMS a tombstoned slot for a new vector: after the caller
// has overwritten the vector bytes at index `id` in the external block, it
// exhaustively unlinks the id from every neighbor list in the graph (a full
// O(total links) sweep - correct by construction, milliseconds at 100k
// vectors - rather than a heuristic local repair), repairs the entry point
// if the reclaimed node held it, draws a fresh level, and re-runs the
// standard insert linking against the new bytes. Requiring the tombstone
// first (remove -> overwrite bytes -> reinsert) makes accidental clobbering
// impossible. Serial only: no queries or builds may run concurrently.
//
// ZERO-ALLOCATION QUERY PATH (CLAUDE.md section 3)
// ------------------------------------------------
// All query-time working memory lives in a SearchContext, pre-allocated when
// the context is created:
//   - a visited-epoch array (O(1) reset by bumping an epoch counter; memset
//     only on 32-bit wraparound, which is not an allocation),
//   - a fixed-capacity candidate min-heap sized capacity+1 (each node enters
//     it at most once per search, so it cannot overflow),
//   - a fixed-capacity result max-heap sized ef_limit+1,
//   - the asymmetric score table and a scored-candidate buffer for
//     search_reranked.
// search() / search_reranked() and everything they call perform no heap
// allocation, no std::vector::push_back, and no system calls; heaps are raw
// arrays driven by std::push_heap/std::pop_heap with explicit size counters.
// insert() is the build path, where allocation is permitted.
//
// THREADING
// ---------
// Queries are thread-safe when each thread uses its own SearchContext
// (graph.make_context()) against a graph that is not being mutated. The
// no-context convenience overloads use an internal default context and are
// single-threaded.
//
// Construction: serial insert() calls are the deterministic path.
// insert_batch(ids, count, n_threads) builds concurrently with the classic
// striped-lock scheme (every link mutation and every construction-time
// neighbor-list read happens under that node's stripe lock; exactly one lock
// is ever held at a time, so deadlock is impossible by construction). The
// query hot path takes NO locks and is unchanged. A parallel build is
// nondeterministic in link structure (insertion order interleaves) but must
// pass the same integrity validation and recall gates; validate_integrity()
// runs the loader's full referential check on the in-memory graph.
// Queries, remove()/restore(), and load_graph() must not run concurrently
// with any build. On toolchains without std::thread (MinGW win32 model)
// insert_batch degrades to the serial path.
//
// Ties break by (distance, id) everywhere, so results are deterministic and
// comparable against a brute-force baseline using the same ordering.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

// Thread support probe: libstdc++ advertises gthreads; other mainstream
// standard libraries (libc++) always have <thread>/<mutex>. MinGW's win32
// thread model lacks them, and insert_batch falls back to the serial path.
// Define EDGEVECTOR_NO_THREADS to force the serial fallback.
#if !defined(EDGEVECTOR_NO_THREADS) && \
    (defined(_GLIBCXX_HAS_GTHREADS) || !defined(__GLIBCXX__))
#define EDGEVECTOR_HAS_THREADS 1
#include <atomic>
#include <mutex>
#include <thread>
#endif

#include "edgevector/quantize_math.hpp"

namespace edgevector {

struct SearchResult {
    std::uint32_t id;
    std::uint32_t distance; // Hamming
};

struct ScoredResult {
    std::uint32_t id;
    float score; // dot(q_float, sign(x)); higher = more similar
};

// Status of save_graph()/load_graph(). Like the storage module, graph I/O
// never throws; every failure is a status value and a failed load leaves the
// graph empty rather than half-populated.
enum class GraphIoStatus : std::uint8_t {
    ok = 0,
    io_error,     // open/read/write failed
    bad_magic,
    bad_version,
    incompatible, // file's dim/capacity/M differ from this graph's config
    corrupt       // internal inconsistency (bad level, count, link, or size)
};

namespace detail {

struct Candidate {
    std::uint32_t dist;
    std::uint32_t id;
};

struct Scored {
    float score;
    std::uint32_t id;
};

// Total order: ascending by distance, then by id. Using it on both the graph
// and the brute-force baseline makes tie handling deterministic.
inline bool candidate_less(const Candidate& a, const Candidate& b) noexcept {
    return (a.dist != b.dist) ? (a.dist < b.dist) : (a.id < b.id);
}

// Descending by score, then ascending by id: the output order of
// search_reranked.
inline bool scored_better(const Scored& a, const Scored& b) noexcept {
    return (a.score != b.score) ? (a.score > b.score) : (a.id < b.id);
}

struct MinHeapCmp {
    bool operator()(const Candidate& a, const Candidate& b) const noexcept {
        return candidate_less(b, a); // heap top = smallest
    }
};

struct MaxHeapCmp {
    bool operator()(const Candidate& a, const Candidate& b) const noexcept {
        return candidate_less(a, b); // heap top = largest
    }
};

// Fixed-capacity binary heap over a pre-allocated buffer. No allocation, no
// push_back: an explicit size counter over raw storage, per the critical-path
// rules. The caller guarantees pushes never exceed the capacity it sized.
template <typename Cmp>
class FixedHeap {
public:
    void init(Candidate* data, std::uint32_t cap) noexcept {
        data_ = data;
        cap_ = cap;
        size_ = 0u;
    }

    void clear() noexcept { size_ = 0u; }
    bool empty() const noexcept { return size_ == 0u; }
    std::uint32_t size() const noexcept { return size_; }
    const Candidate* data() const noexcept { return data_; }
    const Candidate& top() const noexcept {
        assert(size_ > 0u);
        return data_[0];
    }

    void push(Candidate c) noexcept {
        assert(size_ < cap_);
        data_[size_] = c;
        ++size_;
        std::push_heap(data_, data_ + size_, Cmp());
    }

    void pop() noexcept {
        assert(size_ > 0u);
        std::pop_heap(data_, data_ + size_, Cmp());
        --size_;
    }

private:
    Candidate* data_ = nullptr;
    std::uint32_t cap_ = 0u;
    std::uint32_t size_ = 0u;
};

} // namespace detail

class HNSWGraph;

// Pre-allocated per-thread query scratch. Create one per querying thread via
// HNSWGraph::make_context() (which allocates); every search through it is
// then allocation-free. Movable, not copyable.
class SearchContext {
public:
    SearchContext(SearchContext&&) noexcept = default;
    SearchContext& operator=(SearchContext&&) noexcept = default;
    SearchContext(const SearchContext&) = delete;
    SearchContext& operator=(const SearchContext&) = delete;

private:
    friend class HNSWGraph;

    SearchContext(std::uint32_t capacity, std::uint32_t ef_limit,
                  std::uint32_t m0, std::size_t table_floats)
        : capacity_(capacity) {
        visited_.reset(new std::uint32_t[capacity]);
        std::memset(visited_.get(), 0,
                    static_cast<std::size_t>(capacity) * 4u);
        cand_buf_.reset(
            new detail::Candidate[static_cast<std::size_t>(capacity) + 1u]);
        res_buf_.reset(
            new detail::Candidate[static_cast<std::size_t>(ef_limit) + 1u]);
        scored_buf_.reset(
            new detail::Scored[static_cast<std::size_t>(ef_limit) + 1u]);
        asym_table_.reset(new float[table_floats]);
        // Build-path scratch: candidate staging for neighbor selection, the
        // selected-id list, and a bounce buffer for lock-guarded copies of
        // neighbor lists during concurrent construction.
        select_buf_.reset(
            new detail::Candidate[static_cast<std::size_t>(ef_limit) +
                                  static_cast<std::size_t>(m0) + 2u]);
        selected_.reset(new std::uint32_t[static_cast<std::size_t>(m0) + 1u]);
        neigh_buf_.reset(new std::uint32_t[static_cast<std::size_t>(m0)]);
        cand_heap_.init(cand_buf_.get(), capacity + 1u);
        res_heap_.init(res_buf_.get(), ef_limit + 1u);
    }

    void next_epoch() noexcept {
        ++epoch_;
        if (epoch_ == 0u) { // 32-bit wrap: hard reset, then restart at 1
            std::memset(visited_.get(), 0,
                        static_cast<std::size_t>(capacity_) * 4u);
            epoch_ = 1u;
        }
    }
    bool is_visited(std::uint32_t id) const noexcept {
        return visited_[id] == epoch_;
    }
    void mark_visited(std::uint32_t id) noexcept { visited_[id] = epoch_; }

    std::uint32_t capacity_ = 0u;
    std::unique_ptr<std::uint32_t[]> visited_;
    std::uint32_t epoch_ = 0u;
    std::unique_ptr<detail::Candidate[]> cand_buf_;
    std::unique_ptr<detail::Candidate[]> res_buf_;
    std::unique_ptr<detail::Scored[]> scored_buf_;
    std::unique_ptr<float[]> asym_table_;
    std::unique_ptr<detail::Candidate[]> select_buf_; // build path
    std::unique_ptr<std::uint32_t[]> selected_;       // build path
    std::unique_ptr<std::uint32_t[]> neigh_buf_;      // build path (locked copies)
    detail::FixedHeap<detail::MinHeapCmp> cand_heap_;
    detail::FixedHeap<detail::MaxHeapCmp> res_heap_;
};

class HNSWGraph {
public:
    // `vectors`: base of the contiguous quantized block (external, outlives
    // the graph). `record_bytes` >= padded_bytes(dim), 8-byte aligned records.
    HNSWGraph(const std::uint8_t* vectors,
              std::size_t record_bytes,
              std::size_t dim,
              std::uint32_t capacity,
              std::uint32_t m = 16u,
              std::uint32_t ef_construction = 200u,
              std::uint32_t max_ef_search = 256u,
              std::uint64_t seed = 42u)
        : vectors_(vectors),
          record_bytes_(record_bytes),
          dim_(dim),
          capacity_(capacity),
          m_(m < 2u ? 2u : m),
          m0_(2u * (m < 2u ? 2u : m)),
          ef_construction_(ef_construction < 1u ? 1u : ef_construction),
          rng_(seed) {
        assert(vectors_ != nullptr);
        assert(capacity_ > 0u);
        assert(dim_ > 0u);
        assert(record_bytes_ >= padded_bytes(dim_));

        ml_ = 1.0 / std::log(static_cast<double>(m_));
        ef_limit_ = (max_ef_search > ef_construction_) ? max_ef_search
                                                       : ef_construction_;

        levels_.reset(new std::uint8_t[capacity_]);
        std::memset(levels_.get(), 0xFF, capacity_); // kNotInserted

        deleted_words_ = (static_cast<std::size_t>(capacity_) + 63u) / 64u;
        deleted_.reset(new std::uint64_t[deleted_words_]);
        std::memset(deleted_.get(), 0, deleted_words_ * 8u);

        links_.resize(capacity_);
        counts_.resize(capacity_);

#if defined(EDGEVECTOR_HAS_THREADS)
        // Striped per-node locks for concurrent construction. Power-of-two
        // stripe count so lock lookup is a mask; 8192 stripes keeps collision
        // probability negligible at a fraction of a megabyte.
        std::uint32_t stripes = 1u;
        while (stripes < capacity_ && stripes < 8192u) {
            stripes <<= 1u;
        }
        lock_mask_ = stripes - 1u;
        locks_.reset(new std::mutex[stripes]);
#endif

        default_ctx_.reset(new SearchContext(make_context()));
    }

    HNSWGraph(const HNSWGraph&) = delete;
    HNSWGraph& operator=(const HNSWGraph&) = delete;

    std::uint32_t size() const noexcept { return size_; }
    std::uint32_t deleted_count() const noexcept { return deleted_count_; }
    std::uint32_t live_size() const noexcept { return size_ - deleted_count_; }
    std::uint32_t capacity() const noexcept { return capacity_; }
    std::size_t dim() const noexcept { return dim_; }
    std::uint32_t max_ef() const noexcept { return ef_limit_; }

    // Allocates a fresh scratch context sized for this graph. Do this once
    // per querying thread, at setup time - never per query. Contexts created
    // before a grow() are undersized for the larger graph; searches through
    // them return 0 results (safely) until they are recreated.
    SearchContext make_context() const {
        return SearchContext(capacity_, ef_limit_, m0_,
                             asymmetric_table_floats(dim_));
    }

    // Grows the slot count to `new_capacity` and rebinds the vector block to
    // `new_vectors` (the caller enlarges or relocates its block FIRST; the
    // first `capacity()` records must hold the same vectors as before).
    // Every existing node, link, and tombstone is preserved untouched; ids
    // [old capacity, new capacity) become insertable. new_capacity equal to
    // the current capacity is a pure rebind. Returns false for a null
    // pointer or a shrinking capacity. Serial-only: no queries, builds, or
    // other mutations may run concurrently. O(capacity) time; may throw
    // std::bad_alloc under memory exhaustion, like insert().
    bool grow(std::uint32_t new_capacity, const std::uint8_t* new_vectors) {
        if (new_vectors == nullptr || new_capacity < capacity_) {
            return false;
        }
        vectors_ = new_vectors;
        if (new_capacity == capacity_) {
            return true;
        }
        const std::uint32_t old_capacity = capacity_;

        std::uint8_t* new_levels = new std::uint8_t[new_capacity];
        std::memcpy(new_levels, levels_.get(), old_capacity);
        std::memset(new_levels + old_capacity, 0xFF,
                    new_capacity - old_capacity);
        levels_.reset(new_levels);

        const std::size_t new_words =
            (static_cast<std::size_t>(new_capacity) + 63u) / 64u;
        std::uint64_t* new_deleted = new std::uint64_t[new_words];
        std::memset(new_deleted, 0, new_words * 8u);
        std::memcpy(new_deleted, deleted_.get(), deleted_words_ * 8u);
        deleted_.reset(new_deleted);
        deleted_words_ = new_words;

        links_.resize(new_capacity);
        counts_.resize(new_capacity);
        capacity_ = new_capacity;

#if defined(EDGEVECTOR_HAS_THREADS)
        // Re-stripe the build locks for the larger id space (only ever
        // grows; builds are not running during grow(), so no lock is held).
        std::uint32_t stripes = 1u;
        while (stripes < capacity_ && stripes < 8192u) {
            stripes <<= 1u;
        }
        if (stripes != lock_mask_ + 1u) {
            lock_mask_ = stripes - 1u;
            locks_.reset(new std::mutex[stripes]);
        }
#endif

        // The internal default context must match the new capacity; user
        // contexts are invalidated (their searches return 0) until recreated.
        default_ctx_.reset(new SearchContext(make_context()));
        return true;
    }

    // BUILD PATH (allocation permitted). Inserts the vector at index `id` of
    // the external block. Each id may be inserted once; returns false for an
    // out-of-range or duplicate id. Serial and deterministic; must not run
    // concurrently with anything.
    bool insert(std::uint32_t id) {
        return insert_impl(*default_ctx_, id, /*locked=*/false);
    }

    // Concurrent build: inserts `count` ids from `ids` using `n_threads`
    // worker threads (0 = hardware concurrency). Returns how many inserts
    // succeeded. The ids array must not contain duplicates. Nondeterministic
    // link structure (insertion order interleaves), but the graph passes
    // validate_integrity() and the same recall gates. No queries, removals,
    // or loads may run concurrently with a build. Falls back to the serial
    // path when built without thread support or with n_threads <= 1.
    std::size_t insert_batch(const std::uint32_t* ids, std::size_t count,
                             unsigned n_threads = 0u) {
        if (ids == nullptr || count == 0u) {
            return 0u;
        }
#if defined(EDGEVECTOR_HAS_THREADS)
        if (n_threads == 0u) {
            n_threads = std::thread::hardware_concurrency();
            if (n_threads == 0u) {
                n_threads = 1u;
            }
        }
        if (n_threads > 1u) {
            std::atomic<std::size_t> next(0u);
            std::atomic<std::size_t> ok_count(0u);
            std::vector<std::thread> workers;
            workers.reserve(n_threads);
            for (unsigned t = 0u; t < n_threads; ++t) {
                workers.emplace_back([&]() {
                    SearchContext ctx = make_context();
                    for (;;) {
                        const std::size_t i =
                            next.fetch_add(1u, std::memory_order_relaxed);
                        if (i >= count) {
                            break;
                        }
                        if (insert_impl(ctx, ids[i], /*locked=*/true)) {
                            ok_count.fetch_add(1u, std::memory_order_relaxed);
                        }
                    }
                });
            }
            for (std::thread& w : workers) {
                w.join();
            }
            return ok_count.load();
        }
#else
        (void)n_threads;
#endif
        std::size_t ok = 0u;
        for (std::size_t i = 0u; i < count; ++i) {
            if (insert_impl(*default_ctx_, ids[i], /*locked=*/false)) {
                ++ok;
            }
        }
        return ok;
    }

    // Runs the loader's full structural validation on the in-memory graph:
    // level bounds, per-layer neighbor counts, referential integrity of every
    // edge (targets inserted, present on that layer, never self), entry-point
    // consistency, and tombstones only on inserted nodes. For tests and
    // post-build sanity; not a hot path.
    bool validate_integrity() const {
        std::uint32_t inserted = 0u;
        for (std::uint32_t id = 0u; id < capacity_; ++id) {
            if (levels_[id] != kNotInserted) {
                ++inserted;
            }
        }
        if (inserted != size_) {
            return false;
        }
        if (has_entry_) {
            if (entry_ >= capacity_ || levels_[entry_] == kNotInserted ||
                static_cast<std::uint32_t>(levels_[entry_]) < entry_level_) {
                return false;
            }
        } else if (size_ > 0u) {
            return false;
        }
        return graph_invariants_ok();
    }

    // Soft delete: the node stops appearing in results but keeps routing
    // traffic. Returns false for an id that is out of range, never inserted,
    // or already deleted. Not thread-safe against concurrent queries.
    bool remove(std::uint32_t id) noexcept {
        if (id >= capacity_ || levels_[id] == kNotInserted || is_deleted(id)) {
            return false;
        }
        deleted_[id >> 6u] |= (1ull << (id & 63u));
        ++deleted_count_;
        return true;
    }

    // Undoes remove(). Returns false unless the id is currently deleted.
    bool restore(std::uint32_t id) noexcept {
        if (id >= capacity_ || levels_[id] == kNotInserted || !is_deleted(id)) {
            return false;
        }
        deleted_[id >> 6u] &= ~(1ull << (id & 63u));
        --deleted_count_;
        return true;
    }

    bool is_deleted(std::uint32_t id) const noexcept {
        return ((deleted_[id >> 6u] >> (id & 63u)) & 1ull) != 0ull;
    }

    // Slot reclamation: relinks a tombstoned id around the NEW vector bytes
    // the caller has already written at index `id` of the external block.
    // Workflow: remove(id) -> overwrite the bytes -> reinsert(id). Returns
    // false unless the id is inserted AND currently deleted. The id keeps
    // counting toward size(); deleted_count() drops by one. Cost is one full
    // sweep of the graph's link lists plus a normal insert. Serial only:
    // must not run concurrently with queries, builds, or other mutations.
    bool reinsert(std::uint32_t id) {
        if (id >= capacity_ || levels_[id] == kNotInserted ||
            !is_deleted(id)) {
            return false;
        }

        unlink_everywhere(id);

        // Entry repair: if the reclaimed node was the entry point, promote
        // the highest-level other node (or fall back to the empty-graph
        // bootstrap when this is the only node).
        if (has_entry_ && entry_ == id) {
            bool found = false;
            std::uint32_t best = 0u;
            std::uint32_t best_level = 0u;
            for (std::uint32_t u = 0u; u < capacity_; ++u) {
                if (u == id || levels_[u] == kNotInserted) {
                    continue;
                }
                const std::uint32_t ul =
                    static_cast<std::uint32_t>(levels_[u]);
                if (!found || ul > best_level) {
                    found = true;
                    best = u;
                    best_level = ul;
                }
            }
            if (found) {
                entry_ = best;
                entry_level_ = best_level;
            } else {
                has_entry_ = false;
                entry_ = 0u;
                entry_level_ = 0u;
            }
        }

        // Fresh level and zeroed link storage, then clear the tombstone and
        // relink against the caller's new bytes.
        const std::uint32_t level = random_level();
        links_[id].assign(static_cast<std::size_t>(m0_) +
                              static_cast<std::size_t>(level) * m_,
                          0u);
        counts_[id].assign(static_cast<std::size_t>(level) + 1u, 0u);
        levels_[id] = static_cast<std::uint8_t>(level);

        deleted_[id >> 6u] &= ~(1ull << (id & 63u));
        --deleted_count_;

        link_node(*default_ctx_, id, level, /*locked=*/false);
        return true;
    }

    // QUERY CRITICAL PATH: zero allocation, no system calls, noexcept.
    // Writes up to k results into `out`, ascending by (Hamming distance, id),
    // and returns how many were written. `ef` is the layer-0 beam width,
    // clamped to [k, max_ef()]; k itself is clamped to max_ef().
    //
    // `allow`, if given, is a caller-owned bitmap of (capacity+63)/64 words:
    // only ids whose bit is set may be returned. Soft-deleted ids are always
    // excluded. Excluded nodes still route.
    std::uint32_t search(SearchContext& ctx,
                         const std::uint8_t* q,
                         std::uint32_t k,
                         std::uint32_t ef,
                         SearchResult* out,
                         const std::uint64_t* allow = nullptr) const noexcept {
        if (!has_entry_ || k == 0u || q == nullptr || out == nullptr ||
            ctx.capacity_ < capacity_) { // context predates a grow()
            return 0u;
        }
        clamp_kef(k, ef);

        std::uint32_t found = beam_layer0(ctx, q, ef, allow);
        while (found > k) { // keep only the k best
            ctx.res_heap_.pop();
            --found;
        }
        for (std::uint32_t i = found; i > 0u; --i) { // max-heap pops worst first
            const detail::Candidate c = ctx.res_heap_.top();
            ctx.res_heap_.pop();
            out[i - 1u] = SearchResult{c.id, c.dist};
        }
        return found;
    }

    // Convenience overload on the internal default context. Single-threaded.
    std::uint32_t search(const std::uint8_t* q, std::uint32_t k,
                         std::uint32_t ef, SearchResult* out) noexcept {
        return search(*default_ctx_, q, k, ef, out, nullptr);
    }

    // Two-stage search: retrieve an ef-wide candidate pool by Hamming beam,
    // then re-rank the whole pool by the asymmetric score
    // dot(q_floats, sign(x)) and return the k best, descending by (score,
    // then ascending id). `q_bits` must be quantize(q_floats). Same zero-
    // allocation guarantee and filtering semantics as search(); the accuracy
    // gain comes from ranking the pool with float-grade resolution instead of
    // Hamming's coarse integer ties. Widen ef to widen the re-ranked pool.
    std::uint32_t search_reranked(SearchContext& ctx,
                                  const std::uint8_t* q_bits,
                                  const float* q_floats,
                                  std::uint32_t k,
                                  std::uint32_t ef,
                                  ScoredResult* out,
                                  const std::uint64_t* allow = nullptr) const noexcept {
        if (!has_entry_ || k == 0u || q_bits == nullptr ||
            q_floats == nullptr || out == nullptr ||
            ctx.capacity_ < capacity_) { // context predates a grow()
            return 0u;
        }
        clamp_kef(k, ef);

        const std::uint32_t found = beam_layer0(ctx, q_bits, ef, allow);
        if (found == 0u) {
            return 0u;
        }

        build_asymmetric_table(q_floats, dim_, ctx.asym_table_.get());
        const detail::Candidate* raw = ctx.res_heap_.data();
        for (std::uint32_t i = 0u; i < found; ++i) {
            ctx.scored_buf_[i] = detail::Scored{
                asymmetric_score(ctx.asym_table_.get(), vec(raw[i].id), dim_),
                raw[i].id};
        }

        const std::uint32_t n_out = (k < found) ? k : found;
        std::partial_sort(ctx.scored_buf_.get(),
                          ctx.scored_buf_.get() + n_out,
                          ctx.scored_buf_.get() + found, detail::scored_better);
        for (std::uint32_t i = 0u; i < n_out; ++i) {
            out[i] = ScoredResult{ctx.scored_buf_[i].id,
                                  ctx.scored_buf_[i].score};
        }
        return n_out;
    }

    // Convenience overload on the internal default context. Single-threaded.
    std::uint32_t search_reranked(const std::uint8_t* q_bits,
                                  const float* q_floats, std::uint32_t k,
                                  std::uint32_t ef, ScoredResult* out) noexcept {
        return search_reranked(*default_ctx_, q_bits, q_floats, k, ef, out,
                               nullptr);
    }

    // Two-stage search with EXACT re-ranking: retrieve an ef-wide candidate
    // pool by Hamming beam, then score the pool with true float32 cosine
    // similarity against caller-provided full-precision vectors and return
    // the k best (descending score, ascending id on ties).
    //
    // `vectors_f32` holds the original floats: vector id starts at
    // vectors_f32 + id * f32_stride (stride in floats, >= dim). The intended
    // deployment keeps this block on flash via a read-only mmap - RAM then
    // holds only the 1-bit codes, and each query demand-pages just the ef
    // candidate vectors (~ef * dim * 4 bytes) it actually touches. This is
    // the accuracy ceiling of the index: recall against float32 ground truth
    // is limited only by whether the Hamming pool contains the true
    // neighbors, not by 1-bit ranking resolution.
    //
    // Zero allocation, noexcept; ~2 * ef * dim flops of re-ranking cost.
    std::uint32_t search_exact_reranked(SearchContext& ctx,
                                        const std::uint8_t* q_bits,
                                        const float* q_floats,
                                        const float* vectors_f32,
                                        std::size_t f32_stride,
                                        std::uint32_t k,
                                        std::uint32_t ef,
                                        ScoredResult* out,
                                        const std::uint64_t* allow = nullptr) const noexcept {
        if (!has_entry_ || k == 0u || q_bits == nullptr ||
            q_floats == nullptr || vectors_f32 == nullptr ||
            f32_stride < dim_ || out == nullptr ||
            ctx.capacity_ < capacity_) { // context predates a grow()
            return 0u;
        }
        clamp_kef(k, ef);

        const std::uint32_t found = beam_layer0(ctx, q_bits, ef, allow);
        if (found == 0u) {
            return 0u;
        }

        const detail::Candidate* raw = ctx.res_heap_.data();
        for (std::uint32_t i = 0u; i < found; ++i) {
            const float* x = vectors_f32 +
                             static_cast<std::size_t>(raw[i].id) * f32_stride;
            ctx.scored_buf_[i] = detail::Scored{
                cosine_similarity_f32(q_floats, x, dim_), raw[i].id};
        }

        const std::uint32_t n_out = (k < found) ? k : found;
        std::partial_sort(ctx.scored_buf_.get(),
                          ctx.scored_buf_.get() + n_out,
                          ctx.scored_buf_.get() + found, detail::scored_better);
        for (std::uint32_t i = 0u; i < n_out; ++i) {
            out[i] = ScoredResult{ctx.scored_buf_[i].id,
                                  ctx.scored_buf_[i].score};
        }
        return n_out;
    }

    // Convenience overload on the internal default context. Single-threaded.
    std::uint32_t search_exact_reranked(const std::uint8_t* q_bits,
                                        const float* q_floats,
                                        const float* vectors_f32,
                                        std::size_t f32_stride,
                                        std::uint32_t k, std::uint32_t ef,
                                        ScoredResult* out) noexcept {
        return search_exact_reranked(*default_ctx_, q_bits, q_floats,
                                     vectors_f32, f32_stride, k, ef, out,
                                     nullptr);
    }

    // ------------------------------------------------------------------------
    // GRAPH PERSISTENCE (little-endian, load-time path)
    //
    // The link structure is saved separately from the vectors: the vector file
    // is owned by mmap_storage.hpp, this file holds only the graph. A saved
    // graph is only meaningful next to the exact vector block it was built
    // over; pair the two files and load them together.
    //
    //   Bytes  0.. 3  char magic[4] = { 'E','V','H','G' }
    //   Bytes  4.. 7  u32  version  (this writer: 2; version 1 also loads)
    //   Bytes  8..15  u64  dim
    //   Bytes 16..19  u32  capacity
    //   Bytes 20..23  u32  M
    //   Bytes 24..27  u32  size (inserted node count)
    //   Bytes 28..31  u32  entry point id
    //   Bytes 32..35  u32  entry point level
    //   Byte  36      u8   has_entry (0 or 1)
    //   Bytes 37..63  zeroed reserved padding
    //   Then per node id in [0, capacity):
    //     u8 level (0xFF = not inserted; nothing else follows for that id)
    //     u32 counts[level + 1]
    //     u32 slots[M0 + level * M]
    //   Version 2 only, after the node records:
    //     u64 deleted_bitmap[(capacity + 63) / 64]   (soft-delete tombstones)
    //
    // load_graph() validates everything it reads (levels, counts, link
    // targets, entry point, node count, tombstone integrity, exact file
    // length) and rejects a file whose dim/capacity/M differ from this
    // graph's construction parameters.
    // ------------------------------------------------------------------------

    GraphIoStatus save_graph(const char* path) const {
        if (path == nullptr) {
            return GraphIoStatus::io_error;
        }
        std::FILE* f = std::fopen(path, "wb");
        if (f == nullptr) {
            return GraphIoStatus::io_error;
        }

        std::uint8_t header[kIoHeaderBytes];
        std::memset(header, 0, sizeof(header));
        header[0] = static_cast<std::uint8_t>('E');
        header[1] = static_cast<std::uint8_t>('V');
        header[2] = static_cast<std::uint8_t>('H');
        header[3] = static_cast<std::uint8_t>('G');
        const std::uint32_t version = kIoVersion;
        const std::uint64_t dim64 = static_cast<std::uint64_t>(dim_);
        const std::uint8_t has_entry = has_entry_ ? 1u : 0u;
        std::memcpy(header + 4u, &version, 4u);
        std::memcpy(header + 8u, &dim64, 8u);
        std::memcpy(header + 16u, &capacity_, 4u);
        std::memcpy(header + 20u, &m_, 4u);
        std::memcpy(header + 24u, &size_, 4u);
        std::memcpy(header + 28u, &entry_, 4u);
        std::memcpy(header + 32u, &entry_level_, 4u);
        std::memcpy(header + 36u, &has_entry, 1u);

        bool ok = std::fwrite(header, 1u, sizeof(header), f) == sizeof(header);
        for (std::uint32_t id = 0u; ok && id < capacity_; ++id) {
            const std::uint8_t lvl = levels_[id];
            ok = std::fwrite(&lvl, 1u, 1u, f) == 1u;
            if (!ok || lvl == kNotInserted) {
                continue;
            }
            const std::size_t n_counts = counts_[id].size();
            const std::size_t n_slots = links_[id].size();
            ok = std::fwrite(counts_[id].data(), 4u, n_counts, f) == n_counts &&
                 std::fwrite(links_[id].data(), 4u, n_slots, f) == n_slots;
        }
        if (ok) {
            ok = std::fwrite(deleted_.get(), 8u, deleted_words_, f) ==
                 deleted_words_;
        }

        if (std::fclose(f) != 0) {
            ok = false;
        }
        return ok ? GraphIoStatus::ok : GraphIoStatus::io_error;
    }

    // Replaces the graph's state with the file's contents. On any failure the
    // graph is left EMPTY (not half-loaded) and the status says why. The file
    // must have been saved by a graph with identical dim, capacity, and M,
    // over the same vector block this graph was constructed with.
    GraphIoStatus load_graph(const char* path) {
        reset_graph_state();
        if (path == nullptr) {
            return GraphIoStatus::io_error;
        }
        std::FILE* f = std::fopen(path, "rb");
        if (f == nullptr) {
            return GraphIoStatus::io_error;
        }
        const GraphIoStatus st = load_graph_body(f);
        std::fclose(f);
        if (st != GraphIoStatus::ok) {
            reset_graph_state();
        }
        return st;
    }

    // Bytes of pre-allocated + per-node graph memory (links, counts, levels,
    // tombstones, and the default context's pools). For footprint reporting;
    // not on any hot path.
    std::size_t graph_memory_bytes() const noexcept {
        std::size_t total = 0u;
        for (std::uint32_t i = 0u; i < capacity_; ++i) {
            total += links_[i].capacity() * sizeof(std::uint32_t);
            total += counts_[i].capacity() * sizeof(std::uint32_t);
        }
        total += static_cast<std::size_t>(capacity_);       // levels_
        total += deleted_words_ * 8u;                       // tombstones
        // default context: visited + heaps + rerank scratch
        total += static_cast<std::size_t>(capacity_) * 4u;
        total += (static_cast<std::size_t>(capacity_) + 1u) * sizeof(detail::Candidate);
        total += (static_cast<std::size_t>(ef_limit_) + 1u) *
                 (sizeof(detail::Candidate) + sizeof(detail::Scored));
        total += asymmetric_table_floats(dim_) * sizeof(float);
        return total;
    }

private:
    static constexpr std::uint8_t kNotInserted = 0xFFu;
    static constexpr std::uint8_t kClaimed = 0xFEu; // id reserved, links pending
    static constexpr std::uint32_t kMaxLevel = 31u;
    static constexpr std::size_t kIoHeaderBytes = 64u;
    static constexpr std::uint32_t kIoVersion = 2u;

#if defined(EDGEVECTOR_HAS_THREADS)
    std::mutex& lock_for(std::uint32_t id) const noexcept {
        return locks_[id & lock_mask_];
    }
#endif

    // The one insert implementation. locked == false is the serial,
    // deterministic path (bit-identical to pre-parallel builds); locked ==
    // true is the concurrent path, where every read or write of any node's
    // link storage happens under that node's stripe lock and the entry
    // point, RNG, id claims, and size counter are mutex-guarded. Exactly one
    // lock is ever held at a time.
    bool insert_impl(SearchContext& ctx, std::uint32_t id, bool locked) {
        if (id >= capacity_) {
            return false;
        }

        std::uint32_t level = 0u;
        if (locked) {
#if defined(EDGEVECTOR_HAS_THREADS)
            std::lock_guard<std::mutex> guard(rng_mutex_);
            if (levels_[id] != kNotInserted) {
                return false; // duplicate; claim check is atomic with the RNG
            }
            levels_[id] = kClaimed; // unreachable until the first reverse link
            level = random_level();
#else
            return false; // locked mode cannot be reached without threads
#endif
        } else {
            if (levels_[id] != kNotInserted) {
                return false;
            }
            level = random_level();
        }

        links_[id].assign(static_cast<std::size_t>(m0_) +
                              static_cast<std::size_t>(level) * m_,
                          0u);
        counts_[id].assign(static_cast<std::size_t>(level) + 1u, 0u);
        levels_[id] = static_cast<std::uint8_t>(level);

        link_node(ctx, id, level, locked);

        if (locked) {
#if defined(EDGEVECTOR_HAS_THREADS)
            std::lock_guard<std::mutex> guard(entry_mutex_);
            ++size_;
#endif
        } else {
            ++size_;
        }
        return true;
    }

    // The linking core, shared by insert_impl() and reinsert(): assumes
    // levels_[id] is set and links_[id]/counts_[id] are allocated and zeroed.
    // Handles the empty-graph bootstrap, the descent, per-layer neighbor
    // selection with bidirectional links, and the entry-point update. Does
    // NOT touch size_ (callers differ on whether the node is new).
    void link_node(SearchContext& ctx, std::uint32_t id, std::uint32_t level,
                   bool locked) {
        // Entry bootstrap / snapshot.
        std::uint32_t ep = 0u;
        std::uint32_t start_level = 0u;
        if (locked) {
#if defined(EDGEVECTOR_HAS_THREADS)
            std::lock_guard<std::mutex> guard(entry_mutex_);
            if (!has_entry_) {
                entry_ = id;
                entry_level_ = level;
                has_entry_ = true;
                return;
            }
            ep = entry_;
            start_level = entry_level_;
#endif
        } else {
            if (!has_entry_) {
                entry_ = id;
                entry_level_ = level;
                has_entry_ = true;
                return;
            }
            ep = entry_;
            start_level = entry_level_;
        }

        const std::uint8_t* q = vec(id);
        std::uint32_t ep_dist = distance_to(q, ep);

        for (std::uint32_t l = start_level; l > level; --l) {
            greedy_descend_build(ctx, q, l, ep, ep_dist, locked);
        }

        const std::uint32_t top = (start_level < level) ? start_level : level;
        for (std::uint32_t l = top;; --l) {
            // Build ignores tombstones: deleted nodes keep routing, and links
            // to them stay valid.
            const std::uint32_t found = search_layer(
                ctx, q, ep, ef_construction_, l, nullptr, false, locked);

            const detail::Candidate* raw = ctx.res_heap_.data();
            for (std::uint32_t i = 0u; i < found; ++i) {
                ctx.select_buf_[i] = raw[i];
            }
            std::sort(ctx.select_buf_.get(), ctx.select_buf_.get() + found,
                      detail::candidate_less);

            // Entry point for the next layer down: the closest thing found
            // here. Saved now because add_link() reuses ctx.select_buf_.
            ep = ctx.select_buf_[0].id;
            ep_dist = ctx.select_buf_[0].dist;

            const std::uint32_t mm = max_m(l);
            const std::uint32_t n_sel = select_neighbors(ctx, found, mm);

            // Stable copy of the selected ids for the reverse-link loop:
            // add_link() reuses ctx.selected_ internally when it prunes, and
            // in concurrent mode even this node's own slot array can be
            // rewritten by other threads' prunes mid-loop. neigh_buf_ is
            // untouched by add_link, so it is the safe staging area.
            for (std::uint32_t i = 0u; i < n_sel; ++i) {
                ctx.neigh_buf_[i] = ctx.selected_[i];
            }

            // The new node's own slots. Under the stripe lock in concurrent
            // mode: once a higher layer's reverse links published this id,
            // other threads may add reverse links here concurrently.
            if (locked) {
#if defined(EDGEVECTOR_HAS_THREADS)
                std::lock_guard<std::mutex> guard(lock_for(id));
                std::uint32_t* slots = link_slots(id, l);
                for (std::uint32_t i = 0u; i < n_sel; ++i) {
                    slots[i] = ctx.neigh_buf_[i];
                }
                counts_[id][l] = n_sel;
#endif
            } else {
                std::uint32_t* slots = link_slots(id, l);
                for (std::uint32_t i = 0u; i < n_sel; ++i) {
                    slots[i] = ctx.neigh_buf_[i];
                }
                counts_[id][l] = n_sel;
            }

            for (std::uint32_t i = 0u; i < n_sel; ++i) {
                add_link(ctx, ctx.neigh_buf_[i], id, l, locked);
            }

            if (l == 0u) {
                break;
            }
        }

        if (locked) {
#if defined(EDGEVECTOR_HAS_THREADS)
            std::lock_guard<std::mutex> guard(entry_mutex_);
            if (level > entry_level_) {
                entry_ = id;
                entry_level_ = level;
            }
#endif
        } else {
            if (level > entry_level_) {
                entry_ = id;
                entry_level_ = level;
            }
        }
    }

    // Removes every link pointing at `id` from every other node's slot lists
    // (order within a list is not meaningful in HNSW, so removal compacts by
    // swapping with the last slot). One full O(total links) sweep; the
    // exhaustive form is chosen over heuristic local repair because it
    // leaves no dangling edge by construction. Serial build path only.
    void unlink_everywhere(std::uint32_t id) {
        for (std::uint32_t u = 0u; u < capacity_; ++u) {
            if (u == id || levels_[u] == kNotInserted) {
                continue;
            }
            const std::uint32_t ul = static_cast<std::uint32_t>(levels_[u]);
            for (std::uint32_t l = 0u; l <= ul; ++l) {
                std::uint32_t& cnt = counts_[u][l];
                std::uint32_t* slots = link_slots(u, l);
                std::uint32_t i = 0u;
                while (i < cnt) {
                    if (slots[i] == id) {
                        slots[i] = slots[cnt - 1u];
                        --cnt;
                    } else {
                        ++i;
                    }
                }
            }
        }
    }

    // Shared structural invariants, used by validate_integrity() and the
    // loader: per-layer counts within bounds, every edge pointing at an
    // inserted node that exists on that layer and is never the node itself,
    // and tombstones only on inserted nodes.
    bool graph_invariants_ok() const {
        for (std::uint32_t id = 0u; id < capacity_; ++id) {
            if (levels_[id] == kNotInserted) {
                continue;
            }
            const std::uint32_t lvl = static_cast<std::uint32_t>(levels_[id]);
            if (lvl > kMaxLevel) {
                return false;
            }
            for (std::uint32_t l = 0u; l <= lvl; ++l) {
                if (counts_[id][l] > max_m(l)) {
                    return false;
                }
                const std::size_t offset =
                    (l == 0u) ? 0u
                              : static_cast<std::size_t>(m0_) +
                                    static_cast<std::size_t>(l - 1u) * m_;
                for (std::uint32_t i = 0u; i < counts_[id][l]; ++i) {
                    const std::uint32_t t = links_[id][offset + i];
                    if (t >= capacity_ || t == id ||
                        levels_[t] == kNotInserted ||
                        static_cast<std::uint32_t>(levels_[t]) < l) {
                        return false;
                    }
                }
            }
        }
        for (std::size_t w = 0u; w < deleted_words_; ++w) {
            std::uint64_t bits = deleted_[w];
            while (bits != 0ull) {
                const std::uint64_t low = bits & (0ull - bits);
                const std::uint32_t id = static_cast<std::uint32_t>(
                    w * 64u + static_cast<std::size_t>(__builtin_ctzll(low)));
                if (id >= capacity_ || levels_[id] == kNotInserted) {
                    return false;
                }
                bits ^= low;
            }
        }
        return true;
    }

    const std::uint8_t* vec(std::uint32_t id) const noexcept {
        return vectors_ + static_cast<std::size_t>(id) * record_bytes_;
    }

    std::uint32_t distance_to(const std::uint8_t* q, std::uint32_t id) const noexcept {
        return hamming_distance(q, vec(id), dim_);
    }

    std::uint32_t max_m(std::uint32_t layer) const noexcept {
        return (layer == 0u) ? m0_ : m_;
    }

    void clamp_kef(std::uint32_t& k, std::uint32_t& ef) const noexcept {
        if (k > ef_limit_) {
            k = ef_limit_;
        }
        if (ef < k) {
            ef = k;
        }
        if (ef > ef_limit_) {
            ef = ef_limit_;
        }
    }

    // May this node appear in results? (Routing is never gated by this.)
    bool eligible(std::uint32_t id, const std::uint64_t* allow,
                  bool respect_deleted) const noexcept {
        if (respect_deleted && is_deleted(id)) {
            return false;
        }
        if (allow != nullptr &&
            ((allow[id >> 6u] >> (id & 63u)) & 1ull) == 0ull) {
            return false;
        }
        return true;
    }

    std::uint32_t* link_slots(std::uint32_t id, std::uint32_t layer) noexcept {
        const std::size_t offset =
            (layer == 0u) ? 0u
                          : static_cast<std::size_t>(m0_) +
                                static_cast<std::size_t>(layer - 1u) * m_;
        return links_[id].data() + offset;
    }

    const std::uint32_t* neighbors_of(std::uint32_t id, std::uint32_t layer,
                                      std::uint32_t& n) const noexcept {
        if (layer > static_cast<std::uint32_t>(levels_[id])) {
            n = 0u;
            return nullptr;
        }
        n = counts_[id][layer];
        const std::size_t offset =
            (layer == 0u) ? 0u
                          : static_cast<std::size_t>(m0_) +
                                static_cast<std::size_t>(layer - 1u) * m_;
        return links_[id].data() + offset;
    }

    // Geometric level distribution, level = floor(-ln(u) * mL), capped.
    std::uint32_t random_level() {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        double u = u01(rng_);
        if (u < 1e-12) {
            u = 1e-12; // avoid -ln(0)
        }
        const double lvl = -std::log(u) * ml_;
        const std::uint32_t level = static_cast<std::uint32_t>(lvl);
        return (level > kMaxLevel) ? kMaxLevel : level;
    }

    // Hill-climb to the closest node on `layer`. Zero-allocation. Filters are
    // irrelevant here: descent only routes. Query path: lock-free.
    void greedy_descend(const std::uint8_t* q, std::uint32_t layer,
                        std::uint32_t& ep, std::uint32_t& ep_dist) const noexcept {
        bool improved = true;
        while (improved) {
            improved = false;
            std::uint32_t n = 0u;
            const std::uint32_t* nb = neighbors_of(ep, layer, n);
            for (std::uint32_t i = 0u; i < n; ++i) {
                const std::uint32_t d = distance_to(q, nb[i]);
                if (d < ep_dist) {
                    ep_dist = d;
                    ep = nb[i];
                    improved = true;
                }
            }
        }
    }

    // Snapshot of a node's layer-l neighbor list into the context's bounce
    // buffer, taken under the node's stripe lock. Used only by concurrent
    // construction; distances are then computed with no lock held.
    const std::uint32_t* neighbors_snapshot(SearchContext& ctx,
                                            std::uint32_t id,
                                            std::uint32_t layer,
                                            std::uint32_t& n) const noexcept {
#if defined(EDGEVECTOR_HAS_THREADS)
        std::lock_guard<std::mutex> guard(lock_for(id));
        if (layer > static_cast<std::uint32_t>(levels_[id])) {
            n = 0u;
            return nullptr;
        }
        n = counts_[id][layer];
        const std::size_t offset =
            (layer == 0u) ? 0u
                          : static_cast<std::size_t>(m0_) +
                                static_cast<std::size_t>(layer - 1u) * m_;
        std::memcpy(ctx.neigh_buf_.get(), links_[id].data() + offset,
                    static_cast<std::size_t>(n) * 4u);
        return ctx.neigh_buf_.get();
#else
        (void)ctx;
        return neighbors_of(id, layer, n);
#endif
    }

    // Descent used during construction; takes snapshots when concurrent.
    void greedy_descend_build(SearchContext& ctx, const std::uint8_t* q,
                              std::uint32_t layer, std::uint32_t& ep,
                              std::uint32_t& ep_dist, bool locked) const noexcept {
        if (!locked) {
            greedy_descend(q, layer, ep, ep_dist);
            return;
        }
        bool improved = true;
        while (improved) {
            improved = false;
            std::uint32_t n = 0u;
            const std::uint32_t* nb = neighbors_snapshot(ctx, ep, layer, n);
            for (std::uint32_t i = 0u; i < n; ++i) {
                const std::uint32_t d = distance_to(q, nb[i]);
                if (d < ep_dist) {
                    ep_dist = d;
                    ep = nb[i];
                    improved = true;
                }
            }
        }
    }

    // Upper-layer descent + layer-0 beam, shared by both search modes.
    // Results are left in ctx.res_heap_; returns its size.
    std::uint32_t beam_layer0(SearchContext& ctx, const std::uint8_t* q,
                              std::uint32_t ef,
                              const std::uint64_t* allow) const noexcept {
        std::uint32_t ep = entry_;
        std::uint32_t ep_dist = distance_to(q, ep);
        for (std::uint32_t l = entry_level_; l > 0u; --l) {
            greedy_descend(q, l, ep, ep_dist);
        }
        return search_layer(ctx, q, ep, ef, 0u, allow, true);
    }

    // Beam search on one layer (Algorithm 2), with result-eligibility
    // filtering: ineligible nodes (tombstoned, or cleared in `allow`) still
    // route within the beam bound but never enter the result heap. Results
    // are left in ctx.res_heap_ (a max-heap of at most `ef` best eligible
    // candidates); returns its size. Zero-allocation.
    std::uint32_t search_layer(SearchContext& ctx, const std::uint8_t* q,
                               std::uint32_t ep, std::uint32_t ef,
                               std::uint32_t layer,
                               const std::uint64_t* allow,
                               bool respect_deleted,
                               bool locked = false) const noexcept {
        ctx.next_epoch();
        ctx.cand_heap_.clear();
        ctx.res_heap_.clear();

        const std::uint32_t d0 = distance_to(q, ep);
        ctx.mark_visited(ep);
        ctx.cand_heap_.push(detail::Candidate{d0, ep});
        if (eligible(ep, allow, respect_deleted)) {
            ctx.res_heap_.push(detail::Candidate{d0, ep});
        }

        while (!ctx.cand_heap_.empty()) {
            const detail::Candidate c = ctx.cand_heap_.top();
            if (ctx.res_heap_.size() >= ef &&
                c.dist > ctx.res_heap_.top().dist) {
                break; // nearest unexpanded candidate is worse than the beam
            }
            ctx.cand_heap_.pop();

            std::uint32_t n = 0u;
            const std::uint32_t* nb =
                locked ? neighbors_snapshot(ctx, c.id, layer, n)
                       : neighbors_of(c.id, layer, n);
            for (std::uint32_t i = 0u; i < n; ++i) {
                const std::uint32_t e = nb[i];
                if (ctx.is_visited(e)) {
                    continue;
                }
                ctx.mark_visited(e);
                const std::uint32_t de = distance_to(q, e);
                const bool beam_open =
                    ctx.res_heap_.size() < ef ||
                    de < ctx.res_heap_.top().dist;
                if (beam_open) {
                    ctx.cand_heap_.push(detail::Candidate{de, e});
                    if (eligible(e, allow, respect_deleted)) {
                        ctx.res_heap_.push(detail::Candidate{de, e});
                        if (ctx.res_heap_.size() > ef) {
                            ctx.res_heap_.pop();
                        }
                    }
                }
            }
        }
        return ctx.res_heap_.size();
    }

    // Neighbor-selection heuristic (Algorithm 4): scanning candidates in
    // ascending distance to the base point, keep one only if it is closer to
    // the base than to every neighbor already kept. Reads select_buf_[0..n),
    // which must be sorted ascending; writes ids into selected_.
    //
    // Includes the paper's keepPrunedConnections extension: unfilled slots
    // are backfilled with the pruned candidates, closest first, so node
    // degree never collapses on distributions where the heuristic prunes
    // aggressively. (Measured note: on both iid-random and clustered 512-bit
    // data this backfill did not change recall — the heuristic was already
    // filling the slots — but it is kept as cheap insurance for adversarial
    // distributions. Low recall on iid random data is NOT a graph defect: it
    // is the intrinsic-dimensionality wall every ANN index hits on
    // structureless data; see the benchmark's two scenarios.)
    std::uint32_t select_neighbors(SearchContext& ctx,
                                   std::uint32_t n_candidates,
                                   std::uint32_t mm) {
        std::uint32_t n_sel = 0u;
        for (std::uint32_t i = 0u; i < n_candidates && n_sel < mm; ++i) {
            const detail::Candidate c = ctx.select_buf_[i];
            bool keep = true;
            for (std::uint32_t s = 0u; s < n_sel; ++s) {
                const std::uint32_t d_cs =
                    hamming_distance(vec(c.id), vec(ctx.selected_[s]), dim_);
                if (d_cs < c.dist) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                ctx.selected_[n_sel] = c.id;
                ++n_sel;
            }
        }

        // keepPrunedConnections: top up with the pruned candidates in
        // ascending distance until the slots are full or candidates run out.
        for (std::uint32_t i = 0u; i < n_candidates && n_sel < mm; ++i) {
            const std::uint32_t cand_id = ctx.select_buf_[i].id;
            bool already = false;
            for (std::uint32_t s = 0u; s < n_sel; ++s) {
                if (ctx.selected_[s] == cand_id) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                ctx.selected_[n_sel] = cand_id;
                ++n_sel;
            }
        }
        return n_sel;
    }

    // Add the reverse link from -> to on `layer`; when the slot array is
    // full, re-select the best max_m(layer) from {existing neighbors, to}
    // with the same heuristic, measured from `from`. Build path only. In
    // concurrent mode the whole read-modify-write runs under `from`'s stripe
    // lock (the only lock held).
    void add_link(SearchContext& ctx, std::uint32_t from, std::uint32_t to,
                  std::uint32_t layer, bool locked) {
#if defined(EDGEVECTOR_HAS_THREADS)
        if (locked) {
            std::lock_guard<std::mutex> guard(lock_for(from));
            add_link_unlocked(ctx, from, to, layer);
            return;
        }
#else
        (void)locked;
#endif
        add_link_unlocked(ctx, from, to, layer);
    }

    void add_link_unlocked(SearchContext& ctx, std::uint32_t from,
                           std::uint32_t to, std::uint32_t layer) {
        const std::uint32_t mm = max_m(layer);
        std::uint32_t& cnt = counts_[from][layer];
        std::uint32_t* slots = link_slots(from, layer);

        if (cnt < mm) {
            slots[cnt] = to;
            ++cnt;
            return;
        }

        const std::uint8_t* base = vec(from);
        std::uint32_t n = 0u;
        for (std::uint32_t i = 0u; i < cnt; ++i) {
            ctx.select_buf_[n] =
                detail::Candidate{distance_to(base, slots[i]), slots[i]};
            ++n;
        }
        ctx.select_buf_[n] = detail::Candidate{distance_to(base, to), to};
        ++n;
        std::sort(ctx.select_buf_.get(), ctx.select_buf_.get() + n,
                  detail::candidate_less);

        const std::uint32_t n_sel = select_neighbors(ctx, n, mm);
        for (std::uint32_t i = 0u; i < n_sel; ++i) {
            slots[i] = ctx.selected_[i];
        }
        cnt = n_sel;
    }

    void reset_graph_state() {
        std::memset(levels_.get(), 0xFF, capacity_);
        std::memset(deleted_.get(), 0, deleted_words_ * 8u);
        for (std::uint32_t i = 0u; i < capacity_; ++i) {
            links_[i].clear();
            links_[i].shrink_to_fit();
            counts_[i].clear();
            counts_[i].shrink_to_fit();
        }
        has_entry_ = false;
        entry_ = 0u;
        entry_level_ = 0u;
        size_ = 0u;
        deleted_count_ = 0u;
    }

    static bool read_exact(std::FILE* f, void* dst, std::size_t n) {
        return std::fread(dst, 1u, n, f) == n;
    }

    GraphIoStatus load_graph_body(std::FILE* f) {
        std::uint8_t header[kIoHeaderBytes];
        if (!read_exact(f, header, sizeof(header))) {
            return GraphIoStatus::io_error;
        }
        if (header[0] != static_cast<std::uint8_t>('E') ||
            header[1] != static_cast<std::uint8_t>('V') ||
            header[2] != static_cast<std::uint8_t>('H') ||
            header[3] != static_cast<std::uint8_t>('G')) {
            return GraphIoStatus::bad_magic;
        }
        std::uint32_t version = 0u;
        std::memcpy(&version, header + 4u, 4u);
        if (version == 0u || version > kIoVersion) {
            return GraphIoStatus::bad_version;
        }
        std::uint64_t dim64 = 0u;
        std::uint32_t capacity = 0u;
        std::uint32_t m = 0u;
        std::uint32_t size = 0u;
        std::uint32_t entry = 0u;
        std::uint32_t entry_level = 0u;
        std::uint8_t has_entry = 0u;
        std::memcpy(&dim64, header + 8u, 8u);
        std::memcpy(&capacity, header + 16u, 4u);
        std::memcpy(&m, header + 20u, 4u);
        std::memcpy(&size, header + 24u, 4u);
        std::memcpy(&entry, header + 28u, 4u);
        std::memcpy(&entry_level, header + 32u, 4u);
        std::memcpy(&has_entry, header + 36u, 1u);

        if (dim64 != static_cast<std::uint64_t>(dim_) ||
            capacity != capacity_ || m != m_) {
            return GraphIoStatus::incompatible;
        }
        if (size > capacity_ || has_entry > 1u ||
            (has_entry == 0u && size > 0u) || entry_level > kMaxLevel ||
            (has_entry == 1u && entry >= capacity_)) {
            return GraphIoStatus::corrupt;
        }

        std::uint32_t loaded = 0u;
        for (std::uint32_t id = 0u; id < capacity_; ++id) {
            std::uint8_t lvl = 0u;
            if (!read_exact(f, &lvl, 1u)) {
                return GraphIoStatus::io_error;
            }
            if (lvl == kNotInserted) {
                continue;
            }
            if (static_cast<std::uint32_t>(lvl) > kMaxLevel) {
                return GraphIoStatus::corrupt;
            }
            counts_[id].assign(static_cast<std::size_t>(lvl) + 1u, 0u);
            links_[id].assign(static_cast<std::size_t>(m0_) +
                                  static_cast<std::size_t>(lvl) * m_,
                              0u);
            if (!read_exact(f, counts_[id].data(), counts_[id].size() * 4u) ||
                !read_exact(f, links_[id].data(), links_[id].size() * 4u)) {
                return GraphIoStatus::io_error;
            }
            for (std::uint32_t l = 0u; l <= static_cast<std::uint32_t>(lvl); ++l) {
                if (counts_[id][l] > max_m(l)) {
                    return GraphIoStatus::corrupt;
                }
            }
            levels_[id] = lvl;
            ++loaded;
        }

        if (version >= 2u) {
            if (!read_exact(f, deleted_.get(), deleted_words_ * 8u)) {
                return GraphIoStatus::io_error;
            }
        }

        std::uint8_t trailing = 0u;
        if (loaded != size || std::fread(&trailing, 1u, 1u, f) != 0u) {
            return GraphIoStatus::corrupt; // node count or file length is off
        }

        // Structural validation: referential integrity of every edge and
        // tombstone integrity, shared with validate_integrity().
        if (!graph_invariants_ok()) {
            return GraphIoStatus::corrupt;
        }
        if (has_entry == 1u &&
            (levels_[entry] == kNotInserted ||
             static_cast<std::uint32_t>(levels_[entry]) < entry_level)) {
            return GraphIoStatus::corrupt;
        }

        std::uint32_t n_deleted = 0u;
        for (std::size_t w = 0u; w < deleted_words_; ++w) {
            n_deleted += static_cast<std::uint32_t>(
                __builtin_popcountll(deleted_[w]));
        }

        has_entry_ = (has_entry == 1u);
        entry_ = entry;
        entry_level_ = entry_level;
        size_ = size;
        deleted_count_ = n_deleted;
        return GraphIoStatus::ok;
    }

    // --- immutable configuration -------------------------------------------
    const std::uint8_t* vectors_;
    std::size_t record_bytes_;
    std::size_t dim_;
    std::uint32_t capacity_;
    std::uint32_t m_;
    std::uint32_t m0_;
    std::uint32_t ef_construction_;
    std::uint32_t ef_limit_ = 0u;
    double ml_ = 0.0;
    std::mt19937_64 rng_;

    // --- graph state --------------------------------------------------------
    bool has_entry_ = false;
    std::uint32_t entry_ = 0u;
    std::uint32_t entry_level_ = 0u;
    std::uint32_t size_ = 0u;
    std::uint32_t deleted_count_ = 0u;
    std::unique_ptr<std::uint8_t[]> levels_;
    std::size_t deleted_words_ = 0u;
    std::unique_ptr<std::uint64_t[]> deleted_; // soft-delete tombstones
    std::vector<std::vector<std::uint32_t>> links_;  // per-node flat slot arrays
    std::vector<std::vector<std::uint32_t>> counts_; // per-node per-layer counts

    // --- build-path infrastructure ------------------------------------------
    std::unique_ptr<SearchContext> default_ctx_;
#if defined(EDGEVECTOR_HAS_THREADS)
    std::unique_ptr<std::mutex[]> locks_; // striped per-node link locks
    std::uint32_t lock_mask_ = 0u;
    std::mutex entry_mutex_; // entry point, has_entry_, size_
    std::mutex rng_mutex_;   // level generation and id claims
#endif
};

} // namespace edgevector

#endif // EDGEVECTOR_HNSW_GRAPH_HPP
