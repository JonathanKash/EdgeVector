#ifndef EDGEVECTOR_HNSW_GRAPH_HPP
#define EDGEVECTOR_HNSW_GRAPH_HPP

// ============================================================================
// EdgeVector :: hnsw_graph.hpp
//
// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016) over
// binary-quantized vectors, using the Hamming kernel from quantize_math.hpp.
//
// MEMORY MODEL
// ------------
// The graph does NOT own vector data. It is constructed over an external,
// contiguous, 8-byte-aligned block of quantized records (an MMapStorage
// mapping or an in-memory buffer): record i lives at
// `vectors + i * record_bytes`, laid out per quantize_math.hpp. Node ids are
// indices into that block. Capacity is fixed at construction.
//
// Per-node links are stored flat: one std::uint32_t array per node holding
// M0 = 2*M slots for layer 0 followed by M slots for each upper layer the
// node participates in. Node levels are drawn from the standard geometric
// distribution with mL = 1 / ln(M), capped at 31.
//
// ZERO-ALLOCATION QUERY PATH (CLAUDE.md section 3)
// ------------------------------------------------
// All query-time working memory is pre-allocated at construction:
//   - a visited-epoch array (O(1) reset by bumping an epoch counter; the
//     array is memset only on 32-bit epoch wraparound, which is not an
//     allocation),
//   - a fixed-capacity candidate min-heap sized capacity+1 (each node enters
//     it at most once per search, so it cannot overflow),
//   - a fixed-capacity result max-heap sized ef_limit+1.
// search() and everything it calls perform no heap allocation, no
// std::vector::push_back, and no system calls; heaps are raw arrays driven
// by std::push_heap/std::pop_heap with explicit size counters. insert() is
// the build path, where allocation is permitted (per-node link arrays are
// sized exactly once, at insertion).
//
// Ties are broken by (distance, id) everywhere, so results are deterministic
// and comparable against a brute-force baseline using the same ordering.
//
// THREADING: not thread-safe. The scratch pools are owned by the graph, so
// one graph instance supports one query at a time.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "edgevector/quantize_math.hpp"

namespace edgevector {

struct SearchResult {
    std::uint32_t id;
    std::uint32_t distance;
};

namespace detail {

struct Candidate {
    std::uint32_t dist;
    std::uint32_t id;
};

// Total order: ascending by distance, then by id. Using it on both the graph
// and the brute-force baseline makes tie handling deterministic.
inline bool candidate_less(const Candidate& a, const Candidate& b) noexcept {
    return (a.dist != b.dist) ? (a.dist < b.dist) : (a.id < b.id);
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

class HNSWGraph {
public:
    // `vectors`: base of the contiguous quantized block (external, outlives
    // the graph). `record_bytes` >= padded_bytes(dim), 8-byte aligned records.
    // Scratch pools and link tables are sized here; capacity is final.
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

        visited_.reset(new std::uint32_t[capacity_]);
        std::memset(visited_.get(), 0, static_cast<std::size_t>(capacity_) * 4u);

        // Candidate heap bound: every node is pushed at most once per search
        // (guarded by the visited set), so capacity+1 can never overflow.
        cand_buf_.reset(new detail::Candidate[static_cast<std::size_t>(capacity_) + 1u]);
        res_buf_.reset(new detail::Candidate[static_cast<std::size_t>(ef_limit_) + 1u]);
        select_buf_.reset(new detail::Candidate[static_cast<std::size_t>(ef_limit_) +
                                                static_cast<std::size_t>(m0_) + 2u]);
        selected_.reset(new std::uint32_t[static_cast<std::size_t>(m0_) + 1u]);

        cand_heap_.init(cand_buf_.get(), capacity_ + 1u);
        res_heap_.init(res_buf_.get(), ef_limit_ + 1u);

        links_.resize(capacity_);
        counts_.resize(capacity_);
    }

    HNSWGraph(const HNSWGraph&) = delete;
    HNSWGraph& operator=(const HNSWGraph&) = delete;

    std::uint32_t size() const noexcept { return size_; }
    std::uint32_t capacity() const noexcept { return capacity_; }
    std::size_t dim() const noexcept { return dim_; }
    std::uint32_t max_ef() const noexcept { return ef_limit_; }

    // BUILD PATH (allocation permitted). Inserts the vector at index `id` of
    // the external block. Each id may be inserted once; returns false for an
    // out-of-range or duplicate id.
    bool insert(std::uint32_t id) {
        if (id >= capacity_ || levels_[id] != kNotInserted) {
            return false;
        }

        const std::uint32_t level = random_level();
        links_[id].assign(static_cast<std::size_t>(m0_) +
                              static_cast<std::size_t>(level) * m_,
                          0u);
        counts_[id].assign(static_cast<std::size_t>(level) + 1u, 0u);
        levels_[id] = static_cast<std::uint8_t>(level);

        if (!has_entry_) {
            entry_ = id;
            entry_level_ = level;
            has_entry_ = true;
            ++size_;
            return true;
        }

        const std::uint8_t* q = vec(id);
        std::uint32_t ep = entry_;
        std::uint32_t ep_dist = distance_to(q, ep);

        for (std::uint32_t l = entry_level_; l > level; --l) {
            greedy_descend(q, l, ep, ep_dist);
        }

        const std::uint32_t top = (entry_level_ < level) ? entry_level_ : level;
        for (std::uint32_t l = top;; --l) {
            const std::uint32_t found =
                search_layer(q, ep, ef_construction_, l);

            const detail::Candidate* raw = res_heap_.data();
            for (std::uint32_t i = 0u; i < found; ++i) {
                select_buf_[i] = raw[i];
            }
            std::sort(select_buf_.get(), select_buf_.get() + found,
                      detail::candidate_less);

            // Entry point for the next layer down: the closest thing found
            // here. Saved now because add_link() reuses select_buf_.
            ep = select_buf_[0].id;
            ep_dist = select_buf_[0].dist;

            const std::uint32_t mm = max_m(l);
            const std::uint32_t n_sel = select_neighbors(found, mm);

            std::uint32_t* slots = link_slots(id, l);
            for (std::uint32_t i = 0u; i < n_sel; ++i) {
                slots[i] = selected_[i];
            }
            counts_[id][l] = n_sel;

            for (std::uint32_t i = 0u; i < n_sel; ++i) {
                add_link(slots[i], id, l);
            }

            if (l == 0u) {
                break;
            }
        }

        if (level > entry_level_) {
            entry_ = id;
            entry_level_ = level;
        }
        ++size_;
        return true;
    }

    // QUERY CRITICAL PATH: zero allocation, no system calls, noexcept.
    // Writes up to k results into `out`, ascending by (distance, id), and
    // returns how many were written. `ef` is the layer-0 beam width, clamped
    // to [k, max_ef()]; k itself is clamped to max_ef().
    std::uint32_t search(const std::uint8_t* q,
                         std::uint32_t k,
                         std::uint32_t ef,
                         SearchResult* out) noexcept {
        if (!has_entry_ || k == 0u || q == nullptr || out == nullptr) {
            return 0u;
        }
        if (k > ef_limit_) {
            k = ef_limit_;
        }
        if (ef < k) {
            ef = k;
        }
        if (ef > ef_limit_) {
            ef = ef_limit_;
        }

        std::uint32_t ep = entry_;
        std::uint32_t ep_dist = distance_to(q, ep);
        for (std::uint32_t l = entry_level_; l > 0u; --l) {
            greedy_descend(q, l, ep, ep_dist);
        }

        std::uint32_t found = search_layer(q, ep, ef, 0u);
        while (found > k) { // keep only the k best
            res_heap_.pop();
            --found;
        }
        for (std::uint32_t i = found; i > 0u; --i) { // max-heap pops worst first
            const detail::Candidate c = res_heap_.top();
            res_heap_.pop();
            out[i - 1u] = SearchResult{c.id, c.dist};
        }
        return found;
    }

    // Bytes of pre-allocated + per-node graph memory (links, counts, levels,
    // visited set, heaps). For footprint reporting; not on any hot path.
    std::size_t graph_memory_bytes() const noexcept {
        std::size_t total = 0u;
        for (std::uint32_t i = 0u; i < capacity_; ++i) {
            total += links_[i].capacity() * sizeof(std::uint32_t);
            total += counts_[i].capacity() * sizeof(std::uint32_t);
        }
        total += static_cast<std::size_t>(capacity_);       // levels_
        total += static_cast<std::size_t>(capacity_) * 4u;  // visited_
        total += (static_cast<std::size_t>(capacity_) + 1u) * sizeof(detail::Candidate);
        total += (static_cast<std::size_t>(ef_limit_) + 1u) * sizeof(detail::Candidate);
        return total;
    }

private:
    static constexpr std::uint8_t kNotInserted = 0xFFu;
    static constexpr std::uint32_t kMaxLevel = 31u;

    const std::uint8_t* vec(std::uint32_t id) const noexcept {
        return vectors_ + static_cast<std::size_t>(id) * record_bytes_;
    }

    std::uint32_t distance_to(const std::uint8_t* q, std::uint32_t id) const noexcept {
        return hamming_distance(q, vec(id), dim_);
    }

    std::uint32_t max_m(std::uint32_t layer) const noexcept {
        return (layer == 0u) ? m0_ : m_;
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

    // Hill-climb to the closest node on `layer`. Zero-allocation.
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

    // Beam search on one layer (Algorithm 2). Results are left in res_heap_
    // (a max-heap of at most `ef` best candidates); returns its size.
    // Zero-allocation: heaps and the visited set are pre-allocated pools.
    std::uint32_t search_layer(const std::uint8_t* q, std::uint32_t ep,
                               std::uint32_t ef, std::uint32_t layer) noexcept {
        next_epoch();
        cand_heap_.clear();
        res_heap_.clear();

        const std::uint32_t d0 = distance_to(q, ep);
        mark_visited(ep);
        cand_heap_.push(detail::Candidate{d0, ep});
        res_heap_.push(detail::Candidate{d0, ep});

        while (!cand_heap_.empty()) {
            const detail::Candidate c = cand_heap_.top();
            if (res_heap_.size() >= ef && c.dist > res_heap_.top().dist) {
                break; // nearest unexpanded candidate is worse than the beam
            }
            cand_heap_.pop();

            std::uint32_t n = 0u;
            const std::uint32_t* nb = neighbors_of(c.id, layer, n);
            for (std::uint32_t i = 0u; i < n; ++i) {
                const std::uint32_t e = nb[i];
                if (is_visited(e)) {
                    continue;
                }
                mark_visited(e);
                const std::uint32_t de = distance_to(q, e);
                if (res_heap_.size() < ef) {
                    cand_heap_.push(detail::Candidate{de, e});
                    res_heap_.push(detail::Candidate{de, e});
                } else if (de < res_heap_.top().dist) {
                    cand_heap_.push(detail::Candidate{de, e});
                    res_heap_.push(detail::Candidate{de, e});
                    res_heap_.pop();
                }
            }
        }
        return res_heap_.size();
    }

    // Neighbor-selection heuristic (Algorithm 4): scanning candidates in
    // ascending distance to the base point, keep one only if it is closer to
    // the base than to every neighbor already kept. Reads select_buf_[0..n),
    // which must be sorted ascending; writes ids into selected_.
    std::uint32_t select_neighbors(std::uint32_t n_candidates, std::uint32_t mm) {
        std::uint32_t n_sel = 0u;
        for (std::uint32_t i = 0u; i < n_candidates && n_sel < mm; ++i) {
            const detail::Candidate c = select_buf_[i];
            bool keep = true;
            for (std::uint32_t s = 0u; s < n_sel; ++s) {
                const std::uint32_t d_cs =
                    hamming_distance(vec(c.id), vec(selected_[s]), dim_);
                if (d_cs < c.dist) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                selected_[n_sel] = c.id;
                ++n_sel;
            }
        }
        return n_sel;
    }

    // Add the reverse link from -> to on `layer`; when the slot array is
    // full, re-select the best max_m(layer) from {existing neighbors, to}
    // with the same heuristic, measured from `from`. Build path only.
    void add_link(std::uint32_t from, std::uint32_t to, std::uint32_t layer) {
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
            select_buf_[n] = detail::Candidate{distance_to(base, slots[i]), slots[i]};
            ++n;
        }
        select_buf_[n] = detail::Candidate{distance_to(base, to), to};
        ++n;
        std::sort(select_buf_.get(), select_buf_.get() + n,
                  detail::candidate_less);

        const std::uint32_t n_sel = select_neighbors(n, mm);
        for (std::uint32_t i = 0u; i < n_sel; ++i) {
            slots[i] = selected_[i];
        }
        cnt = n_sel;
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
    std::unique_ptr<std::uint8_t[]> levels_;
    std::vector<std::vector<std::uint32_t>> links_;  // per-node flat slot arrays
    std::vector<std::vector<std::uint32_t>> counts_; // per-node per-layer counts

    // --- pre-allocated query scratch (the zero-allocation pools) -----------
    std::unique_ptr<std::uint32_t[]> visited_;
    std::uint32_t epoch_ = 0u;
    std::unique_ptr<detail::Candidate[]> cand_buf_;
    std::unique_ptr<detail::Candidate[]> res_buf_;
    std::unique_ptr<detail::Candidate[]> select_buf_; // build path only
    std::unique_ptr<std::uint32_t[]> selected_;       // build path only
    detail::FixedHeap<detail::MinHeapCmp> cand_heap_;
    detail::FixedHeap<detail::MaxHeapCmp> res_heap_;
};

} // namespace edgevector

#endif // EDGEVECTOR_HNSW_GRAPH_HPP
