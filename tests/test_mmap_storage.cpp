// ============================================================================
// Isolated unit test for edgevector/mmap_storage.hpp
//
// Plain main() + hand-rolled checks, no test framework. Returns non-zero if
// any case fails. Test scaffolding is allowed to allocate; the zero-allocation
// rule binds the library query path, not this harness.
//
// Every file this test creates lives in the current working directory and is
// deleted before main() returns. Mappings are always closed before the backing
// file is removed, because Windows refuses to delete a mapped file.
// ============================================================================

#include "edgevector/mmap_storage.hpp"
#include "edgevector/quantize_math.hpp"

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

const char* status_name(edgevector::StorageStatus s) {
    switch (s) {
        case edgevector::StorageStatus::ok:           return "ok";
        case edgevector::StorageStatus::io_error:     return "io_error";
        case edgevector::StorageStatus::bad_magic:    return "bad_magic";
        case edgevector::StorageStatus::bad_version:  return "bad_version";
        case edgevector::StorageStatus::corrupt_size: return "corrupt_size";
    }
    return "unknown";
}

// Test file names, all relative to the current working directory.
const char* const kMainFile      = "ev_test_index.evec";
const char* const kEmptyFile     = "ev_test_empty.evec";
const char* const kBadMagicFile  = "ev_test_badmagic.evec";
const char* const kBadVerFile    = "ev_test_badver.evec";
const char* const kTruncFile     = "ev_test_trunc.evec";
const char* const kMissingFile   = "ev_test_does_not_exist.evec";

// Contiguous, 8-byte-aligned block of `count` quantized records. Backing the
// storage with std::uint64_t guarantees the alignment the library expects.
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

// Fills `block` with `count` quantized random vectors of dimension `dim`.
void build_random_block(RecordBlock& block, std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> raw(dim);
    for (std::size_t i = 0; i < block.count(); ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = dist(rng);
        }
        edgevector::quantize(raw.data(), dim, block.record(i));
    }
}

bool read_whole_file(const char* path, std::vector<std::uint8_t>& out) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    out.clear();
    std::uint8_t buffer[4096];
    std::size_t got = 0;
    while ((got = std::fread(buffer, 1u, sizeof(buffer), f)) > 0u) {
        out.insert(out.end(), buffer, buffer + got);
    }
    std::fclose(f);
    return true;
}

bool write_whole_file(const char* path, const std::uint8_t* data, std::size_t n) {
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = (n == 0u) || (std::fwrite(data, 1u, n, f) == n);
    return (std::fclose(f) == 0) && ok;
}

// ---------------------------------------------------------------------------
// Cases 1-3 share one written file: round trip, distance parity, alignment.
// ---------------------------------------------------------------------------
void test_round_trip_parity_alignment() {
    const std::size_t dim = 100;   // not a multiple of 64: exercises tail padding
    const std::size_t count = 50;

    std::mt19937 rng(42);
    RecordBlock original(dim, count);
    build_random_block(original, dim, rng);

    std::printf("[1] Round-trip integrity (dim = 100, count = 50)\n");

    const edgevector::StorageStatus wrote = edgevector::write_storage_file(
        kMainFile, static_cast<std::uint64_t>(dim),
        static_cast<std::uint64_t>(count), original.base());
    check(wrote == edgevector::StorageStatus::ok, "write_storage_file returns ok");

    edgevector::MMapStorage storage;
    const edgevector::StorageStatus opened = storage.open(kMainFile);

    char msg[160];
    std::snprintf(msg, sizeof(msg), "open returns ok (got %s)", status_name(opened));
    check(opened == edgevector::StorageStatus::ok, msg);
    check(storage.is_open(), "is_open() == true after successful open");
    check(storage.dim() == dim, "dim() == 100");
    check(storage.count() == count, "count() == 50");
    check(storage.record_bytes() == edgevector::padded_bytes(dim),
          "record_bytes() == padded_bytes(100) == 16");

    bool all_identical = true;
    for (std::size_t i = 0; i < count; ++i) {
        if (std::memcmp(storage.vector(i), original.record(i),
                        original.record_bytes()) != 0) {
            all_identical = false;
        }
    }
    check(all_identical, "all 50 records byte-identical to the originals");

    // -- Case 2 -------------------------------------------------------------
    std::printf("[2] Distance parity through the mapping (20 pairs)\n");

    std::uniform_int_distribution<std::size_t> pick(0u, count - 1u);
    bool all_match = true;
    std::uint32_t last_mapped = 0u;
    std::uint32_t last_memory = 0u;
    for (int t = 0; t < 20; ++t) {
        const std::size_t i = pick(rng);
        const std::size_t j = pick(rng);

        last_mapped = edgevector::hamming_distance(
            storage.vector(i), storage.vector(j), dim);
        last_memory = edgevector::hamming_distance(
            original.record(i), original.record(j), dim);

        if (last_mapped != last_memory) {
            all_match = false;
        }
    }
    std::snprintf(msg, sizeof(msg),
                  "mapped vs in-memory Hamming identical for 20 pairs "
                  "(last: %u vs %u)", last_mapped, last_memory);
    check(all_match, msg);

    // -- Case 3 -------------------------------------------------------------
    std::printf("[3] Alignment of every mapped record\n");

    bool all_aligned = true;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uintptr_t address =
            reinterpret_cast<std::uintptr_t>(storage.vector(i));
        if ((address % 8u) != 0u) {
            all_aligned = false;
        }
    }
    check(all_aligned, "every record address is 8-byte aligned");

    storage.close();
    check(!storage.is_open(), "is_open() == false after close()");
}

// ---------------------------------------------------------------------------
// Case 4: header validation
// ---------------------------------------------------------------------------
void test_header_validation() {
    std::printf("[4] Header validation\n");

    std::vector<std::uint8_t> good;
    if (!read_whole_file(kMainFile, good) || good.size() < 64u) {
        check(false, "could not read the reference file back for corruption");
        return;
    }

    char msg[160];

    // -- flipped magic byte -> bad_magic ------------------------------------
    {
        std::vector<std::uint8_t> bytes = good;
        bytes[0] = static_cast<std::uint8_t>('X');
        write_whole_file(kBadMagicFile, bytes.data(), bytes.size());

        edgevector::MMapStorage s;
        const edgevector::StorageStatus st = s.open(kBadMagicFile);
        std::snprintf(msg, sizeof(msg), "flipped magic byte -> bad_magic (got %s)",
                      status_name(st));
        check(st == edgevector::StorageStatus::bad_magic, msg);
        check(!s.is_open(), "  after bad_magic: is_open() == false");
        check(s.count() == 0u, "  after bad_magic: count() == 0");
    }

    // -- version = 2 -> bad_version ----------------------------------------
    {
        std::vector<std::uint8_t> bytes = good;
        const std::uint32_t wrong_version = 2u;
        std::memcpy(bytes.data() + 4, &wrong_version, sizeof(wrong_version));
        write_whole_file(kBadVerFile, bytes.data(), bytes.size());

        edgevector::MMapStorage s;
        const edgevector::StorageStatus st = s.open(kBadVerFile);
        std::snprintf(msg, sizeof(msg), "version = 2 -> bad_version (got %s)",
                      status_name(st));
        check(st == edgevector::StorageStatus::bad_version, msg);
        check(!s.is_open(), "  after bad_version: is_open() == false");
        check(s.count() == 0u, "  after bad_version: count() == 0");
    }

    // -- truncated by one byte -> corrupt_size ------------------------------
    {
        std::vector<std::uint8_t> bytes = good;
        bytes.pop_back();
        write_whole_file(kTruncFile, bytes.data(), bytes.size());

        edgevector::MMapStorage s;
        const edgevector::StorageStatus st = s.open(kTruncFile);
        std::snprintf(msg, sizeof(msg),
                      "truncated by 1 byte -> corrupt_size (got %s)",
                      status_name(st));
        check(st == edgevector::StorageStatus::corrupt_size, msg);
        check(!s.is_open(), "  after corrupt_size: is_open() == false");
        check(s.count() == 0u, "  after corrupt_size: count() == 0");
    }

    // -- nonexistent path -> io_error ---------------------------------------
    {
        edgevector::MMapStorage s;
        const edgevector::StorageStatus st = s.open(kMissingFile);
        std::snprintf(msg, sizeof(msg), "nonexistent path -> io_error (got %s)",
                      status_name(st));
        check(st == edgevector::StorageStatus::io_error, msg);
        check(!s.is_open(), "  after io_error: is_open() == false");
        check(s.count() == 0u, "  after io_error: count() == 0");
    }
}

// ---------------------------------------------------------------------------
// Case 5: empty index
// ---------------------------------------------------------------------------
void test_empty_index() {
    std::printf("[5] Empty index (count = 0)\n");

    const edgevector::StorageStatus wrote =
        edgevector::write_storage_file(kEmptyFile, 100u, 0u, nullptr);
    check(wrote == edgevector::StorageStatus::ok,
          "write_storage_file with count = 0 returns ok");

    edgevector::MMapStorage storage;
    const edgevector::StorageStatus st = storage.open(kEmptyFile);

    char msg[160];
    std::snprintf(msg, sizeof(msg), "open of empty index returns ok (got %s)",
                  status_name(st));
    check(st == edgevector::StorageStatus::ok, msg);
    check(storage.is_open(), "is_open() == true for an empty index");
    check(storage.count() == 0u, "count() == 0");
    check(storage.dim() == 100u, "dim() == 100 on an empty index");

    storage.close();
}

// ---------------------------------------------------------------------------
// Case 6: move semantics
// ---------------------------------------------------------------------------
void test_move_semantics() {
    std::printf("[6] Move semantics\n");

    const std::size_t dim = 100;

    edgevector::MMapStorage source;
    if (source.open(kMainFile) != edgevector::StorageStatus::ok) {
        check(false, "could not reopen the reference file for the move test");
        return;
    }

    const std::uint64_t expected_count = source.count();
    const std::uint32_t expected_first =
        edgevector::hamming_distance(source.vector(0), source.vector(1), dim);

    // -- move construction --------------------------------------------------
    edgevector::MMapStorage moved(std::move(source));

    check(!source.is_open(), "moved-from source: is_open() == false");
    check(source.count() == 0u, "moved-from source: count() == 0");
    check(source.dim() == 0u, "moved-from source: dim() == 0");
    check(source.record_bytes() == 0u, "moved-from source: record_bytes() == 0");

    check(moved.is_open(), "move-constructed destination: is_open() == true");
    check(moved.count() == expected_count,
          "move-constructed destination: count() preserved");
    check(moved.dim() == dim, "move-constructed destination: dim() preserved");
    check(edgevector::hamming_distance(moved.vector(0), moved.vector(1), dim) ==
              expected_first,
          "move-constructed destination serves identical data");

    // -- move assignment ----------------------------------------------------
    edgevector::MMapStorage assigned;
    assigned = std::move(moved);
    check(!moved.is_open(), "move-assigned source: is_open() == false");
    check(assigned.is_open() && assigned.count() == expected_count,
          "move-assigned destination serves correct data");

    // -- double close is safe -----------------------------------------------
    assigned.close();
    assigned.close();
    check(!assigned.is_open(), "double close() is safe and idempotent");

    source.close(); // closing an already-empty, moved-from object
    check(!source.is_open(), "close() on a moved-from object is safe");
}

void cleanup() {
    std::remove(kMainFile);
    std::remove(kEmptyFile);
    std::remove(kBadMagicFile);
    std::remove(kBadVerFile);
    std::remove(kTruncFile);
}

} // namespace

int main() {
    std::printf("=== EdgeVector :: mmap_storage unit tests ===\n\n");

    test_round_trip_parity_alignment();
    test_header_validation();
    test_empty_index();
    test_move_semantics();

    cleanup();

    std::printf("\n=== %s ===\n",
                (g_failures == 0) ? "ALL CASES PASSED" : "FAILURES DETECTED");
    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
    }
    return (g_failures == 0) ? 0 : 1;
}
