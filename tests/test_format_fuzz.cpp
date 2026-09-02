// ============================================================================
// Deterministic fuzz test for the three on-disk format loaders.
//
// The loaders (EVEC vectors, EVHG graphs, EVRT rotations) validate hostile
// input, but until now only against hand-written corruptions. This harness
// generates thousands of seeded random mutations of valid files - byte
// flips, random overwrites, truncations, extensions, biased toward both the
// header and the body - and asserts the loader CONTRACT on every one:
//
//   1. Never crash (run under ASan/UBSan in CI, that check has teeth).
//   2. Always return a status.
//   3. A rejected load leaves the object EMPTY (storage closed, graph empty
//      and integrity-valid, rotation reset to identity).
//   4. An ACCEPTED load - mutations can hit only padding or be no-ops -
//      must yield an object whose own invariants hold (mapped sizes
//      consistent, graph passes the full referential-integrity sweep,
//      rotation orthogonal), and reading its data must stay in bounds.
//
// Deterministic seed, so any failure is reproducible by iteration number.
// ============================================================================

#include "edgevector/edgevector.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
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

const char* const kFuzzFile = "ev_fuzz_input.bin";

bool write_bytes(const std::vector<std::uint8_t>& bytes) {
    std::FILE* f = std::fopen(kFuzzFile, "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = bytes.empty() ||
                    std::fwrite(bytes.data(), 1u, bytes.size(), f) ==
                        bytes.size();
    return (std::fclose(f) == 0) && ok;
}

// Applies 1..8 random mutations to `bytes`. Mutation sites are biased half
// the time toward the first 64 bytes (the headers) and half toward the body,
// so both the header validation and the record/payload validation get
// exercised deeply.
void mutate(std::vector<std::uint8_t>& bytes, std::mt19937& rng) {
    std::uniform_int_distribution<int> n_muts(1, 8);
    std::uniform_int_distribution<int> kind(0, 4);
    const int rounds = n_muts(rng);
    for (int r = 0; r < rounds; ++r) {
        if (bytes.empty()) {
            bytes.push_back(static_cast<std::uint8_t>(rng()));
            continue;
        }
        std::uniform_int_distribution<std::size_t> header_pos(
            0u, bytes.size() < 64u ? bytes.size() - 1u : 63u);
        std::uniform_int_distribution<std::size_t> any_pos(0u,
                                                           bytes.size() - 1u);
        const std::size_t pos =
            (rng() & 1u) ? header_pos(rng) : any_pos(rng);
        switch (kind(rng)) {
            case 0: // flip one bit
                bytes[pos] ^= static_cast<std::uint8_t>(1u << (rng() % 8u));
                break;
            case 1: // random byte
                bytes[pos] = static_cast<std::uint8_t>(rng());
                break;
            case 2: // truncate to a random shorter length
                bytes.resize(any_pos(rng));
                break;
            case 3: { // extend with random bytes
                std::uniform_int_distribution<std::size_t> extra(1u, 64u);
                const std::size_t n = extra(rng);
                for (std::size_t i = 0; i < n; ++i) {
                    bytes.push_back(static_cast<std::uint8_t>(rng()));
                }
                break;
            }
            case 4: { // overwrite a 4-byte window with a random word
                std::uint32_t w = rng();
                const std::size_t span =
                    (bytes.size() - pos < 4u) ? bytes.size() - pos : 4u;
                std::memcpy(bytes.data() + pos, &w, span);
                break;
            }
        }
    }
}

std::vector<std::uint8_t> read_file(const char* path) {
    std::vector<std::uint8_t> bytes;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        return bytes;
    }
    std::uint8_t buf[4096];
    std::size_t got = 0;
    while ((got = std::fread(buf, 1u, sizeof(buf), f)) > 0u) {
        bytes.insert(bytes.end(), buf, buf + got);
    }
    std::fclose(f);
    return bytes;
}

// ---------------------------------------------------------------------------
// EVEC: the vector-storage format.
// ---------------------------------------------------------------------------
void fuzz_storage(int iterations) {
    std::printf("[1] EVEC storage loader (%d mutated files)\n", iterations);

    const std::size_t dim = 64;
    const std::uint64_t count = 20;
    const std::size_t rb = edgevector::padded_bytes(dim);
    std::vector<std::uint64_t> words((rb / 8u) * count, 0u);
    auto* base = reinterpret_cast<std::uint8_t*>(words.data());
    std::mt19937 rng(101);
    for (std::size_t i = 0; i < count * rb; ++i) {
        base[i] = static_cast<std::uint8_t>(rng());
    }
    // Zero the padding-free layout is irrelevant here: the loader validates
    // structure, not bit patterns.
    if (edgevector::write_storage_file(kFuzzFile, dim, count, base) !=
        edgevector::StorageStatus::ok) {
        check(false, "reference EVEC file written");
        return;
    }
    const std::vector<std::uint8_t> valid = read_file(kFuzzFile);

    int accepted = 0;
    bool contract_held = true;
    for (int it = 0; it < iterations && contract_held; ++it) {
        std::vector<std::uint8_t> bytes = valid;
        mutate(bytes, rng);
        if (!write_bytes(bytes)) {
            continue;
        }

        edgevector::MMapStorage s;
        const edgevector::StorageStatus st = s.open(kFuzzFile);
        if (st == edgevector::StorageStatus::ok) {
            ++accepted;
            // Accepted: the object's own invariants must hold, and reading
            // the extremes of the mapping must stay in bounds (ASan-checked).
            if (!s.is_open() ||
                s.record_bytes() != edgevector::padded_bytes(
                                        static_cast<std::size_t>(s.dim()))) {
                contract_held = false;
            } else if (s.count() > 0u) {
                volatile std::uint8_t sink = 0;
                sink += s.vector(0)[0];
                sink += s.vector(s.count() - 1u)[s.record_bytes() - 1u];
                (void)sink;
            }
        } else {
            if (s.is_open() || s.count() != 0u) {
                contract_held = false; // rejected load must leave it closed
            }
        }
        if (!contract_held) {
            std::printf("      contract violated at iteration %d\n", it);
        }
    }
    std::printf("      accepted %d of %d mutants (rest rejected cleanly)\n",
                accepted, iterations);
    check(contract_held, "EVEC loader contract held for every mutant");
}

// ---------------------------------------------------------------------------
// EVHG: the graph format.
// ---------------------------------------------------------------------------
void fuzz_graph(int iterations) {
    std::printf("[2] EVHG graph loader (%d mutated files)\n", iterations);

    const std::size_t dim = 64;
    const std::uint32_t cap = 50;
    const std::size_t rb = edgevector::padded_bytes(dim);
    std::vector<std::uint64_t> words((rb / 8u) * cap, 0u);
    auto* base = reinterpret_cast<std::uint8_t*>(words.data());
    std::mt19937 rng(103);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> raw(dim);
    for (std::uint32_t i = 0; i < cap; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            raw[d] = gauss(rng);
        }
        edgevector::quantize(raw.data(), dim, base + i * rb);
    }

    edgevector::HNSWGraph builder(base, rb, dim, cap, 8u, 50u, 64u, 7u);
    for (std::uint32_t i = 0; i < cap; ++i) {
        builder.insert(i);
    }
    builder.remove(3u); // a tombstone, so the v2 bitmap is non-trivial
    if (builder.save_graph(kFuzzFile) != edgevector::GraphIoStatus::ok) {
        check(false, "reference EVHG file written");
        return;
    }
    const std::vector<std::uint8_t> valid = read_file(kFuzzFile);

    int accepted = 0;
    bool contract_held = true;
    for (int it = 0; it < iterations && contract_held; ++it) {
        std::vector<std::uint8_t> bytes = valid;
        mutate(bytes, rng);
        if (!write_bytes(bytes)) {
            continue;
        }

        edgevector::HNSWGraph g(base, rb, dim, cap, 8u, 50u, 64u, 7u);
        const edgevector::GraphIoStatus st = g.load_graph(kFuzzFile);
        if (st == edgevector::GraphIoStatus::ok) {
            ++accepted;
            // Anything the loader accepts must be a structurally valid graph.
            if (!g.validate_integrity()) {
                contract_held = false;
            }
        } else {
            // Anything rejected must leave the graph empty (and empty is
            // trivially integrity-valid).
            if (g.size() != 0u || g.deleted_count() != 0u ||
                !g.validate_integrity()) {
                contract_held = false;
            }
        }
        if (!contract_held) {
            std::printf("      contract violated at iteration %d\n", it);
        }
    }
    std::printf("      accepted %d of %d mutants (rest rejected cleanly)\n",
                accepted, iterations);
    check(contract_held, "EVHG loader contract held for every mutant");
}

// ---------------------------------------------------------------------------
// EVRT: the rotation format.
// ---------------------------------------------------------------------------
void fuzz_rotation(int iterations) {
    std::printf("[3] EVRT rotation loader (%d mutated files)\n", iterations);

    const std::size_t dim = 32;
    std::mt19937 rng(107);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> data(400 * dim);
    for (float& v : data) {
        v = gauss(rng);
    }
    edgevector::ItqRotation trained(dim);
    if (trained.train(data.data(), 400, dim, 10u, 7u) !=
            edgevector::ItqStatus::ok ||
        trained.save(kFuzzFile) != edgevector::ItqStatus::ok) {
        check(false, "reference EVRT file written");
        return;
    }
    const std::vector<std::uint8_t> valid = read_file(kFuzzFile);

    std::vector<float> probe(dim, 1.0f);
    std::vector<float> out(dim);

    int accepted = 0;
    bool contract_held = true;
    for (int it = 0; it < iterations && contract_held; ++it) {
        std::vector<std::uint8_t> bytes = valid;
        mutate(bytes, rng);
        if (!write_bytes(bytes)) {
            continue;
        }

        edgevector::ItqRotation r(dim);
        const edgevector::ItqStatus st = r.load(kFuzzFile);
        if (st == edgevector::ItqStatus::ok) {
            ++accepted;
            // Accepted: must be (numerically) orthogonal.
            if (!(r.orthogonality_error() < 1e-2)) {
                contract_held = false;
            }
        } else {
            // Rejected: must have reset to the identity.
            r.rotate(probe.data(), out.data());
            for (std::size_t d = 0; d < dim; ++d) {
                if (out[d] != probe[d]) {
                    contract_held = false;
                    break;
                }
            }
        }
        if (!contract_held) {
            std::printf("      contract violated at iteration %d\n", it);
        }
    }
    std::printf("      accepted %d of %d mutants (rest rejected cleanly)\n",
                accepted, iterations);
    check(contract_held, "EVRT loader contract held for every mutant");
}

} // namespace

int main() {
    std::printf("=== EdgeVector :: format-loader fuzz tests ===\n\n");

    fuzz_storage(2000);
    fuzz_graph(2000);
    fuzz_rotation(2000);
    std::remove(kFuzzFile);

    std::printf("\n=== %s ===\n",
                (g_failures == 0) ? "ALL CASES PASSED" : "FAILURES DETECTED");
    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
    }
    return (g_failures == 0) ? 0 : 1;
}
