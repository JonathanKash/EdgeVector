#ifndef EDGEVECTOR_ITQ_ROTATION_HPP
#define EDGEVECTOR_ITQ_ROTATION_HPP

// ============================================================================
// EdgeVector :: itq_rotation.hpp
//
// Iterative Quantization (ITQ) — Gong & Lazebnik, CVPR 2011 — learns an
// orthogonal rotation G that minimizes the binary quantization error
//
//     sum over samples ||sign(G x) - G x||^2
//
// Sign quantization spends exactly one bit per dimension, so on anisotropic
// data (real embedding spectra decay; energy concentrates in a few
// directions) most bits measure directions that carry almost no signal.
// Rotating first spreads the variance evenly across dimensions, making every
// bit informative. Quantize rotated vectors, search as usual, and rotate each
// query with the same G.
//
// DESIGN DECISION: rotation only — no centering, no PCA projection. An
// orthogonal transform preserves inner products and cosines EXACTLY, so:
//   - float32 ground truth is identical in rotated and original space;
//   - search_exact_reranked() may re-rank with the ORIGINAL floats and the
//     ORIGINAL query even when the index stores rotated codes;
//   - only the quality of the 1-bit codes changes.
// (Classic ITQ centers and PCA-projects first; both would break cosine
// preservation, which this library treats as the ranking contract.)
//
// TRAINING (allocation permitted; offline):
//   Alternate the two optimal closed-form steps of the ITQ objective:
//     B = sign(X W)                       (optimal codes for fixed rotation)
//     W = polar(X^T B)                    (optimal rotation for fixed codes:
//                                          orthogonal Procrustes)
//   The polar factor is computed with the inverse-free Newton–Schulz
//   iteration Q <- Q (3I - Q^T Q) / 2, pre-scaled by a power-iteration
//   estimate of the spectral norm so every singular value starts in (0, 1].
//   All matrix arithmetic runs in double precision; convergence of the polar
//   iteration is checked explicitly and failure is reported, never ignored.
//   The alternation never increases the objective (each half-step is optimal
//   for the other), which the tests verify empirically.
//
// QUERY PATH: rotate() / rotate_quantize() are noexcept, allocation-free,
// O(dim^2) with double accumulators. At dim = 512 that is ~262k multiply-adds
// (~tens of microseconds) per query.
//
// PERSISTENCE (EVRT format, version 1, little-endian):
//   Bytes  0.. 3  char magic[4] = { 'E','V','R','T' }
//   Bytes  4.. 7  u32  version  = 1
//   Bytes  8..15  u64  dim
//   Then dim*dim float32, row-major G (y = G x).
//   load() validates magic, version, dimension, exact file length, finiteness
//   of every entry, and orthogonality (max |G^T G - I| < 1e-2), and leaves
//   the object as the identity rotation on any failure.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>

#include "edgevector/quantize_math.hpp"

namespace edgevector {

enum class ItqStatus : std::uint8_t {
    ok = 0,
    bad_input,     // null data, dimension mismatch, or too few samples
    not_converged, // polar iteration failed; rotation reset to identity
    io_error,      // open/read/write failed
    bad_magic,
    bad_version,
    incompatible,  // file's dim differs from this object's dim
    corrupt        // wrong length, non-finite entries, or not orthogonal
};

class ItqRotation {
public:
    // Starts as the identity rotation (rotate() is then a copy).
    explicit ItqRotation(std::size_t dim)
        : dim_(dim), g_(new float[dim * dim]) {
        set_identity();
    }

    ItqRotation(const ItqRotation&) = delete;
    ItqRotation& operator=(const ItqRotation&) = delete;
    ItqRotation(ItqRotation&&) noexcept = default;
    ItqRotation& operator=(ItqRotation&&) noexcept = default;

    std::size_t dim() const noexcept { return dim_; }
    const float* matrix() const noexcept { return g_.get(); } // row-major, y = G x

    // Diagnostics from the last successful train(): mean per-component
    // squared quantization error |sign(y) - y|^2, before (identity rotation)
    // and after training, and the final orthogonality residual.
    double error_before_training() const noexcept { return err_before_; }
    double error_after_training() const noexcept { return err_after_; }
    double orthogonality_error() const noexcept { return orth_err_; }

    // Learns the rotation from `n` row-major samples (n x dim floats).
    // Requires n >= dim (>= 10*dim recommended). On ANY failure the rotation
    // is reset to identity and the status says why. Allocation permitted:
    // this is an offline build step.
    ItqStatus train(const float* data, std::size_t n, std::size_t dim,
                    std::uint32_t iterations = 20, std::uint64_t seed = 42u) {
        if (data == nullptr || dim != dim_ || n < dim_ || iterations == 0u) {
            return ItqStatus::bad_input;
        }
        const std::size_t d = dim_;

        // Baseline error with the identity rotation.
        err_before_ = quantization_error(data, n, nullptr);

        // Working buffers (double precision for all matrix arithmetic).
        std::unique_ptr<double[]> w(new double[d * d]);   // current rotation, y-row = x-row * W
        std::unique_ptr<double[]> m(new double[d * d]);   // X^T B
        std::unique_ptr<double[]> q(new double[d * d]);   // polar iterate
        std::unique_ptr<double[]> t1(new double[d * d]);
        std::unique_ptr<double[]> t2(new double[d * d]);
        std::unique_ptr<float[]> codes(new float[n * d]); // B in {-1,+1}
        std::unique_ptr<double[]> row(new double[d]);

        random_orthogonal(w.get(), d, seed);

        for (std::uint32_t it = 0u; it < iterations; ++it) {
            // B = sign(X W), one sample row at a time.
            for (std::size_t r = 0; r < n; ++r) {
                const float* x = data + r * d;
                float* b = codes.get() + r * d;
                for (std::size_t j = 0; j < d; ++j) {
                    row[j] = 0.0;
                }
                for (std::size_t i = 0; i < d; ++i) {
                    const double xi = static_cast<double>(x[i]);
                    const double* wrow = w.get() + i * d;
                    for (std::size_t j = 0; j < d; ++j) {
                        row[j] += xi * wrow[j];
                    }
                }
                for (std::size_t j = 0; j < d; ++j) {
                    b[j] = (row[j] > 0.0) ? 1.0f : -1.0f;
                }
            }

            // M = X^T B.
            std::memset(m.get(), 0, d * d * sizeof(double));
            for (std::size_t r = 0; r < n; ++r) {
                const float* x = data + r * d;
                const float* b = codes.get() + r * d;
                for (std::size_t i = 0; i < d; ++i) {
                    const double xi = static_cast<double>(x[i]);
                    if (xi == 0.0) {
                        continue;
                    }
                    double* mrow = m.get() + i * d;
                    for (std::size_t j = 0; j < d; ++j) {
                        mrow[j] += xi * static_cast<double>(b[j]);
                    }
                }
            }

            // W = polar(M): the orthogonal Procrustes solution.
            if (!polar_newton_schulz(m.get(), w.get(), q.get(), t1.get(),
                                     t2.get(), d)) {
                set_identity();
                return ItqStatus::not_converged;
            }
        }

        // Adopt: G = W^T (so that y = G x matches y-row = x-row * W).
        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                g_[i * d + j] = static_cast<float>(w[j * d + i]);
            }
        }

        err_after_ = quantization_error(data, n, g_.get());
        orth_err_ = orthogonality_residual(g_.get(), d);
        if (!(orth_err_ < 1e-3)) { // also catches NaN
            set_identity();
            return ItqStatus::not_converged;
        }
        return ItqStatus::ok;
    }

    // y = G x. Zero allocation, noexcept; `out` must not alias `x`.
    void rotate(const float* x, float* out) const noexcept {
        const std::size_t d = dim_;
        for (std::size_t i = 0; i < d; ++i) {
            const float* grow = g_.get() + i * d;
            double acc = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                acc += static_cast<double>(grow[j]) * static_cast<double>(x[j]);
            }
            out[i] = static_cast<float>(acc);
        }
    }

    // rotate + sign-quantize in one call. `scratch` holds dim floats and must
    // not alias `x`; `dst` needs padded_bytes(dim) bytes, 8-byte aligned.
    void rotate_quantize(const float* x, float* scratch,
                         std::uint8_t* dst) const noexcept {
        rotate(x, scratch);
        quantize(scratch, dim_, dst);
    }

    ItqStatus save(const char* path) const {
        if (path == nullptr) {
            return ItqStatus::io_error;
        }
        std::FILE* f = std::fopen(path, "wb");
        if (f == nullptr) {
            return ItqStatus::io_error;
        }
        std::uint8_t header[16];
        std::memset(header, 0, sizeof(header));
        header[0] = static_cast<std::uint8_t>('E');
        header[1] = static_cast<std::uint8_t>('V');
        header[2] = static_cast<std::uint8_t>('R');
        header[3] = static_cast<std::uint8_t>('T');
        const std::uint32_t version = 1u;
        const std::uint64_t dim64 = static_cast<std::uint64_t>(dim_);
        std::memcpy(header + 4u, &version, 4u);
        std::memcpy(header + 8u, &dim64, 8u);

        bool ok = std::fwrite(header, 1u, sizeof(header), f) == sizeof(header);
        if (ok) {
            const std::size_t n_floats = dim_ * dim_;
            ok = std::fwrite(g_.get(), sizeof(float), n_floats, f) == n_floats;
        }
        if (std::fclose(f) != 0) {
            ok = false;
        }
        return ok ? ItqStatus::ok : ItqStatus::io_error;
    }

    // On any failure the object is reset to the identity rotation.
    ItqStatus load(const char* path) {
        set_identity();
        if (path == nullptr) {
            return ItqStatus::io_error;
        }
        std::FILE* f = std::fopen(path, "rb");
        if (f == nullptr) {
            return ItqStatus::io_error;
        }
        const ItqStatus st = load_body(f);
        std::fclose(f);
        if (st != ItqStatus::ok) {
            set_identity();
        }
        return st;
    }

private:
    void set_identity() noexcept {
        std::memset(g_.get(), 0, dim_ * dim_ * sizeof(float));
        for (std::size_t i = 0; i < dim_; ++i) {
            g_[i * dim_ + i] = 1.0f;
        }
    }

    // Mean per-component squared quantization error; g == nullptr means
    // identity rotation.
    double quantization_error(const float* data, std::size_t n,
                              const float* g) const {
        const std::size_t d = dim_;
        std::unique_ptr<float[]> y(new float[d]);
        double total = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            const float* x = data + r * d;
            const float* v = x;
            if (g != nullptr) {
                for (std::size_t i = 0; i < d; ++i) {
                    const float* grow = g + i * d;
                    double acc = 0.0;
                    for (std::size_t j = 0; j < d; ++j) {
                        acc += static_cast<double>(grow[j]) *
                               static_cast<double>(x[j]);
                    }
                    y[i] = static_cast<float>(acc);
                }
                v = y.get();
            }
            for (std::size_t i = 0; i < d; ++i) {
                const double s = (v[i] > 0.0f) ? 1.0 : -1.0;
                const double e = s - static_cast<double>(v[i]);
                total += e * e;
            }
        }
        return total / (static_cast<double>(n) * static_cast<double>(d));
    }

    static double orthogonality_residual(const float* g, std::size_t d) {
        // max |G^T G - I|, computed in double.
        double worst = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = i; j < d; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < d; ++k) {
                    acc += static_cast<double>(g[k * d + i]) *
                           static_cast<double>(g[k * d + j]);
                }
                const double target = (i == j) ? 1.0 : 0.0;
                const double e = std::fabs(acc - target);
                if (!(e <= worst)) { // catches NaN too
                    worst = std::isnan(e) ? 1e30 : e;
                }
            }
        }
        return worst;
    }

    // Random orthogonal matrix: Gaussian entries, then two passes of modified
    // Gram-Schmidt over the rows (the second pass mops up the rounding the
    // first leaves behind).
    static void random_orthogonal(double* w, std::size_t d, std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> gauss(0.0, 1.0);
        for (std::size_t i = 0; i < d * d; ++i) {
            w[i] = gauss(rng);
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t i = 0; i < d; ++i) {
                double* wi = w + i * d;
                for (std::size_t p = 0; p < i; ++p) {
                    const double* wp = w + p * d;
                    double dot = 0.0;
                    for (std::size_t k = 0; k < d; ++k) {
                        dot += wi[k] * wp[k];
                    }
                    for (std::size_t k = 0; k < d; ++k) {
                        wi[k] -= dot * wp[k];
                    }
                }
                double norm = 0.0;
                for (std::size_t k = 0; k < d; ++k) {
                    norm += wi[k] * wi[k];
                }
                norm = std::sqrt(norm);
                if (norm < 1e-12) { // measure-zero; keep deterministic anyway
                    for (std::size_t k = 0; k < d; ++k) {
                        wi[k] = (k == i) ? 1.0 : 0.0;
                    }
                    continue;
                }
                for (std::size_t k = 0; k < d; ++k) {
                    wi[k] /= norm;
                }
            }
        }
    }

    // C = A^T A (A d x d, row-major).
    static void mat_ata(const double* a, double* c, std::size_t d) {
        std::memset(c, 0, d * d * sizeof(double));
        for (std::size_t k = 0; k < d; ++k) {
            const double* ak = a + k * d;
            for (std::size_t i = 0; i < d; ++i) {
                const double aki = ak[i];
                if (aki == 0.0) {
                    continue;
                }
                double* ci = c + i * d;
                for (std::size_t j = 0; j < d; ++j) {
                    ci[j] += aki * ak[j];
                }
            }
        }
    }

    // C = A * B (all d x d, row-major).
    static void mat_mul(const double* a, const double* b, double* c,
                        std::size_t d) {
        std::memset(c, 0, d * d * sizeof(double));
        for (std::size_t i = 0; i < d; ++i) {
            const double* ai = a + i * d;
            double* ci = c + i * d;
            for (std::size_t k = 0; k < d; ++k) {
                const double aik = ai[k];
                if (aik == 0.0) {
                    continue;
                }
                const double* bk = b + k * d;
                for (std::size_t j = 0; j < d; ++j) {
                    ci[j] += aik * bk[j];
                }
            }
        }
    }

    // Spectral norm estimate of M via power iteration on M^T M.
    static double spectral_norm(const double* m, std::size_t d,
                                double* v, double* u) {
        for (std::size_t i = 0; i < d; ++i) {
            v[i] = 1.0 / std::sqrt(static_cast<double>(d));
        }
        double sigma = 0.0;
        for (int it = 0; it < 50; ++it) {
            // u = M v
            for (std::size_t i = 0; i < d; ++i) {
                const double* mi = m + i * d;
                double acc = 0.0;
                for (std::size_t j = 0; j < d; ++j) {
                    acc += mi[j] * v[j];
                }
                u[i] = acc;
            }
            // v = M^T u
            for (std::size_t j = 0; j < d; ++j) {
                v[j] = 0.0;
            }
            for (std::size_t i = 0; i < d; ++i) {
                const double* mi = m + i * d;
                const double ui = u[i];
                for (std::size_t j = 0; j < d; ++j) {
                    v[j] += mi[j] * ui;
                }
            }
            double norm = 0.0;
            for (std::size_t j = 0; j < d; ++j) {
                norm += v[j] * v[j];
            }
            norm = std::sqrt(norm);
            if (norm == 0.0) {
                return 0.0;
            }
            sigma = std::sqrt(norm); // ||M^T M v|| -> sigma^2 at convergence
            for (std::size_t j = 0; j < d; ++j) {
                v[j] /= norm;
            }
        }
        return sigma;
    }

    // out = polar factor of m (the U V^T of its SVD), via Newton-Schulz:
    //   Q <- Q (3I - Q^T Q) / 2
    // after scaling m so every singular value lies in (0, 1]. Quadratic
    // convergence near the fixed point; returns false if the residual
    // max|Q^T Q - I| fails to fall below 1e-9 within 100 iterations (a
    // rank-deficient M — degenerate training data — lands here).
    static bool polar_newton_schulz(const double* m, double* out, double* q,
                                    double* t1, double* t2, std::size_t d) {
        const double sigma = spectral_norm(m, d, t1, t2);
        if (!(sigma > 0.0) || std::isnan(sigma)) {
            return false;
        }
        const double inv = 1.0 / (sigma * 1.01); // strictly inside (0, 1)
        for (std::size_t i = 0; i < d * d; ++i) {
            q[i] = m[i] * inv;
        }

        for (int it = 0; it < 100; ++it) {
            mat_ata(q, t1, d); // t1 = Q^T Q
            double res = 0.0;
            for (std::size_t i = 0; i < d; ++i) {
                for (std::size_t j = 0; j < d; ++j) {
                    const double target = (i == j) ? 1.0 : 0.0;
                    const double e = std::fabs(t1[i * d + j] - target);
                    if (e > res) {
                        res = e;
                    }
                }
            }
            if (std::isnan(res)) {
                return false;
            }
            if (res < 1e-9) {
                std::memcpy(out, q, d * d * sizeof(double));
                return true;
            }
            // t1 <- 3I - Q^T Q ; Q <- Q t1 / 2
            for (std::size_t i = 0; i < d; ++i) {
                for (std::size_t j = 0; j < d; ++j) {
                    t1[i * d + j] = ((i == j) ? 3.0 : 0.0) - t1[i * d + j];
                }
            }
            mat_mul(q, t1, t2, d);
            for (std::size_t i = 0; i < d * d; ++i) {
                q[i] = 0.5 * t2[i];
            }
        }
        return false;
    }

    ItqStatus load_body(std::FILE* f) {
        std::uint8_t header[16];
        if (std::fread(header, 1u, sizeof(header), f) != sizeof(header)) {
            return ItqStatus::io_error;
        }
        if (header[0] != static_cast<std::uint8_t>('E') ||
            header[1] != static_cast<std::uint8_t>('V') ||
            header[2] != static_cast<std::uint8_t>('R') ||
            header[3] != static_cast<std::uint8_t>('T')) {
            return ItqStatus::bad_magic;
        }
        std::uint32_t version = 0u;
        std::uint64_t dim64 = 0u;
        std::memcpy(&version, header + 4u, 4u);
        std::memcpy(&dim64, header + 8u, 8u);
        if (version != 1u) {
            return ItqStatus::bad_version;
        }
        if (dim64 != static_cast<std::uint64_t>(dim_)) {
            return ItqStatus::incompatible;
        }

        const std::size_t n_floats = dim_ * dim_;
        if (std::fread(g_.get(), sizeof(float), n_floats, f) != n_floats) {
            return ItqStatus::io_error;
        }
        std::uint8_t trailing = 0u;
        if (std::fread(&trailing, 1u, 1u, f) != 0u) {
            return ItqStatus::corrupt; // file longer than the format allows
        }
        for (std::size_t i = 0; i < n_floats; ++i) {
            if (!std::isfinite(g_[i])) {
                return ItqStatus::corrupt;
            }
        }
        orth_err_ = orthogonality_residual(g_.get(), dim_);
        if (!(orth_err_ < 1e-2)) {
            return ItqStatus::corrupt; // not an orthogonal matrix
        }
        return ItqStatus::ok;
    }

    std::size_t dim_;
    std::unique_ptr<float[]> g_; // row-major, y = G x
    double err_before_ = 0.0;
    double err_after_ = 0.0;
    double orth_err_ = 0.0;
};

} // namespace edgevector

#endif // EDGEVECTOR_ITQ_ROTATION_HPP
