# PLAN: Fix statistically invalid test cases 4 and 5 in tests/test_quantize_math.cpp

## Context (read first)

Project: EdgeVector, header-only C++17 vector DB. Ground truth rules are in
CLAUDE.md — read it. The library module `include/edgevector/quantize_math.hpp`
is CORRECT and MUST NOT be modified. Do not touch the public API, the bit
layout, or any function body in that header.

Current test run (`make -C tests run`, g++ under MSYS2/MinGW) fails 2 checks:

```
[4] Cosine parity vs float32 baseline
      mean absolute error = 0.042813  (gate < 0.05)   PASS
      pearson correlation = 0.638379  (gate > 0.98)   FAIL
[5] Ranking preservation
      top-10 overlap = 2 / 10  (gate >= 8)            FAIL
```

Root cause: the test harness draws INDEPENDENT Gaussian pairs in 512-dim, so
all true cosine similarities sit in a band of std ~= 1/sqrt(512) ~= 0.044
around zero, below the SimHash estimator noise floor (~0.07 at theta = pi/2).
The correlation gate is unachievable by construction; the kernel is fine. The
fix is to generate test data spanning the full similarity range. Only
`tests/test_quantize_math.cpp` changes. Nothing else.

## Task 1 — Rewrite Case 4 data generation (test_cosine_parity)

Keep: dim = 512, 1000 pairs, `std::mt19937 rng(42)`, the Pearson helper, the
printed MAE / correlation lines, and the existing check() reporting style.

Replace the pair generation with controlled-similarity construction:

1. Draw `a` as 512 iid samples from `std::normal_distribution<float>(0,1)`.
2. Draw target similarity `alpha` from `std::uniform_real_distribution<float>(-1.0f, 1.0f)`.
3. Draw an independent Gaussian vector `g`, project out the `a` component to
   get `g_perp` (Gram-Schmidt: g_perp = g - (dot(g,a)/dot(a,a)) * a), then
   normalize both `a_hat = a/||a||` and `g_hat = g_perp/||g_perp||`.
4. Set `b[i] = alpha * a_hat[i] + sqrt(1 - alpha*alpha) * g_hat[i]`.
   Now cosine(a, b) == alpha (up to float rounding), spanning [-1, 1].
5. Quantize both, compute hamming, compare `approx_cosine_from_hamming` against
   `cosine_similarity_f32` exactly as the current code does.

Gates (unchanged semantics, do NOT loosen them to make the test pass):
- mean absolute error < 0.05
- Pearson correlation > 0.98

Expected result with correct data generation: MAE ~= 0.03-0.045, r >= 0.99.
If the gates do not pass after your rewrite, the bug is in your harness math
(most likely: forgetting to normalize, or reusing `a` unnormalized in step 4)
— fix the harness, never the gates and never the library header. If after
genuine effort you believe a gate is truly unattainable, STOP and report why
instead of changing it.

## Task 2 — Rewrite Case 5 data generation (test_ranking_preservation)

Keep: dim = 512, 200 candidates, top_k = 10, seed 42, the two sort directions
(cosine descending, hamming ascending), the overlap computation and printout.

Replace candidate generation with a planted-relevance design:

1. Build the query `q`, normalize to `q_hat`.
2. Candidates 0..9 ("relevant"): alpha values evenly spaced in [0.70, 0.90]
   (e.g. alpha_c = 0.70 + 0.02222f * c). Construct each candidate with the same
   Gram-Schmidt mixture as Task 1: `cand = alpha * q_hat + sqrt(1-alpha^2) * g_hat`
   with a fresh `g` per candidate.
3. Candidates 10..199 ("background"): alpha drawn from
   `std::uniform_real_distribution<float>(0.0f, 0.30f)`, same construction.
4. The float32-cosine top-10 is then the planted set by a wide margin, and a
   correct Hamming kernel must recover it through quantization noise (~0.07),
   since the similarity gap between 0.70 and 0.30 is ~10x the noise floor.

Gate: overlap >= 9 of 10 (tighten from 8 — the planted design has enough
margin that 9 is a robust requirement). Update the printed gate text to match.

## Task 3 — Update stale comments

Adjust the comment blocks above cases 4 and 5 so they describe the
controlled-similarity construction, not "random pairs". Keep comment style and
density consistent with the rest of the file.

## Task 4 — Rebuild and retest until green

From `tests/`: run `make clean` then `make run` (g++, flags already in the
Makefile — do not change the Makefile). Iterate on the HARNESS ONLY until the
binary prints all PASS and exits 0. Paste the full final output of `make run`
in your report.

## Hard constraints (violations = rejection by the Overseer)

- Only file modified: `tests/test_quantize_math.cpp` (plus the rebuilt binary).
- `include/edgevector/quantize_math.hpp` and `tests/Makefile`: byte-identical.
- No test framework, no new dependencies, C++17, must compile clean under
  `-Wall -Wextra -Werror`.
- Gates stay as written above. Report, don't loosen, if unattainable.
