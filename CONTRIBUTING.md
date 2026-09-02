# Contributing to EdgeVector

Contributions are welcome. The bar here is unusual in one specific way:
**every claim must be measured and every property must be enforced by a
test.** A change that makes the code faster needs a number; a change that
touches the query path needs the allocation gate to still pass; a new file
format field needs hostile-input coverage.

## Build and test

```sh
make -C tests run          # 6 suites, asserts enabled
make -C tests run-release  # same under -DNDEBUG
make -C examples run       # the self-checking quickstart
make -C tests bench        # benchmarks (pass clustered|random|itq|million)
```

Requires GCC or Clang with C++17 (MSVC is unsupported: the code uses
`__builtin_popcountll`/`__builtin_prefetch`). Cross/emulated runs use the
Makefile knobs `ARCH`, `CXX`, `RUNNER`; `tests/arm64.Dockerfile` reproduces
the ARM validation locally.

## The rules the code holds itself to

- **Zero allocation, zero syscalls, `noexcept` on the query path.** The test
  suites replace the global `operator new` and fail on a single allocation in
  any search mode. If your change allocates at query time, it will not merge.
- **No aliasing tricks**: bytes cross into wider types via `std::memcpy`
  only; on-disk bytes are never `reinterpret_cast` to structs.
- **Every failure is a status value**, never an exception; a failed load
  leaves the object empty. The format loaders are fuzzed
  (`tests/test_format_fuzz.cpp`) — new format fields need to survive it.
- **Deterministic ordering**: (distance, id) tie-breaks everywhere, so
  results are reproducible and brute-force-comparable.
- **`-Wall -Wextra -Werror` everywhere**, including under sanitizers.
- Kernel optimizations must be **bit-identical** to a naive reference and
  gated by an equivalence test.

## CI

Every PR runs seven jobs: Linux gcc + clang, native arm64 (with an
on-silicon benchmark), Windows MinGW, the quickstart via make + CMake,
ASan+UBSan, and ThreadSanitizer on the concurrency suite. Green CI is
necessary but not sufficient — a reviewer will also ask what you measured.

## PRs

Branch from `main`, keep commits explanatory (this repo's history reads as
an engineering narrative — including negative results; a falsified
hypothesis documented honestly is a welcome commit message), and update the
README's numbers only with fresh measurements, stating the hardware.
