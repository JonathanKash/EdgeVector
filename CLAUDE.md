# EdgeVector: Local-First Vector Database
**Version:** 0.1.0
**Architecture:** Header-only C++17 Library
**Primary Niche:** Low-latency, ultra-low memory vector search for Edge AI devices using Binary Quantization and HNSW.

## 1. System Architecture
This is a high-performance C++ systems project. The engine relies on applying vector space principles and matrix algebra to execute rapid distance calculations. The system is broken into three core modules:
*   **`mmap_storage.hpp`**: Zero-copy disk I/O handling POSIX memory mapping.
*   **`quantize_math.hpp`**: Mathematical transformations converting float32 arrays into binary vectors, executing distance metrics using hardware SIMD intrinsics (`__builtin_popcountll`).
*   **`hnsw_graph.hpp`**: The probabilistic skip-list logic, min-heaps for node traversal, and Layer 0 neighborhood routing.

## 2. Agentic Workflow Conventions
This project utilizes a multi-agent workflow supervised by an Overseer (Fable). 
*   **Worker Agents (Claude Code):** Responsible for implementing isolated modules and writing unit tests. Workers must NEVER alter the global architecture without Overseer approval.
*   **State Management:** Workers must check their own code by writing a lightweight test script and compiling it before reporting task completion.

## 3. Strict C++ Engineering Rules
*   **Zero-Allocation Critical Path:** Once the HNSW graph is loaded, memory allocation (`new`, `malloc`, `std::vector::push_back`) is strictly forbidden during the search query path. All query memory must be pre-allocated.
*   **Header-Only:** All implementation logic must reside in `.hpp` files. We are building a drop-in library, not a shared `.so` object.
*   **Build System:** Utilize standard C++ Makefiles for all testing and compilation workflows. Ensure `-O3` and `-march=native` flags are enforced to activate AVX/SIMD instructions.
*   **Data Types:** Strictly enforce `std::uint8_t` for quantized vectors and `std::uint64_t` for internal SIMD bitwise operations. Avoid implicit type casting.

## 4. Performance & Testing Targets
*   **Memory:** The RAM footprint of a quantized index must be at least 30x smaller than the equivalent float32 index. 
*   **Validation:** Every module requires an isolated test within a `tests/` directory. The math module must prove 100% mathematical parity (within an acceptable quantization error margin) against a standard float32 cosine similarity baseline.