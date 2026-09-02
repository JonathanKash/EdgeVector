#ifndef EDGEVECTOR_EDGEVECTOR_HPP
#define EDGEVECTOR_EDGEVECTOR_HPP

// ============================================================================
// EdgeVector :: edgevector.hpp - the single-include umbrella header.
//
//     #include <edgevector/edgevector.hpp>
//
// pulls in the whole library:
//   quantize_math.hpp - binary quantization, Hamming kernel, asymmetric scorer
//   mmap_storage.hpp  - zero-copy memory-mapped vector files (EVEC format)
//   hnsw_graph.hpp    - the HNSW index: build, search, persistence, dynamics
//   itq_rotation.hpp  - optional learned rotation for anisotropic data
//
// Each header also stands alone; include only what you use if you prefer.
// See examples/quickstart.cpp for a complete, runnable tour.
// ============================================================================

#define EDGEVECTOR_VERSION_MAJOR 0
#define EDGEVECTOR_VERSION_MINOR 8
#define EDGEVECTOR_VERSION_PATCH 0
#define EDGEVECTOR_VERSION "0.8.0"

#include "edgevector/quantize_math.hpp"
#include "edgevector/mmap_storage.hpp"
#include "edgevector/hnsw_graph.hpp"
#include "edgevector/itq_rotation.hpp"

#endif // EDGEVECTOR_EDGEVECTOR_HPP
