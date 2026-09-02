# EdgeVector :: unit test build.
# The library is header-only, so each test translation unit is a whole build.
#
# Cross-compilation / emulation knobs:
#   ARCH   - architecture flags (default: -march=native). For aarch64 use
#            e.g. `make ARCH=-march=armv8-a CXX=aarch64-linux-gnu-g++`.
#   RUNNER - command prefix for executing test binaries (default: none). For
#            QEMU user-mode: `RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu"`.

CXX ?= g++
ARCH ?= -march=native
CXXFLAGS := -std=c++17 -O3 $(ARCH) -Wall -Wextra -Werror
INCLUDES := -I../include
RUNNER ?=

# Release configuration: identical flags plus -DNDEBUG, which strips the
# asserts from the query hot path (MMapStorage::vector and the HNSW heaps).
# This is the configuration all latency benchmarks must use. Release binaries
# get a _release suffix so both configurations can coexist in the tree.
RELEASE_FLAGS := $(CXXFLAGS) -DNDEBUG

# MinGW g++ appends .exe on Windows; name the target to match so the rule is
# not considered out of date on every invocation.
ifeq ($(OS),Windows_NT)
    EXE := .exe
else
    EXE :=
endif

TARGET := test_quantize_math$(EXE)
SRC := test_quantize_math.cpp
HEADERS := ../include/edgevector/quantize_math.hpp

STORAGE_TARGET := test_mmap_storage$(EXE)
STORAGE_SRC := test_mmap_storage.cpp
STORAGE_HEADERS := ../include/edgevector/mmap_storage.hpp \
                   ../include/edgevector/quantize_math.hpp

HNSW_TARGET := test_hnsw_graph$(EXE)
HNSW_SRC := test_hnsw_graph.cpp
HNSW_HEADERS := ../include/edgevector/hnsw_graph.hpp \
                ../include/edgevector/quantize_math.hpp

INTEG_TARGET := test_integration$(EXE)
INTEG_SRC := test_integration.cpp
INTEG_HEADERS := ../include/edgevector/hnsw_graph.hpp \
                 ../include/edgevector/mmap_storage.hpp \
                 ../include/edgevector/quantize_math.hpp

ITQ_TARGET := test_itq_rotation$(EXE)
ITQ_SRC := test_itq_rotation.cpp
ITQ_HEADERS := ../include/edgevector/itq_rotation.hpp \
               ../include/edgevector/hnsw_graph.hpp \
               ../include/edgevector/quantize_math.hpp

FUZZ_TARGET := test_format_fuzz$(EXE)
FUZZ_SRC := test_format_fuzz.cpp
FUZZ_HEADERS := $(wildcard ../include/edgevector/*.hpp)

RELEASE_TARGETS := test_quantize_math_release$(EXE) \
                   test_mmap_storage_release$(EXE) \
                   test_hnsw_graph_release$(EXE) \
                   test_integration_release$(EXE) \
                   test_itq_rotation_release$(EXE) \
                   test_format_fuzz_release$(EXE)

.PHONY: all run release run-release bench clean

all: $(TARGET) $(STORAGE_TARGET) $(HNSW_TARGET) $(INTEG_TARGET) $(ITQ_TARGET) $(FUZZ_TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

$(STORAGE_TARGET): $(STORAGE_SRC) $(STORAGE_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(STORAGE_SRC) -o $(STORAGE_TARGET)

$(HNSW_TARGET): $(HNSW_SRC) $(HNSW_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(HNSW_SRC) -o $(HNSW_TARGET)

$(INTEG_TARGET): $(INTEG_SRC) $(INTEG_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTEG_SRC) -o $(INTEG_TARGET)

$(ITQ_TARGET): $(ITQ_SRC) $(ITQ_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ITQ_SRC) -o $(ITQ_TARGET)

$(FUZZ_TARGET): $(FUZZ_SRC) $(FUZZ_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(FUZZ_SRC) -o $(FUZZ_TARGET)

# make stops at the first recipe line that exits non-zero, so this fails fast.
run: $(TARGET) $(STORAGE_TARGET) $(HNSW_TARGET) $(INTEG_TARGET) $(ITQ_TARGET) $(FUZZ_TARGET)
	$(RUNNER) ./$(TARGET)
	$(RUNNER) ./$(STORAGE_TARGET)
	$(RUNNER) ./$(HNSW_TARGET)
	$(RUNNER) ./$(INTEG_TARGET)
	$(RUNNER) ./$(ITQ_TARGET)
	$(RUNNER) ./$(FUZZ_TARGET)

# --- release (-DNDEBUG) -----------------------------------------------------

release: $(RELEASE_TARGETS)

test_quantize_math_release$(EXE): $(SRC) $(HEADERS)
	$(CXX) $(RELEASE_FLAGS) $(INCLUDES) $(SRC) -o $@

test_mmap_storage_release$(EXE): $(STORAGE_SRC) $(STORAGE_HEADERS)
	$(CXX) $(RELEASE_FLAGS) $(INCLUDES) $(STORAGE_SRC) -o $@

test_hnsw_graph_release$(EXE): $(HNSW_SRC) $(HNSW_HEADERS)
	$(CXX) $(RELEASE_FLAGS) $(INCLUDES) $(HNSW_SRC) -o $@

test_integration_release$(EXE): $(INTEG_SRC) $(INTEG_HEADERS)
	$(CXX) $(RELEASE_FLAGS) $(INCLUDES) $(INTEG_SRC) -o $@

test_itq_rotation_release$(EXE): $(ITQ_SRC) $(ITQ_HEADERS)
	$(CXX) $(RELEASE_FLAGS) $(INCLUDES) $(ITQ_SRC) -o $@

test_format_fuzz_release$(EXE): $(FUZZ_SRC) $(FUZZ_HEADERS)
	$(CXX) $(RELEASE_FLAGS) $(INCLUDES) $(FUZZ_SRC) -o $@

run-release: $(RELEASE_TARGETS)
	$(RUNNER) ./test_quantize_math_release$(EXE)
	$(RUNNER) ./test_mmap_storage_release$(EXE)
	$(RUNNER) ./test_hnsw_graph_release$(EXE)
	$(RUNNER) ./test_integration_release$(EXE)
	$(RUNNER) ./test_itq_rotation_release$(EXE)
	$(RUNNER) ./test_format_fuzz_release$(EXE)

# --- benchmark (always -DNDEBUG: assert-enabled numbers are meaningless) ----

BENCH_TARGET := benchmark_100k$(EXE)

$(BENCH_TARGET): benchmark_100k.cpp $(HNSW_HEADERS) ../include/edgevector/itq_rotation.hpp
	$(CXX) $(RELEASE_FLAGS) $(INCLUDES) benchmark_100k.cpp -o $@

bench: $(BENCH_TARGET)
	$(RUNNER) ./$(BENCH_TARGET)

clean:
	rm -f $(TARGET) $(STORAGE_TARGET) $(HNSW_TARGET) $(INTEG_TARGET) \
	      $(ITQ_TARGET) $(FUZZ_TARGET) $(RELEASE_TARGETS) $(BENCH_TARGET)
