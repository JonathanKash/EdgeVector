# EdgeVector :: unit test build for the quantize_math module.
# The library is header-only, so the test translation unit is the whole build.

CXX ?= g++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra -Werror
INCLUDES := -I../include

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

RELEASE_TARGETS := test_quantize_math_release$(EXE) \
                   test_mmap_storage_release$(EXE) \
                   test_hnsw_graph_release$(EXE) \
                   test_integration_release$(EXE)

.PHONY: all run release run-release clean

all: $(TARGET) $(STORAGE_TARGET) $(HNSW_TARGET) $(INTEG_TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

$(STORAGE_TARGET): $(STORAGE_SRC) $(STORAGE_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(STORAGE_SRC) -o $(STORAGE_TARGET)

$(HNSW_TARGET): $(HNSW_SRC) $(HNSW_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(HNSW_SRC) -o $(HNSW_TARGET)

$(INTEG_TARGET): $(INTEG_SRC) $(INTEG_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(INTEG_SRC) -o $(INTEG_TARGET)

# make stops at the first recipe line that exits non-zero, so this fails fast.
run: $(TARGET) $(STORAGE_TARGET) $(HNSW_TARGET) $(INTEG_TARGET)
	./$(TARGET)
	./$(STORAGE_TARGET)
	./$(HNSW_TARGET)
	./$(INTEG_TARGET)

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

run-release: $(RELEASE_TARGETS)
	./test_quantize_math_release$(EXE)
	./test_mmap_storage_release$(EXE)
	./test_hnsw_graph_release$(EXE)
	./test_integration_release$(EXE)

clean:
	rm -f $(TARGET) $(STORAGE_TARGET) $(HNSW_TARGET) $(INTEG_TARGET) \
	      $(RELEASE_TARGETS)
