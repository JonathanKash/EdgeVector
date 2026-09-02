# Build and run the EdgeVector quickstart. Same knobs as tests/Makefile:
#   ARCH   (default -march=native), CXX, RUNNER for cross/emulated runs.
CXX ?= g++
ARCH ?= -march=native
CXXFLAGS := -std=c++17 -O2 $(ARCH) -Wall -Wextra
RUNNER ?=

ifeq ($(OS),Windows_NT)
    EXE := .exe
else
    EXE :=
endif

quickstart$(EXE): quickstart.cpp $(wildcard ../include/edgevector/*.hpp)
	$(CXX) $(CXXFLAGS) -I../include quickstart.cpp -o $@

itq_pipeline$(EXE): itq_pipeline.cpp $(wildcard ../include/edgevector/*.hpp)
	$(CXX) $(CXXFLAGS) -I../include itq_pipeline.cpp -o $@

run: quickstart$(EXE) itq_pipeline$(EXE)
	$(RUNNER) ./quickstart$(EXE)
	$(RUNNER) ./itq_pipeline$(EXE)

clean:
	rm -f quickstart$(EXE) itq_pipeline$(EXE) quickstart.evec quickstart.evhg \
	      itq_example.evrt

.PHONY: run clean
