# EdgeVector: header-only, so consumption is one INTERFACE target.
#
#   FetchContent:
#     FetchContent_Declare(edgevector
#       GIT_REPOSITORY https://github.com/JonathanKash/EdgeVector.git
#       GIT_TAG main)
#     FetchContent_MakeAvailable(edgevector)
#     target_link_libraries(your_app PRIVATE edgevector::edgevector)
#
#   Or vendor the repo and: add_subdirectory(EdgeVector)
#
# -DEDGEVECTOR_BUILD_EXAMPLES=ON builds examples/quickstart.
cmake_minimum_required(VERSION 3.14)
project(edgevector VERSION 0.8.0 LANGUAGES CXX)

if(MSVC)
  message(FATAL_ERROR
    "EdgeVector requires GCC or Clang (it uses __builtin_popcountll and "
    "__builtin_prefetch). On Windows, use MinGW-w64 or clang.")
endif()

add_library(edgevector INTERFACE)
add_library(edgevector::edgevector ALIAS edgevector)
target_include_directories(edgevector INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(edgevector INTERFACE cxx_std_17)

# Concurrent build/queries use std::thread; harmless where already implicit.
find_package(Threads REQUIRED)
target_link_libraries(edgevector INTERFACE Threads::Threads)

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

option(EDGEVECTOR_BUILD_EXAMPLES "Build the EdgeVector examples" OFF)
if(EDGEVECTOR_BUILD_EXAMPLES)
  add_executable(quickstart examples/quickstart.cpp)
  target_link_libraries(quickstart PRIVATE edgevector::edgevector)
  # -march=native activates POPCNT/AVX2 (x86) or NEON codegen; correctness
  # does not depend on it, speed does.
  target_compile_options(quickstart PRIVATE -O2 -march=native -Wall -Wextra)
endif()
