cmake_minimum_required(VERSION 3.23)
project(benchmark-kernels LANGUAGES CXX)

add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/kokkos-kernels)
add_subdirectory(tpls/googletest)
add_subdirectory(tpls/benchmark)

add_subdirectory(perf_test)