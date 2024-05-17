add_executable(benchmark_kernels
    PerfTest_tbsv.cpp
)

target_link_libraries(benchmark_kernels
	PUBLIC
		benchmark::benchmark
        Kokkos::kokkos
        Kokkos::kokkoskernels
)