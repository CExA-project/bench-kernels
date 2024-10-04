add_executable(benchmark_kernels
    #PerfTest_tbsv.cpp
    PerfTest_gesv.cpp
)

target_link_libraries(benchmark_kernels
	PUBLIC
		benchmark::benchmark
        Kokkos::kokkos
        Kokkos::kokkoskernels
)