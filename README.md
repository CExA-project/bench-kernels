# bench-kernels

# Compile
## CPU (OpenMP backend)

```bash
mkdir build_cpu && cd build_cpu
cmake -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=<path-to-kokkos> \
      -DKokkos_ENABLE_OPENMP=ON \
      -DKokkos_ARCH_SKX=ON \
      -DKokkosKernels_INST_DOUBLE=ON \
      -DKokkosKernels_INST_ORDINAL_INT=ON \
      -DKokkosKernels_INST_OFFSET_INT=ON \
      ..

cmake --build . -j 8
cd build_cpu
perf_test/benchmark_kernels --benchmark_format=json --benchmark_out=tbsv_bench.json
```

## GPU (CUDA backend)

```bash
mkdir build_gpu && cd build_gpu
cmake -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=<path-to-kokkos> \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON \
      -DKokkosKernels_INST_DOUBLE=ON \
      -DKokkosKernels_INST_ORDINAL_INT=ON \
      -DKokkosKernels_INST_OFFSET_INT=ON \
      ..

cmake --build . -j 8
cd build_gpu
perf_test/benchmark_kernels --benchmark_format=json --benchmark_out=tbsv_bench.json
```

## Post script

```bash
python analysis.py -dirname build_gpu
```