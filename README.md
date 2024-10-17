# bench-kernels

# Compile and Run
## Installation
First of all, you need to clone this repo.
```bash
git clone --recursive https://github.com/CExA-project/bench-kernels.git
```

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

## Results

<div style style=”line-height: 25%” align="center">
<h3> Icelake (OpenMP) </h3>
<img src=imgs/Tbsv_Icelake_l_n_n.png>
</div>

<div style style=”line-height: 25%” align="center">
<h3> A100 (CUDA) </h3>
<img src=imgs/Tbsv_A100_l_n_n.png>
</div>
