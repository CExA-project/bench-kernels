#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <benchmark/benchmark.h>
#include <KokkosBatched_Tbsv.hpp>
#include <KokkosBatched_Util.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;
using namespace KokkosBatched;

template <typename U, typename T, typename D>
 struct ParamTag {
   using uplo  = U;
   using trans = T;
   using diag  = D;
 };

/**
 * \brief Report throughput and amount of data processed for simple View
 * operations
 */
template <typename AViewType, typename XViewType>
void report_results(benchmark::State& state, AViewType A, XViewType x,
                    double time) {
  // data processed in megabytes
  const double A_data_processed =
      static_cast<double>(A.size() * sizeof(typename AViewType::value_type)) /
      1.0e6;
  const double x_data_processed =
      static_cast<double>(x.size() *
                          sizeof(typename XViewType::value_type)) /
      1.0e6;

  state.SetIterationTime(time);
  state.counters["MB (A)"] = benchmark::Counter(A_data_processed);
  state.counters["MB (x)"] = benchmark::Counter(x_data_processed);
  state.counters["GB/s"] =
      benchmark::Counter((A_data_processed + x_data_processed) / 1.0e3,
                         benchmark::Counter::kIsIterationInvariantRate);
}

template <typename InViewType, typename OutViewType, typename UploType>
void create_banded_triangular_matrix(InViewType& in, OutViewType& out,
                                    int k = 1, bool band_storage = true) {
    auto h_in   = Kokkos::create_mirror_view(in);
    auto h_out  = Kokkos::create_mirror_view(out);
    const int N = in.extent(0), BlkSize = in.extent(1);

    Kokkos::deep_copy(h_in, in);
    if (band_storage) {
        assert(out.extent(0) == in.extent(0));
        assert(out.extent(1) == static_cast<std::size_t>(k + 1));
        assert(out.extent(2) == in.extent(2));
        if constexpr (std::is_same_v<UploType, KokkosBatched::Uplo::Upper>) {
        for (int i0 = 0; i0 < N; i0++) {
            for (int i1 = 0; i1 < k + 1; i1++) {
            for (int i2 = i1; i2 < BlkSize; i2++) {
                h_out(i0, k - i1, i2) = h_in(i0, i2 - i1, i2);
            }
            }
        }
        } else {
        for (int i0 = 0; i0 < N; i0++) {
            for (int i1 = 0; i1 < k + 1; i1++) {
            for (int i2 = 0; i2 < BlkSize - i1; i2++) {
                h_out(i0, i1, i2) = h_in(i0, i2 + i1, i2);
            }
            }
        }
        }
    } else {
        for (std::size_t i = 0; i < InViewType::rank(); i++) {
        assert(out.extent(i) == in.extent(i));
        }

        if constexpr (std::is_same_v<UploType, KokkosBatched::Uplo::Upper>) {
        for (int i0 = 0; i0 < N; i0++) {
            for (int i1 = 0; i1 < BlkSize; i1++) {
            for (int i2 = i1; i2 < Kokkos::min(i1 + k + 1, BlkSize); i2++) {
                h_out(i0, i1, i2) = h_in(i0, i1, i2);
            }
            }
        }
        } else {
        for (int i0 = 0; i0 < N; i0++) {
            for (int i1 = 0; i1 < BlkSize; i1++) {
            for (int i2 = Kokkos::max(0, i1 - k); i2 <= i1; i2++) {
                h_out(i0, i1, i2) = h_in(i0, i1, i2);
            }
            }
        }
        }
    }
    Kokkos::deep_copy(out, h_out);
}

template <typename AViewType, typename BViewType, typename ParamTagType>
 struct Functor_BatchedSerialTbsvFirst {
   AViewType _a;
   BViewType _b;
   int _k, _incx;

   KOKKOS_INLINE_FUNCTION
   Functor_BatchedSerialTbsvFirst(const AViewType &a, const BViewType &b, const int k,
                                  const int incx)
       : _a(a), _b(b), _k(k), _incx(incx) {}

   KOKKOS_INLINE_FUNCTION
   void operator()(const ParamTagType &, const int k) const {
     auto bb = Kokkos::subview(_b, k, Kokkos::ALL());

     KokkosBatched::SerialTbsv<
         typename ParamTagType::uplo, typename ParamTagType::trans,
         typename ParamTagType::diag, typename Algo::Tbsv::Unblocked>::invoke(_a, bb, _k, _incx);
   }

   inline void run() {
     std::string name_region("KokkosBatched::Bench::SerialTbsv");
     Kokkos::Profiling::pushRegion(name_region.c_str());
     Kokkos::RangePolicy<execution_space, ParamTagType> policy(0, _b.extent(0));
     Kokkos::parallel_for(name_region.c_str(), policy, *this);
     Kokkos::Profiling::popRegion();
   }
 };

 template <typename AViewType, typename BViewType, typename ParamTagType>
 struct Functor_BatchedSerialTbsvLast {
   AViewType _a;
   BViewType _b;
   int _k, _incx;

   KOKKOS_INLINE_FUNCTION
   Functor_BatchedSerialTbsvLast(const AViewType &a, const BViewType &b, const int k,
                                  const int incx)
       : _a(a), _b(b), _k(k), _incx(incx) {}

   KOKKOS_INLINE_FUNCTION
   void operator()(const ParamTagType &, const int k) const {
     auto bb = Kokkos::subview(_b, Kokkos::ALL(), k);

     KokkosBatched::SerialTbsv<
         typename ParamTagType::uplo, typename ParamTagType::trans,
         typename ParamTagType::diag, typename Algo::Tbsv::Unblocked>::invoke(_a, bb, _k, _incx);
   }

   inline void run() {
     std::string name_region("KokkosBatched::Bench::SerialTbsv");
     Kokkos::Profiling::pushRegion(name_region.c_str());
     Kokkos::RangePolicy<execution_space, ParamTagType> policy(0, _b.extent(1));
     Kokkos::parallel_for(name_region.c_str(), policy, *this);
     Kokkos::Profiling::popRegion();
   }
 };

template <typename ScalarType, typename LayoutType, typename ParamTagType>
static void TBSV_1D_batch_first(benchmark::State& state) {
  using View2DType = Kokkos::View<ScalarType**, LayoutType, execution_space>;
  using View3DType = Kokkos::View<ScalarType***, LayoutType, execution_space>;

  const int n = state.range(0);
  const int batch = state.range(1);
  const int k = 3;
  
  // Reference is created by trsv from triangular matrix
  View3DType A("A", 1, n, n);
  View3DType Ab("Ab", 1, k+1, n); // Banded storage
  View2DType x0("x0", batch, n);  // Solutions

  Kokkos::Random_XorShift64_Pool<execution_space> random(13718);
  Kokkos::fill_random(A, random, ScalarType(1.0));
  Kokkos::fence();

  create_banded_triangular_matrix<View3DType, View3DType, typename ParamTagType::uplo>(A, Ab, k, true);
  auto _Ab = Kokkos::subview(Ab, 0, Kokkos::ALL(), Kokkos::ALL());

  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;

    Functor_BatchedSerialTbsvFirst<View2DType, View2DType, ParamTagType>(_Ab, x0, k, 1)
       .run();

    Kokkos::fence();
    report_results(state, _Ab, x0, timer.seconds());
  }
}

template <typename ScalarType, typename LayoutType, typename ParamTagType>
static void TBSV_1D_batch_last(benchmark::State& state) {
  using View2DType = Kokkos::View<ScalarType**, LayoutType, execution_space>;
  using View3DType = Kokkos::View<ScalarType***, LayoutType, execution_space>;

  const int n = state.range(0);
  const int batch = state.range(1);
  const int k = 3;
  
  // Reference is created by trsv from triangular matrix
  View3DType A("A", 1, n, n);
  View3DType Ab("Ab", 1, k+1, n); // Banded storage
  View2DType x0("x0", n, batch);  // Solutions

  Kokkos::Random_XorShift64_Pool<execution_space> random(13718);
  Kokkos::fill_random(A, random, ScalarType(1.0));
  Kokkos::fence();

  create_banded_triangular_matrix<View3DType, View3DType, typename ParamTagType::uplo>(A, Ab, k, true);
  auto _Ab = Kokkos::subview(Ab, 0, Kokkos::ALL(), Kokkos::ALL());

  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;

    Functor_BatchedSerialTbsvLast<View2DType, View2DType, ParamTagType>(_Ab, x0, k, 1)
       .run();

    Kokkos::fence();
    report_results(state, _Ab, x0, timer.seconds());
  }
}

using param_l_n_n = ParamTag<Uplo::Lower, Trans::NoTranspose, Diag::NonUnit>;
using param_l_t_n = ParamTag<Uplo::Lower, Trans::Transpose, Diag::NonUnit>;
using param_u_n_n = ParamTag<Uplo::Upper, Trans::NoTranspose, Diag::NonUnit>;
using param_u_t_n = ParamTag<Uplo::Upper, Trans::Transpose, Diag::NonUnit>;

// Batch first
BENCHMARK(TBSV_1D_batch_first<double, Kokkos::LayoutLeft, param_l_n_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_first<double, Kokkos::LayoutLeft, param_l_t_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_first<double, Kokkos::LayoutLeft, param_u_n_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_first<double, Kokkos::LayoutLeft, param_u_t_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_first<double, Kokkos::LayoutRight, param_l_n_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_first<double, Kokkos::LayoutRight, param_l_t_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_first<double, Kokkos::LayoutRight, param_u_n_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_first<double, Kokkos::LayoutRight, param_u_t_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// Batch last
BENCHMARK(TBSV_1D_batch_last<double, Kokkos::LayoutLeft, param_l_n_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_last<double, Kokkos::LayoutLeft, param_l_t_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_last<double, Kokkos::LayoutLeft, param_u_n_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_last<double, Kokkos::LayoutLeft, param_u_t_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_last<double, Kokkos::LayoutRight, param_l_n_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_last<double, Kokkos::LayoutRight, param_l_t_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_last<double, Kokkos::LayoutRight, param_u_n_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(TBSV_1D_batch_last<double, Kokkos::LayoutRight, param_u_t_n>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {100, 200000}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    benchmark::Initialize(&argc, argv);
    benchmark::SetDefaultTimeUnit(benchmark::kSecond);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
  }
  Kokkos::finalize();
  return 0;
}