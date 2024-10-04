#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <benchmark/benchmark.h>
#include <KokkosBatched_Gesv.hpp>
#include <KokkosBatched_Getrf.hpp>
#include <KokkosBatched_Getrs.hpp>
#include <KokkosBatched_Util.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;

namespace KokkosBatched {
struct DynamicPivoting {};
struct DynamicPivoting2 {};

} // namespace KokkosBatched

using namespace KokkosBatched;

template <typename P>
 struct ParamTag {
   using pivot = P;
 };

template <typename MatrixViewType, typename VectorViewType>
void create_tridiagonal_batched_matrices(const MatrixViewType& A, const VectorViewType& B) {
  Kokkos::Random_XorShift64_Pool<typename VectorViewType::device_type::execution_space> random(13718);
  Kokkos::fill_random(B, random, Kokkos::reduction_identity<typename VectorViewType::value_type>::prod());

  auto A_host = Kokkos::create_mirror_view(A);

  const int N       = A.extent(0);
  const int BlkSize = A.extent(1);

  for (int l = 0; l < N; ++l) {
    for (int i = 0; i < BlkSize; ++i) {
      for (int j = i; j < BlkSize; ++j) {
        if (i == j)
          A_host(l, i, j) = typename VectorViewType::value_type(2.0);
        else if (i == j - 1) {
          A_host(l, i, j) = typename VectorViewType::value_type(-1.0);
          A_host(l, j, i) = typename VectorViewType::value_type(-1.0);
        } else {
          A_host(l, i, j) = typename VectorViewType::value_type(0.0);
          A_host(l, j, i) = typename VectorViewType::value_type(0.0);
        }
      }
    }
  }

  Kokkos::fence();

  Kokkos::deep_copy(A, A_host);

  Kokkos::fence();
}

template <typename AViewType, typename PivViewType, typename AlgoTagType>
struct Functor_BatchedSerialGetrf {
  AViewType _a;
  PivViewType _ipiv;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGetrf(const AViewType &a, const PivViewType &ipiv) : _a(a), _ipiv(ipiv) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto aa   = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto ipiv = Kokkos::subview(_ipiv, k, Kokkos::ALL());

    KokkosBatched::SerialGetrf<AlgoTagType>::invoke(aa, ipiv);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name("KokkosBatched::Test::SerialGetrs");
    Kokkos::RangePolicy<execution_space> policy(0, _a.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename AViewType, typename PivViewType, typename BViewType, typename AlgoTagType>
struct Functor_BatchedSerialGetrs {
  AViewType _a;
  BViewType _b;
  PivViewType _ipiv;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGetrs(const AViewType &a, const PivViewType &ipiv, const BViewType &b)
      : _a(a), _b(b), _ipiv(ipiv) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto aa   = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto ipiv = Kokkos::subview(_ipiv, k, Kokkos::ALL());
    auto bb   = Kokkos::subview(_b, k, Kokkos::ALL());

    KokkosBatched::SerialGetrs<Trans::NoTranspose, AlgoTagType>::invoke(aa, ipiv, bb);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name("KokkosBatched::Test::SerialGetrs");
    Kokkos::RangePolicy<execution_space> policy(0, _b.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename AViewType, typename PivViewType, typename BViewType, typename AlgoTagType>
struct Functor_BatchedSerialGesv2 {
  AViewType _a;
  BViewType _b;
  PivViewType _ipiv;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGesv2(const AViewType &a, const PivViewType &ipiv, const BViewType &b)
      : _a(a), _b(b), _ipiv(ipiv) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto aa   = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto ipiv = Kokkos::subview(_ipiv, k, Kokkos::ALL());
    auto bb   = Kokkos::subview(_b, k, Kokkos::ALL());

    KokkosBatched::SerialGetrf<AlgoTagType>::invoke(aa, ipiv);
    KokkosBatched::SerialGetrs<Trans::NoTranspose, AlgoTagType>::invoke(aa, ipiv, bb);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name("KokkosBatched::Test::SerialGesv2");
    Kokkos::RangePolicy<execution_space> policy(0, _b.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename MatrixType, typename VectorType, typename AlgoTagType>
struct Functor_BatchedSerialGesv {
  const MatrixType _A;
  const MatrixType _tmp;
  const VectorType _X;
  const VectorType _B;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGesv(const MatrixType &A, const MatrixType &tmp, const VectorType &X, const VectorType &B)
      : _A(A), _tmp(tmp), _X(X), _B(B) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto A   = Kokkos::subview(_A, k, Kokkos::ALL, Kokkos::ALL);
    auto x   = Kokkos::subview(_X, k, Kokkos::ALL);
    auto b   = Kokkos::subview(_B, k, Kokkos::ALL);
    auto tmp = Kokkos::subview(_tmp, k, Kokkos::ALL, Kokkos::ALL);

    KokkosBatched::SerialGesv<AlgoTagType>::invoke(A, x, b, tmp);
  }

  inline void run() {
    typedef typename VectorType::value_type value_type;
    std::string name("KokkosBatched::Test::SerialGesv");
    Kokkos::RangePolicy<execution_space> policy(0, _X.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

/**
 * \brief Report throughput and amount of data processed for simple View
 * operations
 */
template <typename AViewType, typename XViewType>
void report_results(benchmark::State& state, AViewType A, XViewType x, double time) {
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

template <typename ScalarType, typename LayoutType, typename ParamTagType>
static void GESV_serial_batch(benchmark::State& state) {
  using View2DType = Kokkos::View<ScalarType**, LayoutType, execution_space>;
  using View3DType = Kokkos::View<ScalarType***, LayoutType, execution_space>;
  using PivView2DType = Kokkos::View<int **, LayoutType, execution_space>;
  using PivotingType = typename ParamTagType::pivot;

  const int n = state.range(0);
  const int batch = state.range(1);

  // Reference is created by gesv
  View3DType A("A", batch, n, n), tmp("tmp", batch, n, n+4);
  View2DType x("x", batch, n), b("b", batch, n);  // 
  PivView2DType piv("piv", batch, n);

  Kokkos::Random_XorShift64_Pool<execution_space> random(13718);
  Kokkos::fill_random(A, random, ScalarType(1.0));
  Kokkos::fence();

  create_tridiagonal_batched_matrices(A, b);
  Kokkos::deep_copy(x, b);

  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;

    if constexpr (std::is_same_v<PivotingType, KokkosBatched::DynamicPivoting>) {
      Functor_BatchedSerialGetrf<View3DType, PivView2DType, Algo::Getrf::Unblocked>(A, piv).run();
      Functor_BatchedSerialGetrs<View3DType, PivView2DType, View2DType, Algo::Getrs::Unblocked>(
                  A, piv, x).run();
    } else if constexpr (std::is_same_v<PivotingType, KokkosBatched::DynamicPivoting2>) {
      Functor_BatchedSerialGesv2<View3DType, PivView2DType, View2DType, Algo::Getrs::Unblocked>(
                  A, piv, x).run();
    } else {
      Functor_BatchedSerialGesv<View3DType, View2DType, PivotingType>(A, tmp, x, b)
       .run();
    }

    Kokkos::fence();
    report_results(state, A, x, timer.seconds());
  }
}

using param_s = ParamTag<KokkosBatched::Gesv::StaticPivoting>;
using param_d = ParamTag<KokkosBatched::DynamicPivoting>;
using param_d2 = ParamTag<KokkosBatched::DynamicPivoting2>;

BENCHMARK(GESV_serial_batch<double, Kokkos::LayoutLeft, param_s>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {16, 1024}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(GESV_serial_batch<double, Kokkos::LayoutRight, param_s>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {16, 1024}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

/*
BENCHMARK(GESV_serial_batch<double, Kokkos::LayoutLeft, param_d>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {16, 1024}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(GESV_serial_batch<double, Kokkos::LayoutRight, param_d>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {16, 1024}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
*/

BENCHMARK(GESV_serial_batch<double, Kokkos::LayoutLeft, param_d2>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {16, 1024}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(GESV_serial_batch<double, Kokkos::LayoutRight, param_d2>)
    ->ArgNames({"N", "batch"})
    ->RangeMultiplier(2)
    ->Ranges(
        {{64, 1024},
         {16, 1024}})
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