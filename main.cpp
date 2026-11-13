#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <immintrin.h> //SIMD
#include <omp.h>

//пороги для параллелизации (элементов меньше порога--> считаем в один поток), т.к. запуск потока дорогой 
long threshold_arith = 20000000; // + - * /: и на тестовых размерах всегда в один поток
long threshold_sum = 2000000;
long threshold_activation = 250000; // silu, gelu
long threshold_matmul = 128; // размер матрицы n

template <typename T>
struct Simd;

template <>
struct Simd<float> {
    using reg = __m256;
    static constexpr int width = 8;

    static reg load(const float* p) { return _mm256_loadu_ps(p); }
    static void store(float* p, reg v) { _mm256_storeu_ps(p, v); }
    static reg set1(float x) { return _mm256_set1_ps(x); }
    static reg add(reg a, reg b) { return _mm256_add_ps(a, b); }
    static reg sub(reg a, reg b) { return _mm256_sub_ps(a, b); }
    static reg mul(reg a, reg b) { return _mm256_mul_ps(a, b); }
    static reg div(reg a, reg b) { return _mm256_div_ps(a, b); }

    static reg exp(reg x) {
        x = _mm256_min_ps(x, _mm256_set1_ps(80.0f));
        x = _mm256_max_ps(x, _mm256_set1_ps(-80.0f));
        //exp(x) = 2^n*exp(r), где x = n*ln2+ r и r маленькое
        reg n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.4426950f)),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        reg r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.6931472f)));
        //exp(r) по ряду тейлора
        float c[] = {1.0f / 720, 1.0f / 120, 1.0f / 24, 1.0f / 6, 0.5f, 1.0f, 1.0f};
        reg p = _mm256_set1_ps(c[0]);
        for (int k = 1; k < 7; k++) {
            p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(c[k]));
        }
        //2^n собираем в экспоненте float
        __m256i pow2 = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
        return _mm256_mul_ps(p, _mm256_castsi256_ps(pow2));
    }
};

template <>
struct Simd<double> {
    using reg = __m256d;
    static constexpr int width = 4;

    static reg load(const double* p) { return _mm256_loadu_pd(p); }
    static void store(double* p, reg v) { _mm256_storeu_pd(p, v); }
    static reg set1(double x) { return _mm256_set1_pd(x); }
    static reg add(reg a, reg b) { return _mm256_add_pd(a, b); }
    static reg sub(reg a, reg b) { return _mm256_sub_pd(a, b); }
    static reg mul(reg a, reg b) { return _mm256_mul_pd(a, b); }
    static reg div(reg a, reg b) { return _mm256_div_pd(a, b); }

    static reg exp(reg x) {
        x = _mm256_min_pd(x, _mm256_set1_pd(700.0));
        x = _mm256_max_pd(x, _mm256_set1_pd(-700.0));
        reg n = _mm256_round_pd(_mm256_mul_pd(x, _mm256_set1_pd(1.4426950408889634)),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        reg r = _mm256_sub_pd(x, _mm256_mul_pd(n, _mm256_set1_pd(0.6931471805599453)));
        double c[] = {1.0 / 3628800, 1.0 / 362880, 1.0 / 40320, 1.0 / 5040, 1.0 / 720,
                      1.0 / 120, 1.0 / 24, 1.0 / 6, 0.5, 1.0, 1.0};
        reg p = _mm256_set1_pd(c[0]);
        for (int k = 1; k < 11; k++) {
            p = _mm256_add_pd(_mm256_mul_pd(p, r), _mm256_set1_pd(c[k]));
        }
        //у double экспонента 11 бит, сдвиг на 52
        __m256i n64 = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(n));
        __m256i pow2 = _mm256_slli_epi64(_mm256_add_epi64(n64, _mm256_set1_epi64x(1023)), 52);
        return _mm256_mul_pd(p, _mm256_castsi256_pd(pow2));
    }
};

template <>
struct Simd<int> {
    using reg = __m256i;
    static constexpr int width = 8;

    static reg load(const int* p) { return _mm256_loadu_si256((const __m256i*)p); }
    static void store(int* p, reg v) { _mm256_storeu_si256((__m256i*)p, v); }
    static reg set1(int x) { return _mm256_set1_epi32(x); }
    static reg add(reg a, reg b) { return _mm256_add_epi32(a, b); }
    static reg sub(reg a, reg b) { return _mm256_sub_epi32(a, b); }
    static reg mul(reg a, reg b) { return _mm256_mullo_epi32(a, b); }

    //в avx2 нет целочисленного деления, поэтому делим через double (по 4 числа)
    static reg div(reg a, reg b) {
        __m256d a_low = _mm256_cvtepi32_pd(_mm256_castsi256_si128(a));
        __m256d a_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(a, 1));
        __m256d b_low = _mm256_cvtepi32_pd(_mm256_castsi256_si128(b));
        __m256d b_high = _mm256_cvtepi32_pd(_mm256_extracti128_si256(b, 1));
        //cvtt обрезает дробную часть
        __m128i low = _mm256_cvttpd_epi32(_mm256_div_pd(a_low, b_low));
        __m128i high = _mm256_cvttpd_epi32(_mm256_div_pd(a_high, b_high));
        return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
    }
};


//операции
template <typename T>
struct Plus {
    using reg = typename Simd<T>::reg;
    T scalar_op(T a, T b) { return a + b; }
    reg simd_op(reg a, reg b) { return Simd<T>::add(a, b); }
};

template <typename T>
struct Minus {
    using reg = typename Simd<T>::reg;
    T scalar_op(T a, T b) { return a - b; }
    reg simd_op(reg a, reg b) { return Simd<T>::sub(a, b); }
};

template <typename T>
struct Mult {
    using reg = typename Simd<T>::reg;
    T scalar_op(T a, T b) { return a * b; }
    reg simd_op(reg a, reg b) { return Simd<T>::mul(a, b); }
};

template <typename T>
struct Divide {
    using reg = typename Simd<T>::reg;
    T scalar_op(T a, T b) { return a / b; }
    reg simd_op(reg a, reg b) { return Simd<T>::div(a, b); }
};

//SiLU(x) = x/(1+ exp(-x))
template <typename T>
struct SiLU {
    using reg = typename Simd<T>::reg;
    T scalar_op(T x) { return x / (1 + std::exp(-x)); }
    reg simd_op(reg x) {
        using S = Simd<T>;
        reg e = S::exp(S::sub(S::set1(0), x));
        return S::div(x, S::add(S::set1(1), e));
    }
};

//GELU(x) = 0.5*x * (1+ tanh(sqrt(2/pi) *(x + 0.044715 * x^3)))
//в simd версии tanh(u) считаем как 1 -2/(exp(2u) +1)
template <typename T>
struct GELU {
    using reg = typename Simd<T>::reg;
    T scalar_op(T x) {
        T u = 0.7978845608 * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1 + std::tanh(u));
    }
    reg simd_op(reg x) {
        using S = Simd<T>;
        reg x3 = S::mul(S::mul(x, x), x);
        reg u = S::mul(S::set1(0.7978845608), S::add(x, S::mul(S::set1(0.044715), x3)));
        reg e = S::exp(S::mul(S::set1(2), u));
        reg tanh_u = S::sub(S::set1(1), S::div(S::set1(2), S::add(e, S::set1(1))));
        return S::mul(S::mul(S::set1(0.5), x), S::add(S::set1(1), tanh_u));
    }
};


//арифметика для двух массивов
template <typename T, typename Action>
void do_math_scalar(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& res, Action action) {
    long n = a.size();
    #pragma omp parallel for if(n >= threshold_arith)
    for (long i = 0; i < n; i++) {
        res[i] = action.scalar_op(a[i], b[i]);
    }
}

template <typename T, typename Action>
void do_math_simd(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& res, Action action) {
    using S = Simd<T>;
    long n = a.size();
    long blocks = n / S::width;

    #pragma omp parallel for if(n >= threshold_arith)
    for (long k = 0; k < blocks; k++) {
        long i = k * S::width;
        S::store(&res[i], action.simd_op(S::load(&a[i]), S::load(&b[i])));
    }
    //хвост, если размер не делится на width
    for (long i = blocks * S::width; i < n; i++) {
        res[i] = action.scalar_op(a[i], b[i]);
    }
}

//то же самое для функций активации 
template <typename T, typename Action>
void do_unary_scalar(const std::vector<T>& a, std::vector<T>& res, Action action) {
    long n = a.size();
    #pragma omp parallel for if(n >= threshold_activation)
    for (long i = 0; i < n; i++) {
        res[i] = action.scalar_op(a[i]);
    }
}

template <typename T, typename Action>
void do_unary_simd(const std::vector<T>& a, std::vector<T>& res, Action action) {
    using S = Simd<T>;
    long n = a.size();
    long blocks = n / S::width;

    #pragma omp parallel for if(n >= threshold_activation)
    for (long k = 0; k < blocks; k++) {
        long i = k * S::width;
        S::store(&res[i], action.simd_op(S::load(&a[i])));
    }
    for (long i = blocks * S::width; i < n; i++) {
        res[i] = action.scalar_op(a[i]);
    }
}


//сумма и среднее
template <typename T>
T sum_scalar(const std::vector<T>& a) {
    T total = 0;
    long n = a.size();
    #pragma omp parallel for reduction(+:total) if(n >= threshold_sum)
    for (long i = 0; i < n; i++) {
        total += a[i];
    }
    return total;
}

template <typename T>
T sum_simd(const std::vector<T>& a) {
    using S = Simd<T>;
    long n = a.size();
    long blocks = n / S::width;
    T total = 0;

    #pragma omp parallel if(n >= threshold_sum)
    {
        //у каждого потока свой регистр с частичными суммами
        typename S::reg acc = S::set1(0);
        #pragma omp for nowait
        for (long k = 0; k < blocks; k++) {
            acc = S::add(acc, S::load(&a[k * S::width]));
        }
        T parts[8];
        S::store(parts, acc);
        T mine = 0;
        for (int j = 0; j < S::width; j++) {
            mine += parts[j];
        }
        #pragma omp atomic
        total += mine;
    }
    for (long i = blocks * S::width; i < n; i++) {
        total += a[i];
    }
    return total;
}

template <typename T>
double mean_scalar(const std::vector<T>& a) {
    return (double)sum_scalar(a) / a.size();
}

template <typename T>
double mean_simd(const std::vector<T>& a) {
    return (double)sum_simd(a) / a.size();
}


//умножение квадратных матриц n x n, хранятся по строкам
//порядок циклов i-k-j, чтобы по b и c шли подряд по памяти
template <typename T>
void matmul_scalar(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c, int n) {
    #pragma omp parallel for if(n >= threshold_matmul)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0;
        }
        for (int k = 0; k < n; k++) {
            T a_ik = a[i * n + k];
            for (int j = 0; j < n; j++) {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
}

template <typename T>
void matmul_simd(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c, int n) {
    using S = Simd<T>;
    #pragma omp parallel for if(n >= threshold_matmul)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0;
        }
        for (int k = 0; k < n; k++) {
            typename S::reg a_ik = S::set1(a[i * n + k]);
            int j = 0;
            for (; j + S::width <= n; j += S::width) {
                typename S::reg sum = S::add(S::load(&c[i * n + j]), S::mul(a_ik, S::load(&b[k * n + j])));
                S::store(&c[i * n + j], sum);
            }
            for (; j < n; j++) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}


//заполнение случайными числами; для float/double значения дробные, для int целые
template <typename T>
void fill_random(std::vector<T>& v, bool no_zeros) {
    for (size_t i = 0; i < v.size(); i++) {
        int r = rand() % 19 - 9;
        if (no_zeros && r == 0) {
            r = 1;
        }
        v[i] = (T)r;
        if (!std::is_integral<T>::value) {
            v[i] = v[i] * (T)0.37;
        }
    }
}


//тесты на корректность 
int errors = 0;

bool same(int a, int b) { return a == b; }
bool same(float a, float b) { return std::fabs(a - b) <= 1e-3f * (1 + std::fabs(b)); }
bool same(double a, double b) { return std::fabs(a - b) <= 1e-6 * (1 + std::fabs(b)); }

void report(const char* type_name, const char* what, bool ok) {
    if (!ok) {
        printf("%-8s %-10s FAIL\n", type_name, what);
        errors++;
    }
}

template <typename T>
bool same_arrays(const std::vector<T>& x, const std::vector<T>& y) {
    for (size_t i = 0; i < x.size(); i++) {
        if (!same(x[i], y[i])) {
            return false;
        }
    }
    return true;
}

template <typename T, typename Action>
void test_binary(const char* type_name, const char* op_name, const std::vector<T>& a, const std::vector<T>& b, Action action) {
    std::vector<T> expected(a.size()), got(a.size());
    do_math_scalar(a, b, expected, action);
    do_math_simd(a, b, got, action);
    report(type_name, op_name, same_arrays(expected, got));
}

template <typename T>
void test_basic(const char* type_name) {
    int errors_before = errors;
    //размер не кратен 8, чтобы проверить хвост
    size_t n = 100003;
    std::vector<T> a(n), b(n);
    fill_random(a, false);
    fill_random(b, true); //b без нулей из-за деления

    test_binary(type_name, "plus", a, b, Plus<T>());
    test_binary(type_name, "minus", a, b, Minus<T>());
    test_binary(type_name, "mult", a, b, Mult<T>());
    test_binary(type_name, "divide", a, b, Divide<T>());

    report(type_name, "sum", same(sum_scalar(a), sum_simd(a)));
    report(type_name, "mean", same(mean_scalar(a), mean_simd(a)));

    int m = 37; //тоже не кратно 8
    std::vector<T> ma(m * m), mb(m * m), mc1(m * m), mc2(m * m);
    fill_random(ma, false);
    fill_random(mb, false);
    matmul_scalar(ma, mb, mc1, m);
    matmul_simd(ma, mb, mc2, m);
    report(type_name, "matmul", same_arrays(mc1, mc2));
    printf("%-8s arithmetic, sum, mean, matmul: %s\n", type_name, errors == errors_before ? "ok" : "FAIL");
}

template <typename T>
void test_activations(const char* type_name) {
    int errors_before = errors;
    size_t n = 100003;
    std::vector<T> a(n), expected(n), got(n);
    fill_random(a, false);

    do_unary_scalar(a, expected, SiLU<T>());
    do_unary_simd(a, got, SiLU<T>());
    report(type_name, "silu", same_arrays(expected, got));

    do_unary_scalar(a, expected, GELU<T>());
    do_unary_simd(a, got, GELU<T>());
    report(type_name, "gelu", same_arrays(expected, got));
    printf("%-8s silu, gelu: %s\n", type_name, errors == errors_before ? "ok" : "FAIL");
}


//нагрузочные тесты
template <typename F>
double measure(F func, long repeats) {
    func(); //прогрев
    double start = omp_get_wtime();
    for (long r = 0; r < repeats; r++) {
        func();
    }
    return (omp_get_wtime() - start) / repeats;
}

//чтобы маленькие размеры не мерились один раз, а большие не считались вечность
long repeats_for(long work) {
    long r = 20000000 / work;
    if (r < 3) {
        r = 3;
    }
    if (r > 1000) {
        r = 1000;
    }
    return r;
}

void print_line(const char* type_name, const char* op_name, long size, double t_scalar, double t_simd) {
    printf("%-8s %-8s %10ld %12.4f %12.4f %8.2f\n", type_name, op_name, size, t_scalar * 1000, t_simd * 1000, t_scalar / t_simd);
}

template <typename T, typename Action>
void bench_binary(const char* type_name, const char* op_name, const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& res, Action action) {
    long repeats = repeats_for(a.size());
    double t_scalar = measure([&]() { do_math_scalar(a, b, res, action); }, repeats);
    double t_simd = measure([&]() { do_math_simd(a, b, res, action); }, repeats);
    print_line(type_name, op_name, a.size(), t_scalar, t_simd);
}

template <typename T, typename Action>
void bench_unary(const char* type_name, const char* op_name, const std::vector<T>& a, std::vector<T>& res, Action action) {
    long repeats = repeats_for(a.size());
    double t_scalar = measure([&]() { do_unary_scalar(a, res, action); }, repeats);
    double t_simd = measure([&]() { do_unary_simd(a, res, action); }, repeats);
    print_line(type_name, op_name, a.size(), t_scalar, t_simd);
}

template <typename T>
void bench_basic(const char* type_name) {
    long sizes[] = {1000, 100000, 10000000};
    for (long size : sizes) {
        std::vector<T> a(size), b(size), res(size);
        fill_random(a, false);
        fill_random(b, true);

        bench_binary(type_name, "plus", a, b, res, Plus<T>());
        bench_binary(type_name, "minus", a, b, res, Minus<T>());
        bench_binary(type_name, "mult", a, b, res, Mult<T>());
        bench_binary(type_name, "divide", a, b, res, Divide<T>());

        long repeats = repeats_for(size);
        T keep = 0; //результат сохраняем, чтобы компилятор не выкинул вызов
        double t_scalar = measure([&]() { keep += sum_scalar(a); }, repeats);
        double t_simd = measure([&]() { keep += sum_simd(a); }, repeats);
        print_line(type_name, "sum", size, t_scalar, t_simd);
        if (keep == 12345) {
            printf(" ");
        }
    }

    int matrix_sizes[] = {64, 256, 512};
    for (int n : matrix_sizes) {
        std::vector<T> a(n * n), b(n * n), c(n * n);
        fill_random(a, false);
        fill_random(b, false);
        long repeats = repeats_for((long)n * n * n);
        double t_scalar = measure([&]() { matmul_scalar(a, b, c, n); }, repeats);
        double t_simd = measure([&]() { matmul_simd(a, b, c, n); }, repeats);
        print_line(type_name, "matmul", n, t_scalar, t_simd);
    }
}

template <typename T>
void bench_activations(const char* type_name) {
    long sizes[] = {1000, 100000, 10000000};
    for (long size : sizes) {
        std::vector<T> a(size), res(size);
        fill_random(a, false);
        bench_unary(type_name, "silu", a, res, SiLU<T>());
        bench_unary(type_name, "gelu", a, res, GELU<T>());
    }
}


int main() {
    //на время тестов пороги не учитываем, иначе маленькие тестовые массивы не попадут в параллельную ветку
    long saved_arith = threshold_arith;
    long saved_sum = threshold_sum;
    long saved_activation = threshold_activation;
    long saved_matmul = threshold_matmul;
    threshold_arith = 0;
    threshold_sum = 0;
    threshold_activation = 0;
    threshold_matmul = 0;

    printf("--- correctness tests ---\n");
    test_basic<int>("int32");
    test_basic<float>("float32");
    test_basic<double>("float64");
    test_activations<float>("float32");
    test_activations<double>("float64");
    if (errors == 0) {
        printf("all tests passed\n\n");
    } else {
        printf("failed tests: %d\n\n", errors);
    }

    threshold_arith = saved_arith;
    threshold_sum = saved_sum;
    threshold_activation = saved_activation;
    threshold_matmul = saved_matmul;
    printf("thresholds: arith %ld, sum %ld, activation %ld, matmul %ld\n", threshold_arith, threshold_sum, threshold_activation, threshold_matmul);

    printf("--- benchmarks (time in ms, matmul size = n for n x n) ---\n");
    int thread_counts[] = {1, omp_get_max_threads()};
    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        printf("\nthreads: %d\n", threads);
        printf("%-8s %-8s %10s %12s %12s %8s\n", "type", "op", "size", "scalar", "simd", "speedup");
        bench_basic<int>("int32");
        bench_basic<float>("float32");
        bench_basic<double>("float64");
        bench_activations<float>("float32");
        bench_activations<double>("float64");
    }

    return 0;
}