#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cblas.h>

template <typename T>
void my_symm(int M, int N, T alpha, const T* A, int lda, 
             const T* B, int ldb, T beta, T* C, int ldc) {
    if (beta == 0) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) C[i + j * ldc] = 0;
    } else if (beta != 1) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) C[i + j * ldc] *= beta;
    }
    
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            T sum = 0;
            for (int k = 0; k <= i; ++k) sum += A[i + k * lda] * B[k + j * ldb];
            for (int k = i + 1; k < M; ++k) sum += A[k + i * lda] * B[k + j * ldb];
            C[i + j * ldc] += alpha * sum;ac
        }
    }
}

double geometric(const std::vector<double>& v) {
    double log_sum = 0;
    for (double x : v) log_sum += std::log(x);
    return std::exp(log_sum / v.size());
}

int main() {
    const int size = 3400;
    const double alpha = 1.0, beta = 0.0;
    const int iterations = 10;
    std::vector<int> threads = {1, 2, 4, 8, 16};

    std::cout << "Тестирование SYMM " << size << "x" << size << "\n";

    std::vector<double> A(size * size, 1.0), B(size * size, 2.0), C0(size * size, 0.0);
    for (int i = 0; i < size; ++i) A[i + i * size] = size * 2.0;

    std::cout << "\n[1/2] my_symm...\n";
    std::vector<double> my_times;
    for (int i = 0; i < iterations; ++i) {
        auto C = C0;
        auto start = std::chrono::high_resolution_clock::now();
        my_symm(size, size, alpha, A.data(), size, B.data(), size, beta, C.data(), size);
        double t = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        my_times.push_back(t);
        std::cout << "  " << i+1 << ": " << std::fixed << std::setprecision(4) << t << " с\n";
    }
    double my_g = geometric(my_times);
    std::cout << ">> Среднее геом: " << my_g << " с\n";

    std::cout << "\n[2/2] OpenBLAS dsymm:\n";
    for (int t : threads) {
        openblas_set_num_threads(t);
        std::vector<double> blas_times;
        std::cout << "\n--- Потоков: " << t << " ---\n";
        
        for (int i = 0; i < iterations; ++i) {
            auto C = C0;
            auto start = std::chrono::high_resolution_clock::now();
            cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, size, size, 
                       alpha, A.data(), size, B.data(), size, beta, C.data(), size);
            double t = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
            blas_times.push_back(t);
            std::cout << "  " << i+1 << ": " << std::setprecision(6) << t << " с\n";
        }
        
        double blas_g = geometric(blas_times);
        std::cout << "Ср. геом: " << blas_g << " с\n";
        std::cout << "my_symm/BLAS: " << (my_g/blas_g)*100 << "%\n";
    }

    std::cout << "\n=== Проверка корректности ===\n";
    const int testM = 4;
    std::vector<double> testA(testM * testM, 1.0), testB(testM * testM, 2.0);
    std::vector<double> C1(testM * testM, 0.0), C2(testM * testM, 0.0);
    for (int i = 0; i < testM; ++i) testA[i + i * testM] = 5.0;
    
    my_symm(testM, testM, 1.0, testA.data(), testM, testB.data(), testM, 0.0, C1.data(), testM);
    cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, testM, testM, 1.0, 
                testA.data(), testM, testB.data(), testM, 0.0, C2.data(), testM);
    
    double max_diff = 0;
    for (int i = 0; i < testM * testM; ++i) 
        max_diff = std::max(max_diff, std::abs(C1[i] - C2[i]));
    std::cout << "Макс. разница: " << max_diff << " - " 
              << (max_diff < 1e-10 ? "OK" : "FAIL") << "\n";

    return 0;
}
