#include "bits/stdc++.h"
#include <omp.h>

using namespace std;

/// @brief Esta é um função executada em device(gpu). Funções executadas em devices se chamam kernel.
/// @param A Matriz A de entrada da multiplicação de matrizes
/// @param B Matriz B de entrada da multiplicação de matrizes
/// @param C Matriz C de saída da multiplicação de matrizes
/// n, k e m são as dimensões das matrizes. A é n*k, B é k*m, e C é n*m
__global__ void gpu_mult(int *A, int *B, int *C, int n, int k, int m){
    // Toda thread nesse kernel é representado por um par de coordenadas (x, y)
    int I = threadIdx.x + blockIdx.x * blockDim.x, J = threadIdx.y + blockIdx.y * blockDim.y;
    for(int i = I; i < n; i += gridDim.x * blockDim.x) {
        for(int j = J; j < m; j += gridDim.y * blockDim.y) {
            C[i*m+j] = 0;
            for(int p = 0; p < k; p++) C[i*m+j] += A[i*k+p] * B[p*m+j];
        }
    }
}

signed main(){



    return 0;
}