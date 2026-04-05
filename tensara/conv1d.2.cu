#include <cuda_runtime.h>

__global__ void solve(const float* A, const float* B, float* C, const size_t N, const size_t K, const int r){
    size_t p = threadIdx.x + ((size_t)blockIdx.x) * blockDim.x;
    int i = p/K, j = p%K;
    int i2 = i+j-r;
    if(i2 < 0 || i2 >= N || i >= N) return;
    atomicAdd(C+i, A[i2]*B[j]);
}

// Note: A, B, C are device pointers
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    int r = (K-1)/2;
    size_t sz = N*K;
    size_t grid = (sz+127)/128, block = 128;
    solve<<<grid, block>>>(A, B, C, N, K, r);
    cudaDeviceSynchronize();
}