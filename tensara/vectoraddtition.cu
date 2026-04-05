#include <cuda_runtime.h>

__global__ void solve(const float *A, const float *B, float *C, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= n) return;
    C[i] = A[i]+B[i];
}

// Note: d_input1, d_input2, d_output are device pointers
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    int grid = (n+255)/256, block = 256;

    solve<<<grid, block>>>(d_input1, d_input2, d_output, n);
    cudaDeviceSynchronize();
}