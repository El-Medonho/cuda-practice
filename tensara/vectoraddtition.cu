#include <cuda_runtime.h>
#include "bits/stdc++.h"

using namespace std;

__global__ void solve(const float *A, const float *B, float *C, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= n) return;
    C[i] = A[i]+B[i];
}

// Note: d_input1, d_input2, d_output are device pointers
// extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
//     int grid = (n+255)/256, block = 256;

//     solve<<<grid, block>>>(d_input1, d_input2, d_output, n);
//     cudaDeviceSynchronize();
// }

void execute_kernel(const float *A, const float *B, float *C, int n){
    // the 'd' suffix indicates that this points to a device variable
    float *Ad, *Bd, *Cd;

    // this allocs VRAM memory to our variables
    cudaMalloc((void**)Ad, n*sizeof(float));
    cudaMalloc((void**)Bd, n*sizeof(float));
    cudaMalloc((void**)Cd, n*sizeof(float));

    // we copy the input vectors to device
    cudaMemcpy(Ad, A, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, n*sizeof(float), cudaMemcpyHostToDevice);

    // now we should determine the size of the grid and block
    // gridDim determines the amount of blocks in the grid
    // blockDim determines the amount of threads in a block. This number should be a multiple of 32

    dim3 gridDim((n+63)/64), blockDim(64);

    // execute the kernel
    solve<<<gridDim, blockDim>>>(Ad, Bd, Cd, n);

    // copy the result vector Cd to host

    cudaMemcpy(C, Cd, n*sizeof(float), cudaMemcpyDeviceToHost);

    return;
}