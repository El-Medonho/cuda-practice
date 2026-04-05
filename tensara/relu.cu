#include <cuda_runtime.h>

__global__ void solve(const float* input, float* output){
    size_t p = threadIdx.x + blockIdx.x * blockDim.x;
    output[p] = max((float)0, input[p]);
}

// Note: input, output are device pointers
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t sz = n*m;
    int grid = (sz+127)/128, block = 128;
    solve<<<grid, block>>>(input, output);
}