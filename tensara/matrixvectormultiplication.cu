#include <cuda_runtime.h>

__global__ void solve(const float *A, const float *B, float *C, size_t m, size_t k){
    __shared__ float Bs[256];
    __shared__ float As[256][33];
    A += blockIdx.x * blockDim.x * k;
    float ans = 0;
    int cc = 0, tgb = 0, tga = 0, mn = blockDim.x/8, j = 0, jj = 0;
    while(cc < k){
        __syncthreads();
        if(cc == tgb){
            jj = 0;
            tgb += blockDim.x;
            Bs[threadIdx.x] = B[cc+threadIdx.x];
            __syncthreads();
        }
        if(cc == tga){
            j = 0;
            tga += mn;
            int add = threadIdx.x % mn;
            for(int i = threadIdx.x/mn; i < blockDim.x; i += 8){
                As[i][add] = A[i*k+add];
            }
            A += mn;
            __syncthreads();
        }

        ans += As[threadIdx.x][j] * Bs[jj];
        j++; jj++; cc++;
    }
    C[blockIdx.x * blockDim.x + threadIdx.x] = ans;
}

// Note: input_a, input_b, output_c are device pointers
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    dim3 block(256), grid((m+255)/256);
    solve<<<grid, block>>>(input_a, input_b, output_c, m, k);
}