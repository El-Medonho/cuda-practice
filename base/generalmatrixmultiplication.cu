#include "bits/stdc++.h"
#include <omp.h>

using namespace std;

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_int_distribution<int> uid(0, 1<<30);

//#region 
#pragma region CPU_FUNCS

void def_mult(int *A, int *B, int *C, int n, int k, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            C[i*m+j] = 0;
            for(int p = 0; p < k; p++) C[i*m+j] += A[i*k+p] * B[p*m+j];
        }
    }

    return;
}

// I asked gemini to make the best cpu matrix multiplication that he could
// Using __restrict__ tells the compiler the arrays don't overlap in memory,
// which allows it to safely generate fast SIMD (vectorized) instructions.
void opt_cpu_mult(const int* __restrict__ A, const int* __restrict__ B, int* __restrict__ C, int n, int k, int m) {
    
    // 1. Initialize C array to 0 in parallel
    #pragma omp parallel for
    for (int i = 0; i < n * m; ++i) {
        C[i] = 0;
    }

    // 2. Block size for cache tiling (64 works beautifully for most L1/L2 caches)
    const int BLOCK_SIZE = 64; 

    // 3. Collapse the two outer loops to distribute the 64x64 blocks across CPU threads
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < m; j += BLOCK_SIZE) {
            for (int p = 0; p < k; p += BLOCK_SIZE) {
                
                // 4. Multiply the mini 64x64 blocks
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, n); ++ii) {
                    for (int pp = p; pp < std::min(p + BLOCK_SIZE, k); ++pp) {
                        
                        int a_val = A[ii * k + pp];
                        
                        // Because 'jj' is the innermost loop, B and C are accessed perfectly sequentially!
                        #pragma omp simd
                        for (int jj = j; jj < std::min(j + BLOCK_SIZE, m); ++jj) {
                            C[ii * m + jj] += a_val * B[pp * m + jj];
                        }
                    }
                }
                
            }
        }
    }
}

#pragma endregion CPU_FUNCS
// #endregion

__global__ void gpu_mult(int *A, int *B, int *C, int n, int k, int m){
    int I = threadIdx.x + blockIdx.x * blockDim.x, J = threadIdx.y + blockIdx.y * blockDim.y;
    for(int i = I; i < n; i += gridDim.x * blockDim.x) {
        for(int j = J; j < m; j += gridDim.y * blockDim.y) {
            C[i*m+j] = 0;
            for(int p = 0; p < k; p++) C[i*m+j] += A[i*k+p] * B[p*m+j];
        }
    }
}

__global__ void gpu_mem_coalescing(int *A, int *B, int *C, int n, int k, int m){
    int J = threadIdx.x + blockIdx.x * blockDim.x, I = threadIdx.y + blockIdx.y * blockDim.y;
    for(int i = I; i < n; i += gridDim.x * blockDim.x) {
        for(int j = J; j < m; j += gridDim.y * blockDim.y) {
            C[i*m+j] = 0;
            for(int p = 0; p < k; p++) C[i*m+j] += A[i*k+p] * B[p*m+j];
        }
    }
}


__global__ void gpu_single_mult(int *A, int *B, int *C, int n, int k, int m){
    int Q = threadIdx.x + blockIdx.x * blockDim.x;

    for(int q = Q; q < n*m; q += gridDim.x * blockDim.x){
        C[q] = 0;
        int i = q/m, j = q%m;
        for(int p = 0; p < k; p++) C[q] += A[i*k+p] * B[p*m+j];
    }
}

// this function is not set to work in every matrix. All sides must be multiples of blockDim (prob 16)
__global__ void gpu_2d_tiling(int *A, int *B, int *C, int n, int k, int m){
    int blockSize = blockDim.x;
    __shared__ int As[16*16], Bs[16*16];
    A += blockIdx.x * blockSize * k; B += blockIdx.y * blockSize;
    C += blockIdx.x * blockSize * m + blockIdx.y * blockSize;

    int temp = 0, d = 0;

    while(d < k){
        __syncthreads();
        
        As[threadIdx.y*blockSize+threadIdx.x] = A[threadIdx.y*k + threadIdx.x];
        Bs[threadIdx.y*blockSize+threadIdx.x] = B[threadIdx.y*m + threadIdx.x];
    
        __syncthreads();

        for(int i = 0; i < blockSize; i++) temp += As[threadIdx.x * blockSize + i] * Bs[threadIdx.y + i*blockSize];
        d += blockSize;
        A += blockSize; B += blockSize * m;
    }

    C[threadIdx.x * m + threadIdx.y] = temp;
}


// this function is not set to work in every matrix. All sides must be multiples of 64
__global__ void gpu_2d_tiling_mult_per_thread(int *A, int *B, int *C, int n, int k, int m){
    __shared__ int As[64*8], Bs[64*8];
    int Ct[8];
    for(int i = 0; i < 8; i++) Ct[i] = 0;
    int ind = threadIdx.y * blockDim.x + threadIdx.x;

    A += blockIdx.x * 64 * k; B += blockIdx.y * 64;
    C += blockIdx.x * 64 * m + blockIdx.y * 64;

    int d = 0;

    while(d < k){
        __syncthreads();
        
        As[ind] = A[ind/8*k + ind%8];
        Bs[threadIdx.y*blockDim.x+threadIdx.x] = B[threadIdx.y*m + threadIdx.x];
    
        __syncthreads();

        for(int i = 0; i < 8; i++){
            int bs_val = Bs[ind%64 + i * 64];
            for(int j = 0; j < 8; j++){
                Ct[j] += As[i + j*8 + ind/64*8*8] * bs_val;
            } 
        }
        A += 8; B += 8 * m;
        d += 8;
    }

    for(int i = 0; i < 8; i++){
        C[ind%64 + ind/64*m*8 + i*m] = Ct[i];
    }
}

const int TileWidth = 16;

signed main(){

    int runCnt = 100, warmUpCnt = 5;

    float defTime = 0, geminiCpuTime = 0, gpuTime = 0, gpuSingleTime = 0, gpuMemCoalTime = 0, gpu2dTilingTime = 0, gpu2dTilingMultThreadTime = 0;

    int n = 256, k = 512, m = 128;
    // int n = 128, k = 256, m = 64;
    vector<int> A(n*k), B(k*m), CDef(n*m), CGpu(n*m), CGpuSingle(n*m), CGpuMemCoal(n*m), CGpu2dTiling(n*m), CGpu2dTilingMultThread(n*m);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < k; j++) {
            A[i*k+j] = uid(rng);
        }
    }

    for(int i = 0; i < k; i++){
        for(int j = 0; j < m; j++) {
            B[i*m+j] = uid(rng);
        }
    }

    // run CPU def version
    for(int i = 0; i < warmUpCnt; i++) def_mult(A.data(), B.data(), CDef.data(), n, k, m);

    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        def_mult(A.data(), B.data(), CDef.data(), n, k, m);
        auto end = chrono::steady_clock::now();
        defTime += (chrono::duration<double>(end-start)).count();
    }
    defTime /= runCnt;

    // run gemini CPU version
    for(int i = 0; i < warmUpCnt; i++) opt_cpu_mult(A.data(), B.data(), CDef.data(), n, k, m);

    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        opt_cpu_mult(A.data(), B.data(), CDef.data(), n, k, m);
        auto end = chrono::steady_clock::now();
        geminiCpuTime += (chrono::duration<double>(end-start)).count();
    }
    geminiCpuTime /= runCnt;

    // run GPU matrix version
    int *Ad, *Bd, *Cd;

    cudaMalloc((void**) &Ad, n*k*sizeof(int));
    cudaMalloc((void**) &Bd, k*m*sizeof(int));
    cudaMalloc((void**) &Cd, n*m*sizeof(int));

    cudaMemcpy(Ad, A.data(), n*k*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B.data(), m*k*sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid((n + TileWidth - 1)/TileWidth, (m + TileWidth - 1)/TileWidth), gridSingle(n*m/(TileWidth*TileWidth)), 
    gridMemCoal((m + TileWidth - 1)/TileWidth, (n + TileWidth - 1)/TileWidth), grid2dTilingMultThread(n/64, m/64),
    block(TileWidth, TileWidth), blockSingle(TileWidth * TileWidth), block2dTilingMultThread(64, 8);

    // cout << "Tamanhos: " << grid.x << ' ' << grid.y << '\n' << gridSingle.x << '\n';

    for(int i = 0; i < warmUpCnt; i++) {
        gpu_mult<<<grid, block>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }
    
    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        gpu_mult<<<grid, block>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        gpuTime += (chrono::duration<double>(end-start)).count();
    }
    gpuTime /= runCnt;

    cudaMemcpy(CGpu.data(), Cd, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n*m; i++){
        if(CGpu[i] != CDef[i]){
            cout << "Gpu and Def answers differ!\n";
            goto cleanup;
        }
    }

    // run GPU single line version

    cudaMemcpy(Cd, CGpuSingle.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);
    for(int i = 0; i < warmUpCnt; i++) {
        gpu_single_mult<<<gridSingle, blockSingle>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }
    
    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        gpu_single_mult<<<gridSingle, blockSingle>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        gpuSingleTime += (chrono::duration<double>(end-start)).count();
    }
    gpuSingleTime /= runCnt;

    cudaMemcpy(CGpuSingle.data(), Cd, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n*m; i++){
        if(CGpuSingle[i] != CDef[i]){
            cout << "Gpu single run and Def answers differ!\n";
            goto cleanup;
        }
    }

    // run GPU memory coalescing version

    cudaMemcpy(Cd, CGpuMemCoal.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);
    for(int i = 0; i < warmUpCnt; i++) {
        gpu_mem_coalescing<<<gridMemCoal, block>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }
    
    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        gpu_mem_coalescing<<<gridMemCoal, block>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        gpuMemCoalTime += (chrono::duration<double>(end-start)).count();
    }
    gpuMemCoalTime /= runCnt;

    cudaMemcpy(CGpuMemCoal.data(), Cd, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n*m; i++){
        if(CGpuMemCoal[i] != CDef[i]){
            cout << "Gpu memory coalescing and Def answers differ!\n";
            goto cleanup;
        }
    }

    // run GPU 2d tiling version

    cudaMemcpy(Cd, CGpu2dTiling.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);
    for(int i = 0; i < warmUpCnt; i++) {
        gpu_2d_tiling<<<grid, block>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }
    
    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        gpu_2d_tiling<<<grid, block>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        gpu2dTilingTime += (chrono::duration<double>(end-start)).count();
    }
    gpu2dTilingTime /= runCnt;

    cudaMemcpy(CGpu2dTiling.data(), Cd, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n*m; i++){
        if(CGpu2dTiling[i] != CDef[i]){
            cout << "Gpu 2d tiling and Def answers differ!\n";
            goto cleanup;
        }
    }

    // run GPU 2d tiling with multiple work per processor version

    cudaMemcpy(Cd, CGpu2dTilingMultThread.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);
    for(int i = 0; i < warmUpCnt; i++) {
        gpu_2d_tiling_mult_per_thread<<<grid2dTilingMultThread, block2dTilingMultThread>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }
    
    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        gpu_2d_tiling_mult_per_thread<<<grid2dTilingMultThread, block2dTilingMultThread>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        gpu2dTilingMultThreadTime += (chrono::duration<double>(end-start)).count();
    }
    gpu2dTilingMultThreadTime /= runCnt;

    cudaMemcpy(CGpu2dTilingMultThread.data(), Cd, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n*m; i++){
        if(CGpu2dTilingMultThread[i] != CDef[i]){
            cout << "Gpu 2d tiling with multiple work per processor and Def answers differ!\n";
            goto cleanup;
        }
    }

    cout << "Runs successful! Results match!\n";
    cout << fixed << setprecision(8) << 
    "Cpu naive time: " << defTime << '\n' << 
    "Cpu gemini time: " << geminiCpuTime << '\n' << 
    "Gpu naive time: " << gpuTime << '\n' << 
    "Gpu naive 1d naive time: " << gpuSingleTime << '\n' <<
    "Gpu memory coalescing time: " << gpuMemCoalTime << '\n' <<
    "Gpu 2d tiling time: " << gpu2dTilingTime << '\n' <<
    "Gpu 2d tiling time with multiple work per processor: " << gpu2dTilingMultThreadTime << '\n'
    ;

    cout << "Printing first few results:\n";
    for(int i = 0; i < min((size_t)8, CDef.size()); i++) cout << CDef[i] << ' ';
    cout << '\n';

cleanup:
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    return 0;
}