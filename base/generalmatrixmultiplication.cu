#include "bits/stdc++.h"
#include <omp.h>
#include <cublas_v2.h>
#include <cmath>

using namespace std;

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_real_distribution<float> uid(-0.5f, 0.5f);
const float eps = 1e-1;

//#region 
#pragma region CPU_FUNCS

void def_mult(float *A, float *B, float *C, int n, int k, int m){
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
void opt_cpu_mult(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n, int k, int m) {
    
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
                        
                        float a_val = A[ii * k + pp];
                        
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

__global__ void gpu_mult(float *A, float *B, float *C, int n, int k, int m){
    int I = threadIdx.x + blockIdx.x * blockDim.x, J = threadIdx.y + blockIdx.y * blockDim.y;
    for(int i = I; i < n; i += gridDim.x * blockDim.x) {
        for(int j = J; j < m; j += gridDim.y * blockDim.y) {
            C[i*m+j] = 0;
            for(int p = 0; p < k; p++) C[i*m+j] += A[i*k+p] * B[p*m+j];
        }
    }
}

__global__ void gpu_mem_coalescing(float *A, float *B, float *C, int n, int k, int m){
    int J = threadIdx.x + blockIdx.x * blockDim.x, I = threadIdx.y + blockIdx.y * blockDim.y;
    for(int i = I; i < n; i += gridDim.x * blockDim.x) {
        for(int j = J; j < m; j += gridDim.y * blockDim.y) {
            C[i*m+j] = 0;
            for(int p = 0; p < k; p++) C[i*m+j] += A[i*k+p] * B[p*m+j];
        }
    }
}


__global__ void gpu_single_mult(float *A, float *B, float *C, int n, int k, int m){
    int Q = threadIdx.x + blockIdx.x * blockDim.x;

    for(int q = Q; q < n*m; q += gridDim.x * blockDim.x){
        C[q] = 0;
        int i = q/m, j = q%m;
        for(int p = 0; p < k; p++) C[q] += A[i*k+p] * B[p*m+j];
    }
}

// this function is not set to work in every matrix. All sides must be multiples of blockDim (prob 16)
__global__ void gpu_2d_tiling(float *A, float *B, float *C, int n, int k, int m){
    int blockSize = blockDim.x;
    __shared__ float As[16*16], Bs[16*16];
    A += blockIdx.x * blockSize * k; B += blockIdx.y * blockSize;
    C += blockIdx.x * blockSize * m + blockIdx.y * blockSize;

    float temp = 0;
    int d = 0;

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
__global__ void gpu_2d_tiling_mult_per_thread(float *A, float *B, float *C, int n, int k, int m){
    __shared__ float As[64*8], Bs[64*8];
    float Ct[8];
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
            float bs_val = Bs[ind%64 + i * 64];
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

// this function is not set to work in every matrix. All sides must be multiples of 64
__global__ void gpu_2d_blocktiling(float *A, float *B, float *C, int n, int k, int m){
    
    // Dimensions of a single thread block
    const int TN = 8, TM = 8;
    // Dimensions of output tile computed by whole block
    const int BN = 128, BM = 64;
    int id = threadIdx.x + threadIdx.y*blockDim.x;
    const int BK = 32;


    __shared__ float As[BN*BK], Bs[BK*BM];
    float Cr[TN*TM] = {0};
    float Ar[TN] = {0}, Br[TM] = {0};

    A += BN*blockIdx.x*k, B += BM*blockIdx.y;
    C += BN*blockIdx.x*m + BM*blockIdx.y;

    int rowA = id/BK, colA = id%BK;
    int rowB = id/64, colB = id%64;

    for(int d = 0; d < k; d += BK){
        __syncthreads();

        for(int j = 0; j*128 < BN*BK; j++){
            As[j*128 + rowA * BK + colA] = A[j*4*k + rowA * k + colA];
        }

        for(int j = 0; j*128 < BM*BK; j++){
            Bs[j*128 + rowB * BM + colB] = B[j*2*m + rowB * m + colB];
        }

        __syncthreads();

        for(int cDot = id&31, cnt = 0; cnt < BK; cnt++, cDot = (cDot == 31) ? 0 : cDot+1 ){
            for(int i = 0; i < TN; i++) Ar[i] = As[(threadIdx.x*TN+i)*BK + cDot];
            for(int i = 0, idx = id&7; i < TM; i++, idx = (idx == 7) ? 0 : idx+1) Br[idx] = Bs[(threadIdx.y*TM+idx) + cDot*BM];

            for(int i = 0; i < TN; i++){
                for(int j = 0; j < TM; j++){
                    Cr[i*8+j] += Ar[i]*Br[j];
                }
            }
        }

        A += BK; B += BK*m;
    }

    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 8; j++){
            C[(threadIdx.x*8+i)*m + threadIdx.y*8+j] = Cr[i*8+j];
        }
    }

    return;
}

const int TileWidth = 16;

signed main(int argc, char** argv){

    int runCnt = 20, warmUpCnt = 3;

    float cublasTime = 0, defTime = 0, geminiCpuTime = 0, gpuTime = 0, gpuSingleTime = 0, gpuMemCoalTime = 0, gpu2dTilingTime = 0, gpu2dTilingMultThreadTime = 0, gpu2dBlocktilingTime = 0;
    bool benchmarking = false;

    long long n = 1024, k = 2048, m = 512;
    if (argc >= 5) {
        benchmarking = stoi(argv[1]);
        n = stoi(argv[2]);
        k = stoi(argv[3]);
        m = stoi(argv[4]);
    }


    vector<float> A(n*k), B(k*m), CDef(n*m), CGpu(n*m), CGpuSingle(n*m), CGpuMemCoal(n*m), CGpu2dTiling(n*m), CGpu2dTilingMultThread(n*m), CGpu2dBlocktiling(n*m);
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

    
    
    if(n*m*k <= (1LL<<30)){
        // run CPU def version
        def_mult(A.data(), B.data(), CDef.data(), n, k, m);
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
    } else {
        defTime = -1.0;
        geminiCpuTime = -1.0;
    }
    
    // run GPU matrix version
    float *Ad, *Bd, *Cd;
    
    cudaMalloc((void**) &Ad, n*k*sizeof(float));
    cudaMalloc((void**) &Bd, k*m*sizeof(float));
    cudaMalloc((void**) &Cd, n*m*sizeof(float));
    
    cudaMemcpy(Ad, A.data(), n*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B.data(), m*k*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 grid((n + TileWidth - 1)/TileWidth, (m + TileWidth - 1)/TileWidth), gridSingle(n*m/(TileWidth*TileWidth)), 
    gridMemCoal((m + TileWidth - 1)/TileWidth, (n + TileWidth - 1)/TileWidth), grid2dTilingMultThread(n/64, m/64),
    block(TileWidth, TileWidth), blockSingle(TileWidth * TileWidth), block2dTilingMultThread(64, 8);
    dim3 grid2dBlockTiling(n/128, m/64), block2dBlockTiling(16, 8);
    
    #pragma region CUBLAS

    vector<float> CCublas(n*m, 0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // Warmup cuBLAS (Note the swap of A and B for Row-Major to Col-Major trick)
    for(int i = 0; i < warmUpCnt; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, Bd, m, Ad, k, &beta, Cd, m);
    }
    cudaDeviceSynchronize();

    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, Bd, m, Ad, k, &beta, Cd, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        cublasTime += (chrono::duration<double>(end-start)).count();
    }
    cublasTime /= runCnt;
    cudaMemcpy(CCublas.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);

    #pragma endregion CUBLAS
    
    
    if(n*m*k <= (1LL<<36)){
        
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
        
        cudaMemcpy(CGpu.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);
    
        
        // run GPU single line version
        
        cudaMemcpy(Cd, CGpuSingle.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);
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
        
        cudaMemcpy(CGpuSingle.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);
        
        
        // run GPU memory coalescing version
        
        cudaMemcpy(Cd, CGpuMemCoal.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);
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
        
        cudaMemcpy(CGpuMemCoal.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);

        // run GPU 2d tiling version
        
        cudaMemcpy(Cd, CGpu2dTiling.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);
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
        
        cudaMemcpy(CGpu2dTiling.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);
    } else{
        gpuTime = -1.0;
        gpuSingleTime = -1.0;
        gpuMemCoalTime = -1.0;
        gpu2dTilingTime = -1.0;
    }
    
    
    // run GPU 2d tiling with multiple work per processor version
    
    cudaMemcpy(Cd, CGpu2dTilingMultThread.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);
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
    
    cudaMemcpy(CGpu2dTilingMultThread.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);
    
    
    // run GPU 2d block tiling version
    
    cudaMemcpy(Cd, CGpu2dBlocktiling.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);
    for(int i = 0; i < warmUpCnt; i++) {
        gpu_2d_blocktiling<<<grid2dBlockTiling, block2dBlockTiling>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }
    
    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        gpu_2d_blocktiling<<<grid2dBlockTiling, block2dBlockTiling>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        gpu2dBlocktilingTime += (chrono::duration<double>(end-start)).count();
    }
    gpu2dBlocktilingTime /= runCnt;
    
    cudaMemcpy(CGpu2dBlocktiling.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);
    
    
    if(!benchmarking && (n*m*k <= (1<<30))){
        for(int i = 0; i < n*m; i++){
            if(abs(CGpu[i] - CDef[i]) > eps){
                cout << "Gpu and Def answers differ!\n";
                goto cleanup;
            }
        }
        for(int i = 0; i < n*m; i++){
            if(abs(CGpuSingle[i] - CDef[i]) > eps){
                cout << "Gpu single run and Def answers differ!\n";
                goto cleanup;
            }
        }
        for(int i = 0; i < n*m; i++){
            if(abs(CGpuMemCoal[i] - CDef[i]) > eps){
                cout << "Gpu memory coalescing and Def answers differ!\n";
                goto cleanup;
            }
        }
        for(int i = 0; i < n*m; i++){
            if(abs(CGpu2dTiling[i] - CDef[i]) > eps){
                cout << "Gpu 2d tiling and Def answers differ!\n";
                goto cleanup;
            }
        }
        for(int i = 0; i < n*m; i++){
            if(abs(CGpu2dTilingMultThread[i] - CDef[i]) > eps){
                cout << "Gpu 2d tiling with multiple work per processor and Def answers differ!\n";
                goto cleanup;
            }
        }
        for(int i = 0; i < n*m; i++){
            if(abs(CGpu2dBlocktiling[i] - CDef[i]) > eps){
                cout << "Gpu 2d block tiling and Def answers differ!\n";
                goto cleanup;
            }
        }
    }



    if(!benchmarking){
        cout << "Runs successful! Results match!\n";
        cout << fixed << setprecision(8) << 
        "Cpu naive time: " << defTime << '\n' << 
        "Cpu gemini time: " << geminiCpuTime << '\n' << 
        "Gpu naive time: " << gpuTime << '\n' << 
        "Gpu naive 1d naive time: " << gpuSingleTime << '\n' <<
        "Gpu memory coalescing time: " << gpuMemCoalTime << '\n' <<
        "Gpu 2d tiling time: " << gpu2dTilingTime << '\n' <<
        "Gpu 2d tiling time with multiple work per processor: " << gpu2dTilingMultThreadTime << '\n' << 
        "Gpu 2d block tiling time: " << gpu2dBlocktilingTime << '\n' <<
        "Gpu cuBLAS reference time: " << cublasTime << '\n'
        ;
    
        cout << "Printing first few results:\n";
        for(int i = 0; i < min((size_t)8, CDef.size()); i++) cout << CDef[i] << ' ';
        cout << '\n';
    }

cleanup:
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    if(benchmarking){
        cout << n << "," << k << "," << m << ","
             << fixed << setprecision(6)
             << defTime << "," 
             << geminiCpuTime << "," 
             << gpuTime << "," 
             << gpuSingleTime << ","
             << gpuMemCoalTime << ","
             << gpu2dTilingTime << ","
             << gpu2dTilingMultThreadTime << "," 
             << gpu2dBlocktilingTime << "," 
             << cublasTime << "\n";
    }

    return 0;
}