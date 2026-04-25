#include "bits/stdc++.h"
#include <omp.h>
#include <cublas_v2.h>
#include <cmath>

using namespace std;

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_real_distribution<float> uid(-0.5f, 0.5f);
const float eps = 1e-1;

__global__ void GMM(float *A, float *B, float *C, int n, int k, int m){
    
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
            // for(int i = 0, idx = id&7; i < TM; i++, idx = (idx == 7) ? 0 : idx+1) Br[idx] = Bs[(threadIdx.y*TM+idx) + cDot*BM];
            for(int i = 0; i < TM; i++) Br[i] = Bs[(threadIdx.y*TM+i) + cDot*BM];

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

/// @brief  This is the reference function for my GMM. I know that this function works
__global__ void GMM_ref(float *A, float *B, float *C, int n, int k, int m){
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

signed main(){

    // int runCnt = 100, warmUpCnt = 5;
    int runCnt = 1, warmUpCnt = 1;

    float defTime = 0, refTime = 0, cublasTime = 0;

    int n = 16384, k = 16384, m = 16384;
    // int n = 8192, k = 16384, m = 4096;
    // int n = 1024, k = 2048, m = 512;
    // int n = 256, k = 512, m = 128;
    
    vector<float> A(n*k), B(k*m), C(n*m, 0), CRef(n*m, 0), CCublas(n*m, 0);
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

    float *Ad, *Bd, *Cd;
    cudaMalloc((void**)&Ad, n*k*sizeof(float));
    cudaMalloc((void**)&Bd, k*m*sizeof(float));
    cudaMalloc((void**)&Cd, n*m*sizeof(float));

    cudaMemcpy(Ad, A.data(), n*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B.data(), k*m*sizeof(float), cudaMemcpyHostToDevice);

    #pragma region CUBLAS

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

    #pragma endregion CPU_FUNCS

    dim3 gridRef(n/64, m/64), blockRef(64, 8);
    GMM_ref<<<gridRef, blockRef>>>(Ad, Bd, Cd, n, k, m);
    for(int i = 0; i < warmUpCnt; i++) {
        GMM_ref<<<gridRef, blockRef>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }

    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        GMM_ref<<<gridRef, blockRef>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        refTime += (chrono::duration<double>(end-start)).count();
    }
    refTime /= runCnt;
    cudaMemcpy(CRef.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);

    dim3 grid(n/128, m/64), block(16, 8);
    cudaMemcpy(Cd, C.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);
    for(int i = 0; i < warmUpCnt; i++) {
        GMM<<<grid, block>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }

    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        GMM<<<grid, block>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        defTime += (chrono::duration<double>(end-start)).count();
    }
    defTime /= runCnt;
    cudaMemcpy(C.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n*m; i++){
        if(abs(C[i] - CRef[i]) > eps){
            cout << "GEMM results are different!\n";
            goto cleanup;
        }
        if(abs(CCublas[i] - CRef[i]) > eps){
            cout << "Cublas results are different!\n";
            goto cleanup;
        }
    }

    cout << "Runs successful! Results match!\n";
    cout << fixed << setprecision(10) << 
    "Reference time spent in ms: " << (refTime*1000) << '\n' <<
    "Time spent in ms: " << (defTime*1000) << '\n' <<
    "cuBLAS time spent in ms: " << (cublasTime*1000) << '\n';

    
    cleanup:

    cout << "Printing first few results:\n";
    for(int i = 0; i < min((size_t)8, CRef.size()); i++) cout << CRef[i] << ' ';
    cout << '\n';
    for(int i = 0; i < min((size_t)8, C.size()); i++) cout << C[i] << ' ';
    cout << '\n';
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    cublasDestroy(handle);

    return 0;
}