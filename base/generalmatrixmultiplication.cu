#include "bits/stdc++.h"

using namespace std;

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_int_distribution<int> uid(0, 1<<30);

void def_mult(int *A, int *B, int *C, int n, int k, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            C[i*m+j] = 0;
            for(int p = 0; p < k; p++) C[i*m+j] += A[i*k+p] * B[p*m+j];
        }
    }

    return;
}

__global__ void gpu_mult(int *A, int *B, int *C, int n, int k, int m){
    int i = threadIdx.x, j = threadIdx.y;
    C[i*m+j] = 0;
    for(int p = 0; p < k; p++) C[i*m+j] += A[i*k+p] * B[p*m+j];
}

__global__ void gpu_single_mult(int *A, int *B, int *C, int n, int k, int m){
    int q = threadIdx.x;
    int i = q/m, j = q%m;
    C[q] = 0;
    for(int p = 0; p < k; p++) C[q] += A[i*k+p] * B[p*m+j];
}


signed main(){

    int runCnt = 100, warmUpCnt = 5;

    float defTime = 0, gpuTime = 0, gpuSingleTime = 0;

    int n = 4, k = 6, m = 9;
    int A[n*k], B[k*m], CDef[n*m], CGpu[n*m], CGpuSingle[n*m];

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
    for(int i = 0; i < warmUpCnt; i++) def_mult(A, B, CDef, n, k, m);

    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        def_mult(A, B, CDef, n, k, m);
        auto end = chrono::steady_clock::now();
        defTime += (chrono::duration<double>(end-start)).count();
    }
    defTime /= runCnt;

    // run GPU matrix version
    int *Ad, *Bd, *Cd;

    cudaMalloc((void**) &Ad, n*k*sizeof(int));
    cudaMalloc((void**) &Bd, k*m*sizeof(int));
    cudaMalloc((void**) &Cd, n*m*sizeof(int));

    cudaMemcpy(Ad, A, n*k*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, m*k*sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1), threads(n, m), singleThreads(n*m);

    for(int i = 0; i < warmUpCnt; i++) {
        gpu_mult<<<grid, threads>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }
    
    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        gpu_mult<<<grid, threads>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        gpuTime += (chrono::duration<double>(end-start)).count();
    }
    gpuTime /= runCnt;

    cudaMemcpy(CGpu, Cd, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n*m; i++){
        if(CGpu[i] != CDef[i]){
            cout << "Gpu and Def answers differ!\n";
            return 0;
        }
    }

    // run GPU single line version

    cudaMemcpy(Cd, CGpuSingle, n*m*sizeof(int), cudaMemcpyHostToDevice);
    for(int i = 0; i < warmUpCnt; i++) {
        gpu_single_mult<<<grid, singleThreads>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
    }
    
    for(int i = 0; i < runCnt; i++){
        auto start = chrono::steady_clock::now();
        gpu_single_mult<<<grid, singleThreads>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize();
        auto end = chrono::steady_clock::now();
        gpuTime += (chrono::duration<double>(end-start)).count();
    }
    gpuTime /= runCnt;

    cudaMemcpy(CGpu, Cd, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n*m; i++){
        if(CGpu[i] != CDef[i]){
            cout << "Gpu single run and Def answers differ!\n";
            return 0;
        }
    }

    cout << "Runs successful! Results match!\n";
    cout << fixed << setprecision(4) << 
    "DefTime: " << defTime << '\n' << 
    "GpuTime: " << gpuTime << '\n' << 
    "GpuSingleTime: " << gpuSingleTime << '\n';

    return 0;
}