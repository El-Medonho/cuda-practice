#include "bits/stdc++.h"

using namespace std;

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_int_distribution<int> uid(0, 1<<30);

// A, B and C should be pointers to positions in GPU memory
__global__ void MatrixMultiplication(int *A, int *B, int *C, int n, int m) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sum = 0;
    printf("%d %d\n", tx, ty);
    for(int i = 0; i < m; i++) sum += A[ty*m+i] * B[i*n+tx]; 
    C[ty*n+tx] = sum;

    return;
}

signed main(){
    bool receiveInput = false;
    receiveInput = true;
    
    int n, m; 
    if(receiveInput) cin >> n >> m;
    else m = 4, n = 8;

    int *N = new int[n*m], *M = new int[m*n], *P = new int[n*n];

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++) {
            if(receiveInput) cin >> N[i*m+j];
            else N[i*m+j] = uid(rng);
        }
    }

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++) {
            if(receiveInput) cin >> M[i*n+j];
            else M[i*n+j] = uid(rng);
        }
    }
    
    int size = n * m * sizeof(int), finalSize = n * n * sizeof(int);
    int *Nd, *Md, *Pd;
    cudaMalloc((void**)&Nd, size); cudaMalloc((void**)&Md, size); cudaMalloc((void**)&Pd, finalSize);
    
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    
    // Setup the grid to run the function

    dim3 dimBlock(n, n);
    dim3 dimGrid(1, 1);

    MatrixMultiplication<<<dimGrid, dimBlock>>>(Nd, Md, Pd, n, m);
    cudaDeviceSynchronize();
    
    cudaMemcpy(P, Pd, finalSize, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++) cout << P[i*n+j] << ' ';
        cout << '\n';
    }
    
    cudaFree(Nd); cudaFree(Md); cudaFree(Pd);
    
    return 0;
}