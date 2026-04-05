#include "bits/stdc++.h"

using namespace std;

// vector addition
__global__ void solve(int *A, int *B, int *C){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    C[i] = A[i]+B[i];
}

signed main(){
    int n; cin >> n;

    vector<int> a(n), b(n), c(n);
    int *A, *B, *C;

    cudaMalloc((void**)&A, n*sizeof(int));
    cudaMalloc((void**)&B, n*sizeof(int));
    cudaMalloc((void**)&C, n*sizeof(int));

    for(int &i: a) cin >> i;

    for(int &i: b) cin >> i;

    cudaMemcpy(A, a.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B, b.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid((n+511)/512), block(512);

    solve<<<grid, block>>>(A, B, C);

    cudaMemcpy(c.data(), C, n*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i: c) cout << i << ' ';
    cout << '\n';

    return 0;
}