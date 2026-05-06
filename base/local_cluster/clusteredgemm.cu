#include "bits/stdc++.h"
#include <omp.h>
#include <mpi.h>

using namespace std;

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
uniform_real_distribution<float> uid(-0.5f, 0.5f);
const long long seed = 998244353, seed2 = 998244853, seed3 = 1e9+7;
const float half = (seed3+1)/2;

const float eps = 1.0f;
const int softSize = 512;

__global__ void GMM_hard(float *A, float *B, float *C, int n, int k, int m, bool baseline = false){
    if(blockIdx.x < softSize/128 && blockIdx.y < softSize/64 && !baseline) return;

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

/// @brief  This gives less work to the worse gpu
__global__ void GMM_soft(float *A, float *B, float *C, int n, int k, int m){
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

signed main(int argc, char** argv){
    int n = 8192, k = 65536, m = 8192;
    // int n = 16384, k = 32768, m = 16384;
    // int n = 16384, k = 16384, m = 16384;
    cout << "n = " << n << 
    ", m = " << m << 
    ", k = " << k << endl;
    
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(MPI_COMM_WORLD);
    double start_global = MPI_Wtime();
    
    
    if(rank == 0) {
        vector<float> A(n*k), B(k*m), C(n*m, 0);
        dim3 grid(n/128, m/64), block(16, 8);
        float *Ad, *Bd, *Cd;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < k; j++) {
                A[i*k+j] = ((seed*i*k+j)%seed3-half)/(seed3*10);
            }
        }
        for(int i = 0; i < k; i++){
            for(int j = 0; j < m; j++) {
                B[i*m+j] = ((seed2*i*k+j)%seed3-half)/(seed3*10);
            }
        }
        cudaMalloc(((void**) &Ad), n*k*sizeof(float));
        cudaMalloc(((void**) &Bd), m*k*sizeof(float));
        cudaMalloc(((void**) &Cd), n*m*sizeof(float));

        cudaMemcpy(Ad, A.data(), n*k*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Bd, B.data(), m*k*sizeof(float), cudaMemcpyHostToDevice);

        vector<float> Cl(softSize*softSize, 0);
        MPI_Request request; 

        MPI_Irecv(Cl.data(), softSize * softSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &request);

        double start_gpu = MPI_Wtime();
        GMM_hard<<<grid, block>>>(Ad, Bd, Cd, n, k, m);
        cudaDeviceSynchronize(); 
        double end_gpu = MPI_Wtime();

        cudaMemcpy(C.data(), Cd, n*m*sizeof(float), cudaMemcpyDeviceToHost);

        MPI_Wait(&request, MPI_STATUS_IGNORE);

        for(int i = 0; i < softSize; i++){
            for(int j = 0; j < softSize; j++){
                C[i*m + j] = Cl[i*softSize + j];
            }
        }

        double end_global = MPI_Wtime();
        
        cout << "[Rank 0 - RTX 5060 Ti] GPU time: " << (end_gpu - start_gpu) << " seconds\n";
        cout << "[Distributed] Total time: " << (end_global - start_global) <<  " seconds\n";

        // free some memory
        cudaFree(Ad);
        cudaFree(Bd);
        cudaFree(Cd);
        vector<float>().swap(A);
        vector<float>().swap(B);

        end_global = MPI_Wtime();

        vector<float> A_base(n*k), B_base(k*m), C_base(n*m, 0);
        float *Ad_base, *Bd_base, *Cd_base;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < k; j++) {
                A_base[i*k+j] = ((seed*i*k+j)%seed3-half)/(seed3*10);
            }
        }
        for(int i = 0; i < k; i++){
            for(int j = 0; j < m; j++) {
                B_base[i*m+j] = ((seed2*i*k+j)%seed3-half)/(seed3*10);
            }
        }
        cudaMalloc(((void**) &Ad_base), n*k*sizeof(float));
        cudaMalloc(((void**) &Bd_base), m*k*sizeof(float));
        cudaMalloc(((void**) &Cd_base), n*m*sizeof(float));

        cudaMemcpy(Ad_base, A_base.data(), n*k*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Bd_base, B_base.data(), m*k*sizeof(float), cudaMemcpyHostToDevice);

        double start_base = MPI_Wtime();
        GMM_hard<<<grid, block>>>(Ad_base, Bd_base, Cd_base, n, k, m, true);
        cudaDeviceSynchronize();
        double end_base = MPI_Wtime();

        cudaMemcpy(C_base.data(), Cd_base, n*m*sizeof(float), cudaMemcpyDeviceToHost);
        
        cout << "[Rank 0 - RTX 5060 Ti only] GPU time: " << (end_base - start_base) << " seconds\n";
        cout << "[Single] Total time: " << (MPI_Wtime() - end_global) << " seconds\n";

        float maxDiff = 0;
        for(int i = 0; i < n*m; i++){
            maxDiff = max(maxDiff, abs(C[i] - C_base[i]));
        }
        
        if(maxDiff < eps) cout << "=> Successfully validated \n";
        else cout << "Validation failed: " << maxDiff << "\n";

        cudaFree(Cd_base);
    }

    
    else{
        vector<float> Al(softSize*k), Bl(k*softSize), Cl(softSize*softSize, 0);
        float *Ad, *Bd, *Cd;
        for(int i = 0; i < softSize; i++){
            for(int j = 0; j < k; j++) {
                Al[i*k+j] = ((seed*i*k+j)%seed3-half)/(seed3*10);
            }
        }
        for(int i = 0; i < k; i++){
            for(int j = 0; j < softSize; j++) {
                Bl[i*softSize+j] = ((seed2*i*k+j)%seed3-half)/(seed3*10);
            }
        }

        cudaMalloc(((void**) &Ad), softSize*k*sizeof(float));
        cudaMalloc(((void**) &Bd), softSize*k*sizeof(float));
        cudaMalloc(((void**) &Cd), softSize*softSize*sizeof(float));

        cudaMemcpy(Ad, Al.data(), softSize*k*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Bd, Bl.data(), softSize*k*sizeof(float), cudaMemcpyHostToDevice);
        dim3 softGrid(softSize/64, softSize/64), softBlock(64, 8);

        double start_gpu = MPI_Wtime();
        cout << "[Rank 1 - GTX 1050 Ti] Checkpoint at " << (start_gpu - start_global) << " seconds.\n";
        GMM_soft<<<softGrid, softBlock>>>(Ad, Bd, Cd, softSize, k, softSize);
        cudaDeviceSynchronize();
        double end_gpu = MPI_Wtime();

        cudaMemcpy(Cl.data(), Cd, softSize*softSize*sizeof(float), cudaMemcpyDeviceToHost);

        MPI_Send(Cl.data(), softSize * softSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

        double end_global = MPI_Wtime();
        cout << "[Rank 1 - GTX 1050 Ti] GPU time: " << (end_gpu - start_gpu) << " seconds\n";
        cout << "[Rank 1 - GTX 1050 Ti] Ready at " << (end_global - start_global) << " seconds.\n";
    }

    MPI_Finalize();



    return 0;
}