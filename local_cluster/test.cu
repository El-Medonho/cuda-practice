#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>
#include <unistd.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        cout << "[Node " << rank << " / " << size << "] Host: " << hostname 
            << " -> No GPU was found!" << endl;
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cout << "[Node " << rank << " / " << size << "] Host: " << hostname 
            << " -> GPU ready: " << prop.name << endl;
    }

    MPI_Finalize();
    return 0;
}