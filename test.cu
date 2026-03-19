#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    
    if (error_id != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error_id) << "\n";
        return 1;
    }
    
    std::cout << "Success! Found " << deviceCount << " CUDA capable device(s).\n";
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name << "\n";
    }
    
    return 0;
}