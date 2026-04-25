Repo for my codes while learning cuda.

To have better results while benchmarking, use this:
nvcc example.cu -lcublas -O3 -arch=native
-lcublas: Links the cuBLAS library.
-O3: Applies maximum optimization to your host code.
-arch=native: Tells the compiler to optimize the PTX code specifically for the GPU architecture you currently have installed in your machine.
If you would like to profile your kernels, add the -lineinfo flag.
ncu --page details -f -o best-gmm.profout ./best-gmm