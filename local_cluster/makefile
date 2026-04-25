-include cluster_config.mk

CUDA_PATH = /usr/local/cuda
MPI_INCLUDE = -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi
WORKER_IP ?= 127.0.0.1
WORKER_USER ?= user

.PHONY: all clean run

all: test

test: test.cu
	nvcc -c test.cu $(MPI_INCLUDE) -o test.o
	mpicxx test.o -o test -L$(CUDA_PATH)/lib64 -lcudart_static -lrt -lpthread -ldl

clean:
	rm -f test.o test

run: test
	scp ./test bfred@100.107.8.11:~/cuda-practice/local_cluster/
	mpirun --mca btl tcp,self --mca btl_tcp_if_include tailscale0 --mca oob_tcp_if_include tailscale0 --prefix /usr -x LD_LIBRARY_PATH=/usr/lib/wsl/lib --hostfile hostfile ./test