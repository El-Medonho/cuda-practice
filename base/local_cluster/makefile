-include cluster_config.mk

WORKER_IP ?= 127.0.0.1
WORKER_USER ?= bfred
WORKER_DIR ?= ~/cuda-practice/local_cluster/

CUDA_PATH ?= /usr/local/cuda
MPI_INCLUDE = -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi

NVCC = nvcc
MPICXX = mpicxx
NVCC_FLAGS = -O3 $(MPI_INCLUDE) 
LD_FLAGS = -L$(CUDA_PATH)/lib64 -lcudart_static -lrt -lpthread -ldl
MPI_RUN_FLAGS = --mca btl tcp,self --mca btl_tcp_if_include tailscale0 --mca oob_tcp_if_include tailscale0 --prefix /usr -x LD_LIBRARY_PATH=/usr/lib/wsl/lib --hostfile hostfile

TARGETS = test clusteredgemm

.PHONY: all clean run-test run-gemm

all: $(TARGETS)
%.o: %.cu
	$(NVCC) -c $< $(NVCC_FLAGS) -o $@

%: %.o
	$(MPICXX) $< -o $@ $(LD_FLAGS)

clean:
	rm -f *.o $(TARGETS)

run-test: test
	scp ./test $(WORKER_USER)@$(WORKER_IP):$(WORKER_DIR)
	mpirun $(MPI_RUN_FLAGS) ./test

run-gemm: clusteredgemm
	scp ./clusteredgemm $(WORKER_USER)@$(WORKER_IP):$(WORKER_DIR)
	mpirun $(MPI_RUN_FLAGS) ./clusteredgemm