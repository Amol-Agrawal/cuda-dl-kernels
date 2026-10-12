NVCC  ?= nvcc
ARCH  ?= sm_75
FLAGS  = -O3 -std=c++17 -arch=$(ARCH) -lineinfo

all: sgemm hgemm softmax

sgemm: src/sgemm.cu src/kernels.cuh src/common.cuh
	$(NVCC) $(FLAGS) -o $@ src/sgemm.cu -lcublas

hgemm: src/hgemm.cu src/kernels_wmma.cuh src/common.cuh
	$(NVCC) $(FLAGS) -o $@ src/hgemm.cu -lcublas

softmax: src/softmax.cu src/kernels_softmax.cuh src/common.cuh
	$(NVCC) $(FLAGS) -o $@ src/softmax.cu -lcublas

clean:
	rm -f sgemm hgemm softmax
