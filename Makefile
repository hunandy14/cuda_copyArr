# /*****************************************************************
# Name : 
# Date : 2018/05/29
# By   : CharlotteHonG
# Final: 2018/05/29
# *****************************************************************/
NVCCFLAGS :=
NVCCFLAGS += -std=c++11
NVCCFLAGS += -Xcompiler -fopenmp

# ================================================================
all: uda_copyArr.out

run: uda_copyArr.out
	./uda_copyArr.out
clear:
	rm -f *.o *.out

# ================================================================
uda_copyArr.out: cuda_copyArr.o
	nvcc $(NVCCFLAGS) *.o -o uda_copyArr.out

cuda_copyArr.o: cuda_copyArr.cu
	nvcc $(NVCCFLAGS) -c cuda_copyArr.cu
