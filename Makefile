all: 
	nvcc cuda_copyArr.cu -std=c++11
	./a.out