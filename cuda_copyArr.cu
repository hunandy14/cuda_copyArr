#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
using namespace std;

#include "Timer.hpp"
#include "cudaData.cuh"

__global__ void cudacopy(float* b, float* a, int size){
    // 乾式寫法
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size){
        b[idx]=a[idx];
    }
    // 迴圈寫法
    for(int i=threadIdx.x; i<size; i+=blockDim.x){
        b[i]=a[i];
    }
}
void cpucopoy(float* b, float* a, int size) {
    for(int i=0; i<size; ++i){
        b[i]=a[i];
    }
}
void testCuda(size_t size) {
    Timer T;

    // 配置主機記憶體
    vector<float> img_data(size), cpu_data(size), gpu_data(size);
    float* a = img_data.data(); // 原始資料
    float* b = cpu_data.data(); // CPU計算後資料
    float* c = gpu_data.data(); // GPU輸出回來資料
    // 設置初值
    float test_val=7;
    for(int i=0; i<size; i++){
        a[i]=test_val;
    }

    // 配置顯示記憶體, 載入資料.
    T.start();
    CudaData<float> gpuDataIn(a, size), gpuDataOut(size);
    gpuDataOut.memset(0, size);
    T.print("  Cuda Data malloc and copy");

    // 網格區塊設定. (與 kernel for 的次數有關)
    const size_t blkDim=16;
    int grid(size/blkDim+1);  // 網格要含蓋所有範圍, 所以除完要加 1.
    int block(blkDim);        // 區塊設定 16x16.

    // Cuda Kernel 執行運算
    T.start();
    cudacopy<<<grid,block>>>(gpuDataOut, gpuDataIn, size);
    T.print("  Cuda-copy");
    // 取出GPU資料
    gpuDataOut.memcpyOut(c, size);

    // CPU 執行運算
    T.start();
    cpucopoy(b, a, size);
    T.print("  Cpu-copy");

    // 測試
    bool f=0;
    for(size_t i = 0; i < size; i++) {
        if(c[i] != b[i]) {f=1;}
        cout << a[i] << "| ";
        cout << c[i] << ", ";
        cout << b[i] << endl;
    }

    // 測試報告
    if(f==0) {
        cout << "test ok" << endl;
    } else {
        cout << "test Error" << endl;
    }
}

int main(){
    Timer T; 
    testCuda(10);
    T.print("ALL time.");

    system("pause");
    return 0;
}