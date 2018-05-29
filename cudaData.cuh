
/*****************************************************************
Name : cudaData.cuh
Date : 2017/12/19
By   : CharlotteHonG
Final: 2017/12/19
*****************************************************************/
// Cuda 記憶體自動管理程序
template <class T>
class CudaData {
public:
    CudaData(){}
    CudaData(size_t size){
        malloc(size);
    }
    CudaData(T* dataIn ,size_t size): len(size){
        memcpyInAuto(dataIn, size);
    }
    ~CudaData(){
        if(gpuData!=nullptr) {
            cudaFree(gpuData);
            gpuData = nullptr;
            len = 0;
        }
    }
public:
    void malloc(size_t size) {
        this->~CudaData();
        len = size;
        cudaMalloc((void**)&gpuData, size*sizeof(T));
    }
    void memcpyIn(T* dataIn ,size_t size) {
        if(size > len) {throw out_of_range("memcpyIn input size > curr size.");}
        cudaMemcpy(gpuData, dataIn, size*sizeof(T), cudaMemcpyHostToDevice);
    }
    void memcpyInAuto(T* dataIn ,size_t size) {
        malloc(size);
        memcpyIn(dataIn, size);
    }
    void memcpyOut(T* dataIn ,size_t size) {
        cudaMemcpy(dataIn, gpuData, size*sizeof(T), cudaMemcpyDeviceToHost);
    }
    void memset(int value, size_t size) {
        if(size>len) {
            throw out_of_range("memset input size > curr size.");
        }
        cudaMemset(gpuData, value, size*sizeof(T));
    }
    size_t size() {
        return this->len;
    }
public:
    operator T*() {
        return gpuData;
    }
private:
    T* gpuData;
    size_t len=0;
};