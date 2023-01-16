#ifndef HOST_UTILITIES_HPP
#define HOST_UTILITIES_HPP

#include <cuda_runtime.h>
#include <iostream>

#define CUDASafeCall() cuda_safe_call(cudaGetLastError(), __FILE__, __LINE__);

inline void cuda_safe_call(const cudaError& error, const char* const file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error %s : %d %s\n", file, line, cudaGetErrorString(error));
    }
}

#endif // HOST_UTILITIES_HPP
