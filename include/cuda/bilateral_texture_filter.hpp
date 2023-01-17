#ifndef CUDA_BILATERAL_TEXTURE_FILTER_CUH
#define CUDA_BILATERAL_TEXTURE_FILTER_CUH

#include <cstdint>

class CudaBilateralTextureFilter {
public:
    CudaBilateralTextureFilter(const int width, const int height, const int ksize = 9, const int nitr = 3);
    ~CudaBilateralTextureFilter();

    void execute(const std::uint8_t* const d_src, std::uint8_t* const d_dst);

protected:
    class Impl;
    Impl* impl_;
};

#endif // CUDA_BILATERAL_TEXTURE_FILTER_CUH
