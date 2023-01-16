#ifndef CUDA_BILATERAL_TEXTURE_FILTER_CUH
#define CUDA_BILATERAL_TEXTURE_FILTER_CUH

#include <cstdint>

class CudaBilateralTextureFilter {
public:
    using ElemType = ::std::uint8_t;

    CudaBilateralTextureFilter(const int width, const int height, const int ksize = 9, const int nitr = 3);
    ~CudaBilateralTextureFilter();

    void execute(const ElemType* const d_src, ElemType* const d_dst);

protected:
    class Impl;
    Impl* impl_;
};

#endif // CUDA_BILATERAL_TEXTURE_FILTER_CUH
