#ifndef CUDA_BILATERAL_TEXTURE_FILTER_CUH
#define CUDA_BILATERAL_TEXTURE_FILTER_CUH

#include <cstdint>

namespace cuda {

class BilateralTextureFilter {
public:
    using ElemType = ::std::uint8_t;

    BilateralTextureFilter(const int width, const int height, const int ksize = 9, const int nitr = 3);
    ~BilateralTextureFilter();

    void execute(const ElemType* const d_src, ElemType* const d_dst);

private:
    class Impl;
    Impl* impl_;
};

} // namespace cuda

#endif // CUDA_BILATERAL_TEXTURE_FILTER_CUH
