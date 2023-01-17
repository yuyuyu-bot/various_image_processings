#ifndef CUDA_BILATERAL_FILTER_HPP
#define CUDA_BILATERAL_FILTER_HPP

#include <cstdint>

class CudaBilateralFilter {
public:
    CudaBilateralFilter(
        const int width,
        const int height,
        const int ksize = 9,
        const float sigma_space = 10.f,
        const float sigma_color = 30.f);
    ~CudaBilateralFilter();

    void bilateral_filter(
        const std::uint8_t* const d_src,
        std::uint8_t* const d_dst) const;

    void joint_bilateral_filter(
        const std::uint8_t* const d_src,
        const std::uint8_t* const d_guide,
        std::uint8_t* const d_dst) const;

protected:
    class Impl;
    Impl* impl_;
};

#endif // CUDA_BILATERAL_FILTER_HPP
