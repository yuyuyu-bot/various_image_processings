#ifndef BILATERAL_FILTER_IMPL_CUH
#define BILATERAL_FILTER_IMPL_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "cuda/bilateral_filter.hpp"

class CudaBilateralFilter::Impl {
public:
    Impl(
        const int width,
        const int height,
        const int ksize = 9,
        const float sigma_space = 10.f,
        const float sigma_color = 30.f);

    void bilateral_filter(
        const std::uint8_t* const d_src,
        std::uint8_t* const d_dst) const;

    void joint_bilateral_filter(
        const std::uint8_t* const d_src,
        const std::uint8_t* const d_guide,
        std::uint8_t* const d_dst) const;

private:
    const int   width_;
    const int   height_;
    const int   ksize_;
    const float sigma_space_;
    const float sigma_color_;

    thrust::device_vector<float> d_kernel_space_;
    thrust::device_vector<float> d_kernel_color_table_;
};

#endif // BILATERAL_FILTER_IMPL_CUH
