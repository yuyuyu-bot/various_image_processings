#ifndef CUDA_ADAPTIVE_BILATERAL_FILTER_HPP
#define CUDA_ADAPTIVE_BILATERAL_FILTER_HPP

#include <cstdint>
#include <memory>

class CudaAdaptiveBilateralFilter {
public:
    CudaAdaptiveBilateralFilter(
        const int width,
        const int height,
        const int ksize = 9,
        const float sigma_space = 10.f,
        const float sigma_color = 30.f);
    ~CudaAdaptiveBilateralFilter();

    void execute(
        const std::uint8_t* const d_src,
        std::uint8_t* const d_dst) const;

protected:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // CUDA_ADAPTIVE_BILATERAL_FILTER_HPP
