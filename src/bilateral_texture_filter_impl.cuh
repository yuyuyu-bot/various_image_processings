#ifndef BILATERAL_TEXTURE_FILTER_IMPL_CUH
#define BILATERAL_TEXTURE_FILTER_IMPL_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "cuda/bilateral_filter.hpp"
#include "cuda/bilateral_texture_filter.hpp"

class CudaBilateralTextureFilter::Impl {
public:
    Impl(const int width, const int height, const int ksize = 9, const int nitr = 3);
    ~Impl();

    void execute(
        const std::uint8_t* const d_src,
        std::uint8_t* const d_dst);

    void compute_blur_and_rtv(
        const thrust::device_vector<std::uint8_t>& d_image,
        const thrust::device_vector<float>& d_magnitude,
        thrust::device_vector<float>& d_blurred,
        thrust::device_vector<float>& d_rtv);

    void compute_guide(
        const thrust::device_vector<float>& d_blurred,
        const thrust::device_vector<float>& d_rtv,
        thrust::device_vector<std::uint8_t>& d_guide);

private:
    static constexpr auto jbf_sigma_color = 1.73205080757f; // sqrt(3)

    const int width_;
    const int height_;
    const int ksize_;
    const int nitr_;

    thrust::device_vector<std::uint8_t> d_src_n_;
    thrust::device_vector<std::uint8_t> d_dst_n_;
    thrust::device_vector<float>        d_blurred_;
    thrust::device_vector<float>        d_magnitude_;
    thrust::device_vector<float>        d_rtv_;
    thrust::device_vector<std::uint8_t> d_guide_;

    const CudaBilateralFilter jbf_executor_;
};

#endif // BILATERAL_TEXTURE_FILTER_IMPL_CUH
