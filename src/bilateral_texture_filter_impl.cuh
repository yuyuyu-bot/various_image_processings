#ifndef BILATERAL_TEXTURE_FILTER_IMPL_CUH
#define BILATERAL_TEXTURE_FILTER_IMPL_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "cuda/bilateral_texture_filter.hpp"

class CudaBilateralTextureFilter::Impl {
public:
    Impl(const int width, const int height, const int ksize = 9, const int nitr = 3);
    ~Impl();

    void execute(
        const ElemType* const d_src,
        ElemType* const d_dst);

    void compute_magnitude(
        const thrust::device_vector<ElemType>& d_image,
        thrust::device_vector<float>& d_magnitude);

    template <typename BlurredType, typename RTVType>
    void compute_blur_and_rtv(
        const thrust::device_vector<ElemType>& d_image,
        const thrust::device_vector<float>& d_magnitude,
        thrust::device_vector<BlurredType>& d_blurred,
        thrust::device_vector<RTVType>& d_rtv);

    template <typename BlurredType, typename RTVType, typename GuideType>
    void compute_guide(
        const thrust::device_vector<BlurredType>& d_blurred,
        const thrust::device_vector<RTVType>& d_rtv,
        thrust::device_vector<GuideType>& d_guide);

    template <typename GuideType>
    void joint_bilateral_filter(
        const thrust::device_vector<ElemType>& d_src,
        const thrust::device_vector<GuideType>& d_guide,
        thrust::device_vector<ElemType>& d_dst,
        const int ksize,
        const float sigma_space,
        const float sigma_color);

private:
    static constexpr auto jbf_sigma_color = 1.73205080757f; // sqrt(3)

    const int   width_;
    const int   height_;
    const int   ksize_;
    const int   nitr_;
    const float sigma_space_;
    const float sigma_color_;

    thrust::device_vector<ElemType> d_src_n_;
    thrust::device_vector<ElemType> d_dst_n_;
    thrust::device_vector<float>    d_blurred_;
    thrust::device_vector<float>    d_magnitude_;
    thrust::device_vector<float>    d_rtv_;
    thrust::device_vector<ElemType> d_guide_;
};

#endif // BILATERAL_TEXTURE_FILTER_IMPL_CUH
