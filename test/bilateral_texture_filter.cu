#include <gtest/gtest.h>
#include <memory>
#include <opencv2/core.hpp>
#include <random>

#include "bilateral_texture_filter_impl.cuh"

class RefImpl {
public:
    RefImpl(const int width, const int height, const int ksize = 9, const int nitr = 3)
    : width_(width), height_(height), ksize_(ksize), nitr_(nitr)
    {}

    void compute_magnitude(const std::uint8_t* const image, float* magnitude) {
        const auto compute_del =
            [width = this->width_, height = this->height_, image]
            (const int x0, const int y0, const int x1, const int y1) {
                const auto x0_clamped = std::clamp(x0, 0, width - 1);
                const auto y0_clamped = std::clamp(y0, 0, height - 1);
                const auto x1_clamped = std::clamp(x1, 0, width - 1);
                const auto y1_clamped = std::clamp(y1, 0, height - 1);

                const auto diff0 = image[width * 3 * y0_clamped + x0_clamped * 3 + 0] -
                                   image[width * 3 * y1_clamped + x1_clamped * 3 + 0];
                const auto diff1 = image[width * 3 * y0_clamped + x0_clamped * 3 + 1] -
                                   image[width * 3 * y1_clamped + x1_clamped * 3 + 1];
                const auto diff2 = image[width * 3 * y0_clamped + x0_clamped * 3 + 2] -
                                   image[width * 3 * y1_clamped + x1_clamped * 3 + 2];
                return diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
            };

        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                const auto del_x = compute_del(x - 1, y, x + 1, y);
                const auto del_y = compute_del(x, y - 1, x, y + 1);
                magnitude[width_ * y + x] = std::sqrt(del_x + del_y);
            }
        }
    }

private:
    const int width_;
    const int height_;
    const int ksize_;
    const int nitr_;
};

class CudaImpl : public CudaBilateralTextureFilter {
public:
    CudaImpl(const int width, const int height, const int ksize = 9, const int nitr = 3)
    : CudaBilateralTextureFilter(width, height, ksize, nitr)
    {}

    void compute_magnitude(
        const thrust::device_vector<ElemType>& d_image,
        thrust::device_vector<float>& d_magnitude)
    {
        impl_->compute_magnitude(d_image, d_magnitude);
        cudaDeviceSynchronize();
    }

    template <typename BlurredType, typename RTVType>
    void compute_blur_and_rtv(
        const thrust::device_vector<ElemType>& d_image,
        const thrust::device_vector<float>& d_magnitude,
        thrust::device_vector<BlurredType>& d_blurred,
        thrust::device_vector<RTVType>& d_rtv)
    {
        impl_->compute_blur_and_rtv(d_image, d_magnitude, d_blurred, d_rtv);
        cudaDeviceSynchronize();
    }

    template <typename BlurredType, typename RTVType, typename GuideType>
    void compute_guide(
        const thrust::device_vector<BlurredType>& d_blurred,
        const thrust::device_vector<RTVType>& d_rtv,
        thrust::device_vector<GuideType>& d_guide)
    {
        impl_->compute_guide(d_blurred, d_rtv, d_guide);
        cudaDeviceSynchronize();
    }

    template <typename GuideType>
    void joint_bilateral_filter(
        const thrust::device_vector<ElemType>& d_src,
        const thrust::device_vector<GuideType>& d_guide,
        thrust::device_vector<ElemType>& d_dst,
        const int ksize,
        const float sigma_space,
        const float sigma_color)
    {
        impl_->joint_bilateral_filter(d_src, d_guide, d_dst, ksize, sigma_space, sigma_color);
        cudaDeviceSynchronize();
    }
};

template <typename ElemType>
auto random_array(const std::size_t len) {
    std::random_device seed_gen;
    std::mt19937 rand_gen(seed_gen());

    auto array = std::make_unique<ElemType[]>(len);
    for (int i = 0; i < len; i++) {
        array[i] = rand_gen() % std::numeric_limits<ElemType>::max();
    }

    return array;
}

TEST(BilateralTextureFilterTest, ComputeMagnitude) {
    std::random_device seed_gen;
    std::mt19937 rand_gen(seed_gen());

    constexpr auto width  = 50;
    constexpr auto height = 50;

    const auto input_array = random_array<std::uint8_t>(width * height * 3);
    const auto actual_array = std::make_unique<float[]>(width * height);
    const auto expected_array = std::make_unique<float[]>(width * height);
    thrust::device_vector<std::uint8_t> d_input_array(width * height * 3);
    thrust::device_vector<float> d_actual(width * height);
    thrust::copy(input_array.get(), input_array.get() + width * height * 3, d_input_array.begin());

    CudaImpl cuda_impl(width, height);
    cuda_impl.compute_magnitude(d_input_array, d_actual);
    thrust::copy(d_actual.begin(), d_actual.end(), actual_array.get());

    RefImpl ref_impl(width, height);
    ref_impl.compute_magnitude(input_array.get(), expected_array.get());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual   = actual_array[width * y + x];
            const auto expected = expected_array[width * y + x];
            EXPECT_EQ(actual, expected) << "(x, y) = (" << x << ", " << y << ")";
        }
    }
}

