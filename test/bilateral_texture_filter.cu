#include <gtest/gtest.h>
#include <memory>
#include <opencv2/core.hpp>
#include <random>

#include "bilateral_texture_filter_impl.cuh"

constexpr auto epsilon = 1e-20f;
constexpr auto rel_diff_threshold = 1e-10f;

class RefImpl {
public:
    RefImpl(const int width, const int height, const int ksize = 9, const int nitr = 3)
    : width_(width), height_(height), ksize_(ksize), nitr_(nitr) {}

    void compute_magnitude(
        const std::uint8_t* const image,
        float* magnitude
    ) {
        const auto compute_del =
            [width=this->width_, height=this->height_, image]
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

    void compute_blur_and_rtv(
        const std::uint8_t* const image,
        const float* const magnitude,
        float* const blurred,
        float* const rtv
    ) {
        const auto radius = ksize_ / 2;

        const auto get_intensity = [width=this->width_, image](const int x, const int y) {
            const auto bgr = &image[width * 3 * y + x * 3];
            return (bgr[0] + bgr[1] + bgr[2]) / 3.f;
        };

        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                auto sum0 = 0.f;
                auto sum1 = 0.f;
                auto sum2 = 0.f;

                auto intensity_max = 0.f;
                auto intensity_min = std::numeric_limits<float>::max();
                auto magnitude_max = 0.f;
                auto magnitude_sum = 0.f;

                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        const auto x_clamped = std::clamp(x + kx, 0, width_ - 1);
                        const auto y_clamped = std::clamp(y + ky, 0, height_ - 1);

                        sum0 += image[width_ * 3 * y_clamped + x_clamped * 3 + 0];
                        sum1 += image[width_ * 3 * y_clamped + x_clamped * 3 + 1];
                        sum2 += image[width_ * 3 * y_clamped + x_clamped * 3 + 2];

                        intensity_max  = std::max(intensity_max, get_intensity(x_clamped, y_clamped));
                        intensity_min  = std::min(intensity_min, get_intensity(x_clamped, y_clamped));
                        magnitude_max  = std::max(magnitude_max, magnitude[width_ * y_clamped + x_clamped]);
                        magnitude_sum += magnitude[width_ * y_clamped + x_clamped];
                    }
                }

                blurred[width_ * 3 * y + x * 3 + 0] = sum0 / (ksize_ * ksize_);
                blurred[width_ * 3 * y + x * 3 + 1] = sum1 / (ksize_ * ksize_);
                blurred[width_ * 3 * y + x * 3 + 2] = sum2 / (ksize_ * ksize_);
                rtv[width_ * y + x] = (intensity_max - intensity_min) * magnitude_max / (magnitude_sum + epsilon);
            }
        }
    }

    void compute_guide(
        const float* const blurred,
        const float* const rtv,
        std::uint8_t* const guide
    ) {
        const auto radius = ksize_ / 2;
        const auto sigma_alpha = 1.f / (5 * ksize_);

        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                auto rtv_min = std::numeric_limits<float>::max();
                auto rtv_min_x = 0;
                auto rtv_min_y = 0;

                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        const auto x_clamped = std::clamp(x + kx, 0, width_ - 1);
                        const auto y_clamped = std::clamp(y + ky, 0, height_ - 1);

                        if (rtv_min > rtv[width_ * y_clamped + x_clamped]) {
                            rtv_min = rtv[width_ * y_clamped + x_clamped];
                            rtv_min_x = x_clamped;
                            rtv_min_y = y_clamped;
                        }
                    }
                }

                const auto alpha =
                    2 / (1 + std::exp(sigma_alpha * (rtv[width_ * y + x] - rtv[width_ * rtv_min_y + rtv_min_x]))) - 1.f;
                guide[width_ * 3 * y + x * 3 + 0] =      alpha  * blurred[width_ * 3 * rtv_min_y + rtv_min_x * 3 + 0] +
                                                    (1 - alpha) * blurred[width_ * 3 *         y +         x * 3 + 0];
                guide[width_ * 3 * y + x * 3 + 1] =      alpha  * blurred[width_ * 3 * rtv_min_y + rtv_min_x * 3 + 1] +
                                                    (1 - alpha) * blurred[width_ * 3 *         y +         x * 3 + 1];
                guide[width_ * 3 * y + x * 3 + 2] =      alpha  * blurred[width_ * 3 * rtv_min_y + rtv_min_x * 3 + 2] +
                                                    (1 - alpha) * blurred[width_ * 3 *         y +         x * 3 + 2];
            }
        }
    }

private:
    static constexpr auto epsilon = 1e-9f;

    const int width_;
    const int height_;
    const int ksize_;
    const int nitr_;
};

class CudaImpl : public CudaBilateralTextureFilter {
public:
    CudaImpl(const int width, const int height, const int ksize = 9, const int nitr = 3)
    : CudaBilateralTextureFilter(width, height, ksize, nitr) {}

    void compute_magnitude(
        const thrust::device_vector<std::uint8_t>& d_image,
        thrust::device_vector<float>& d_magnitude
    ) {
        impl_->compute_magnitude(d_image, d_magnitude);
        cudaDeviceSynchronize();
    }

    void compute_blur_and_rtv(
        const thrust::device_vector<std::uint8_t>& d_image,
        const thrust::device_vector<float>& d_magnitude,
        thrust::device_vector<float>& d_blurred,
        thrust::device_vector<float>& d_rtv
    ) {
        impl_->compute_blur_and_rtv(d_image, d_magnitude, d_blurred, d_rtv);
        cudaDeviceSynchronize();
    }

    void compute_guide(
        const thrust::device_vector<float>& d_blurred,
        const thrust::device_vector<float>& d_rtv,
        thrust::device_vector<std::uint8_t>& d_guide
    ) {
        impl_->compute_guide(d_blurred, d_rtv, d_guide);
        cudaDeviceSynchronize();
    }

    void joint_bilateral_filter(
        const thrust::device_vector<std::uint8_t>& d_src,
        const thrust::device_vector<std::uint8_t>& d_guide,
        thrust::device_vector<std::uint8_t>& d_dst,
        const int ksize,
        const float sigma_space,
        const float sigma_color
    ) {
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

template <>
auto random_array<float>(const std::size_t len) {
    std::random_device seed_gen;
    std::mt19937 rand_gen(seed_gen());

    auto array = std::make_unique<float[]>(len);
    for (int i = 0; i < len; i++) {
        array[i] = rand_gen() / std::numeric_limits<std::uint32_t>::max();
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

TEST(BilateralTextureFilterTest, ComputeBlurAndRTV) {
    std::random_device seed_gen;
    std::mt19937 rand_gen(seed_gen());

    constexpr auto width  = 50;
    constexpr auto height = 50;

    const auto input_image      = random_array<std::uint8_t>(width * height * 3);
    const auto input_magnitude  = random_array<float>(width * height);
    const auto actual_blurred   = std::make_unique<float[]>(width * height * 3);
    const auto actual_rtv       = std::make_unique<float[]>(width * height);
    const auto expected_blurred = std::make_unique<float[]>(width * height * 3);
    const auto expected_rtv     = std::make_unique<float[]>(width * height);
    thrust::device_vector<std::uint8_t> d_input_image(width * height * 3);
    thrust::device_vector<float>        d_input_magnitude(width * height);
    thrust::device_vector<float>        d_actual_blurred(width * height * 3);
    thrust::device_vector<float>        d_actual_rtv(width * height);
    thrust::copy(input_image.get(), input_image.get() + width * height * 3, d_input_image.begin());
    thrust::copy(input_magnitude.get(), input_magnitude.get() + width * height, d_input_magnitude.begin());

    CudaImpl cuda_impl(width, height);
    cuda_impl.compute_blur_and_rtv(d_input_image, d_input_magnitude, d_actual_blurred, d_actual_rtv);
    thrust::copy(d_actual_blurred.begin(), d_actual_blurred.end(), actual_blurred.get());
    thrust::copy(d_actual_rtv.begin(), d_actual_rtv.end(), actual_rtv.get());

    RefImpl ref_impl(width, height);
    ref_impl.compute_blur_and_rtv(input_image.get(), input_magnitude.get(), expected_blurred.get(), expected_rtv.get());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            {
                // blurred
                const auto actual   = &actual_blurred[width * 3 * y + x * 3];
                const auto expected = &expected_blurred[width * 3 * y + x * 3];
                const auto rel_diff0 = std::abs(actual[0] - expected[0]) / (expected[0] + epsilon);
                const auto rel_diff1 = std::abs(actual[1] - expected[1]) / (expected[1] + epsilon);
                const auto rel_diff2 = std::abs(actual[2] - expected[2]) / (expected[2] + epsilon);
                EXPECT_LE(rel_diff0, rel_diff_threshold) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
                EXPECT_LE(rel_diff1, rel_diff_threshold) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
                EXPECT_LE(rel_diff2, rel_diff_threshold) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
            }
            {
                // rtv
                const auto actual   = actual_rtv[width * y + x];
                const auto expected = expected_rtv[width * y + x];
                const auto rel_diff = std::abs(actual - expected) / (expected + epsilon);
                EXPECT_LE(rel_diff, rel_diff_threshold) << "(x, y) = (" << x << ", " << y << ")";
            }
        }
    }
}

TEST(BilateralTextureFilterTest, ComputeGuide) {
    std::random_device seed_gen;
    std::mt19937 rand_gen(seed_gen());

    constexpr auto width  = 50;
    constexpr auto height = 50;

    const auto input_blurred  = random_array<float>(width * height * 3);
    const auto input_rtv      = random_array<float>(width * height);
    const auto actual_guide   = std::make_unique<std::uint8_t[]>(width * height * 3);
    const auto expected_guide = std::make_unique<std::uint8_t[]>(width * height * 3);
    thrust::device_vector<float>        d_input_blurred(width * height * 3);
    thrust::device_vector<float>        d_input_rtv(width * height);
    thrust::device_vector<std::uint8_t> d_actual_guide(width * height * 3);
    thrust::copy(input_blurred.get(), input_blurred.get() + width * height * 3, d_input_blurred.begin());
    thrust::copy(input_rtv.get(), input_rtv.get() + width * height, d_input_rtv.begin());

    CudaImpl cuda_impl(width, height);
    cuda_impl.compute_guide(d_input_blurred, d_input_rtv, d_actual_guide);
    thrust::copy(d_actual_guide.begin(), d_actual_guide.end(), actual_guide.get());

    RefImpl ref_impl(width, height);
    ref_impl.compute_guide(input_blurred.get(), input_rtv.get(), expected_guide.get());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual   = &actual_guide[width * 3 * y + x * 3];
            const auto expected = &expected_guide[width * 3 * y + x * 3];
            EXPECT_EQ(actual[0], expected[0]) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
            EXPECT_EQ(actual[1], expected[1]) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
            EXPECT_EQ(actual[2], expected[2]) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
        }
    }
}
