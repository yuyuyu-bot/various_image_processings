#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "random_array.hpp"

#include "cpp/bilateral_texture_filter.hpp"
#include "bilateral_texture_filter_impl.cuh"

class RefBilateralTextureFilterImpl {
public:
    RefBilateralTextureFilterImpl(const int width, const int height, const int ksize = 9, const int nitr = 3)
    : width_(width), height_(height), ksize_(ksize), nitr_(nitr) {}

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
                guide[width_ * 3 * y + x * 3 + 0] =
                    std::clamp<int>(     alpha  * blurred[width_ * 3 * rtv_min_y + rtv_min_x * 3 + 0] +
                                    (1 - alpha) * blurred[width_ * 3 *         y +         x * 3 + 0] + 0.5f,
                                    0, 255);
                guide[width_ * 3 * y + x * 3 + 1] =
                    std::clamp<int>(     alpha  * blurred[width_ * 3 * rtv_min_y + rtv_min_x * 3 + 1] +
                                    (1 - alpha) * blurred[width_ * 3 *         y +         x * 3 + 1] + 0.5f,
                                    0, 255);
                guide[width_ * 3 * y + x * 3 + 2] =
                    std::clamp<int>(     alpha  * blurred[width_ * 3 * rtv_min_y + rtv_min_x * 3 + 2] +
                                    (1 - alpha) * blurred[width_ * 3 *         y +         x * 3 + 2] + 0.5f,
                                    0, 255);
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

class CudaBilateralTextureFilterImpl : public CudaBilateralTextureFilter {
public:
    CudaBilateralTextureFilterImpl(const int width, const int height, const int ksize = 9, const int nitr = 3)
    : CudaBilateralTextureFilter(width, height, ksize, nitr) {}

    void compute_blur_and_rtv(
        const thrust::device_vector<std::uint8_t>& d_image,
        const thrust::device_vector<float>& d_magnitude,
        thrust::device_vector<float>& d_blurred,
        thrust::device_vector<float>& d_rtv
    ) {
        impl_->compute_blur_and_rtv(d_image, d_magnitude, d_blurred, d_rtv);
    }

    void compute_guide(
        const thrust::device_vector<float>& d_blurred,
        const thrust::device_vector<float>& d_rtv,
        thrust::device_vector<std::uint8_t>& d_guide
    ) {
        impl_->compute_guide(d_blurred, d_rtv, d_guide);
    }
};

TEST(BilateralTextureFilterTest, CppComputeBlurAndRTV) {
    constexpr auto width  = 50;
    constexpr auto height = 50;
    constexpr auto ksize  = 9;

    const auto input_image      = random_array<std::uint8_t>(width * height * 3);
    const auto input_magnitude  = random_array<float>(width * height);
    const auto actual_blurred   = std::make_unique<float[]>(width * height * 3);
    const auto actual_rtv       = std::make_unique<float[]>(width * height);
    const auto expected_blurred = std::make_unique<float[]>(width * height * 3);
    const auto expected_rtv     = std::make_unique<float[]>(width * height);

    cv::Mat3b input_mat(height, width, reinterpret_cast<cv::Vec3b*>(input_image.get()));
    cv::Mat1f magnitude_mat(height, width, input_magnitude.get());
    cv::Mat3f blurred_mat(height, width, reinterpret_cast<cv::Vec3f*>(actual_blurred.get()));
    cv::Mat1f rtv_mat(height, width, actual_rtv.get());
    internal::compute_blur_and_rtv(input_mat, magnitude_mat, blurred_mat, rtv_mat, ksize);

    RefBilateralTextureFilterImpl ref_impl(width, height, ksize);
    ref_impl.compute_blur_and_rtv(input_image.get(), input_magnitude.get(), expected_blurred.get(), expected_rtv.get());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            {
                // blurred
                const auto actual   = &actual_blurred[width * 3 * y + x * 3];
                const auto expected = &expected_blurred[width * 3 * y + x * 3];
                EXPECT_FLOAT_EQ(actual[0], expected[0]) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
                EXPECT_FLOAT_EQ(actual[1], expected[1]) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
                EXPECT_FLOAT_EQ(actual[2], expected[2]) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
            }
            {
                // rtv
                const auto actual   = actual_rtv[width * y + x];
                const auto expected = expected_rtv[width * y + x];
                EXPECT_FLOAT_EQ(actual, expected) << "(x, y) = (" << x << ", " << y << ")";
            }
        }
    }
}

TEST(BilateralTextureFilterTest, CppComputeGuide) {
    constexpr auto width  = 50;
    constexpr auto height = 50;
    constexpr auto ksize  = 9;

    const auto input_blurred  = random_array<float>(width * height * 3);
    const auto input_rtv      = random_array<float>(width * height, 1);
    const auto actual_guide   = std::make_unique<std::uint8_t[]>(width * height * 3);
    const auto expected_guide = std::make_unique<std::uint8_t[]>(width * height * 3);

    cv::Mat3f blurred_mat(height, width, reinterpret_cast<cv::Vec3f*>(input_blurred.get()));
    cv::Mat1f rtv_mat(height, width, input_rtv.get());
    cv::Mat3b guide_mat(height, width, reinterpret_cast<cv::Vec3b*>(actual_guide.get()));
    internal::compute_guide(blurred_mat, rtv_mat, guide_mat, ksize);

    RefBilateralTextureFilterImpl ref_impl(width, height, ksize);
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

TEST(BilateralTextureFilterTest, CudaComputeBlurAndRTV) {
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

    CudaBilateralTextureFilterImpl cuda_impl(width, height);
    cuda_impl.compute_blur_and_rtv(d_input_image, d_input_magnitude, d_actual_blurred, d_actual_rtv);
    thrust::copy(d_actual_blurred.begin(), d_actual_blurred.end(), actual_blurred.get());
    thrust::copy(d_actual_rtv.begin(), d_actual_rtv.end(), actual_rtv.get());

    RefBilateralTextureFilterImpl ref_impl(width, height);
    ref_impl.compute_blur_and_rtv(input_image.get(), input_magnitude.get(), expected_blurred.get(), expected_rtv.get());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            {
                // blurred
                const auto actual   = &actual_blurred[width * 3 * y + x * 3];
                const auto expected = &expected_blurred[width * 3 * y + x * 3];
                EXPECT_FLOAT_EQ(actual[0], expected[0]) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
                EXPECT_FLOAT_EQ(actual[1], expected[1]) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
                EXPECT_FLOAT_EQ(actual[2], expected[2]) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
            }
            {
                // rtv
                const auto actual   = actual_rtv[width * y + x];
                const auto expected = expected_rtv[width * y + x];
                EXPECT_FLOAT_EQ(actual, expected) << "(x, y) = (" << x << ", " << y << ")";
            }
        }
    }
}

TEST(BilateralTextureFilterTest, CudaComputeGuide) {
    constexpr auto width  = 50;
    constexpr auto height = 50;

    const auto input_blurred  = random_array<float>(width * height * 3);
    const auto input_rtv      = random_array<float>(width * height, 1);
    const auto actual_guide   = std::make_unique<std::uint8_t[]>(width * height * 3);
    const auto expected_guide = std::make_unique<std::uint8_t[]>(width * height * 3);
    thrust::device_vector<float>        d_input_blurred(width * height * 3);
    thrust::device_vector<float>        d_input_rtv(width * height);
    thrust::device_vector<std::uint8_t> d_actual_guide(width * height * 3);
    thrust::copy(input_blurred.get(), input_blurred.get() + width * height * 3, d_input_blurred.begin());
    thrust::copy(input_rtv.get(), input_rtv.get() + width * height, d_input_rtv.begin());

    CudaBilateralTextureFilterImpl cuda_impl(width, height);
    cuda_impl.compute_guide(d_input_blurred, d_input_rtv, d_actual_guide);
    thrust::copy(d_actual_guide.begin(), d_actual_guide.end(), actual_guide.get());

    RefBilateralTextureFilterImpl ref_impl(width, height);
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
