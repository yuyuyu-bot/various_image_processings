#include <gtest/gtest.h>
#include "random_array.hpp"

#include "bilateral_filter_impl.cuh"

class RefBilateralFilterImpl {
private:
    auto get_kernel_space(const int kx, const int ky) {
        const int radius = ksize_ / 2;
        return kernel_space_[(ky + radius) * ksize_ + (kx + radius)];
    }

    auto get_kernel_color(const std::uint8_t* const a, const std::uint8_t* const b) {
        const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
        const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
        const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
        const auto color_distance = std::abs(diff0) + std::abs(diff1) + std::abs(diff2);
        return kernel_color_table_[color_distance];
    }

public:
    RefBilateralFilterImpl(
        const int width,
        const int height,
        const int ksize = 9,
        const float sigma_space = 10.f,
        const float sigma_color = 30.f)
    : width_(width),
      height_(height),
      ksize_(ksize),
      kernel_space_(new float[ksize_ * ksize_]),
      kernel_color_table_(new float[256 * 3]) {
        const auto gauss_color_coeff = -1.f / (2 * sigma_color * sigma_color);
        const auto gauss_space_coeff = -1.f / (2 * sigma_space * sigma_space);
        const auto radius  = ksize_ / 2;

        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                const auto kidx = (ky + radius) * ksize_ + (kx + radius);
                const auto r2 = kx * kx + ky * ky;
                if (r2 > radius * radius) {
                    kernel_space_[kidx] = 0.f;
                    continue;
                }
                kernel_space_[kidx] = std::exp(r2 * gauss_space_coeff);
            }
        }

        for (int i = 0; i < 256 * 3; i++) {
            kernel_color_table_[i] = std::exp((i * i) * gauss_color_coeff);
        }
    }

    void bilateral_filter(
        const std::uint8_t* const src,
        std::uint8_t* const dst
    ) {
        const auto stride_3ch = width_ * 3;
        const auto radius = ksize_ / 2;

        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                const auto src_center_pix = src + stride_3ch * y + x * 3;
                auto sum0 = 0.f;
                auto sum1 = 0.f;
                auto sum2 = 0.f;
                auto sumk = 0.f;

                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        const auto x_clamped = std::clamp(x + kx, 0, width_ - 1);
                        const auto y_clamped = std::clamp(y + ky, 0, height_ - 1);
                        const auto src_pix   = src + stride_3ch * y_clamped + x_clamped * 3;
                        const auto kernel    = get_kernel_space(kx, ky) * get_kernel_color(src_center_pix, src_pix);

                        sum0 += src_pix[0] * kernel;
                        sum1 += src_pix[1] * kernel;
                        sum2 += src_pix[2] * kernel;
                        sumk += kernel;
                    }
                }

                dst[stride_3ch * y + x * 3 + 0] = static_cast<std::uint8_t>(sum0 / sumk);
                dst[stride_3ch * y + x * 3 + 1] = static_cast<std::uint8_t>(sum1 / sumk);
                dst[stride_3ch * y + x * 3 + 2] = static_cast<std::uint8_t>(sum2 / sumk);
            }
        }
    }

    void joint_bilateral_filter(
        const std::uint8_t* const src,
        const std::uint8_t* const guide,
        std::uint8_t* const dst
    ) {
        const auto stride_3ch = width_ * 3;
        const auto radius = ksize_ / 2;

        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                const auto guide_center_pix = guide + stride_3ch * y + x * 3;
                auto sum0 = 0.f;
                auto sum1 = 0.f;
                auto sum2 = 0.f;
                auto sumk = 0.f;

                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        const auto x_clamped = std::clamp(x + kx, 0, width_ - 1);
                        const auto y_clamped = std::clamp(y + ky, 0, height_ - 1);
                        const auto src_pix   = src + stride_3ch * y_clamped + x_clamped * 3;
                        const auto guide_pix = guide + stride_3ch * y_clamped + x_clamped * 3;
                        const auto kernel    = get_kernel_space(kx, ky) * get_kernel_color(guide_center_pix, guide_pix);

                        sum0 += src_pix[0] * kernel;
                        sum1 += src_pix[1] * kernel;
                        sum2 += src_pix[2] * kernel;
                        sumk += kernel;
                    }
                }

                dst[stride_3ch * y + x * 3 + 0] = static_cast<std::uint8_t>(sum0 / sumk);
                dst[stride_3ch * y + x * 3 + 1] = static_cast<std::uint8_t>(sum1 / sumk);
                dst[stride_3ch * y + x * 3 + 2] = static_cast<std::uint8_t>(sum2 / sumk);
            }
        }
    }

private:
    const int width_;
    const int height_;
    const int ksize_;

    std::unique_ptr<float[]> kernel_space_;
    std::unique_ptr<float[]> kernel_color_table_;
};

class CudaBilateralFilterImpl : public CudaBilateralFilter {
public:
    CudaBilateralFilterImpl(
        const int width,
        const int height,
        const int ksize = 9,
        const float sigma_space = 10.f,
        const float sigma_color = 30.f)
    : CudaBilateralFilter(width, height, ksize, sigma_space, sigma_color) {}

    void bilateral_filter(
        const std::uint8_t* const d_src,
        std::uint8_t* const d_dst
    ) {
        impl_->bilateral_filter(d_src, d_dst);
    }

    void joint_bilateral_filter(
        const std::uint8_t* const d_src,
        const std::uint8_t* const d_guide,
        std::uint8_t* const d_dst
    ) {
        impl_->joint_bilateral_filter(d_src, d_guide, d_dst);
    }
};

TEST(BilateralFilterTest, BilateralFilter) {
    constexpr auto width  = 50;
    constexpr auto height = 50;
    constexpr auto len    = width * height * 3;

    const auto src      = random_array<std::uint8_t>(len);
    const auto actual   = std::make_unique<std::uint8_t[]>(len);
    const auto expected = std::make_unique<std::uint8_t[]>(len);
    auto d_src    = thrust::device_vector<std::uint8_t>(len);
    auto d_actual = thrust::device_vector<std::uint8_t>(len);
    thrust::copy(src.get(), src.get() + len, d_src.begin());

    CudaBilateralFilterImpl cuda_impl(width, height);
    cuda_impl.bilateral_filter(d_src.data().get(), d_actual.data().get());
    thrust::copy(d_actual.begin(), d_actual.end(), actual.get());

    RefBilateralFilterImpl ref_impl(width, height);
    ref_impl.bilateral_filter(src.get(), expected.get());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual_ptr   = &actual[width * 3 * y + x * 3];
            const auto expected_ptr = &expected[width * 3 * y + x * 3];
            EXPECT_EQ(actual_ptr[0], expected_ptr[0]) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
            EXPECT_EQ(actual_ptr[1], expected_ptr[1]) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
            EXPECT_EQ(actual_ptr[2], expected_ptr[2]) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
        }
    }
}

TEST(BilateralFilterTest, JointBilateralFilter) {
    constexpr auto width  = 50;
    constexpr auto height = 50;
    constexpr auto len    = width * height * 3;

    const auto src      = random_array<std::uint8_t>(len);
    const auto guide    = random_array<std::uint8_t>(len);
    const auto actual   = std::make_unique<std::uint8_t[]>(len);
    const auto expected = std::make_unique<std::uint8_t[]>(len);
    auto d_src    = thrust::device_vector<std::uint8_t>(len);
    auto d_guide  = thrust::device_vector<std::uint8_t>(len);
    auto d_actual = thrust::device_vector<std::uint8_t>(len);
    thrust::copy(src.get(), src.get() + len, d_src.begin());
    thrust::copy(guide.get(), guide.get() + len, d_guide.begin());

    CudaBilateralFilterImpl cuda_impl(width, height);
    cuda_impl.joint_bilateral_filter(d_src.data().get(), d_guide.data().get(), d_actual.data().get());
    thrust::copy(d_actual.begin(), d_actual.end(), actual.get());

    RefBilateralFilterImpl ref_impl(width, height);
    ref_impl.joint_bilateral_filter(src.get(), guide.get(), expected.get());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual_ptr   = &actual[width * 3 * y + x * 3];
            const auto expected_ptr = &expected[width * 3 * y + x * 3];
            EXPECT_EQ(actual_ptr[0], expected_ptr[0]) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
            EXPECT_EQ(actual_ptr[1], expected_ptr[1]) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
            EXPECT_EQ(actual_ptr[2], expected_ptr[2]) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
        }
    }
}
