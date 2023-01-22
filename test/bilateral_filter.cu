#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include "random_array.hpp"

#include "cpp/bilateral_filter.hpp"
#include "bilateral_filter_impl.cuh"

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

TEST(BilateralFilterTest, CppBilateralFilter) {
    constexpr auto width       = 50;
    constexpr auto height      = 50;
    constexpr auto len         = width * height * 3;
    constexpr auto ksize       = 9;
    constexpr auto sigma_space = 10.f;
    constexpr auto sigma_color = 30.f;

    const auto src      = random_array<std::uint8_t>(len);
    const auto actual   = std::make_unique<std::uint8_t[]>(len);
    const auto expected = std::make_unique<std::uint8_t[]>(len);

    cv::Mat3b src_mat(height, width, reinterpret_cast<cv::Vec3b*>(src.get()));
    cv::Mat3b actual_mat(height, width, reinterpret_cast<cv::Vec3b*>(actual.get()));
    bilateral_filter(src_mat, actual_mat, ksize, sigma_space, sigma_color);

    cv::Mat3b expected_mat(height, width, reinterpret_cast<cv::Vec3b*>(expected.get()));
    cv::bilateralFilter(src_mat, expected_mat, ksize, sigma_color, sigma_space, cv::BORDER_REPLICATE);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual_ptr   = &actual[width * 3 * y + x * 3];
            const auto expected_ptr = &expected[width * 3 * y + x * 3];
            EXPECT_NEAR(actual_ptr[0], expected_ptr[0], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
            EXPECT_NEAR(actual_ptr[1], expected_ptr[1], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
            EXPECT_NEAR(actual_ptr[2], expected_ptr[2], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
        }
    }
}

TEST(BilateralFilterTest, CppJointBilateralFilter) {
    constexpr auto width       = 50;
    constexpr auto height      = 50;
    constexpr auto len         = width * height * 3;
    constexpr auto ksize       = 9;
    constexpr auto sigma_space = 10.f;
    constexpr auto sigma_color = 30.f;

    const auto src      = random_array<std::uint8_t>(len);
    const auto guide    = random_array<std::uint8_t>(len);
    const auto actual   = std::make_unique<std::uint8_t[]>(len);
    const auto expected = std::make_unique<std::uint8_t[]>(len);

    cv::Mat3b src_mat(height, width, reinterpret_cast<cv::Vec3b*>(src.get()));
    cv::Mat3b guide_mat(height, width, reinterpret_cast<cv::Vec3b*>(guide.get()));
    cv::Mat3b actual_mat(height, width, reinterpret_cast<cv::Vec3b*>(actual.get()));
    joint_bilateral_filter(src_mat, guide_mat, actual_mat);

    cv::Mat3b expected_mat(height, width, reinterpret_cast<cv::Vec3b*>(expected.get()));
    cv::ximgproc::jointBilateralFilter(guide_mat, src_mat, expected_mat, ksize, sigma_color, sigma_space, cv::BORDER_REPLICATE);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual_ptr   = &actual[width * 3 * y + x * 3];
            const auto expected_ptr = &expected[width * 3 * y + x * 3];
            EXPECT_NEAR(actual_ptr[0], expected_ptr[0], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
            EXPECT_NEAR(actual_ptr[1], expected_ptr[1], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
            EXPECT_NEAR(actual_ptr[2], expected_ptr[2], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
        }
    }
}

TEST(BilateralFilterTest, CudaBilateralFilter) {
    constexpr auto width       = 50;
    constexpr auto height      = 50;
    constexpr auto len         = width * height * 3;
    constexpr auto ksize       = 9;
    constexpr auto sigma_space = 10.f;
    constexpr auto sigma_color = 30.f;

    const auto src      = random_array<std::uint8_t>(len);
    const auto actual   = std::make_unique<std::uint8_t[]>(len);
    const auto expected = std::make_unique<std::uint8_t[]>(len);
    auto d_src    = thrust::device_vector<std::uint8_t>(len);
    auto d_actual = thrust::device_vector<std::uint8_t>(len);
    thrust::copy(src.get(), src.get() + len, d_src.begin());

    CudaBilateralFilterImpl cuda_impl(width, height);
    cuda_impl.bilateral_filter(d_src.data().get(), d_actual.data().get());
    thrust::copy(d_actual.begin(), d_actual.end(), actual.get());

    cv::Mat3b src_mat(height, width, reinterpret_cast<cv::Vec3b*>(src.get()));
    cv::Mat3b expected_mat(height, width, reinterpret_cast<cv::Vec3b*>(expected.get()));
    cv::bilateralFilter(src_mat, expected_mat, ksize, sigma_color, sigma_space, cv::BORDER_REPLICATE);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual_ptr   = &actual[width * 3 * y + x * 3];
            const auto expected_ptr = &expected[width * 3 * y + x * 3];
            EXPECT_NEAR(actual_ptr[0], expected_ptr[0], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
            EXPECT_NEAR(actual_ptr[1], expected_ptr[1], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
            EXPECT_NEAR(actual_ptr[2], expected_ptr[2], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
        }
    }
}

TEST(BilateralFilterTest, CudaJointBilateralFilter) {
    constexpr auto width       = 50;
    constexpr auto height      = 50;
    constexpr auto len         = width * height * 3;
    constexpr auto ksize       = 9;
    constexpr auto sigma_space = 10.f;
    constexpr auto sigma_color = 30.f;

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

    cv::Mat3b src_mat(height, width, reinterpret_cast<cv::Vec3b*>(src.get()));
    cv::Mat3b guide_mat(height, width, reinterpret_cast<cv::Vec3b*>(guide.get()));
    cv::Mat3b expected_mat(height, width, reinterpret_cast<cv::Vec3b*>(expected.get()));
    cv::ximgproc::jointBilateralFilter(guide_mat, src_mat, expected_mat, ksize, sigma_color, sigma_space, cv::BORDER_REPLICATE);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual_ptr   = &actual[width * 3 * y + x * 3];
            const auto expected_ptr = &expected[width * 3 * y + x * 3];
            EXPECT_NEAR(actual_ptr[0], expected_ptr[0], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 0 << ")";
            EXPECT_NEAR(actual_ptr[1], expected_ptr[1], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 1 << ")";
            EXPECT_NEAR(actual_ptr[2], expected_ptr[2], 1) << "(x, y, ch) = (" << x << ", " << y << ", " << 2 << ")";
        }
    }
}
