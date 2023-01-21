#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <thrust/device_vector.h>
#include "random_array.hpp"

#include "cpp/gradient.hpp"
#include "cuda/gradient.hpp"

template <typename SrcType>
void ref_gradient(const SrcType* const src, float* const dst, const int width, const int height, const int src_ch = 1) {
    const auto compute_del = [src, width, height, src_ch](const int x0, const int y0, const int x1, const int y1) {
        const auto x0_clamped = std::clamp(x0, 0, width - 1);
        const auto y0_clamped = std::clamp(y0, 0, height - 1);
        const auto x1_clamped = std::clamp(x1, 0, width - 1);
        const auto y1_clamped = std::clamp(y1, 0, height - 1);

        const auto pix0 = src + (width * src_ch * y0_clamped + x0_clamped * src_ch);
        const auto pix1 = src + (width * src_ch * y1_clamped + x1_clamped * src_ch);

        auto diff = 0.f;
        for (int ch = 0; ch < src_ch; ch++) {
            diff += (pix0[ch] - pix1[ch]) * (pix0[ch] - pix1[ch]);
        }
        return diff;
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto del_x = compute_del(x - 1, y, x + 1, y);
            const auto del_y = compute_del(x, y - 1, x, y + 1);
            dst[width * y + x] = std::sqrt(del_x + del_y);
        }
    }
}

struct GradientTest : ::testing::TestWithParam<int> {};

TEST_P(GradientTest, CppRandomU8) {
    cv::setNumThreads(1);
    using SrcType = std::uint8_t;
    constexpr auto width  = 50;
    constexpr auto height = 50;
    const auto src_ch = GetParam();

    const auto input_array = random_array<SrcType>(width * height * src_ch);
    const auto actual_array = std::make_unique<float[]>(width * height);
    const auto expected_array = std::make_unique<float[]>(width * height);

    cv::Mat input_mat(height, width, CV_8UC(src_ch), input_array.get());
    cv::Mat actual_mat(height, width, CV_32FC1, actual_array.get());
    gradient(input_mat, actual_mat);

    ref_gradient(input_array.get(), expected_array.get(), width, height, src_ch);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual   = actual_array[width * y + x];
            const auto expected = expected_array[width * y + x];
            EXPECT_FLOAT_EQ(actual, expected) << "(x, y) = (" << x << ", " << y << ")";
        }
    }
}

TEST_P(GradientTest, CppRandomF32) {
    using SrcType = float;
    constexpr auto width  = 50;
    constexpr auto height = 50;
    const auto src_ch = GetParam();

    const auto input_array = random_array<SrcType>(width * height * src_ch);
    const auto actual_array = std::make_unique<float[]>(width * height);
    const auto expected_array = std::make_unique<float[]>(width * height);

    cv::Mat input_mat(height, width, CV_32FC(src_ch), input_array.get());
    cv::Mat actual_mat(height, width, CV_32FC1, actual_array.get());
    gradient(input_mat, actual_mat);

    ref_gradient(input_array.get(), expected_array.get(), width, height, src_ch);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual   = actual_array[width * y + x];
            const auto expected = expected_array[width * y + x];
            EXPECT_FLOAT_EQ(actual, expected) << "(x, y) = (" << x << ", " << y << ")";
        }
    }
}

TEST_P(GradientTest, CudaRandomU8) {
    using SrcType = std::uint8_t;
    constexpr auto width  = 50;
    constexpr auto height = 50;
    const auto src_ch = GetParam();

    const auto input_array = random_array<SrcType>(width * height * src_ch);
    const auto actual_array = std::make_unique<float[]>(width * height);
    const auto expected_array = std::make_unique<float[]>(width * height);
    thrust::device_vector<SrcType> d_input_array(width * height * src_ch);
    thrust::device_vector<float> d_actual(width * height);
    thrust::copy(input_array.get(), input_array.get() + width * height * src_ch, d_input_array.begin());

    cuda_gradient(d_input_array.data().get(), d_actual.data().get(), width, height, src_ch);
    thrust::copy(d_actual.begin(), d_actual.end(), actual_array.get());

    ref_gradient(input_array.get(), expected_array.get(), width, height, src_ch);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual   = actual_array[width * y + x];
            const auto expected = expected_array[width * y + x];
            EXPECT_FLOAT_EQ(actual, expected) << "(x, y) = (" << x << ", " << y << ")";
        }
    }
}

TEST_P(GradientTest, CudaRandomF32) {
    using SrcType = float;
    constexpr auto width  = 50;
    constexpr auto height = 50;
    const auto src_ch = GetParam();

    const auto input_array = random_array<SrcType>(width * height * src_ch);
    const auto actual_array = std::make_unique<float[]>(width * height);
    const auto expected_array = std::make_unique<float[]>(width * height);
    thrust::device_vector<SrcType> d_input_array(width * height * src_ch);
    thrust::device_vector<float> d_actual(width * height);
    thrust::copy(input_array.get(), input_array.get() + width * height * src_ch, d_input_array.begin());

    cuda_gradient(d_input_array.data().get(), d_actual.data().get(), width, height, src_ch);
    thrust::copy(d_actual.begin(), d_actual.end(), actual_array.get());

    ref_gradient(input_array.get(), expected_array.get(), width, height, src_ch);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto actual   = actual_array[width * y + x];
            const auto expected = expected_array[width * y + x];
            EXPECT_FLOAT_EQ(actual, expected) << "(x, y) = (" << x << ", " << y << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(GradientTest, GradientTest, testing::Values(1, 3));
