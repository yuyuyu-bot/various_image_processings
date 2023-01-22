#include <gtest/gtest.h>
#include "random_array.hpp"

#include "cpp/border_replicated_integral_image.hpp"

struct BorderReplicatedIntegralImageTest : ::testing::TestWithParam<int> {
    template <typename SrcType, int Channels>
    void test() {
        using VecType = cv::Vec<SrcType, Channels>;
        constexpr auto width    = 5;
        constexpr auto height   = 5;
        const auto radius = GetParam();

        const auto src_array = random_array<SrcType>(width * height * Channels, 255);
        cv::Mat_<VecType> src(height, width, reinterpret_cast<VecType*>(src_array.get()));

        BorderReplicatedIntegralImage<SrcType, Channels> integral_image(src, radius);

        for (int y0 = -radius; y0 < height + radius; y0++) {
            for (int x0 = -radius; x0 < width + radius; x0++) {
                for (int y1 = y0; y1 < height + radius; y1++) {
                    for (int x1 = x0; x1 < width + radius; x1++) {
                        const auto actual = integral_image.get(x0, y0, x1, y1);

                        if (std::is_floating_point_v<SrcType>) {
                            auto expected = cv::Vec<double, Channels>::all(0);
                            for (int y = y0; y <= y1; y++) {
                                for (int x = x0; x <= x1; x++) {
                                    const auto cx = std::clamp(x, 0, width - 1);
                                    const auto cy = std::clamp(y, 0, height - 1);
                                    expected += src(cy, cx);
                                }
                            }

                            for (int ch = 0; ch < Channels; ch++) {
                                const auto ref_diff = std::abs(actual[ch] - expected[ch]) / expected[ch];
                                EXPECT_LT(ref_diff, 1e-2); // less than 1%
                            }
                        }
                        else {
                            auto expected = cv::Vec<int, Channels>::all(0);
                            for (int y = y0; y <= y1; y++) {
                                for (int x = x0; x <= x1; x++) {
                                    const auto cx = std::clamp(x, 0, width - 1);
                                    const auto cy = std::clamp(y, 0, height - 1);
                                    expected += src(cy, cx);
                                }
                            }

                            for (int ch = 0; ch < Channels; ch++) {
                                EXPECT_EQ(actual[ch], expected[ch]);
                            }
                        }
                    } // x1 loop
                } // y1 loop
            } // x0 loop
        } // y0 loop
    }
};

TEST_P(BorderReplicatedIntegralImageTest, u8_1ch) {
    test<std::uint8_t, 1>();
}

TEST_P(BorderReplicatedIntegralImageTest, u8_2ch) {
    test<std::uint8_t, 2>();
}

TEST_P(BorderReplicatedIntegralImageTest, u8_3ch) {
    test<std::uint8_t, 2>();
}

TEST_P(BorderReplicatedIntegralImageTest, u16_1ch) {
    test<std::uint16_t, 1>();
}

TEST_P(BorderReplicatedIntegralImageTest, u16_2ch) {
    test<std::uint16_t, 2>();
}

TEST_P(BorderReplicatedIntegralImageTest, u16_3ch) {
    test<std::uint16_t, 2>();
}

TEST_P(BorderReplicatedIntegralImageTest, f32_1ch) {
    test<float, 1>();
}

TEST_P(BorderReplicatedIntegralImageTest, f32_2ch) {
    test<float, 2>();
}

TEST_P(BorderReplicatedIntegralImageTest, f32_3ch) {
    test<float, 2>();
}

INSTANTIATE_TEST_SUITE_P(ParametrizeRadius, BorderReplicatedIntegralImageTest, testing::Values(1, 3, 5));
