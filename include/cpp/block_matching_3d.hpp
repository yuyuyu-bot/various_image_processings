#ifndef BLOCK_MATCHING_3D_HPP
#define BLOCK_MATCHING_3D_HPP

#include <algorithm>
#include <cstdint>
#include <opencv2/core.hpp>

template <typename SrcType>
auto wavelet_2D_biorthogonal_1p5(
    const cv::Mat_<SrcType>& src,
    cv::Mat1f& dst_lo,
    cv::Mat1f& dst_hi0,
    cv::Mat1f& dst_hi1
) {
    constexpr int coeff_radius = 5;
    constexpr float bior1p5_low_coeff[11] = {
        0.01657281518405971,
        -0.01657281518405971,
        -0.12153397801643787,
        0.12153397801643787,
        0.7071067811865476,
        1.0,
        0.7071067811865476,
        0.12153397801643787,
        -0.12153397801643787,
        -0.01657281518405971,
        0.01657281518405971
    };
    constexpr float bior1p5_hi0_coeff[11] = {
        0.0,
        0.0,
        0.0,
        0.0,
        -0.7071067811865476,
        0.0,
        0.7071067811865476,
        0.0,
        0.0,
        0.0,
        0.0,
    };
    constexpr float bior1p5_hi1_coeff[11] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.7071067811865476,
        0.0,
        -0.7071067811865476,
        0.0,
        0.0,
        0.0,
        0.0,
    };

    dst_lo.create(src.size());
    dst_hi0.create(src.size());
    dst_hi1.create(src.size());
    dst_lo.setTo(0);
    dst_hi0.setTo(0);
    dst_hi1.setTo(0);
    const auto width  = src.cols;
    const auto height = src.rows;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // horizontal
            {
                float low_reaction = 0.f;
                float hi0_reaction = 0.f;
                float hi1_reaction = 0.f;
                for (int dx = -coeff_radius; dx <= coeff_radius; dx++) {
                    const auto x_clamped = std::clamp(x + dx, 0, width - 1);
                    low_reaction += bior1p5_low_coeff[dx + coeff_radius] * src(y, x_clamped);
                    hi0_reaction += bior1p5_hi0_coeff[dx + coeff_radius] * src(y, x_clamped);
                    hi1_reaction += bior1p5_hi1_coeff[dx + coeff_radius] * src(y, x_clamped);
                }
                dst_lo(y, x) += low_reaction;
                dst_hi0(y, x) += hi0_reaction;
                dst_hi1(y, x) += hi1_reaction;
            }
            // vertival
            {
                float low_reaction = 0.f;
                float hi0_reaction = 0.f;
                float hi1_reaction = 0.f;
                for (int dy = -coeff_radius; dy <= coeff_radius; dy++) {
                    const auto y_clamped = std::clamp(y + dy, 0, height - 1);
                    low_reaction += bior1p5_low_coeff[dy + coeff_radius] * src(y_clamped, x);
                    hi0_reaction += bior1p5_hi0_coeff[dy + coeff_radius] * src(y_clamped, x);
                    hi1_reaction += bior1p5_hi1_coeff[dy + coeff_radius] * src(y_clamped, x);
                }
                dst_lo(y, x) += low_reaction;
                dst_hi0(y, x) += hi0_reaction;
                dst_hi1(y, x) += hi1_reaction;
            }
        }
    }
}

template <typename ValueType, int Channels>
auto block_distance(
    const cv::Mat_<cv::Vec<ValueType, Channels>>& data,
    const int x0,
    const int y0,
    const int x1,
    const int y1,
    const int block_diameter,
    const float lambda_2d,
    const float sigma
) {
    const auto block_radius = block_diameter / 2;
    auto distance = 0.f;

    if (sigma < 40.f) {
        for (int dy = -block_radius; dy <= block_radius; dy++) {
            for (int dx = -block_radius; dx <= block_radius; dx++) {
                const auto x0_offset = std::clamp(x0 + dx, 0, data.cols - 1);
                const auto y0_offset = std::clamp(y0 + dy, 0, data.rows - 1);
                const auto x1_offset = std::clamp(x1 + dx, 0, data.cols - 1);
                const auto y1_offset = std::clamp(y1 + dy, 0, data.rows - 1);

                for (int ch = 0; ch < Channels; ch++) {
                    const auto d = data(y0_offset, x0_offset)[ch] - data(y1_offset, x1_offset)[ch];
                    distance += d * d;
                }
            }
        }
    }
    else {
        const auto threshold = lambda_2d * sigma;
        for (int dy = -block_radius; dy <= block_radius; dy++) {
            for (int dx = -block_radius; dx <= block_radius; dx++) {
                const auto x0_offset = std::clamp(x0 + dx, 0, data.cols - 1);
                const auto y0_offset = std::clamp(y0 + dy, 0, data.rows - 1);
                const auto x1_offset = std::clamp(x1 + dx, 0, data.cols - 1);
                const auto y1_offset = std::clamp(y1 + dy, 0, data.rows - 1);

                for (int ch = 0; ch < Channels; ch++) {
                    auto d0 = data(y0_offset, x0_offset)[ch];
                    d0 = std::abs(d0) < threshold ? 0 : d0;
                    auto d1 = data(y1_offset, x1_offset)[ch];
                    d1 = std::abs(d1) < threshold ? 0 : d1;
                    distance += (d0 - d1) * (d0 - d1);
                }
            }
        }
    }

    return distance / (block_diameter * block_diameter);
}

template <typename ValueType, int Channels>
auto block_matching(
    const cv::Mat_<cv::Vec<ValueType, Channels>>& data,
    const int block_diameter,
    const int x,
    const int y
) {
    std::vector<std::pair<float, std::pair<int, int>>> similar_blocks;

    constexpr auto lambda_2d = 8.f;
    constexpr auto sigma = 10.f;

    constexpr auto match_threshold = 2500.f;
    constexpr auto window_diameter = 39;
    constexpr auto window_radius = window_diameter / 2;
    constexpr auto max_match = 10;

    for (int dy = -window_radius; dy <= window_radius; dy++) {
        for (int dx = -window_radius; dx <= window_radius; dx++) {
            const auto nx = x + dx;
            const auto ny = y + dy;
            if (nx < 0 || nx >= data.cols || ny < 0 || ny >= data.rows) {
                continue;
            }

            const auto distance = block_distance(data, x, y, nx, ny, block_diameter, lambda_2d, sigma);
            if (distance < match_threshold) {
                similar_blocks.push_back(std::make_pair(distance, std::make_pair(nx, ny)));
            }
        }
    }

    if (similar_blocks.size() > max_match) {
        std::sort(similar_blocks.begin(), similar_blocks.end());
        similar_blocks.erase(similar_blocks.begin() + max_match, similar_blocks.end());
    }

    return similar_blocks;
}

class BM3D {
public:



private:


};




#endif // BLOCK_MATCHING_3D_HPP
