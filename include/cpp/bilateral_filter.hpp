#ifndef BILATERAL_FILTER_HPP
#define BILATERAL_FILTER_HPP

#include <cstdint>
#include <cmath>
#include <opencv2/core.hpp>

namespace {

inline void bilateral_filter(const cv::Mat3b& src, cv::Mat3b& dst, const int ksize = 9,
                             const float sigma_space = 10.f, const float sigma_color = 30.f) {
    const auto width  = src.cols;
    const auto height = src.rows;
    const auto radius  = ksize / 2;
    dst.create(src.size());

    const auto kernel_space = std::make_unique<float[]>(ksize * ksize);
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto kidx = (ky + radius) * ksize + (kx + radius);
            kernel_space[kidx] = std::exp(-(kx * kx + ky * ky) / (2 * sigma_space * sigma_space));
        }
    }

    static std::array<float, 255 * 255> kernel_color_table;
    for (int i = 0; i < kernel_color_table.size(); i++) {
        kernel_color_table[i] = std::exp(-i / (2 * sigma_color * sigma_color));
    }

    const auto get_kernel_space = [ksize, radius, &kernel_space](const int kx, const int ky) {
        return kernel_space[(ky + radius) * ksize + (kx + radius)];
    };

    const auto get_kernel_color = [](const cv::Vec3b& a, const cv::Vec3b& b) {
        const auto diff_b = static_cast<int>(a[0]) - static_cast<int>(b[0]);
        const auto diff_g = static_cast<int>(a[1]) - static_cast<int>(b[1]);
        const auto diff_r = static_cast<int>(a[2]) - static_cast<int>(b[2]);
        const auto color_distance = (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r) / 3;
        return kernel_color_table[static_cast<int>(color_distance + 0.5f)];
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto center_bgr = src.at<cv::Vec3b>(y, x);
            auto sum_b = 0.f;
            auto sum_g = 0.f;
            auto sum_r = 0.f;
            auto sum_k = 0.f;

            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    const auto x_clamped = std::clamp(x + kx, 0, width - 1);
                    const auto y_clamped = std::clamp(y + ky, 0, height - 1);
                    const auto bgr       = src.at<cv::Vec3b>(y_clamped, x_clamped);
                    const auto kernel    = get_kernel_space(kx, ky) * get_kernel_color(center_bgr, bgr);

                    sum_b += bgr[0] * kernel;
                    sum_g += bgr[1] * kernel;
                    sum_r += bgr[2] * kernel;
                    sum_k += kernel;
                }
            }

            dst.at<cv::Vec3b>(y, x)[0] = static_cast<std::uint8_t>(sum_b / sum_k);
            dst.at<cv::Vec3b>(y, x)[1] = static_cast<std::uint8_t>(sum_g / sum_k);
            dst.at<cv::Vec3b>(y, x)[2] = static_cast<std::uint8_t>(sum_r / sum_k);
        }
    }
}

inline void joint_bilateral_filter(const cv::Mat3b& src, const cv::Mat3b& guide, cv::Mat3b& dst, const int ksize = 9,
                                   const float sigma_space = 10.f, const float sigma_color = 30.f) {
    const auto width  = src.cols;
    const auto height = src.rows;
    const auto radius  = ksize / 2;
    dst.create(src.size());

    const auto kernel_space = std::make_unique<float[]>(ksize * ksize);
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto kidx = (ky + radius) * ksize + (kx + radius);
            kernel_space[kidx] = std::exp(-(kx * kx + ky * ky) / (2 * sigma_space * sigma_space));
        }
    }

    static std::array<float, 255 * 255> kernel_color_table;
    for (int i = 0; i < kernel_color_table.size(); i++) {
        kernel_color_table[i] = std::exp(-i / (2 * sigma_color * sigma_color));
    }

    const auto get_kernel_space = [ksize, radius, &kernel_space](const int kx, const int ky) {
        return kernel_space[(ky + radius) * ksize + (kx + radius)];
    };

    const auto get_kernel_color = [](const cv::Vec3b& a, const cv::Vec3b& b) {
        const auto diff_b = static_cast<int>(a[0]) - static_cast<int>(b[0]);
        const auto diff_g = static_cast<int>(a[1]) - static_cast<int>(b[1]);
        const auto diff_r = static_cast<int>(a[2]) - static_cast<int>(b[2]);
        const auto color_distance = (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r) / 3;
        return kernel_color_table[color_distance];
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto guide_center_bgr = guide.at<cv::Vec3b>(y, x);
            auto sum_b = 0.f;
            auto sum_g = 0.f;
            auto sum_r = 0.f;
            auto sum_k = 0.f;

            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    const auto x_clamped = std::clamp(x + kx, 0, width - 1);
                    const auto y_clamped = std::clamp(y + ky, 0, height - 1);
                    const auto bgr       = src.at<cv::Vec3b>(y_clamped, x_clamped);
                    const auto guide_bgr = guide.at<cv::Vec3b>(y_clamped, x_clamped);
                    const auto kernel    = get_kernel_space(kx, ky) * get_kernel_color(guide_center_bgr, guide_bgr);

                    sum_b += bgr[0] * kernel;
                    sum_g += bgr[1] * kernel;
                    sum_r += bgr[2] * kernel;
                    sum_k += kernel;
                }
            }

            dst.at<cv::Vec3b>(y, x)[0] = static_cast<std::uint8_t>(sum_b / sum_k);
            dst.at<cv::Vec3b>(y, x)[1] = static_cast<std::uint8_t>(sum_g / sum_k);
            dst.at<cv::Vec3b>(y, x)[2] = static_cast<std::uint8_t>(sum_r / sum_k);
        }
    }
}

} // anonymous namespace

#endif // BILATERAL_FILTER_HPP
