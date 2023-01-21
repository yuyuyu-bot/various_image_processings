#ifndef BILATERAL_FILTER_HPP
#define BILATERAL_FILTER_HPP

#include <cstdint>
#include <cmath>
#include <opencv2/core.hpp>

namespace {

template <int ColorTableSize = 256 * 3>
inline auto pre_compute_kernels(const int ksize, const float sigma_space, const float sigma_color) {
    const auto radius  = ksize / 2;
    const auto gauss_color_coeff = -1. / (2 * sigma_color * sigma_color);
    const auto gauss_space_coeff = -1. / (2 * sigma_space * sigma_space);

    std::vector<float> kernel_space(ksize * ksize);
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto kidx = (ky + radius) * ksize + (kx + radius);
            const auto r2 = kx * kx + ky * ky;
            if (r2 > radius * radius) {
                kernel_space[kidx] = 0.f;
                continue;
            }
            kernel_space[kidx] = std::exp(r2 * gauss_space_coeff);
        }
    }

    std::array<float, ColorTableSize> kernel_color_table;
    for (int i = 0; i < kernel_color_table.size(); i++) {
        kernel_color_table[i] = std::exp((i * i) * gauss_color_coeff);
    }

    return std::make_pair(kernel_space, kernel_color_table);
}

inline void bilateral_filter(
    const cv::Mat3b& src,
    cv::Mat3b& dst,
    const int ksize = 9,
    const float sigma_space = 10.f,
    const float sigma_color = 30.f
) {
    struct BilateralFilterCore : cv::ParallelLoopBody {
        BilateralFilterCore(
            const cv::Mat3b& src,
            cv::Mat3b& dst,
            const int radius,
            const float sigma_space,
            const float sigma_color)
        : src_(src),
          dst_(dst),
          width_(src.cols),
          height_(src.rows),
          radius_(radius) {
            const auto ksize = radius_ * 2 + 1;
            std::tie(kernel_space_, kernel_color_table_) = pre_compute_kernels(ksize, sigma_space, sigma_color);

            get_kernel_space_ = [ksize, radius, this](const int kx, const int ky) {
                return kernel_space_[(ky + radius) * ksize + (kx + radius)];
            };

            get_kernel_color_ = [this](const cv::Vec3b& a, const cv::Vec3b& b) {
                const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
                const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
                const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
                const auto color_distance = std::abs(diff0) + std::abs(diff1) + std::abs(diff2);
                return kernel_color_table_[color_distance];
            };
        }

        void operator()(const cv::Range& range) const CV_OVERRIDE {
            for (int r = range.start; r < range.end; r++) {
                const auto x = r % width_;
                const auto y = r / width_;

                const auto src_center_pix = src_.at<cv::Vec3b>(y, x);
                auto sum0 = 0.f;
                auto sum1 = 0.f;
                auto sum2 = 0.f;
                auto sumk = 0.f;

                for (int ky = -radius_; ky <= radius_; ky++) {
                    for (int kx = -radius_; kx <= radius_; kx++) {
                        const auto x_clamped = std::clamp(x + kx, 0, width_ - 1);
                        const auto y_clamped = std::clamp(y + ky, 0, height_ - 1);
                        const auto src_pix   = src_.at<cv::Vec3b>(y_clamped, x_clamped);
                        const auto kernel    = get_kernel_space_(kx, ky) * get_kernel_color_(src_center_pix, src_pix);

                        sum0 += src_pix[0] * kernel;
                        sum1 += src_pix[1] * kernel;
                        sum2 += src_pix[2] * kernel;
                        sumk += kernel;
                    }
                }

                dst_.at<cv::Vec3b>(y, x)[0] = static_cast<std::uint8_t>(sum0 / sumk + 0.5f);
                dst_.at<cv::Vec3b>(y, x)[1] = static_cast<std::uint8_t>(sum1 / sumk + 0.5f);
                dst_.at<cv::Vec3b>(y, x)[2] = static_cast<std::uint8_t>(sum2 / sumk + 0.5f);
            }
        }

        const cv::Mat3b& src_;
        cv::Mat3b& dst_;
        const int width_;
        const int height_;
        const int radius_;

        std::vector<float> kernel_space_;
        std::array<float, 256 * 3> kernel_color_table_;
        std::function<float(int, int)> get_kernel_space_;
        std::function<float(cv::Vec3b, cv::Vec3b)> get_kernel_color_;
    };

    dst.create(src.size());
    cv::parallel_for_(
        cv::Range(0, src.rows * src.cols),
        BilateralFilterCore(src, dst, ksize / 2, sigma_space, sigma_color)
    );
}

inline void joint_bilateral_filter(const cv::Mat3b& src, const cv::Mat3b& guide, cv::Mat3b& dst, const int ksize = 9,
                                   const float sigma_space = 10.f, const float sigma_color = 30.f) {
    struct JointBilateralFilterCore : cv::ParallelLoopBody {
        JointBilateralFilterCore(
            const cv::Mat3b& src,
            const cv::Mat3b& guide,
            cv::Mat3b& dst,
            const int radius,
            const float sigma_space,
            const float sigma_color)
        : src_(src),
          guide_(guide),
          dst_(dst),
          width_(src.cols),
          height_(src.rows),
          radius_(radius) {
            const auto ksize = radius_ * 2 + 1;
            const auto&& [kernel_space, kernel_color_table] = pre_compute_kernels(ksize, sigma_space, sigma_color);

            get_kernel_space_ = [ksize, radius, &kernel_space](const int kx, const int ky) {
                return kernel_space[(ky + radius) * ksize + (kx + radius)];
            };

            get_kernel_color_ = [&kernel_color_table](const cv::Vec3b& a, const cv::Vec3b& b) {
                const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
                const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
                const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
                const auto color_distance = std::abs(diff0) + std::abs(diff1) + std::abs(diff2);
                return kernel_color_table[color_distance];
            };
        }

        void operator()(const cv::Range& range) const CV_OVERRIDE {
            for (int r = range.start; r < range.end; r++) {
                for (int x = 0; x < width_; x++) {
                    const auto guide_center_pix = guide_.at<cv::Vec3b>(r, x);
                    auto sum0 = 0.f;
                    auto sum1 = 0.f;
                    auto sum2 = 0.f;
                    auto sumk = 0.f;

                    for (int ky = -radius_; ky <= radius_; ky++) {
                        for (int kx = -radius_; kx <= radius_; kx++) {
                            const auto x_clamped = std::clamp(x + kx, 0, width_ - 1);
                            const auto y_clamped = std::clamp(r + ky, 0, height_ - 1);
                            const auto src_pix   = src_.at<cv::Vec3b>(y_clamped, x_clamped);
                            const auto guide_pix = guide_.at<cv::Vec3b>(y_clamped, x_clamped);
                            const auto kernel    = get_kernel_space_(kx, ky) * get_kernel_color_(guide_center_pix, guide_pix);

                            sum0 += src_pix[0] * kernel;
                            sum1 += src_pix[1] * kernel;
                            sum2 += src_pix[2] * kernel;
                            sumk += kernel;
                        }
                    }

                    dst_.at<cv::Vec3b>(r, x)[0] = static_cast<std::uint8_t>(sum0 / sumk + 0.5f);
                    dst_.at<cv::Vec3b>(r, x)[1] = static_cast<std::uint8_t>(sum1 / sumk + 0.5f);
                    dst_.at<cv::Vec3b>(r, x)[2] = static_cast<std::uint8_t>(sum2 / sumk + 0.5f);
                }
            }
        }

        const cv::Mat3b& src_;
        const cv::Mat3b& guide_;
        cv::Mat3b& dst_;
        const int width_;
        const int height_;
        const int radius_;

        std::function<float(int, int)> get_kernel_space_;
        std::function<float(cv::Vec3b, cv::Vec3b)> get_kernel_color_;
    };

    dst.create(src.size());
    cv::parallel_for_(
        cv::Range(0, src.rows),
        JointBilateralFilterCore(src, guide, dst, ksize / 2, sigma_space, sigma_color)
    );
}

} // anonymous namespace

#endif // BILATERAL_FILTER_HPP
