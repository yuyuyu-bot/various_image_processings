#ifndef ADAPTIVE_BILATERAL_FILTER_HPP
#define ADAPTIVE_BILATERAL_FILTER_HPP

#include <cstdint>
#include <cmath>
#include <opencv2/core.hpp>

#include "bilateral_filter.hpp"

namespace {

namespace internal {

class IntegralImage {
public:
    IntegralImage(
        const cv::Mat3b& src,
        const int radius)
    : width_(src.cols + radius * 2 + 1),
      height_(src.rows + radius * 2 + 1),
      radius_(radius) {
        const int dim[3] = { height_, width_, channels_};
        buffer_.create(src.dims, dim);
        buffer_.setTo(cv::Scalar::all(0));

        for (int y = 1; y < height_; y++) {
            for (int x = 1; x < width_; x++) {
                const auto src_x = std::clamp(x - 1 - radius, 0, src.cols - 1);
                const auto src_y = std::clamp(y - 1 - radius, 0, src.rows - 1);

                for (int ch = 0; ch < channels_; ch++) {
                    buffer_.at<cv::Vec3i>(y, x)[ch] = src.at<cv::Vec3b>(src_y, src_x)[ch];
                }
            }
        }

        for (int y = 1; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                for (int ch = 0; ch < channels_; ch++) {
                    buffer_.at<cv::Vec3i>(y, x)[ch] += buffer_.at<cv::Vec3i>(y - 1, x)[ch];
                }
            }
        }

        for (int y = 0; y < height_; y++) {
            for (int x = 1; x < width_; x++) {
                for (int ch = 0; ch < channels_; ch++) {
                    buffer_.at<cv::Vec3i>(y, x)[ch] += buffer_.at<cv::Vec3i>(y, x - 1)[ch];
                }
            }
        }
    }

    auto get(const int x0, const int y0, const int x1, const int y1) const {
        return buffer_.at<cv::Vec3i>(y1 + radius_ + 1, x1 + radius_ + 1) -
               buffer_.at<cv::Vec3i>(y1 + radius_ + 1, x0 + radius_    ) -
               buffer_.at<cv::Vec3i>(y0 + radius_    , x1 + radius_ + 1) +
               buffer_.at<cv::Vec3i>(y0 + radius_    , x0 + radius_    );
    }

private:
    static constexpr int channels_ = 3;
    const int width_;
    const int height_;
    const int radius_;
    cv::Mat3i buffer_;
};

} // namespace internal

inline void adaptive_bilateral_filter(
    const cv::Mat3b& src,
    cv::Mat3b& dst,
    const int ksize = 9,
    const float sigma_space = 10.f,
    const float sigma_color = 30.f
) {
    struct AdaptiveBilateralFilterCore : cv::ParallelLoopBody {
        AdaptiveBilateralFilterCore(
            const cv::Mat3b& src,
            cv::Mat3b& dst,
            const int radius,
            const float sigma_space,
            const float sigma_color)
        : src_(src),
          dst_(dst),
          width_(src.cols),
          height_(src.rows),
          ksize_(radius * 2 + 1),
          radius_(radius),
          src_integral_(src, radius) {
            std::tie(kernel_space_, kernel_color_table_) = internal::pre_compute_kernels<512 * 3>(ksize_, sigma_space, sigma_color);

            get_kernel_space_ = [ksize=this->ksize_, radius, this](const int kx, const int ky) {
                return kernel_space_[(ky + radius) * ksize + (kx + radius)];
            };

            get_kernel_color_ = [this](const cv::Vec3b& a, const cv::Vec3b& b, const cv::Vec3f& offset) {
                const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]) - offset[0];
                const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]) - offset[1];
                const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]) - offset[2];
                const auto color_distance = std::abs(diff0) + std::abs(diff1) + std::abs(diff2);
                return kernel_color_table_[static_cast<int>(color_distance)];
            };
        }

        void operator()(const cv::Range& range) const CV_OVERRIDE {
            for (int r = range.start; r < range.end; r++) {
                for (int x = 0; x < width_; x++) {
                    const auto src_center_pix = src_.at<cv::Vec3b>(r, x);
                    const auto src_sum = src_integral_.get(x - radius_, r - radius_, x + radius_, r + radius_);
                    const auto del0 = src_center_pix[0] - static_cast<float>(src_sum[0]) / (ksize_ * ksize_);
                    const auto del1 = src_center_pix[1] - static_cast<float>(src_sum[1]) / (ksize_ * ksize_);
                    const auto del2 = src_center_pix[2] - static_cast<float>(src_sum[2]) / (ksize_ * ksize_);
                    cv::Vec3f offset(del0, del1, del2);

                    auto sum0 = 0.f;
                    auto sum1 = 0.f;
                    auto sum2 = 0.f;
                    auto sumk = 0.f;
                    for (int ky = -radius_; ky <= radius_; ky++) {
                        for (int kx = -radius_; kx <= radius_; kx++) {
                            const auto x_clamped = std::clamp(x + kx, 0, width_ - 1);
                            const auto y_clamped = std::clamp(r + ky, 0, height_ - 1);
                            const auto src_pix   = src_.at<cv::Vec3b>(y_clamped, x_clamped);
                            const auto kernel    = get_kernel_space_(kx, ky) * get_kernel_color_(src_pix, src_center_pix, offset);

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
        cv::Mat3b& dst_;
        const int width_;
        const int height_;
        const int ksize_;
        const int radius_;

        const internal::IntegralImage src_integral_;
        std::vector<float> kernel_space_;
        std::array<float, 512 * 3> kernel_color_table_;
        std::function<float(int, int)> get_kernel_space_;
        std::function<float(cv::Vec3b, cv::Vec3b, cv::Vec3f)> get_kernel_color_;
    };


    dst.create(src.size());
    cv::parallel_for_(
        cv::Range(0, src.rows),
        AdaptiveBilateralFilterCore(src, dst, ksize / 2, sigma_space, sigma_color)
    );
}

} // anonymous namespace

#endif // ADAPTIVE_BILATERAL_FILTER_HPP
