#ifndef BILATERAL_TEXTURE_FILTER_HPP
#define BILATERAL_TEXTURE_FILTER_HPP

#include <cstdint>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>

class BilateralTextureFilterImpl {
public:
    void execute(const cv::Mat3b& src, cv::Mat3b& dst, const int ksize = 9, const int nitr = 3) {
        cv::Mat3b src_n;
        src.copyTo(dst);

        for (int itr = 0; itr < nitr; itr++) {
            dst.copyTo(src_n);
            compute_magnitude(src_n, magnitude_);
            compute_blur_and_rtv(src_n, magnitude_, blurred_, rtv_, ksize);
            compute_guide(blurred_, rtv_, guide_, ksize);
            cv::ximgproc::jointBilateralFilter(guide_, src_n, dst, 2 * ksize - 1, std::sqrt(3), ksize - 1);
        }
    }

private:
    using PixType = cv::Vec3b;
    using ElemType = std::uint8_t;

    class ComputeMagnitude : public cv::ParallelLoopBody {
    public:
        ComputeMagnitude(const cv::Mat3b& image, cv::Mat1f& magnitude) : image_(image), magnitude_(magnitude) {}

        void compute_magnitude_pixel(const cv::Mat3b& image, cv::Mat1f& magnitude, const int x, const int y) const {
            if (x == 0 || x == image.cols - 1 || y == 0 || y == image.rows - 1) {
                magnitude.at<float>(y, x) = 0.f;
                return;
            }

            const auto compute_del = [&image](const int x0, const int y0, const int x1, const int y1) {
                const auto diff_b = image.at<PixType>(y0, x0)[0] - image.at<PixType>(y1, x1)[0];
                const auto diff_g = image.at<PixType>(y0, x0)[1] - image.at<PixType>(y1, x1)[1];
                const auto diff_r = image.at<PixType>(y0, x0)[2] - image.at<PixType>(y1, x1)[2];
                return diff_b * diff_b + diff_g * diff_g + diff_r * diff_r;
            };

            const auto del_x = compute_del(x - 1, y, x + 1, y);
            const auto del_y = compute_del(x, y - 1, x, y + 1);
            magnitude.at<float>(y, x) = std::sqrt(del_x + del_y);
        }

        void operator()(const cv::Range& range) const CV_OVERRIDE {
            for (int r = range.start; r < range.end; r++) {
                const auto x = r % image_.cols;
                const auto y = r / image_.cols;
                compute_magnitude_pixel(image_, magnitude_, x, y);
            }
        }

    private:
        const cv::Mat3b& image_;
        cv::Mat1f& magnitude_;
    };

    void compute_magnitude(const cv::Mat3b& image, cv::Mat1f& magnitude) {
        magnitude.create(image.size());
        cv::parallel_for_(cv::Range(0, image.rows * image.cols), ComputeMagnitude(image, magnitude));
    }

    class ComputeBlurAndRTV : public cv::ParallelLoopBody {
    public:
        ComputeBlurAndRTV(
            const cv::Mat3b& image, const cv::Mat1f& magnitude, cv::Mat3f& blurred, cv::Mat1f& rtv, const int ksize)
            : image_(image), magnitude_(magnitude), blurred_(blurred), rtv_(rtv), ksize_(ksize) {}

        void compute_blur_and_rtv_pixel(const cv::Mat3b& image, const cv::Mat1f& magnitude, cv::Mat3f& blurred,
                                        cv::Mat1f& rtv, const int ksize, const int x, const int y) const {
            const auto width  = magnitude.cols;
            const auto height = magnitude.rows;
            const auto khalf  = ksize / 2;
            const auto get_intensity = [&image](const int x, const int y) {
                const auto bgr = image.at<PixType>(y, x);
                return (bgr[0] + bgr[1] + bgr[2]) / 3.f;
            };

            auto b_sum = 0;
            auto g_sum = 0;
            auto r_sum = 0;

            auto intensity_max = 0.f;
            auto intensity_min = 256.f;
            auto magnitude_max = 0.f;
            auto magnitude_sum = 0.f;

            for (int ky = -khalf; ky <= khalf; ky++) {
                for (int kx = -khalf; kx <= khalf; kx++) {
                    const auto x_clamped = std::clamp(x + kx, 0, width - 1);
                    const auto y_clamped = std::clamp(y + ky, 0, height - 1);

                    b_sum += image.at<PixType>(y_clamped, x_clamped)[0];
                    g_sum += image.at<PixType>(y_clamped, x_clamped)[1];
                    r_sum += image.at<PixType>(y_clamped, x_clamped)[2];

                    intensity_max  = std::max(intensity_max, get_intensity(x_clamped, y_clamped));
                    intensity_min  = std::min(intensity_min, get_intensity(x_clamped, y_clamped));
                    magnitude_max  = std::max(magnitude_max, magnitude.at<float>(y_clamped, x_clamped));
                    magnitude_sum += magnitude.at<float>(y_clamped, x_clamped);
                }
            }

            blurred.at<cv::Vec3f>(y, x)[0] = static_cast<ElemType>(b_sum / (ksize * ksize));
            blurred.at<cv::Vec3f>(y, x)[1] = static_cast<ElemType>(g_sum / (ksize * ksize));
            blurred.at<cv::Vec3f>(y, x)[2] = static_cast<ElemType>(r_sum / (ksize * ksize));
            rtv.at<float>(y, x) = (intensity_max - intensity_min) * magnitude_max / (magnitude_sum + epsilon);
        }

        void operator()(const cv::Range& range) const CV_OVERRIDE {
            for (int r = range.start; r < range.end; r++) {
                const auto x = r % image_.cols;
                const auto y = r / image_.cols;
                compute_blur_and_rtv_pixel(image_, magnitude_, blurred_, rtv_, ksize_, x, y);
            }
        }

    private:
        const cv::Mat3b& image_;
        const cv::Mat1f& magnitude_;
        cv::Mat3f& blurred_;
        cv::Mat1f& rtv_;
        const int ksize_;
    };

    void compute_blur_and_rtv(const cv::Mat3b& image, const cv::Mat1f& magnitude, cv::Mat3f& blurred, cv::Mat1f& rtv,
                              const int ksize) {
        blurred.create(image.size());
        rtv.create(image.size());
        cv::parallel_for_(cv::Range(0, image.rows * image.cols), ComputeBlurAndRTV(image, magnitude, blurred, rtv, ksize));
    }

    class ComputeGuide : public cv::ParallelLoopBody {
    public:
        ComputeGuide(const cv::Mat3f& blurred, const cv::Mat1f& rtv, cv::Mat3b& guide, const int ksize)
            : blurred_(blurred), rtv_(rtv), guide_(guide), ksize_(ksize) {}

        void compute_guide_pixel(const cv::Mat3f& blurred, const cv::Mat1f& rtv, cv::Mat3b& guide, const int ksize,
                                 const int x, const int y) const {
            const auto width  = blurred.cols;
            const auto height = blurred.rows;
            const auto khalf  = ksize / 2;
            const auto sigma_alpha = 1.f / (5 * ksize);

            auto rtv_min = std::numeric_limits<float>::max();
            auto rtv_min_x = 0;
            auto rtv_min_y = 0;

            for (int ky = -khalf; ky <= khalf; ky++) {
                for (int kx = -khalf; kx <= khalf; kx++) {
                    const auto x_clamped = std::clamp(x + kx, 0, width - 1);
                    const auto y_clamped = std::clamp(y + ky, 0, height - 1);

                    if (rtv_min > rtv.at<float>(y_clamped, x_clamped)) {
                        rtv_min = rtv.at<float>(y_clamped, x_clamped);
                        rtv_min_x = x_clamped;
                        rtv_min_y = y_clamped;
                    }
                }
            }

            const auto alpha =
                2 / (1 + std::exp(sigma_alpha * (rtv.at<float>(y, x) - rtv.at<float>(rtv_min_y, rtv_min_x)))) - 1.f;
            guide.at<cv::Vec3b>(y, x)[0] =      alpha  * blurred.at<cv::Vec3f>(rtv_min_y, rtv_min_x)[0] +
                                           (1 - alpha) * blurred.at<cv::Vec3f>(y, x)[0];
            guide.at<cv::Vec3b>(y, x)[ 1] =     alpha  * blurred.at<cv::Vec3f>(rtv_min_y, rtv_min_x)[1] +
                                           (1 - alpha) * blurred.at<cv::Vec3f>(y, x)[1];
            guide.at<cv::Vec3b>(y, x)[2] =      alpha  * blurred.at<cv::Vec3f>(rtv_min_y, rtv_min_x)[2] +
                                           (1 - alpha) * blurred.at<cv::Vec3f>(y, x)[2];
        }

        void operator()(const cv::Range& range) const CV_OVERRIDE {
            for (int r = range.start; r < range.end; r++) {
                const auto x = r % blurred_.cols;
                const auto y = r / blurred_.cols;
                compute_guide_pixel(blurred_, rtv_, guide_, ksize_, x, y);
            }
        }

    private:
        const cv::Mat3f& blurred_;
        const cv::Mat1f& rtv_;
        cv::Mat3b& guide_;
        const int ksize_;
    };

    void compute_guide(const cv::Mat3f& blurred, const cv::Mat1f& rtv, cv::Mat3b& guide, const int ksize) {
        guide.create(blurred.size());
        cv::parallel_for_(cv::Range(0, blurred.rows * blurred.cols), ComputeGuide(blurred, rtv, guide, ksize));
    }

private:
    static constexpr auto epsilon = 1e-9f;

    cv::Mat3f blurred_;
    cv::Mat1f magnitude_;
    cv::Mat1f rtv_;
    cv::Mat3b guide_;
};

namespace {

inline void bilateral_texture_filter(const cv::Mat3b& src, cv::Mat3b& dst, const int ksize = 9, const int nitr = 3) {
    BilateralTextureFilterImpl impl;
    impl.execute(src, dst, ksize, nitr);
}

} // anonymous namespace

#endif // BILATERAL_TEXTURE_FILTER_HPP
