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
    void execute(const cv::Mat3b& src, cv::Mat3b& dst, const int ksize = 9, const int nitr = 3,
                 const bool debug_print = false) {
        cv::Mat3b src_n;
        src.copyTo(dst);

        for (int itr = 0; itr < nitr; itr++) {
            if (debug_print) { std::cout << cv::format("itration %d", itr + 1) << std::endl; }

            dst.copyTo(src_n);

            if (debug_print) { std::cout << "\tcompute magnitude ..." << std::endl; }
            compute_magnitude(src_n, magnitude_);

            if (debug_print) { std::cout << "\tcompute rtv ..." << std::endl; }
            compute_blur_and_rtv(src_n, magnitude_, blurred_, rtv_, ksize);

            if (debug_print) { std::cout << "\tcompute guide ..." << std::endl; }
            compute_guide(blurred_, rtv_, ksize, guide_);

            if (debug_print) { std::cout << "\tapply joint bilateral filter ..." << std::endl; }
            cv::ximgproc::jointBilateralFilter(guide_, src_n, dst, 2 * ksize - 1, std::sqrt(3), ksize - 1);
            // joint_bilateral_filter(src_n, guide_, dst, 2 * ksize - 1, ksize - 1, 0.05 * std::sqrt(3));
        }
    }

private:
    using PixType = cv::Vec3b;
    using ElemType = std::uint8_t;

    void compute_magnitude_pixel(const cv::Mat3b& image, cv::Mat1f& magnitude, const int x, const int y) {
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

    void compute_magnitude(const cv::Mat3b& image, cv::Mat1f& magnitude) {
        magnitude.create(image.size());

        for (int y = 0; y < magnitude.rows; y++) {
            for (int x = 0; x < magnitude.cols; x++) {
                compute_magnitude_pixel(image, magnitude, x, y);
            }
        }
    }

    void compute_blur_and_rtv_pixel(const cv::Mat3b& image, const cv::Mat1f& magnitude, cv::Mat3f& blurred,
                                    cv::Mat1f& rtv, const int ksize, const int x, const int y) {
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
        auto intensity_min = 0.f;
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

    void compute_blur_and_rtv(const cv::Mat3b& image, const cv::Mat1f& magnitude, cv::Mat3f& blurred, cv::Mat1f& rtv,
                              const int ksize) {
        blurred.create(image.size());
        rtv.create(image.size());

        for (int y = 0; y < blurred.rows; y++) {
            for (int x = 0; x < blurred.cols; x++) {
                compute_blur_and_rtv_pixel(image, magnitude, blurred, rtv, ksize, x, y);
            }
        }
    }

    void compute_guide_pixel(const cv::Mat3f& blurred, const cv::Mat1f& rtv, const int ksize, cv::Mat3b& guide,
                             const int x, const int y) {
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
        guide.at<cv::Vec3b>(y, x)[1] =      alpha  * blurred.at<cv::Vec3f>(rtv_min_y, rtv_min_x)[1] +
                                       (1 - alpha) * blurred.at<cv::Vec3f>(y, x)[1];
        guide.at<cv::Vec3b>(y, x)[2] =      alpha  * blurred.at<cv::Vec3f>(rtv_min_y, rtv_min_x)[2] +
                                       (1 - alpha) * blurred.at<cv::Vec3f>(y, x)[2];
    }

    void compute_guide(const cv::Mat3f& blurred, const cv::Mat1f& rtv, const int ksize, cv::Mat3b& guide) {
        guide.create(blurred.size());

        for (int y = 0; y < guide.rows; y++) {
            for (int x = 0; x < guide.cols; x++) {
                compute_guide_pixel(blurred, rtv, ksize, guide, x, y);
            }
        }
    }

private:
    static constexpr auto epsilon = 1e-9f;

    cv::Mat3f blurred_;
    cv::Mat1f magnitude_;
    cv::Mat1f rtv_;
    cv::Mat3b guide_;
};

namespace {

inline void bilateral_texture_filter(const cv::Mat3b& src, cv::Mat3b& dst, const int ksize = 9, const int nitr = 3,
                                     const bool debug_print = false) {
    BilateralTextureFilterImpl impl;
    impl.execute(src, dst, ksize, nitr, debug_print);
}

} // anonymous namespace

#endif // BILATERAL_TEXTURE_FILTER_HPP
