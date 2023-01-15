#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>

#include "cuda/device_image.hpp"
#include "cuda/bilateral_texture_filter.hpp"
#include "bilateral_texture_filter.hpp"

static auto compare(const cv::Mat3b& a, const cv::Mat3b& b, const int radius = 0) {
    const auto width  = a.cols;
    const auto height = a.rows;

    auto max_diff = -1;
    auto diff_count = 0;

    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
            if (a.at<cv::Vec3b>(y, x) != b.at<cv::Vec3b>(y, x)) {
                max_diff = std::max(max_diff, static_cast<int>(a.at<cv::Vec3b>(y, x)[0]) - static_cast<int>(b.at<cv::Vec3b>(y, x)[0]));
                max_diff = std::max(max_diff, static_cast<int>(a.at<cv::Vec3b>(y, x)[1]) - static_cast<int>(b.at<cv::Vec3b>(y, x)[1]));
                max_diff = std::max(max_diff, static_cast<int>(a.at<cv::Vec3b>(y, x)[2]) - static_cast<int>(b.at<cv::Vec3b>(y, x)[2]));
                diff_count++;
            }
        }
    }

    std::cout << "diff count : " << diff_count << std::endl;
    std::cout << "max diff   : " << max_diff << std::endl;
}

static void bench_bilateral_texture_filter(const cv::Mat3b& input_image, const int ksize, const int nitr) {
    constexpr auto measurent_times = 5;

    cv::setNumThreads(2);

    cv::Mat3b dst_ref(input_image.size());
    cv::Mat3b dst_cuda(input_image.size());

    cuda::DeviceImage<std::uint8_t> d_input_image(input_image.cols, input_image.rows, 3);
    cuda::DeviceImage<std::uint8_t> d_dst(input_image.cols, input_image.rows, 3);
    d_input_image.upload(input_image.ptr<std::uint8_t>());
    cuda::BilateralTextureFilter filter(input_image.cols, input_image.rows, ksize, nitr);

    std::int64_t sum_duration_ref  = 0ll;
    std::int64_t sum_duration_cuda = 0ll;
    for (int i = 0; i <= measurent_times; i++) {
        const auto start_ref = std::chrono::system_clock::now();
        bilateral_texture_filter(input_image, dst_ref, ksize, nitr);
        const auto end_ref = std::chrono::system_clock::now();

        const auto start_cuda = std::chrono::system_clock::now();
        filter.execute(d_input_image.get(), d_dst.get());
        const auto end_cuda = std::chrono::system_clock::now();

        if (i != 0) {
            sum_duration_ref  += std::chrono::duration_cast<std::chrono::milliseconds>(end_ref - start_ref).count();
            sum_duration_cuda += std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda - start_cuda).count();
        }
    }
    std::cout << "duration ref  : " << sum_duration_ref / measurent_times << " [msec]" << std::endl;
    std::cout << "duration cuda : " << sum_duration_cuda / measurent_times << " [msec]" << std::endl;

    compare(dst_ref, dst_cuda, ksize / 2);
}

void benchmark(const cv::Mat3b& input_image) {
    bench_bilateral_texture_filter(input_image, 9, 3);
}
