#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>

#include "cuda/device_image.hpp"
#include "cuda/bilateral_texture_filter.hpp"
#include "bilateral_texture_filter.hpp"

#define MEASURE(num_itr, fn, duration)                                                                                 \
do {                                                                                                                   \
    std::int64_t sum_duration = 0ll;                                                                                   \
    for (int i = 0; i <= num_itr; i++) {                                                                               \
        const auto start = std::chrono::system_clock::now();                                                           \
        fn;                                                                                                            \
        const auto end = std::chrono::system_clock::now();                                                             \
                                                                                                                       \
        if (i != 0) {                                                                                                  \
            sum_duration  += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();                \
        }                                                                                                              \
    }                                                                                                                  \
    duration = sum_duration / num_itr / 1e6f;                                                                          \
} while (false)

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
    constexpr auto measurement_times = 5;

    cv::setNumThreads(2);

    cv::Mat3b dst_ref(input_image.size());
    cv::Mat3b dst_cuda(input_image.size());

    cuda::DeviceImage<std::uint8_t> d_input_image(input_image.cols, input_image.rows, 3);
    cuda::DeviceImage<std::uint8_t> d_dst(input_image.cols, input_image.rows, 3);
    d_input_image.upload(input_image.ptr<std::uint8_t>());
    cuda::BilateralTextureFilter filter(input_image.cols, input_image.rows, ksize, nitr);

    float duration_ref  = 0.f;
    float duration_cuda = 0.f;

    MEASURE(measurement_times, bilateral_texture_filter(input_image, dst_ref, ksize, nitr), duration_ref);
    MEASURE(measurement_times, filter.execute(d_input_image.get(), d_dst.get()), duration_cuda);
    d_dst.download(dst_cuda.ptr<std::uint8_t>());

    std::cout << "duration ref  : " << duration_ref << " [msec]" << std::endl;
    std::cout << "duration cuda : " << duration_cuda << " [msec]" << std::endl;

    compare(dst_ref, dst_cuda, ksize / 2 * nitr);
}

void benchmark(const cv::Mat3b& input_image) {
    bench_bilateral_texture_filter(input_image, 9, 3);
}
