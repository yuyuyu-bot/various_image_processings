#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "toml.hpp"

#include "cpp/gradient.hpp"
#include "cpp/bilateral_filter.hpp"
#include "cpp/adaptive_bilateral_filter.hpp"
#include "cpp/bilateral_texture_filter.hpp"
#include "cpp/slic.hpp"

#include "cuda/device_image.hpp"
#include "cuda/gradient.hpp"
#include "cuda/bilateral_filter.hpp"
#include "cuda/adaptive_bilateral_filter.hpp"
#include "cuda/bilateral_texture_filter.hpp"

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

static void print_duration(const std::string& name, const float duration) {
    std::printf("%-40s : %10.6f [msec]\n", name.c_str(), duration);
}

struct Parameters {
    int execute_times = 50;

    struct {
        int ksize = 9;
    } BilateralFilter;

    struct {
        int ksize = 9;
    } AdaptiveBilateralFilter;

    struct {
        int ksize = 9;
        int nitr = 3;
    } BilateralTextureFilter;

    struct {
        int superpixel_size = 10;
        int num_iteration = 10;
    } SuperpixelSLIC;
};

static auto parse_config(const std::string& filename) {
    const auto toml_all = toml::parse(filename);

    Parameters params;
    toml::value data;

    // common
    params.execute_times = toml::find<int>(toml_all, "execute_times");

    // bilateral filter
    data = toml::find(toml_all, "BilateralFilter");
    params.BilateralFilter.ksize = toml::find<int>(data, "ksize");

    // adaptive bilateral filter
    data = toml::find(toml_all, "AdaptiveBilateralFilter");
    params.AdaptiveBilateralFilter.ksize = toml::find<int>(data, "ksize");

    // bilateral texture filter
    data = toml::find(toml_all, "BilateralTextureFilter");
    params.BilateralTextureFilter.ksize = toml::find<int>(data, "ksize");
    params.BilateralTextureFilter.nitr = toml::find<int>(data, "nitr");

    // superpixel SLIC
    data = toml::find(toml_all, "SuperpixelSLIC");
    params.SuperpixelSLIC.superpixel_size = toml::find<int>(data, "superpixel_size");
    params.SuperpixelSLIC.num_iteration = toml::find<int>(data, "num_iteration");

    return params;
}

static auto convert_to_3ch(const cv::Mat& image) {
    cv::Mat3b image_3ch;
    if (image.channels() == 1) {
        cv::cvtColor(image, image_3ch, cv::COLOR_GRAY2BGR);
    }
    else if (image.channels() == 3) {
        image.copyTo(image_3ch);
    }
    else if (image.channels() == 4) {
        cv::cvtColor(image, image_3ch, cv::COLOR_BGRA2BGR);
    }
    return image_3ch;
}

static void bench_gradient(
    const int measurement_times,
    const cv::Mat& input_image
) {
    const cv::Mat3b input_image_color = convert_to_3ch(input_image);
    cv::Mat1f dst(input_image_color.size());

    float duration = 0.f;
    MEASURE(measurement_times, gradient(input_image_color, dst), duration);
    print_duration("gradient [cpp]", duration);

    DeviceImage<std::uint8_t> d_input_image(input_image.cols, input_image.rows, 3);
    DeviceImage<float> d_dst(input_image.cols, input_image.rows);
    d_input_image.upload(input_image_color.ptr<std::uint8_t>());

    MEASURE(measurement_times, cuda_gradient(d_input_image.get(), d_dst.get(), input_image.cols, input_image.rows, 3), duration);
    print_duration("gradient [cuda]", duration);
}

static void bench_bilateral_filter(
    const int measurement_times,
    const cv::Mat& input_image,
    const int ksize
) {
    const cv::Mat3b input_image_color = convert_to_3ch(input_image);
    cv::Mat3b dst(input_image_color.size());

    float duration = 0.f;
    MEASURE(measurement_times, bilateral_filter(input_image_color, dst, ksize), duration);
    print_duration("bilateral filter [cpp]", duration);

    DeviceImage<std::uint8_t> d_input_image(input_image.cols, input_image.rows, 3);
    DeviceImage<std::uint8_t> d_dst(input_image.cols, input_image.rows, 3);
    d_input_image.upload(input_image_color.ptr<std::uint8_t>());
    CudaBilateralFilter filter(input_image.cols, input_image.rows, ksize);

    MEASURE(measurement_times, filter.bilateral_filter(d_input_image.get(), d_dst.get()), duration);
    print_duration("bilateral filter [cuda]", duration);
}

static void bench_adaptive_bilateral_filter(
    const int measurement_times,
    const cv::Mat& input_image,
    const int ksize
) {
    const cv::Mat3b input_image_color = convert_to_3ch(input_image);
    cv::Mat3b dst(input_image_color.size());

    float duration = 0.f;
    MEASURE(measurement_times, adaptive_bilateral_filter(input_image_color, dst, ksize), duration);
    print_duration("adaptive bilateral filter [cpp]", duration);

    DeviceImage<std::uint8_t> d_input_image(input_image.cols, input_image.rows, 3);
    DeviceImage<std::uint8_t> d_dst(input_image.cols, input_image.rows, 3);
    d_input_image.upload(input_image_color.ptr<std::uint8_t>());
    CudaAdaptiveBilateralFilter filter(input_image.cols, input_image.rows, ksize);

    MEASURE(measurement_times, filter.execute(d_input_image.get(), d_dst.get()), duration);
    print_duration("adaptive bilateral filter [cuda]", duration);
}

static void bench_bilateral_texture_filter(
    const int measurement_times,
    const cv::Mat& input_image,
    const int ksize,
    const int nitr
) {
    const cv::Mat3b input_image_color = convert_to_3ch(input_image);
    cv::Mat3b dst(input_image_color.size());

    float duration = 0.f;
    MEASURE(measurement_times, bilateral_texture_filter(input_image_color, dst, ksize, nitr), duration);
    print_duration("bilateral texture filter [cpp]", duration);

    DeviceImage<std::uint8_t> d_input_image(input_image.cols, input_image.rows, 3);
    DeviceImage<std::uint8_t> d_dst(input_image.cols, input_image.rows, 3);
    d_input_image.upload(input_image_color.ptr<std::uint8_t>());
    CudaBilateralTextureFilter filter(input_image.cols, input_image.rows, ksize, nitr);

    MEASURE(measurement_times, filter.execute(d_input_image.get(), d_dst.get()), duration);
    print_duration("bilateral texture filter [cuda]", duration);
}

static void bench_superpixel_slic(
    const int measurement_times,
    const cv::Mat& input_image,
    const int superpixel_size,
    const int num_iteration
) {
    const cv::Mat3b input_image_color = convert_to_3ch(input_image);
    cv::Mat1i dst;

    float duration = 0.f;

    MEASURE(measurement_times, superpixel_slic(input_image, dst, superpixel_size, num_iteration), duration);
    print_duration("superpixel SLIC [cpp]", duration);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "[Usage] ./benchmark [config_path]" << std::endl;
    }

    const auto params = argc == 3 ? parse_config(argv[2]) : Parameters();

    constexpr auto width  = 100;
    constexpr auto height = 100;
    cv::Mat3b input_image(height, width);
    cv::randu(input_image, cv::Scalar::all(100), cv::Scalar::all(120));

    std::cout << "Parameters" << std::endl;
    std::cout << "\twidth         : " << input_image.cols << std::endl;
    std::cout << "\theight        : " << input_image.rows << std::endl;
    std::cout << "\texecute times : " << params.execute_times << std::endl;
    std::cout << "\t[bilateral filter]          ksize : " << params.BilateralFilter.ksize << std::endl;
    std::cout << "\t[adaptive bilateral filter] ksize : " << params.AdaptiveBilateralFilter.ksize << std::endl;
    std::cout << "\t[bilateral texture filter]  ksize : " << params.BilateralTextureFilter.ksize << std::endl;
    std::cout << "\t[bilateral texture filter]  nitr  : " << params.BilateralTextureFilter.nitr << std::endl;
    std::cout << "\t[superpixel SLIC] superpixel_size : " << params.SuperpixelSLIC.superpixel_size << std::endl;
    std::cout << "\t[superpixel SLIC] num_iteration   : " << params.SuperpixelSLIC.num_iteration << std::endl;
    std::cout << std::endl;

    bench_gradient(params.execute_times, input_image);
    bench_bilateral_filter(params.execute_times, input_image, params.BilateralFilter.ksize);
    bench_adaptive_bilateral_filter(params.execute_times, input_image, params.AdaptiveBilateralFilter.ksize);
    bench_bilateral_texture_filter(params.execute_times, input_image, params.BilateralTextureFilter.ksize, params.BilateralTextureFilter.nitr);
    bench_superpixel_slic(params.execute_times, input_image, params.SuperpixelSLIC.superpixel_size, params.SuperpixelSLIC.num_iteration);

    return 0;
}
