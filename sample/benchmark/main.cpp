#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "toml.hpp"

#include "cuda/device_image.hpp"
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

struct Parameters {
    int execute_times = 100;

    struct {
        int ksize = 9;
        int nitr = 3;
    } BilateralTextureFilter;
};

static auto parse_config(const std::string& filename) {
    const auto toml_all = toml::parse(filename);

    Parameters params;
    toml::value data;

    // common
    params.execute_times = toml::find<int>(toml_all, "execute_times");

    // bilateral texture filter
    data = toml::find(toml_all, "BilateralTextureFilter");
    params.BilateralTextureFilter.ksize = toml::find<int>(data, "ksize");
    params.BilateralTextureFilter.nitr = toml::find<int>(data, "nitr");

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

static void print_duration(const std::string& name, const float duration) {
    std::printf("%-30s : %10.3f [msec]\n", name.c_str(), duration);
}

static void bench_bilateral_texture_filter(const int measurement_times, const cv::Mat& input_image,
                                           const int ksize, const int nitr) {
    const cv::Mat3b input_image_color = convert_to_3ch(input_image);
    cv::Mat3b dst(input_image_color.size());

    const auto width  = input_image.cols;
    const auto height = input_image.rows;

    DeviceImage<std::uint8_t> d_input_image(width, height, 3);
    DeviceImage<std::uint8_t> d_dst(width, height, 3);
    CudaBilateralTextureFilter filter(width, height, ksize, nitr);

    d_input_image.upload(input_image_color.ptr<std::uint8_t>());

    float duration = 0.f;
    MEASURE(measurement_times, filter.execute(d_input_image.get(), d_dst.get()), duration);
    print_duration("bilateral texture filter", duration);

    d_dst.download(dst.ptr<std::uint8_t>());
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "[Usage] ./benchmark image_path [config_path]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const auto filename = std::string(argv[1]);
    const auto params = argc == 3 ? parse_config(argv[2]) : Parameters();

    cv::Mat input_image = cv::imread(filename, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Failed to load " << filename << std::endl;
        return 0;
    }

    std::cout << "Parameters" << std::endl;
    std::cout << "\twidth         : " << input_image.cols << std::endl;
    std::cout << "\theight        : " << input_image.rows << std::endl;
    std::cout << "\texecute times : " << params.execute_times << std::endl;
    std::cout << "\t[bilateral texture filter] ksize : " << params.BilateralTextureFilter.ksize << std::endl;
    std::cout << "\t[bilateral texture filter] nitr  : " << params.BilateralTextureFilter.nitr << std::endl;
    std::cout << std::endl;

    bench_bilateral_texture_filter(
        params.execute_times, input_image, params.BilateralTextureFilter.ksize, params.BilateralTextureFilter.nitr);

    return 0;
}
