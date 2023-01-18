#include <iostream>
#include <opencv2/highgui.hpp>

#include "cuda/device_image.hpp"

#include "cpp/bilateral_filter.hpp"
#include "cuda/bilateral_filter.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "[Usage] ./bilateral_filter filename [ksize] [sigma_space] [sigma_color]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const auto filename    = std::string(argv[1]);
    const auto ksize       = argc >= 3 ? std::stoi(argv[2]) : 9;
    const auto sigma_space = argc >= 4 ? std::stof(argv[3]) : 10.f;
    const auto sigma_color = argc >= 5 ? std::stof(argv[4]) : 30.f;

    cv::Mat input_image = cv::imread(filename, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Failed to load " << filename << std::endl;
        return 0;
    }

    // cpp
    cv::Mat3b dst_cpp;
    bilateral_filter(input_image, dst_cpp, ksize, sigma_space, sigma_color);

    // cuda
    DeviceImage<std::uint8_t> d_input_image(input_image.cols, input_image.rows, 3);
    DeviceImage<std::uint8_t> d_dst(input_image.cols, input_image.rows, 3);
    CudaBilateralFilter filter(input_image.cols, input_image.rows, ksize, sigma_space, sigma_color);
    cv::Mat3b dst_cuda(input_image.size());

    d_input_image.upload(input_image.ptr<std::uint8_t>());
    filter.bilateral_filter(d_input_image.get(), d_dst.get());
    d_dst.download(dst_cuda.ptr<std::uint8_t>());

    // show result
    cv::imshow("input", input_image);
    cv::imshow("cpp", dst_cpp);
    cv::imshow("cuda", dst_cuda);

    while (true) {
        const auto entered_key = cv::waitKey(0);
        if (entered_key == 27) {
            break;
        }
    }

    cv::imwrite("bilateral_filter_cpp.png", dst_cpp);
    cv::imwrite("bilateral_filter_cuda.png", dst_cuda);

    return 0;
}
