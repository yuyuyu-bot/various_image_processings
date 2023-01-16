#include <iostream>
#include <opencv2/highgui.hpp>

#include "cuda/device_image.hpp"

#include "bilateral_texture_filter.hpp"
#include "cuda/bilateral_texture_filter.hpp"

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "[Usage] ./bilateral_texture_filter filename [ksize] [nitr]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const auto filename = std::string(argv[1]);
    const auto ksize    = argc >= 3 ? std::stoi(argv[2]) : 9;
    const auto nitr     = argc >= 4 ? std::stof(argv[3]) : 3;

    cv::Mat input_image = cv::imread(filename, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Failed to load " << filename << std::endl;
        return 0;
    }

    // cpp
    cv::Mat3b dst_cpp;
    bilateral_texture_filter(input_image, dst_cpp, ksize, nitr);

    // cuda
    cuda::DeviceImage<std::uint8_t> d_input_image(input_image.cols, input_image.rows, 3);
    cuda::DeviceImage<std::uint8_t> d_dst(input_image.cols, input_image.rows, 3);
    cuda::BilateralTextureFilter filter(input_image.cols, input_image.rows, ksize, nitr);
    cv::Mat3b dst_cuda(input_image.size());

    d_input_image.upload(input_image.ptr<std::uint8_t>());
    filter.execute(d_input_image.get(), d_dst.get());
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

    cv::imwrite("bilateral_texture_filter_cpp.png", dst_cpp);
    cv::imwrite("bilateral_texture_filter_cuda.png", dst_cuda);

    return 0;
}
