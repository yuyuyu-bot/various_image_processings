#include <iostream>
#include <opencv2/highgui.hpp>

#include "cuda/device_image.hpp"

#include "cpp/gradient.hpp"
#include "cuda/gradient.hpp"

static auto convert_to_u8(const cv::Mat1f& img) {
    double max;
    cv::minMaxLoc(img, nullptr, &max);
    cv::Mat img_scaled = img * 255 / max;

    cv::Mat img_u8;
    img_scaled.convertTo(img_u8, CV_8UC1);

    return img_u8;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "[Usage] ./gradient filename" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const auto filename = std::string(argv[1]);

    cv::Mat3b input_image = cv::imread(filename, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Failed to load " << filename << std::endl;
        return 0;
    }

    // cpp
    cv::Mat1f dst_cpp;
    gradient(input_image, dst_cpp);

    // cuda
    DeviceImage<std::uint8_t> d_src(input_image.cols, input_image.rows, 3);
    DeviceImage<float> d_dst(input_image.cols, input_image.rows);
    d_src.upload(input_image.ptr<std::uint8_t>());

    cuda_gradient(d_src.get(), d_dst.get(), input_image.cols, input_image.rows, 3);

    cv::Mat1f dst_cuda(input_image.size());
    d_dst.download(dst_cuda.ptr<float>());

    // show result
    cv::imshow("input", input_image);
    cv::imshow("cpp", convert_to_u8(dst_cpp));
    cv::imshow("cuda", convert_to_u8(dst_cuda));

    while (true) {
        const auto entered_key = cv::waitKey(0);
        if (entered_key == 27) {
            break;
        }
    }

    return 0;
}
