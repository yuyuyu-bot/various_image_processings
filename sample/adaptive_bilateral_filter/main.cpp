#include <iostream>
#include <opencv2/highgui.hpp>

#include "cpp/adaptive_bilateral_filter.hpp"

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

    cv::Mat3b dst_cpp;
    adaptive_bilateral_filter(input_image, dst_cpp);

    // show result
    cv::imshow("input", input_image);
    cv::imshow("cpp", dst_cpp);

    while (true) {
        const auto entered_key = cv::waitKey(0);
        if (entered_key == 27) {
            break;
        }
    }

    return 0;
}
