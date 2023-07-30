#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

#include "cpp/wexler_inpainting.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: ./wexler_inpainting [image] [mask]" << std::endl;
        exit(EXIT_FAILURE);
    }

    const cv::Mat3b image = cv::imread(argv[1]);
    const cv::Mat1b mask  = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    cv::Mat3b result;

    inpainting_wexler(image, mask, result);

    cv::imwrite("result.png", result);

    return 0;
}
