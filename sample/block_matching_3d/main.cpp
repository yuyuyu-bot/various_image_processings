#include <iostream>
#include <opencv2/highgui.hpp>

#include "cpp/block_matching_3d.hpp"

static auto convert_to_u8(const cv::Mat1f& img) {
    double max;
    cv::minMaxLoc(img, nullptr, &max);
    cv::Mat img_scaled = img * 255 / max;

    cv::Mat img_u8;
    img_scaled.convertTo(img_u8, CV_8UC1);

    return img_u8;
}

static auto convert_to_u8_offset(const cv::Mat1f& img) {
    double max;
    cv::minMaxLoc(img, nullptr, &max);
    cv::Mat img_scaled = img * 128 / max + 127;

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

    cv::Mat1b input_image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (input_image.empty()) {
        std::cerr << "Failed to load " << filename << std::endl;
        return 0;
    }

    cv::Mat1f low, hi0, hi1;
    wavelet_2D_biorthogonal_1p5(input_image, low, hi0, hi1);

    constexpr auto block_radius = 9;
    const auto similar_blocks = block_matching<std::uint8_t, 1>(input_image, block_radius, 127, 127);
    for (auto& block_info : similar_blocks) {
        std::cout << block_info.second.first << " "  << block_info.second.second << std::endl;
    }

    // show result
    cv::imshow("input", input_image);
    cv::imshow("low", convert_to_u8(low));
    cv::imshow("hi0", convert_to_u8_offset(hi0));
    cv::imshow("hi1", convert_to_u8_offset(hi1));

    while (true) {
        const auto entered_key = cv::waitKey(0);
        if (entered_key == 27) {
            break;
        }
    }

    return 0;
}
