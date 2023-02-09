#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

#include "cpp/split_and_merge.hpp"

static cv::Mat1b draw_contour(const cv::Mat1i& labels) {
    const auto rows = labels.rows;
    const auto cols = labels.cols;
    cv::Mat1b dst(rows, cols);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            const auto label_center = labels(y, x);
            const auto label_right  = x + 1 < cols ? labels(y, x + 1) : -1;
            const auto label_down   = y + 1 < rows ? labels(y + 1, x) : -1;

            if (label_center != label_right || label_center != label_down) {
                dst(y, x) = 255u;
            }
            else {
                dst(y, x) = 0u;
            }
        }
    }

    return dst;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./split_and_merge [image]" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat image = cv::imread(argv[1]);
    cv::Mat1i label;

    SplitAndMerge splitAndMerge(image, SplitAndMerge::Parameters());
    splitAndMerge.apply();
    splitAndMerge.get_labels(label);

    image.setTo(cv::Scalar(0, 0, 255), draw_contour(label));
    cv::imshow("superpixel contour", image);
    cv::waitKey(0);

    return 0;
}
