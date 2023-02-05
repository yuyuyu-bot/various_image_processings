#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

#include "cpp/slic.hpp"

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

static cv::Mat3b draw_superpixel(const cv::Mat3b& image, const cv::Mat1i& labels) {
    const auto rows = image.rows;
    const auto cols = image.cols;
    cv::Mat3b dst(rows, cols);

    double max_label;
    cv::minMaxLoc(labels, nullptr, &max_label);

    const auto colors = std::make_unique<cv::Vec3i[]>(static_cast<int>(max_label) + 1);
    const auto sizes = std::make_unique<int[]>(static_cast<int>(max_label) + 1);

    for (int y = 0; y != rows; ++y) {
        const int* label_row = &labels(y, 0);
        const cv::Vec3b* image_row = &image(y, 0);

        for (int x = 0; x != cols; ++x) {
            colors[label_row[x]] += image_row[x];
            sizes[label_row[x]]++;
        }
    }

    for (std::size_t i = 0; i < static_cast<int>(max_label) + 1; i++) {
        if (sizes[i] != 0) {
            colors[i] /= sizes[i];
        }
    }

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            dst(y, x) = colors[labels(y, x)];
        }
    }

    return dst;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./slic.exe [image]" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat image = cv::imread(argv[1]);

    const int superpixel_size = 30;
    const int iterate = 10;
    const float m = 20.f;

    // set SLIC parameters
    SLIC::Parameters param;
    param.superpixel_size = superpixel_size;
    param.iterate = iterate;
    param.color_scale = m;
    // create SLIC
    SLIC slic(image.rows, image.cols, param);
    // perform SLIC
    slic.apply(image);

    cv::Mat labels;
    slic.getLabels(labels);

    cv::imshow("Superpixel image", draw_superpixel(image, labels));
    image.setTo(cv::Scalar(0, 0, 255), draw_contour(labels));
    cv::imshow("Superpixel contour", image);
    cv::waitKey(0);

    return 0;
}
