#ifndef DEBUG_SHOW_HPP
#define DEBUG_SHOW_HPP

#include <opencv2/highgui.hpp>
#include <string>

void debug_show_float_image(const std::string& name, const cv::Mat1f& image) {
    double min;
    double max;
    cv::minMaxLoc(image, &min, &max);
    const cv::Mat1f image_scaled = 255 * (image - min) / (max - min);
    cv::Mat1b image_u8;
    image_scaled.convertTo(image_u8, CV_8U);
    cv::imshow(name, image_u8);
    cv::waitKey(1000);
}

void debug_show_float_image(const std::string& name, const cv::Mat3f& image) {
    cv::Mat3b image_u8;
    image.convertTo(image_u8, CV_8UC3);
    cv::imshow(name, image_u8);
    cv::waitKey(1000);
}

#endif // DEBUG_SHOW_HPP
