#ifndef BORDER_REPLICATED_INTEGRAL_IMAGE_HPP
#define BORDER_REPLICATED_INTEGRAL_IMAGE_HPP

#include <opencv2/core.hpp>

template <typename SrcType, int Channels>
class BorderReplicatedIntegralImage {
public:
    using SrcVecType = cv::Vec<SrcType, Channels>;

    BorderReplicatedIntegralImage(
        const cv::Mat_<SrcVecType>& src,
        const int radius)
    : width_(src.cols + radius * 2 + 1),
      height_(src.rows + radius * 2 + 1),
      radius_(radius) {
        const int dim[3] = { height_, width_, Channels};
        if constexpr (std::is_floating_point_v<SrcType>) {
            buffer_.create(src.dims, dim, CV_32FC(Channels));
        }
        else {
            buffer_.create(src.dims, dim, CV_32SC(Channels));
        }
        buffer_.setTo(cv::Scalar::all(0));

        if constexpr (std::is_floating_point_v<SrcType>) {
            compute<cv::Vec<float, Channels>>(src);
        }
        else {
            compute<cv::Vec<int, Channels>>(src);
        }
    }

    auto get(const int x0, const int y0, const int x1, const int y1) const {
        if constexpr (std::is_floating_point_v<SrcType>) {
            return buffer_.at<cv::Vec<float, Channels>>(y1 + radius_ + 1, x1 + radius_ + 1) -
                   buffer_.at<cv::Vec<float, Channels>>(y1 + radius_ + 1, x0 + radius_    ) -
                   buffer_.at<cv::Vec<float, Channels>>(y0 + radius_    , x1 + radius_ + 1) +
                   buffer_.at<cv::Vec<float, Channels>>(y0 + radius_    , x0 + radius_    );
        }
        else {
            return buffer_.at<cv::Vec<int, Channels>>(y1 + radius_ + 1, x1 + radius_ + 1) -
                   buffer_.at<cv::Vec<int, Channels>>(y1 + radius_ + 1, x0 + radius_    ) -
                   buffer_.at<cv::Vec<int, Channels>>(y0 + radius_    , x1 + radius_ + 1) +
                   buffer_.at<cv::Vec<int, Channels>>(y0 + radius_    , x0 + radius_    );
        }
    }

private:
    template <typename AccmVecType>
    void compute(const cv::Mat_<SrcVecType>& src) {
        for (int y = 1; y < height_; y++) {
            for (int x = 1; x < width_; x++) {
                const auto src_x = std::clamp(x - 1 - radius_, 0, src.cols - 1);
                const auto src_y = std::clamp(y - 1 - radius_, 0, src.rows - 1);

                for (int ch = 0; ch < Channels; ch++) {
                    buffer_.at<AccmVecType>(y, x)[ch] = src(src_y, src_x)[ch];
                }
            }
        }

        for (int y = 1; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                for (int ch = 0; ch < Channels; ch++) {
                    buffer_.at<AccmVecType>(y, x)[ch] += buffer_.at<AccmVecType>(y - 1, x)[ch];
                }
            }
        }

        for (int y = 0; y < height_; y++) {
            for (int x = 1; x < width_; x++) {
                for (int ch = 0; ch < Channels; ch++) {
                    buffer_.at<AccmVecType>(y, x)[ch] += buffer_.at<AccmVecType>(y, x - 1)[ch];
                }
            }
        }
    }

private:
    const int width_;
    const int height_;
    const int radius_;
    cv::Mat buffer_;
};

#endif // BORDER_REPLICATED_INTEGRAL_IMAGE_HPP
