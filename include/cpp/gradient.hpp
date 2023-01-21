#ifndef GRADIENT_HPP
#define GRADIENT_HPP


#include <cstdint>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>

namespace {

namespace internal {

template <typename ImgType, int ImgChannels>
inline void gradient_impl(
    const cv::Mat& src,
    cv::Mat& dst
) {
    class ComputeGradient : public cv::ParallelLoopBody {
    public:
        ComputeGradient(
            const cv::Mat& src,
            cv::Mat& dst
        ) : src_(src), dst_(dst) {}

        void compute_gradient_row(
            const ImgType* const src_row_m1,
            const ImgType* const src_row,
            const ImgType* const src_row_p1,
            float* const dst_row
        ) const {
            // left col
            int x = 0;
            {
                auto sum = 0.f;
                for (int ch = 0; ch < ImgChannels; ch++) {
                    const auto horizontal_diff =
                        src_row[(x + 1) * ImgChannels + ch] - src_row[x * ImgChannels + ch];
                    const auto vertical_diff =
                        src_row_p1[x * ImgChannels + ch] - src_row_m1[x * ImgChannels + ch];
                    sum += horizontal_diff * horizontal_diff + vertical_diff * vertical_diff;
                }
                dst_row[x] = std::sqrt(static_cast<float>(sum));
            }
            // middle cols
            for (x = 1; x < src_.cols - 1; x++) {
                auto sum = 0.f;
                for (int ch = 0; ch < ImgChannels; ch++) {
                    const auto horizontal_diff =
                        src_row[(x + 1) * ImgChannels + ch] - src_row[(x - 1) * ImgChannels + ch];
                    const auto vertical_diff =
                        src_row_p1[x * ImgChannels + ch] - src_row_m1[x * ImgChannels + ch];
                    sum += horizontal_diff * horizontal_diff + vertical_diff * vertical_diff;
                }
                dst_row[x] = std::sqrt(static_cast<float>(sum));
            }
            // right col
            {
                auto sum = 0.f;
                for (int ch = 0; ch < ImgChannels; ch++) {
                    const auto horizontal_diff = src_row[x * ImgChannels + ch] - src_row[(x - 1) * ImgChannels + ch];
                    const auto vertical_diff = src_row_p1[x * ImgChannels + ch] - src_row_m1[x * ImgChannels + ch];
                    sum += horizontal_diff * horizontal_diff + vertical_diff * vertical_diff;
                }
                dst_row[x] = std::sqrt(static_cast<float>(sum));
            }
        }

        void operator()(const cv::Range& range) const CV_OVERRIDE {
            for (int r = range.start; r < range.end; r++) {
                const auto src_row    = src_.ptr<ImgType>() + src_.cols * ImgChannels * r;
                const auto src_row_m1 = (r > 0)             ? src_row - src_.cols * ImgChannels : src_row;
                const auto src_row_p1 = (r < src_.rows - 1) ? src_row + src_.cols * ImgChannels : src_row;
                auto dst_row = dst_.ptr<float>() + dst_.cols * r;
                compute_gradient_row(src_row_m1, src_row, src_row_p1, dst_row);
            }
        }

    private:
        const cv::Mat& src_;
        cv::Mat& dst_;
    };

    dst.create(src.size(), CV_32FC1);
    cv::parallel_for_(cv::Range(0, src.rows), ComputeGradient(src, dst));
}

} // namespace internal

inline void gradient(
    const cv::Mat& src,
    cv::Mat& dst
) {
    if (src.type() == CV_8UC1) {
        internal::gradient_impl<std::uint8_t, 1>(src, dst);
    }
    else if (src.type() == CV_8UC3) {
        internal::gradient_impl<std::uint8_t, 3>(src, dst);
    }
    else if (src.type() == CV_32FC1) {
        internal::gradient_impl<float, 1>(src, dst);
    }
    else if (src.type() == CV_32FC3) {
        internal::gradient_impl<float, 3>(src, dst);
    }
    else {
        std::cout << "Invalid src type." << std::endl;
    }
}

} // annonymous namespace

#endif // GRADIENT_HPP
