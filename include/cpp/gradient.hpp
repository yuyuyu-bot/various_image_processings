#ifndef GRADIENT_HPP
#define GRADIENT_HPP


#include <cstdint>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>

namespace {

template <typename ImgType, int ImgChannels>
inline void gradient(
    const cv::Mat_<cv::Vec<ImgType, ImgChannels>>& src,
    cv::Mat1f& dst
) {
    class ComputeGradient : public cv::ParallelLoopBody {
    public:
        ComputeGradient(
            const cv::Mat_<cv::Vec<ImgType, ImgChannels>>& src,
            cv::Mat1f& dst
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
                    const auto horizontal_diff =
                        src_row[x * ImgChannels + ch] - src_row[(x - 1) * ImgChannels + ch];
                    const auto vertical_diff =
                        src_row_p1[x * ImgChannels + ch] - src_row_m1[x * ImgChannels + ch];
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
        const cv::Mat3b& src_;
        cv::Mat1f& dst_;
    };


    dst.create(src.size());
    cv::parallel_for_(cv::Range(0, src.rows), ComputeGradient(src, dst));
}

} // annonymous namespace


#endif // GRADIENT_HPP
