#ifndef SPLIT_AND_MERGE_HPP
#define SPLIT_AND_MERGE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>

class SplitAndMerge {
public:
    struct Parameters {
        float split_thresh;
        float merge_thresh;
        int minimum_label_size;

        Parameters(const float split_thresh = 0.99f, const float merge_thresh = 5.f, const int minimum_label_size = 64)
        : split_thresh(split_thresh), merge_thresh(merge_thresh), minimum_label_size(minimum_label_size) {
        }
    };

    SplitAndMerge(const cv::Mat3b& image, const Parameters& param)
    : rows_(image.rows), cols_(image.cols), param_(param), label_(image.rows, image.cols), num_label_(0) {
        label_.setTo(-1);
        cv::cvtColor(image, gray_, cv::COLOR_BGR2GRAY);
    }

    void apply() {
        split(0, 0, cols_ - 1, rows_ - 1);
        compute_block_info();
        merge();
        absorb();
    }

    void get_labels(cv::Mat& label_out) const {
        label_.copyTo(label_out);
    }

private:
    struct mergeResources {
        std::vector<float> intensity_average;
        std::vector<int> size;
        std::vector<cv::Point> begin;
        std::vector<cv::Point> end;

        mergeResources(const int label_num)
        : intensity_average(label_num, 0.f), size(label_num, 0), begin(label_num), end(label_num) {
        }
    };

    bool is_divisible(const int l, const int t, const int r, const int b) const {
        const auto smooth_histogram = [](std::array<int, RANGE_U8>& histogram, const int iteration) {
            for (int itr = 0; itr < iteration; itr++) {
                std::array<int, RANGE_U8> smoothened = {};
                for (int i = 0; i < RANGE_U8; i++) {
                    int n = 0;
                    for (int s = -2; s <= 2; s++) {
                        if (i + s < 0 || i + s >= RANGE_U8) {
                            continue;
                        }

                        smoothened[i] += histogram[i + s];
                        n++;
                    }
                    smoothened[i] /= n;
                }
                std::copy_n(smoothened.begin(), RANGE_U8, histogram.begin());
            }
        };

        std::array<int, RANGE_U8> histogram = {};
        for (int y = t; y <= b; y++) {
            for (int x = l; x <= r; x++) {
                histogram[gray_.ptr<uchar>(y)[x]]++;
            }
        }

        smooth_histogram(histogram, 3);

        // search peak index
        int max_idx = 0;
        for (int i = 0; i < RANGE_U8; i++) {
            if (histogram[max_idx] < histogram[i]) {
                max_idx = i;
            }
        }

        // search left valley
        int begin = 0;
        for (int i = max_idx - 1; i > 0; i--) {
            if (histogram[i - 1] > histogram[i]) {
                begin = i;
                break;
            }
        }

        // search right valley
        int end = 255;
        for (int i = max_idx + 1; i < 255; i++) {
            if (histogram[i] < histogram[i + 1]) {
                end = i;
                break;
            }
        }

        if (begin == 0 && end == 255) {
            return false;
        }

        auto all = 0ull;
        auto mount = 0ull;
        for (int i = 0; i < RANGE_U8; i++) {
            all += histogram[i];
            if (begin <= i && i <= end) {
                mount += histogram[i];
            }
        }

        return (static_cast<float>(mount) / all) < param_.split_thresh;
    }

    void split(const int l, const int t, const int r, const int b) {
        if (!is_divisible(l, t, r, b)) {
            if (label_(t, l) < 0) {
                // labeling the rectangle
                for (int y = t; y <= b; y++) {
                    for (int x = l; x <= r; x++) {
                        label_(y, x) = num_label_;
                    }
                }
                num_label_++;
            }
            return;
        }

        const int x = (l + r) / 2;
        const int y = (t + b) / 2;
        split(l, t, x, y);
        split(x + 1, t, r, y);
        split(l, y + 1, x, b);
        split(x + 1, y + 1, r, b);
    }

    void compute_block_info() {
        resources_ = std::unique_ptr<mergeResources>(new mergeResources(num_label_));

        for (int y = 0; y < rows_; y++) {
            for (int x = 0; x < cols_; x++) {
                const int l = label_.ptr<int>(y)[x];
                if (resources_->size[l] == 0) {
                    resources_->begin[l] = { x, y };
                }

                resources_->end[l] = { x, y };
                resources_->intensity_average[l] += gray_(y, x);
                resources_->size[l]++;
            }
        }

        for (int i = 0; i < num_label_; i++) {
            if (resources_->size[i] > 0) {
                resources_->intensity_average[i] /= static_cast<float>(resources_->size[i]);
            }
        }
    }

    bool is_combinable(const int a, const int b) const {
        return std::abs(resources_->intensity_average[a] - resources_->intensity_average[b]) < param_.merge_thresh;
    }

    void relabel(int a, int b) {
        int small;
        int large;
        if (resources_->size[a] < resources_->size[b]) {
            small = a;
            large = b;
        }
        else {
            large = a;
            small = b;
        }

        // overwrite the label
        for (int y = resources_->begin[small].y; y <= resources_->end[small].y; y++) {
            for (int x = 0; x < cols_; x++) {
                if (label_.ptr<int>(y)[x] == small) {
                    label_.ptr<int>(y)[x] = large;
                }
            }
        }

        // set new properties
        const auto new_intensity = resources_->intensity_average[small] * resources_->size[small] +
                                   resources_->intensity_average[large] * resources_->size[large];
        const auto new_size = resources_->size[small] + resources_->size[large];
        resources_->intensity_average[large] = new_intensity / new_size;
        resources_->size[large] = new_size;

        const auto small_begin_index = resources_->begin[small].y * rows_ + resources_->begin[small].x;
        const auto large_begin_index = resources_->begin[large].y * rows_ + resources_->begin[large].x;
        if (small_begin_index < large_begin_index) {
            resources_->begin[large] = resources_->begin[small];
        }

        const auto small_end_index = resources_->end[small].y * rows_ + resources_->end[small].x;
        const auto large_end_idnex = resources_->end[large].y * rows_ + resources_->end[large].x;
        if (small_end_index > large_end_idnex) {
            resources_->end[large] = resources_->end[small];
        }

        // invalidate small block
        resources_->intensity_average[small] = 0.f;
        resources_->size[small] = 0;
        resources_->begin[small] = resources_->end[small] = { 0, 0 };
    }

    void merge() {
        int merged_count;
        do {
            merged_count = 0;
            for (int y = 0; y < rows_; y++) {
                for (int x = 0; x < cols_; x++) {
                    int center = label_(y, x);

                    if (x + 1 < cols_) {
                        int right = label_(y, x + 1);
                        if (center != right && is_combinable(center, right)) {
                            relabel(center, right);
                            merged_count++;
                            // re-fetch center label
                            center = label_(y, x);
                        }
                    }


                    if (y + 1 < rows_) {
                        int down = label_(y + 1, x);
                        if (center != down && is_combinable(center, down)) {
                            relabel(center, down);
                            merged_count++;
                        }
                    }
                }
            }
        } while (merged_count > 0);
    }

    int trace_contour(int target) {
        cv::Point begin = resources_->begin[target];
        begin.x -= 1;
        cv::Point curr = begin;
        int from = 0, to, n = 0;
        int vec = 2;
        int answer = 0;
        float sub, min = 256.f;

        const auto x_isValid = [this](const int x) { return (x >= 0 && x < cols_); };
        const auto y_isValid = [this](const int y) { return (y >= 0 && y < rows_); };

        while (true) {
            if (n != 0 && curr == begin) {
                return answer;
            }

            if (n > rows_ * cols_) {
                // stuck in an infinite loop ?
                return -1;
            }

            switch (vec) {
            case 6:
                from = (x_isValid(curr.x + 1) && y_isValid(curr.y)) ? label_.ptr<int>(curr.y)[curr.x + 1] : 0;
                to = (x_isValid(curr.x + 1) && y_isValid(curr.y - 1)) ? label_.ptr<int>(curr.y - 1)[curr.x + 1] : 0;
                if (from != target && to == target) {
                    curr.x = curr.x + 1;
                    vec = 7;
                    break;
                }

            case 3:
                from = (x_isValid(curr.x + 1) && y_isValid(curr.y - 1)) ? label_.ptr<int>(curr.y - 1)[curr.x + 1] : 0;
                to = (x_isValid(curr.x) && y_isValid(curr.y - 1)) ? label_.ptr<int>(curr.y - 1)[curr.x] : 0;
                if (from != target && to == target) {
                    curr.x = curr.x + 1;
                    curr.y = curr.y - 1;
                    vec = 8;
                    break;
                }

            case 2:
                from = (x_isValid(curr.x) && y_isValid(curr.y - 1)) ? label_.ptr<int>(curr.y - 1)[curr.x] : 0;
                to = (x_isValid(curr.x - 1) && y_isValid(curr.y - 1)) ? label_.ptr<int>(curr.y - 1)[curr.x - 1] : 0;
                if (from != target && to == target) {
                    curr.y = curr.y - 1;
                    vec = 9;
                    break;
                }

            case 1:
                from = (x_isValid(curr.x - 1) && y_isValid(curr.y - 1)) ? label_.ptr<int>(curr.y - 1)[curr.x - 1] : 0;
                to = (x_isValid(curr.x - 1) && y_isValid(curr.y)) ? label_.ptr<int>(curr.y)[curr.x - 1] : 0;
                if (from != target && to == target) {
                    curr.x = curr.x - 1;
                    curr.y = curr.y - 1;
                    vec = 6;
                    break;
                }

            case 4:
                from = (x_isValid(curr.x - 1) && y_isValid(curr.y)) ? label_.ptr<int>(curr.y)[curr.x - 1] : 0;
                to = (x_isValid(curr.x - 1) && y_isValid(curr.y + 1)) ? label_.ptr<int>(curr.y + 1)[curr.x - 1] : 0;
                if (from != target && to == target) {
                    curr.x = curr.x - 1;
                    vec = 3;
                    break;
                }

            case 7:
                from = (x_isValid(curr.x - 1) && y_isValid(curr.y + 1)) ? label_.ptr<int>(curr.y + 1)[curr.x - 1] : 0;
                to = (x_isValid(curr.x) && y_isValid(curr.y + 1)) ? label_.ptr<int>(curr.y + 1)[curr.x] : 0;
                if (from != target && to == target) {
                    curr.x = curr.x - 1;
                    curr.y = curr.y + 1;
                    vec = 2;
                    break;
                }

            case 8:
                from = (x_isValid(curr.x) && y_isValid(curr.y + 1)) ? label_.ptr<int>(curr.y + 1)[curr.x] : 0;
                to = (x_isValid(curr.x + 1) && y_isValid(curr.y + 1)) ? label_.ptr<int>(curr.y + 1)[curr.x + 1] : 0;
                if (from != target && to == target) {
                    curr.y = curr.y + 1;
                    vec = 1;
                    break;
                }

            case 9:
                from = (x_isValid(curr.x + 1) && y_isValid(curr.y + 1)) ? label_.ptr<int>(curr.y + 1)[curr.x + 1] : 0;
                to = (x_isValid(curr.x + 1) && y_isValid(curr.y)) ? label_.ptr<int>(curr.y)[curr.x + 1] : 0;
                if (from != target && to == target) {
                    curr.x = curr.x + 1;
                    curr.y = curr.y + 1;
                    vec = 4;
                    break;
                }

                vec = 6;
                continue;
            }

            if (from != 0) {
                sub = std::abs(resources_->intensity_average[target] - resources_->intensity_average[from]);
                if (sub < min) {
                    min = sub;
                    answer = from;
                }
            }
            n++;
        }
    }

    void absorb() {
        if (param_.minimum_label_size <= 1) {
            // no need to absorb
            return;
        }

        int absorbed_count;
        do {
            absorbed_count = 0;
            for (int i = 0; i < num_label_; i++) {
                if (resources_->size[i] > 0 && resources_->size[i] < param_.minimum_label_size) {
                    const int neighbor = trace_contour(i);
                    if (neighbor >= 0) {
                        relabel(i, neighbor);
                        absorbed_count++;
                    }
                }
            }
        } while (absorbed_count > 0);
    }

private:
    static constexpr auto RANGE_U8 = 256;

    cv::Mat1b gray_;
    cv::Mat1i label_;

    Parameters param_;
    std::unique_ptr<mergeResources> resources_;

    int rows_, cols_;
    int num_label_;
};

#endif // SPLIT_AND_MERGE_HPP
