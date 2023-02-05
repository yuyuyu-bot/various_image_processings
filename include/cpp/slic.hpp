#ifndef SLIC_HPP
#define SLIC_HPP

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>

class SLIC {
public:
    struct Parameters {
        int superpixel_size;
        int iterate;
        float color_scale;

        Parameters(int superpixel_size = 20, int iterate = 10, float color_scale = 30.0f)
            : superpixel_size(superpixel_size), iterate(iterate), color_scale(color_scale) {
            if (this->superpixel_size <= 0) {
                this->superpixel_size = 30;
            }
            if (this->iterate <= 0) {
                this->iterate = 10;
            }
            if (this->color_scale <= 0) {
                this->color_scale = 20.0f;
            }
        }
    };

    SLIC(const int width, const int height, const Parameters& param)
    : rows_(width), cols_(height), superpixel_size_(param.superpixel_size), num_iteration_(param.iterate), color_scale_(param.color_scale) {
        const auto superpixel_per_col = (cols_ + superpixel_size_ - 1) / superpixel_size_;
        const auto superpixel_per_row = (rows_ + superpixel_size_ - 1) / superpixel_size_;
        num_superpixels_ = superpixel_per_col * superpixel_per_row;

        centers_.reset(new ClusterCenter[num_superpixels_]);
        centers_tmp_.reset(new ClusterCenter[num_superpixels_]);

        space_norm_ = 1.f / (superpixel_size_ * superpixel_size_);
        color_norm_ = 1.f / (color_scale_ * color_scale_);
    }

    void apply(const cv::Mat3b& image) {
        init(image);

        int itr = 0;
        while (itr++ != num_iteration_ && iterate() != 0) {}
        enforce_connectivity();
    }

    void getLabels(cv::Mat& label_out) const {
        labels_.copyTo(label_out);
    }

private:
    struct ClusterCenter {
        int x, y;
        int l, a, b;
        ClusterCenter() : x(0), y(0), l(0), a(0), b(0) {};
        ClusterCenter(int _x, int _y, cv::Vec3b _lab) : x(_x), y(_y), l(_lab[0]), a(_lab[1]), b(_lab[2]) {}
        ClusterCenter(int _x, int _y, uchar _l, uchar _a, uchar _b) : x(_x), y(_y), l(_l), a(_a), b(_b) {}

        template <typename DevisorType>
        ClusterCenter operator / (DevisorType n) const {
            return ClusterCenter(this->x / n, this->y / n, this->l / n, this->a / n, this->b / n);
        }

        ClusterCenter& operator = (const ClusterCenter& obj) {
            this->x = obj.x;
            this->y = obj.y;
            this->l = obj.l;
            this->a = obj.a;
            this->b = obj.b;
            return *this;
        }

        ClusterCenter& operator += (const ClusterCenter& obj)
        {
            this->x += obj.x;
            this->y += obj.y;
            this->l += obj.l;
            this->a += obj.a;
            this->b += obj.b;
            return *this;
        }

        void set(int _x, int _y, cv::Vec3b _lab) {
            this->x = _x;
            this->y = _y;
            this->l = _lab[0];
            this->a = _lab[1];
            this->b = _lab[2];
        }
    };

    void init(const cv::Mat3b& image) {
        cv::cvtColor(image, lab_, cv::COLOR_BGR2Lab);
        labels_.create(rows_, cols_);
        labels_.setTo(-1);
        distance_.create(rows_, cols_);
        distance_.setTo(std::numeric_limits<float>::max());

        int superpixel_idx = 0;
        const int offset = superpixel_size_ / 2;
        for (int top = 0; top < rows_; top += superpixel_size_) {
            for (int left = 0; left < cols_; left += superpixel_size_) {
                const auto bottom = std::min(top + superpixel_size_ - 1, rows_ - 1);
                const auto right = std::min(left + superpixel_size_ - 1, cols_ - 1);
                const auto x = (left + right) / 2;
                const auto y = (top + bottom) / 2;
                centers_[superpixel_idx] = ClusterCenter(x, y, lab_(y, x));
                superpixel_idx++;
            }
        }

        // reset centers on lower gradient
        cv::Mat3f gradient_image_;
        cv::Laplacian(lab_, gradient_image_, CV_32F, 1);

        for (int i = 0; i < num_superpixels_; i++) {
            auto& center = centers_[i];
            const auto x = center.x;
            const auto y = center.y;
            auto min_grad_x = x;
            auto min_grad_y = y;
            auto center_grad = gradient_image_(y, x);
            auto min_grad = center_grad[0] + center_grad[1] + center_grad[2];

            const auto xs = std::max(x - 1, 0);
            const auto xe = std::min(x + 2, cols_);
            const auto ys = std::max(y - 1, 0);
            const auto ye = std::min(y + 2, rows_);

            for (int yj = ys; yj != ye; ++yj) {
                const auto* const grad_row = gradient_image_.ptr<cv::Vec3f>(yj);
                for (int xi = xs; xi != xe; ++xi) {
                    const auto grad = grad_row[xi][0] + grad_row[xi][1] + grad_row[xi][2];

                    if (min_grad > grad) {
                        min_grad = grad;
                        min_grad_x = xi;
                        min_grad_y = yj;
                    }
                }
            }

            center.set(min_grad_x, min_grad_y, lab_.at<cv::Vec3b>(min_grad_y, min_grad_x));
        }
    }

    float getDistance(ClusterCenter& center, int x, int y) const {
        const cv::Vec3b p_color = lab_.ptr<cv::Vec3b>(y)[x];
        const auto ldiff = center.l - p_color[0];
        const auto adiff = center.a - p_color[1];
        const auto bdiff = center.b - p_color[2];
        const auto xdiff = center.x - x;
        const auto ydiff = center.y - y;

        return space_norm_ * (xdiff * xdiff + ydiff * ydiff) +
               color_norm_ * (ldiff * ldiff + adiff * adiff + bdiff * bdiff) ;
    }

    void updateCenters() {
        std::vector<int> min_dist(num_superpixels_, INT_MAX);

        for (int y = 0; y < rows_; y++) {
            const auto* const label_row = labels_.ptr<int>(y);
            const auto* const lab_row = lab_.ptr<cv::Vec3b>(y);

            for (int x = 0; x < cols_; x++) {
                const auto label = label_row[x];
                const auto diff_l = centers_tmp_[label].l - lab_row[x][0];
                const auto diff_a = centers_tmp_[label].a - lab_row[x][1];
                const auto diff_b = centers_tmp_[label].b - lab_row[x][2];
                const auto dist = diff_l * diff_l + diff_a * diff_a + diff_b * diff_b;

                if (min_dist[label] > dist) {
                    min_dist[label] = dist;
                    centers_[label].set(x, y, lab_row[x]);
                }
            }
        }
    }

    int iterate() {
        int num_updated = 0;
        for (int center_num = 0; center_num != num_superpixels_; ++center_num) {
            ClusterCenter center = centers_[center_num];
            ClusterCenter new_center;
            int count = 0;

            const int xs = std::max(center.x - superpixel_size_, 0);
            const int xe = std::min(center.x + superpixel_size_ + 1, cols_);
            const int ys = std::max(center.y - superpixel_size_, 0);
            const int ye = std::min(center.y + superpixel_size_ + 1, rows_);

            for (int y = ys; y != ye; ++y) {
                float* _dist = distance_.ptr<float>(y);
                int* label_row = labels_.ptr<int>(y);
                const cv::Vec3b* lab_row = lab_.ptr<cv::Vec3b>(y);

                for (int x = xs; x != xe; ++x) {
                    const float dist = getDistance(center, x, y);

                    if (_dist[x] > dist) {
                        _dist[x] = dist;
                        label_row[x] = center_num;
                        num_updated++;
                    }

                    if (label_row[x] == center_num) {
                        new_center += ClusterCenter(x, y, lab_row[x]);
                        count++;
                    }
                }
            }

            centers_tmp_[center_num] = new_center / count;
        }

        updateCenters();

        return num_updated;
    }

    int labeling(cv::Mat1i& new_label, const int x, const int y, const int n) {
        new_label(y, x) = n;

        int count = 0;
        for (int i = 0; i < 4; i++) {
            const auto nx = x + nx4[i];
            const auto ny = y + ny4[i];
            if (nx < 0 || nx >= cols_ || ny < 0 || ny >= rows_) {
                continue;
            }

            if (new_label(ny, nx) < 0 && labels_(y, x) == labels_(ny, nx)) {
                count += labeling(new_label, nx, ny, n);
            }
        }

        return count + 1;
    }

    void relabeling(cv::Mat1i& new_label, const int x, const int y, const int old_n, const int new_n) {
        new_label(y, x) = new_n;

        for (int i = 0; i < 4; i++) {
            const auto nx = x + nx4[i];
            const auto ny = y + ny4[i];
            if (nx < 0 || nx >= cols_ || ny < 0 || ny >= rows_) {
                continue;
            }

            if (new_label(ny, nx) == old_n) {
                relabeling(new_label, nx, ny, old_n, new_n);
            }
        }
    }

    void get_neighbor_labels(
        const cv::Mat1i& label, cv::Mat_<bool>& scanned, const int x, const int y, const int n,
        std::vector<int>& neighbor_labels
    ) {
        scanned(y, x) = true;

        for (int i = 0; i < 4; i++) {
            const auto nx = x + nx4[i];
            const auto ny = y + ny4[i];
            if (nx < 0 || nx >= cols_ || ny < 0 || ny >= rows_) {
                continue;
            }

            const auto next_label = label(ny, nx);
            if (scanned(ny, nx)) {
                continue;
            }
            else if (next_label == n) {
                get_neighbor_labels(label, scanned, nx, ny, n, neighbor_labels);
            }
            else {
                auto exist = false;
                for (int nidx = 0; nidx < neighbor_labels.size(); nidx++) {
                    if (neighbor_labels[nidx] == next_label) {
                        exist = true;
                    }
                }

                if (!exist) {
                    neighbor_labels.push_back(next_label);
                }
            }
        }
    }

    void enforce_connectivity() {
        cv::Mat1i new_label(rows_, cols_, -1);
        std::vector<int> sizes;

        // final labeling
        int number = 0;
        for (int y = 0; y < rows_; y++) {
            for (int x = 0; x < cols_; x++) {
                if (new_label(y, x) < 0) {
                    sizes.push_back(labeling(new_label, x, y, number));
                    number++;
                }
            }
        }

        // final centers
        const auto new_centers = std::make_unique<ClusterCenter[]>(number);
        for (int y = 0; y < rows_; y++) {
            for (int x = 0; x < cols_; x++) {
                const auto label = new_label(y, x);
                const auto lab = lab_(y, x);
                new_centers[label].x += x;
                new_centers[label].y += y;
                new_centers[label].l += lab[0];
                new_centers[label].a += lab[1];
                new_centers[label].b += lab[2];
            }
        }

        for (int i = 0; i < number; i++) {
            new_centers[i].x /= number;
            new_centers[i].y /= number;
            new_centers[i].l /= number;
            new_centers[i].a /= number;
            new_centers[i].b /= number;
        }

        const auto minimum_superpixel_area = (superpixel_size_ * superpixel_size_) / 20;

        cv::Mat_<bool> scanned(rows_, cols_);
        for (int y = 0; y < rows_; y++) {
            for (int x = 0; x < cols_; x++) {
                const auto label_center = new_label(y, x);

                if (sizes[label_center] < minimum_superpixel_area) {
                    scanned.setTo(false);
                    std::vector<int> neighbor_labels;
                    get_neighbor_labels(new_label, scanned, x, y, label_center, neighbor_labels);

                    if (neighbor_labels.size() == 0) {
                        std::cerr << "Failed to extract neighbors." << std::endl;
                        continue;
                    }

                    auto min_diff = std::numeric_limits<int>::max();
                    auto min_idx = label_center;
                    for (int nidx = 0; nidx < neighbor_labels.size(); nidx++) {
                        const auto diff_l = new_centers[label_center].l - new_centers[neighbor_labels[nidx]].l;
                        const auto diff_a = new_centers[label_center].a - new_centers[neighbor_labels[nidx]].a;
                        const auto diff_b = new_centers[label_center].b - new_centers[neighbor_labels[nidx]].b;
                        const auto diff = diff_l * diff_l + diff_a * diff_a + diff_b * diff_b;
                        if (min_diff > diff) {
                            min_diff = diff;
                            min_idx = neighbor_labels[nidx];
                        }
                    }

                    relabeling(new_label, x, y, label_center, min_idx);
                }
            }
        }

        new_label.copyTo(labels_);
    }

private:
    cv::Mat3b lab_;
    cv::Mat1i labels_;
    cv::Mat1f distance_;
    std::unique_ptr<ClusterCenter[]> centers_;
    std::unique_ptr<ClusterCenter[]> centers_tmp_;

    float color_norm_, space_norm_;
    int num_superpixels_;
    int rows_, cols_;

    // parameters
    const int superpixel_size_;
    const int num_iteration_;
    const float color_scale_;

    static constexpr int nx4[4] = { +1, +0, -1, +0 };
    static constexpr int ny4[4] = { +0, +1, +0, -1 };
};

#endif // SLIC_HPP
