#ifndef SLIC_HPP
#define SLIC_HPP

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>

float euclidean_distance(const int l1, const int a1, const int b1, const int l2, const int a2, const int b2) {
    const auto diff_l = (l1 - l2) * 2.55f;
    const auto diff_a = a1 - a2;
    const auto diff_b = b1 - b2;
    return diff_l * diff_l + diff_a * diff_a + diff_b * diff_b;
}

float CIE_DeltaE2000_square(const int l1, const int a1, const int b1, const int l2, const int a2, const int b2) {
    const auto degree_to_radian = [] (const float degree) {
        return degree * std::numbers::pi_v<float>;
    };

    constexpr float k_L = 1.f;
    constexpr float k_C = 1.f;
    constexpr float k_H = 1.f;
    constexpr float deg360InRad = degree_to_radian(360.f);
    constexpr float deg180InRad = degree_to_radian(180.f);
    constexpr float pow25To7 = 6103515625.f; // pow(25, 7)

    const auto C1 = std::sqrt((a1 * a1) + (b1 * b1));
    const auto C2 = std::sqrt((a2 * a2) + (b2 * b2));
    const auto barC = (C1 + C2) / 2.f;
    const auto G = 0.5f * (1 - std::sqrt(std::pow(barC, 7) / (std::pow(barC, 7) + pow25To7)));
    const auto a1Prime = (1.f + G) * a1;
    const auto a2Prime = (1.f + G) * a2;
    const auto CPrime1 = std::sqrt((a1Prime * a1Prime) + (b1 * b1));
    const auto CPrime2 = std::sqrt((a2Prime * a2Prime) + (b2 * b2));
    float hPrime1;
    if (b1 == 0 && a1Prime == 0) {
        hPrime1 = 0.f;
    }
    else {
        hPrime1 = std::atan2(b1, a1Prime);
        if (hPrime1 < 0) {
            hPrime1 += deg360InRad;
        }
    }
    float hPrime2;
    if (b2 == 0 && a2Prime == 0)
        hPrime2 = 0.f;
    else {
        hPrime2 = std::atan2(b2, a2Prime);
        if (hPrime2 < 0) {
            hPrime2 += deg360InRad;
        }
    }

    const auto deltaLPrime = l2 - l1;
    const auto deltaCPrime = CPrime2 - CPrime1;
    float deltahPrime;
    const auto CPrimeProduct = CPrime1 * CPrime2;
    if (CPrimeProduct == 0) {
        deltahPrime = 0;
    }
    else {
        deltahPrime = hPrime2 - hPrime1;
        if (deltahPrime < -deg180InRad) {
            deltahPrime += deg360InRad;
        }
        else if (deltahPrime > deg180InRad) {
            deltahPrime -= deg360InRad;
        }
    }
    const auto deltaHPrime = 2.f * std::sqrt(CPrimeProduct) * std::sin(deltahPrime / 2.f);

    const auto barLPrime = (l1 + l2) / 2.f;
    const auto barCPrime = (CPrime1 + CPrime2) / 2.f;
    const auto hPrimeSum = hPrime1 + hPrime2;
    float barhPrime;
    if (CPrime1 * CPrime2 == 0) {
        barhPrime = hPrimeSum;
    } else {
        if (std::abs(hPrime1 - hPrime2) <= deg180InRad)
            barhPrime = hPrimeSum / 2.0;
        else {
            if (hPrimeSum < deg360InRad) {
                barhPrime = (hPrimeSum + deg360InRad) / 2.f;
            }
            else {
                barhPrime = (hPrimeSum - deg360InRad) / 2.f;
            }
        }
    }
#define SQUARE(x) ((x) * (x))
    const auto T = 1.0 -
        (0.17f * std::cos(barhPrime - degree_to_radian(30.f))) +
        (0.24f * std::cos(2.f * barhPrime)) +
        (0.32f * std::cos((3.f * barhPrime) + degree_to_radian(6.f))) -
        (0.20f * std::cos((4.f * barhPrime) - degree_to_radian(63.f)));
    const auto deltaTheta = degree_to_radian(30.f) * std::exp(-std::pow((barhPrime - degree_to_radian(275.f)) / degree_to_radian(25.0), 2.0));
    const auto R_C = 2.f * std::sqrt(std::pow(barCPrime, 7) / (std::pow(barCPrime, 7) + pow25To7));
    const auto S_L = 1 + ((0.015f * SQUARE(barLPrime - 50.f)) / std::sqrt(20 + SQUARE(barLPrime - 50.f)));
    const auto S_C = 1 + (0.045f * barCPrime);
    const auto S_H = 1 + (0.015f * barCPrime * T);
    const auto R_T = (-std::sin(2.f * deltaTheta)) * R_C;

    const auto deltaE =
        SQUARE(deltaLPrime / (k_L * S_L)) +
        SQUARE(deltaCPrime / (k_C * S_C)) +
        SQUARE(deltaHPrime / (k_H * S_H)) +
        (R_T * (deltaCPrime / (k_C * S_C)) * (deltaHPrime / (k_H * S_H)));

    return deltaE;
}

class SuperpixelSLIC {
public:
    SuperpixelSLIC(
        const int width,
        const int height,
        const int superpixel_size = 30,
        const int num_iteration = 10,
        const float color_scale = 20.f)
    : height_(width),
      width_(height),
      superpixel_size_(superpixel_size),
      num_iteration_(num_iteration),
      color_scale_(color_scale)
    {
        const auto superpixel_per_col = (width_ + superpixel_size_ - 1) / superpixel_size_;
        const auto superpixel_per_row = (height_ + superpixel_size_ - 1) / superpixel_size_;
        num_superpixels_ = superpixel_per_col * superpixel_per_row;

        centers_.reset(new ClusterCenter[num_superpixels_]);
        new_centers_.reset(new ClusterCenter[num_superpixels_]);

        space_norm_ = 1.f / (superpixel_size_ * superpixel_size_);
        color_norm_ = 1.f / (color_scale_ * color_scale_);

        distance_function_ = euclidean_distance;
    }

    void apply(const cv::Mat3b& image) {
        init(image);
        for (int itr = 0; itr < num_iteration_; itr++) {
            if (iterate() == 0) {
                break;
            }
        }
        enforce_connectivity();
    }

    void get_label(cv::Mat& label_out) const {
        label_image_.copyTo(label_out);
    }

private:
    struct ClusterCenter {
        int x, y;
        int l, a, b;

        ClusterCenter(const int _x = 0, const int _y = 0,
                      const std::uint8_t _l = 0, const std::uint8_t _a = 0, const std::uint8_t _b = 0)
        : x(_x), y(_y), l(_l), a(_a), b(_b) {}
    };

    void init(const cv::Mat3b& image) {
        cv::cvtColor(image, lab_image_, cv::COLOR_BGR2Lab);
        label_image_.create(height_, width_);
        label_image_.setTo(-1);
        distance_image_.create(height_, width_);
        distance_image_.setTo(std::numeric_limits<float>::max());

        int superpixel_idx = 0;
        const int offset = superpixel_size_ / 2;
        for (int top = 0; top < height_; top += superpixel_size_) {
            for (int left = 0; left < width_; left += superpixel_size_) {
                const auto bottom = std::min(top + superpixel_size_ - 1, height_ - 1);
                const auto right = std::min(left + superpixel_size_ - 1, width_ - 1);
                const auto x = (left + right) / 2;
                const auto y = (top + bottom) / 2;
                const auto lab = lab_image_(y, x);
                centers_[superpixel_idx] = ClusterCenter(x, y, lab[0], lab[1], lab[2]);
                superpixel_idx++;
            }
        }

        // reset centers on lower gradient
        cv::Mat3f gradient_image_;
        cv::Laplacian(lab_image_, gradient_image_, CV_32F, 1);

        for (int i = 0; i < num_superpixels_; i++) {
            auto& center = centers_[i];
            const auto x = center.x;
            const auto y = center.y;
            auto min_grad_x = x;
            auto min_grad_y = y;
            auto center_grad = gradient_image_(y, x);
            auto min_grad = center_grad[0] + center_grad[1] + center_grad[2];

            const auto xs = std::max(x - 1, 0);
            const auto xe = std::min(x + 2, width_);
            const auto ys = std::max(y - 1, 0);
            const auto ye = std::min(y + 2, height_);

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

            const auto lab = lab_image_(min_grad_y, min_grad_x);
            center.x = x;
            center.y = y;
            center.l = lab[0];
            center.a = lab[1];
            center.b = lab[2];
        }
    }

    float get_distance(ClusterCenter& center, int x, int y) const {
        const cv::Vec3b p_color = lab_image_.ptr<cv::Vec3b>(y)[x];
        const auto diff_color = distance_function_(center.l, center.a, center.b, p_color[0], p_color[1], p_color[2]);
        const auto xdiff = center.x - x;
        const auto ydiff = center.y - y;

        return space_norm_ * (xdiff * xdiff + ydiff * ydiff) +
               color_norm_ * diff_color;
    }

    int association() {
        auto num_updated = 0;
        for (int center_idx = 0; center_idx != num_superpixels_; ++center_idx) {
            ClusterCenter center = centers_[center_idx];
            ClusterCenter new_center;
            auto count = 0;

            const auto xs = std::max(center.x - superpixel_size_, 0);
            const auto xe = std::min(center.x + superpixel_size_ + 1, width_);
            const auto ys = std::max(center.y - superpixel_size_, 0);
            const auto ye = std::min(center.y + superpixel_size_ + 1, height_);

            for (int y = ys; y != ye; ++y) {
                auto* const       distance_row = distance_image_.ptr<float>(y);
                auto* const       label_row    = label_image_.ptr<int>(y);
                const auto* const lab_row      = lab_image_.ptr<cv::Vec3b>(y);

                for (int x = xs; x != xe; ++x) {
                    const auto dist = get_distance(center, x, y);

                    if (distance_row[x] > dist) {
                        distance_row[x] = dist;
                        label_row[x] = center_idx;
                        num_updated++;
                    }

                    if (label_row[x] == center_idx) {
                        new_center.x += x;
                        new_center.y += y;
                        new_center.l += lab_row[x][0];
                        new_center.a += lab_row[x][1];
                        new_center.b += lab_row[x][2];
                        count++;
                    }
                }
            }

            new_centers_[center_idx].x = new_center.x / count;
            new_centers_[center_idx].y = new_center.y / count;
            new_centers_[center_idx].l = new_center.l / count;
            new_centers_[center_idx].a = new_center.a / count;
            new_centers_[center_idx].b = new_center.b / count;
        }

        return num_updated;
    }

    void updateCenters() {
        std::vector<int> min_dist(num_superpixels_, INT_MAX);

        for (int y = 0; y < height_; y++) {
            const auto* const label_row = label_image_.ptr<int>(y);
            const auto* const lab_row = lab_image_.ptr<cv::Vec3b>(y);

            for (int x = 0; x < width_; x++) {
                const auto label = label_row[x];
                const auto dist = distance_function_(
                    new_centers_[label].l, new_centers_[label].a, new_centers_[label].b,
                    lab_row[x][0], lab_row[x][1], lab_row[x][2]);

                if (min_dist[label] > dist) {
                    min_dist[label] = dist;
                    centers_[label].x = x;
                    centers_[label].y = y;
                    centers_[label].l = lab_row[x][0];
                    centers_[label].a = lab_row[x][1];
                    centers_[label].b = lab_row[x][2];
                }
            }
        }
    }

    int iterate() {

        const auto num_updated = association();
        updateCenters();

        return num_updated;
    }

    int labeling(cv::Mat1i& new_label, const int x, const int y, const int n) {
        new_label(y, x) = n;

        int count = 0;
        for (int i = 0; i < 4; i++) {
            const auto nx = x + nx4[i];
            const auto ny = y + ny4[i];
            if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) {
                continue;
            }

            if (new_label(ny, nx) < 0 && label_image_(y, x) == label_image_(ny, nx)) {
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
            if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) {
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
            if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) {
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
        cv::Mat1i new_label(height_, width_, -1);
        std::vector<int> sizes;

        // final labeling
        int number = 0;
        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                if (new_label(y, x) < 0) {
                    sizes.push_back(labeling(new_label, x, y, number));
                    number++;
                }
            }
        }

        // final centers
        const auto new_centers = std::make_unique<ClusterCenter[]>(number);
        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                const auto label = new_label(y, x);
                const auto lab = lab_image_(y, x);
                new_centers[label].x += x;
                new_centers[label].y += y;
                new_centers[label].l += lab[0];
                new_centers[label].a += lab[1];
                new_centers[label].b += lab[2];
            }
        }

        for (int i = 0; i < number; i++) {
            new_centers[i].x /= sizes[i];
            new_centers[i].y /= sizes[i];
            new_centers[i].l /= sizes[i];
            new_centers[i].a /= sizes[i];
            new_centers[i].b /= sizes[i];
        }

        const auto minimum_superpixel_area = (superpixel_size_ * superpixel_size_) / 20;

        cv::Mat_<bool> scanned(height_, width_);
        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
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
                        const auto diff = distance_function_(
                            new_centers[label_center].l, new_centers[label_center].a, new_centers[label_center].b,
                            new_centers[neighbor_labels[nidx]].l, new_centers[neighbor_labels[nidx]].a, new_centers[neighbor_labels[nidx]].b);
                        if (min_diff > diff) {
                            min_diff = diff;
                            min_idx = neighbor_labels[nidx];
                        }
                    }

                    relabeling(new_label, x, y, label_center, min_idx);
                }
            }
        }

        new_label.copyTo(label_image_);
    }

private:
    static constexpr int nx4[4] = { +1, +0, -1, +0 };
    static constexpr int ny4[4] = { +0, +1, +0, -1 };

    std::function<float(int, int, int, int, int, int)> distance_function_;

    cv::Mat3b lab_image_;
    cv::Mat1i label_image_;
    cv::Mat1f distance_image_;
    std::unique_ptr<ClusterCenter[]> centers_;
    std::unique_ptr<ClusterCenter[]> new_centers_;

    float color_norm_, space_norm_;
    int num_superpixels_;

    const int width_;
    const int height_;
    const int superpixel_size_;
    const int num_iteration_;
    const float color_scale_;
};

void superpixel_slic(
    const cv::Mat3b& image,
    cv::Mat1i& label,
    const int superpixel_size = 30,
    const int num_iteration = 10,
    const float color_scale = 20.f
) {
    SuperpixelSLIC slic(image.rows, image.cols, superpixel_size, num_iteration, color_scale);
    slic.apply(image);
    slic.get_label(label);
}

#endif // SLIC_HPP
