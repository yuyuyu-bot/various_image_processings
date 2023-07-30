#ifndef WEXLER_INPAINTING_HPP
#define WEXLER_INPAINTING_HPP

#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>

class WexlerInpaintingImpl {
public:
    WexlerInpaintingImpl(const cv::Mat3b& src, const cv::Mat1b& mask) {
        assert(src.size() == mask.size());

        construct_pyramid(src, mask);
        const auto num_layers = src_pyramid_.size();

        bool do_initial_filling = true;
        for (int layer = num_layers - 1; layer >= 0; layer--) {
            std::cout << "Layer " << layer << "..." << std::endl;

            const auto&& weight = calculate_weight(mask_pyramid_[layer]);

            if (do_initial_filling) {
                std::cout << "\tInitial filling" << "... ";
                const auto energy = exemplar_based_inpainting(src_pyramid_[layer], mask_pyramid_[layer], weight, true);
                if (energy < 0) {
                    std::cout << "failed to inpaint layer " << layer << std::endl;
                }
                else {
                    std::cout << "done." << std::endl;;
                    do_initial_filling = false;
                }
            }

            int loop_count = 0;
            auto current_energy = std::numeric_limits<EnergyType>::max();

            // Energy minimization loop
            while (loop_count++ < max_loop) {
                std::cout << "\tLoop " << loop_count << "... ";
                auto src_tmp = src_pyramid_[layer].clone();
                const auto new_energy = exemplar_based_inpainting(src_tmp, mask_pyramid_[layer], weight, false);
                std::cout << "done. New energy: " << new_energy << std::endl;
                if (current_energy <= new_energy) {
                    break;
                }
                current_energy = new_energy;
                src_tmp.copyTo(src_pyramid_[layer], mask_pyramid_[layer]);
            }

            if (layer > 0) {
                // Copy filled to next layer
                cv::Mat3b up_sampled;
                cv::pyrUp(src_pyramid_[layer], up_sampled);
                up_sampled.copyTo(src_pyramid_[layer - 1], mask_pyramid_[layer - 1]);
            }
        }
    }

    void copy_result(cv::Mat3b& dst) const {
        src_pyramid_[0].copyTo(dst);
    }

private:
    using EnergyType = double;

    void construct_pyramid(const cv::Mat3b& src, const cv::Mat1b& mask) {
        src_pyramid_.push_back(src.clone());
        mask_pyramid_.push_back(mask.clone());

        int layer = 0;
        while (true) {
            const auto new_rows = src_pyramid_[layer].rows / 2;
            const auto new_cols = src_pyramid_[layer].cols / 2;

            if (std::min(new_rows, new_cols) < pyramid_bottom_size) {
                break;
            }

            cv::Mat3b src_new_layer;
            cv::pyrDown(src_pyramid_[layer], src_new_layer);
            src_pyramid_.push_back(std::move(src_new_layer));

            cv::Mat1b mask_new_layer;
            cv::pyrDown(mask_pyramid_[layer], mask_new_layer);
            mask_pyramid_.push_back(std::move(mask_new_layer));

            layer++;
        }
    }


    auto extract_mask_contour(const cv::Mat1b& mask, const int start_x, const int start_y) {
        std::vector<cv::Point2i> contour_pixels;

        // Freeman chain code
        constexpr int chain_code[8][2] = {{ 1, 0}, { 1, -1}, {0, -1}, {-1, -1},
                                          {-1, 0}, {-1,  1}, {0,  1}, { 1,  1}};
        constexpr int next_code[8] = {7, 7, 1, 1, 3, 3, 5, 5};
        int code_index = 5;

        int current_x = start_x;
        int current_y = start_y;
        int contour_length = 0;

        while (true) {
            if (current_x == start_x && current_y == start_y && contour_length > 0) {
                break;
            }

            if (contour_length > mask.rows * mask.cols) {
                std::cerr << "Error: contour did not converged." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            contour_pixels.emplace_back(current_x, current_y);

            // Search next contour
            int x = current_x + chain_code[code_index][0];
            int y = current_y + chain_code[code_index][1];

            int search_count = 0;
            while (x >= 0 && x < mask.cols && y >= 0 && y < mask.rows &&
                   mask(y, x) == 0 && search_count < 8) {
                code_index = (code_index + 1) % 8;
                x = current_x + chain_code[code_index][0];
                y = current_y + chain_code[code_index][1];
                search_count++;
            }

            if (search_count >= 8) {
                std::cerr << "Error: next contour not found." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // Next contour found
            current_x = x;
            current_y = y;
            code_index = next_code[code_index];
            contour_length++;
        }

        return contour_pixels;
    }

    cv::Mat1d calculate_weight(const cv::Mat1b& mask) {
        int start_x = -1;
        int start_y;
        for (int y = 0; y < mask.rows; y++) {
            for (int x = 0; x < mask.cols; x++) {
                if (mask(y, x) > 0) {
                    start_x = x;
                    start_y = y;
                    break;
                }
            }

            if (start_x >= 0) {
                break;
            }
        }

        if (start_x < 0) {
            // Target not found
            return mask;
        }

        const auto&& contour_pixels = extract_mask_contour(mask, start_x, start_y);

        cv::Mat1d weight(mask.size());
        constexpr auto constant_c = 1.2;

        for (int y = 0; y < mask.rows; y++) {
            for (int x = 0; x < mask.cols; x++) {
                if (mask(y, x) > 0) {
                    double minimum_distance = mask.rows * mask.cols;
                    for (const auto& contour_pixel : contour_pixels) {
                        const auto dist_sq = (x - contour_pixel.x) * (x - contour_pixel.x)
                                           + (y - contour_pixel.y) * (y - contour_pixel.y);
                        minimum_distance = std::min(minimum_distance, std::sqrt(dist_sq));
                    }
                    weight(y, x) = std::pow(constant_c, -minimum_distance);
                }
            }
        }

        return weight;
    }

    auto extract_mask_contour_with_priority(const cv::Mat1b& mask, const int start_x, const int start_y) {
        const auto compare = [](const auto& l, const auto& r) { return l.first < r.first; };
        using ElemType = std::pair<int, cv::Point2i>;
        std::priority_queue<ElemType, std::vector<ElemType>, decltype(compare)> target_pixels{compare};

        const auto&& contour_pixels = extract_mask_contour(mask, start_x, start_y);
        for (const auto& contour_pixel : contour_pixels) {
            // Count valid pixel around the pixel
            int valid_pixel_num = 0;
            for (int dy = -window_size_half; dy <= window_size_half; dy++) {
                for (int dx = -window_size_half; dx <= window_size_half; dx++) {
                    // Range check
                    if (contour_pixel.x + dx < 0 || contour_pixel.x + dx >= mask.cols ||
                        contour_pixel.y + dy < 0 || contour_pixel.y + dy >= mask.rows) {
                        continue;
                    }

                    if (mask(contour_pixel.y + dy, contour_pixel.x + dx) == 0) {
                        valid_pixel_num++;
                    }
                }
            }

            target_pixels.push(std::make_pair(valid_pixel_num, contour_pixel));
        }

        return target_pixels;
    }

    auto serach_exemplar(const cv::Mat3b& image, const cv::Mat1b& mask, const cv::Point2i& target_pixel, const bool is_initial_filling) {
        // Search exemplar
        int minimum_energy = std::numeric_limits<int>::max();
        cv::Point2i exemplar_position{-1, -1};

        for (int y = window_size_half; y < image.rows - window_size_half; y++) {
            for (int x = window_size_half; x < image.cols - window_size_half; x++) {
                int energy = 0;
                bool invalid = false;
                for (int dy = -window_size_half; dy <= window_size_half; dy++) {
                    for (int dx = -window_size_half; dx <= window_size_half; dx++) {
                        // Range check
                        if (target_pixel.x + dx < 0 || target_pixel.x + dx >= image.cols ||
                            target_pixel.y + dy < 0 || target_pixel.y + dy >= image.rows) {
                            continue;
                        }

                        // Reject if the candidate patch contains lack pixel
                        if (mask(y + dy, x + dx) > 0) {
                            invalid = true;
                            break;
                        }

                        // Skip lack pixel
                        if (is_initial_filling && mask(target_pixel.y + dy, target_pixel.x + dx) > 0) {
                            continue;
                        }

                        const auto pixel0 = image(y + dy, x + dx);
                        const auto pixel1 = image(target_pixel.y + dy, target_pixel.x + dx);
                        energy +=
                            (static_cast<int>(pixel0[0]) - static_cast<int>(pixel1[0])) * (static_cast<int>(pixel0[0]) - static_cast<int>(pixel1[0])) +
                            (static_cast<int>(pixel0[1]) - static_cast<int>(pixel1[1])) * (static_cast<int>(pixel0[1]) - static_cast<int>(pixel1[1])) +
                            (static_cast<int>(pixel0[2]) - static_cast<int>(pixel1[2])) * (static_cast<int>(pixel0[2]) - static_cast<int>(pixel1[2]));
                    }

                    if (invalid) {
                        break;
                    }
                }

                if (minimum_energy > energy && !invalid) {
                    minimum_energy = energy;
                    exemplar_position = cv::Point2i{x, y};
                }
            }
        }

        return std::make_pair(minimum_energy, exemplar_position);
    }

    EnergyType exemplar_based_inpainting(cv::Mat3b& image, const cv::Mat1b& mask, const cv::Mat1d& weight, const bool is_initial_filling) {
        auto remained = mask.clone();
        auto total_energy = EnergyType{0};

        while (true) {
            int start_x = -1;
            int start_y;
            for (int y = 0; y < remained.rows; y++) {
                for (int x = 0; x < remained.cols; x++) {
                    if (remained(y, x) > 0) {
                        start_x = x;
                        start_y = y;
                        break;
                    }
                }

                if (start_x >= 0) {
                    break;
                }
            }

            if (start_x < 0) {
                // Target not found
                break;
            }

            // Extract contour pixels
            auto&& target_pixels = extract_mask_contour_with_priority(remained, start_x, start_y);

            // Fill target pixels
            while (!target_pixels.empty()) {
                const auto target_pixel = target_pixels.top().second;
                target_pixels.pop();

                // Search exemplar
                const auto [minimum_energy, exemplar_position] = serach_exemplar(image, remained, target_pixel, is_initial_filling);

                if (minimum_energy == std::numeric_limits<int>::max()) {
                    // Exemplar not found
                    return EnergyType{-1};
                }

                total_energy += minimum_energy * weight(target_pixel);

                // Fill
                image(target_pixel) = image(exemplar_position);
                remained(target_pixel) = 0;
            }
        }

        return total_energy;
    }

    static constexpr auto pyramid_bottom_size = 32;
    static constexpr auto max_loop = 5;
    static constexpr auto window_size = 13;
    static constexpr auto window_size_half = window_size / 2;

    std::vector<cv::Mat3b> src_pyramid_;
    std::vector<cv::Mat1b> mask_pyramid_;
    std::vector<cv::Mat3b> dst_pyramid_;
};

namespace {

void inpainting_wexler(const cv::Mat3b& src, const cv::Mat1b& mask, cv::Mat3b& dst) {
    WexlerInpaintingImpl(src, mask).copy_result(dst);
}

} // anonymous namespace

#endif // WEXLER_INPAINTING_HPP
