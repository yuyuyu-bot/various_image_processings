#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <sys/stat.h>

#include "toml.hpp"

#include "bilateral_filter.hpp"
#include "bilateral_texture_filter.hpp"

#include "cuda/device_image.hpp"
#include "cuda/bilateral_texture_filter.hpp"

void benchmark(const cv::Mat3b& input_image);

struct Args {
    std::string filename;
    std::string out_dir;
    bool        imshow;

    struct {
        bool   run;
        int    ksize;
        float  sigma_space;
        float  sigma_color;
    } bilateral_filter;

    struct {
        bool   run;
        int    ksize;
        float  sigma_space;
        float  sigma_color;
    } joint_bilateral_filter;

    struct {
        bool run;
        bool run_cuda;
        int  ksize;
        int  nitr;
    } bilateral_texture_filter;
};

auto parse_args(const std::string& toml_filename) {
    const auto toml_data = toml::parse(toml_filename);

    Args args;

    args.filename = toml::find<std::string>(toml_data, "filename");
    args.out_dir  = toml::find<std::string>(toml_data, "out_dir");
    args.imshow   = toml::find<bool>(toml_data, "imshow");

    // bilateral filter
    {
        const auto& data = toml::find(toml_data, "BilateralFilter");
        args.bilateral_filter.run         = toml::find<bool>(data, "run");
        args.bilateral_filter.ksize       = toml::find<int>(data, "ksize");
        args.bilateral_filter.sigma_space = toml::find<float>(data, "sigma_space");
        args.bilateral_filter.sigma_color = toml::find<float>(data, "sigma_color");
    }
    // joint bilateral filter
    {
        const auto& data = toml::find(toml_data, "JointBilateralFilter");
        args.joint_bilateral_filter.run         = toml::find<bool>(data, "run");
        args.joint_bilateral_filter.ksize       = toml::find<int>(data, "ksize");
        args.joint_bilateral_filter.sigma_space = toml::find<float>(data, "sigma_space");
        args.joint_bilateral_filter.sigma_color = toml::find<float>(data, "sigma_color");
    }
    // bilateral filter
    {
        const auto& data = toml::find(toml_data, "BilateralTextureFilter");
        args.bilateral_texture_filter.run         = toml::find<bool>(data, "run");
        args.bilateral_texture_filter.run_cuda    = toml::find<bool>(data, "run_cuda");
        args.bilateral_texture_filter.ksize       = toml::find<int>(data, "ksize");
        args.bilateral_texture_filter.nitr        = toml::find<int>(data, "nitr");
    }

    return args;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "[Usage] ./image_processing config_toml_file" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const auto args = parse_args(argv[1]);

    mkdir(args.out_dir.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH);

    cv::Mat input_image = cv::imread(args.filename, cv::IMREAD_COLOR);
    if (args.imshow) {
        cv::imshow("input", input_image);
    }

    // bilateral filter
    if (args.bilateral_filter.run) {
        std::cout << "bilateral filter" << std::endl;
        cv::Mat3b dst;
        bilateral_filter(
            input_image, dst, args.bilateral_filter.ksize, args.bilateral_filter.sigma_space,
            args.bilateral_filter.sigma_color);
        if (args.imshow) {
            cv::imshow("bilateral filter", dst);
        }
        cv::imwrite(args.out_dir + "/bilateral_filter.png", dst);
    }

    // joint bilateral filter
    if (args.joint_bilateral_filter.run) {
        std::cout << "joint bilateral filter" << std::endl;
        cv::Mat3b dst;
        joint_bilateral_filter(
            input_image, input_image, dst, args.joint_bilateral_filter.ksize, args.joint_bilateral_filter.sigma_space,
            args.joint_bilateral_filter.sigma_color);
        if (args.imshow) {
            cv::imshow("joint bilateral filter", dst);
        }
        cv::imwrite(args.out_dir + "/joint_bilateral_filter.png", dst);
    }

    // bilateral texture filter
    if (args.bilateral_texture_filter.run) {
        std::cout << "bilateral texture filter" << std::endl;
        cv::Mat3b dst;
        bilateral_texture_filter(
            input_image, dst, args.bilateral_texture_filter.ksize, args.bilateral_texture_filter.nitr);
        if (args.imshow) {
            cv::imshow("bilateral texture filter", dst);
        }
        cv::imwrite(args.out_dir + "/bilateral_texture_filter.png", dst);
    }

    // cuda bilateral texture filter
    if (args.bilateral_texture_filter.run_cuda) {
        std::cout << "cuda bilateral texture filter" << std::endl;

        cuda::DeviceImage<std::uint8_t> d_input_image(input_image.cols, input_image.rows, 3);
        cuda::DeviceImage<std::uint8_t> d_dst(input_image.cols, input_image.rows, 3);
        d_input_image.upload(input_image.ptr<std::uint8_t>());

        cuda::BilateralTextureFilter filter(input_image.cols, input_image.rows, args.bilateral_texture_filter.ksize,
                                            args.bilateral_texture_filter.nitr);
        filter.execute(d_input_image.get(), d_dst.get());

        cv::Mat3b dst(input_image.size());
        d_dst.download(dst.ptr<std::uint8_t>());

        if (args.imshow) {
            cv::imshow("cuda bilateral texture filter", dst);
        }
        cv::imwrite(args.out_dir + "/cuda_bilateral_texture_filter.png", dst);
    }

    while (args.imshow) {
        const auto entered_key = cv::waitKey(0);
        if (entered_key == 27) {
            break;
        }
    }

    benchmark(input_image);

    return 0;
}
