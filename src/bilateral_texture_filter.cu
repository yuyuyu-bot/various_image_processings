#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda/bilateral_texture_filter.hpp"

#define CUDASafeCall() cuda_safe_call(cudaGetLastError(), __FILE__, __LINE__);

static constexpr auto epsilon = 1e-9;

inline void cuda_safe_call(const cudaError& error, const char* const file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error %s : %d %s\n", file, line, cudaGetErrorString(error));
    }
}

template <typename T>
inline __device__ T clamp(T v, T min, T max) {
    return v < min ? min :
           v > max ? max :
           v;
}

template <typename ImageType, typename MagnitudeType>
__global__ void compute_magnitude_kernel(const ImageType* const image, MagnitudeType* const magnitude,
                                         const int width, const int height) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride_3ch = width * 3;
    const int stride = width;
    const int x = idx % width;
    const int y = idx / width;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        magnitude[stride * y + x] = 0.f;
        return;
    }

    const auto compute_del = [image, stride_3ch](const int x0, const int y0, const int x1, const int y1) {
        const auto diff0 = image[stride_3ch * y0 + x0 * 3 + 0] - image[stride_3ch * y1 + x1 * 3 + 0];
        const auto diff1 = image[stride_3ch * y0 + x0 * 3 + 1] - image[stride_3ch * y1 + x1 * 3 + 1];
        const auto diff2 = image[stride_3ch * y0 + x0 * 3 + 2] - image[stride_3ch * y1 + x1 * 3 + 2];
        return diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
    };

    const auto del_x = compute_del(x - 1, y, x + 1, y);
    const auto del_y = compute_del(x, y - 1, x, y + 1);
    magnitude[stride * y + x] = sqrtf(del_x + del_y);
}

template <typename ImageType, typename MagnitudeType, typename BlurredType, typename RTVType>
__global__ void compute_blur_and_rtv_kernel(const ImageType* const image, const MagnitudeType* const magnitude,
                                            BlurredType* const blurred, RTVType* const rtv, const int ksize,
                                            const int width, const int height) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride_3ch = width * 3;
    const int stride = width;
    const int x = idx % width;
    const int y = idx / width;
    const int radius  = ksize / 2;

    const auto get_intensity = [image, stride_3ch](const int x0, const int y0) {
        const auto pix = image + stride_3ch * y0 + x0 * 3;
        return (pix[0] + pix[1] + pix[2]) / 3.f;
    };

    auto sum0 = 0;
    auto sum1 = 0;
    auto sum2 = 0;

    auto intensity_max = 0.f;
    auto intensity_min = 0.f;
    auto magnitude_max = 0.f;
    auto magnitude_sum = 0.f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto x_clamped = clamp(x + kx, 0, width - 1);
            const auto y_clamped = clamp(y + ky, 0, height - 1);

            sum0 += image[stride_3ch * y_clamped + x_clamped * 3 + 0];
            sum1 += image[stride_3ch * y_clamped + x_clamped * 3 + 1];
            sum2 += image[stride_3ch * y_clamped + x_clamped * 3 + 2];

            intensity_max  = max(intensity_max, get_intensity(x_clamped, y_clamped));
            intensity_min  = min(intensity_min, get_intensity(x_clamped, y_clamped));
            magnitude_max  = max(magnitude_max, magnitude[stride * y_clamped + x_clamped]);
            magnitude_sum += magnitude[stride * y_clamped + x_clamped];
        }
    }

    blurred[stride_3ch * y + x * 3 + 0] = static_cast<BlurredType>(sum0 / (ksize * ksize));
    blurred[stride_3ch * y + x * 3 + 1] = static_cast<BlurredType>(sum1 / (ksize * ksize));
    blurred[stride_3ch * y + x * 3 + 2] = static_cast<BlurredType>(sum2 / (ksize * ksize));
    rtv[stride * y + x] = (intensity_max - intensity_min) * magnitude_max / (magnitude_sum + epsilon);
}

template <typename BlurredType, typename RTVType, typename GuideType>
__global__ void compute_guide_kernel(const BlurredType* const blurred, const RTVType* const rtv, GuideType* const guide,
                                     const int ksize, const int width, const int height) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride_3ch = width * 3;
    const int stride = width;
    const int x = idx % width;
    const int y = idx / width;
    const auto radius  = ksize / 2;
    const auto sigma_alpha = 1.f / (5 * ksize);

    auto rtv_min = 1e10f;
    auto rtv_min_x = 0;
    auto rtv_min_y = 0;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto x_clamped = clamp(x + kx, 0, width - 1);
            const auto y_clamped = clamp(y + ky, 0, height - 1);

            if (rtv_min > rtv[stride * y_clamped + x_clamped]) {
                rtv_min = rtv[stride * y_clamped + x_clamped];
                rtv_min_x = x_clamped;
                rtv_min_y = y_clamped;
            }
        }
    }

    const auto alpha =
        2 / (1 + exp(sigma_alpha * (rtv[stride * y + x] - rtv[stride * rtv_min_y + rtv_min_x]))) - 1.f;
    guide[stride_3ch * y + x * 3 + 0] =      alpha  * blurred[stride_3ch * rtv_min_y + rtv_min_x * 3 + 0] +
                                        (1 - alpha) * blurred[stride_3ch * y + x * 3 + 0];
    guide[stride_3ch * y + x * 3 + 1] =      alpha  * blurred[stride_3ch * rtv_min_y + rtv_min_x * 3 + 1] +
                                        (1 - alpha) * blurred[stride_3ch * y + x * 3 + 1];
    guide[stride_3ch * y + x * 3 + 2] =      alpha  * blurred[stride_3ch * rtv_min_y + rtv_min_x * 3 + 2] +
                                        (1 - alpha) * blurred[stride_3ch * y + x * 3 + 2];
}

template <typename ImageType, typename GuideType>
__global__ void joint_bilateral_filter_kernel(const ImageType* const src, const GuideType* const guide,
                                              ImageType* const dst, const int ksize, const float* const kernel_space,
                                              const float* const kernel_color_table, const int width,
                                              const int height) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride_3ch = width * 3;
    const int x = idx % width;
    const int y = idx / width;
    const auto radius  = ksize / 2;

    const auto get_kernel_space = [ksize, radius, kernel_space](const int kx, const int ky) {
        return kernel_space[(ky + radius) * ksize + (kx + radius)];
    };

    const auto get_kernel_color = [kernel_color_table](const auto a, const auto b) {
        const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
        const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
        const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
        // const auto color_distance = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2) / 3;
        const auto color_distance = abs(diff0) + abs(diff1) + abs(diff2);
        return kernel_color_table[color_distance];
    };

    const auto guide_center_pix = guide + stride_3ch * y + x * 3;
    auto sum0 = 0.f;
    auto sum1 = 0.f;
    auto sum2 = 0.f;
    auto sum_k = 0.f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto x_clamped = clamp(x + kx, 0, width - 1);
            const auto y_clamped = clamp(y + ky, 0, height - 1);
            const auto pix       = src + stride_3ch * y_clamped + x_clamped * 3;
            const auto guide_pix = guide + stride_3ch * y_clamped + x_clamped * 3;
            const auto kernel    = get_kernel_space(kx, ky) * get_kernel_color(guide_center_pix, guide_pix);

            sum0 += pix[0] * kernel;
            sum1 += pix[1] * kernel;
            sum2 += pix[2] * kernel;
            sum_k += kernel;
        }
    }

    dst[stride_3ch * y + x * 3 + 0] = static_cast<ImageType>(sum0 / sum_k);
    dst[stride_3ch * y + x * 3 + 1] = static_cast<ImageType>(sum1 / sum_k);
    dst[stride_3ch * y + x * 3 + 2] = static_cast<ImageType>(sum2 / sum_k);
}

namespace cuda {

class BilateralTextureFilter::Impl {
public:
    Impl(const int width, const int height, const int ksize = 9, const int nitr = 3)
    : width_(width), height_(height), ksize_(ksize), nitr_(nitr), sigma_space_(ksize - 1), sigma_color_(jbf_sigma_color) {
        d_src_n_     = thrust::device_vector<ElemType>(width_ * height_ * 3);
        d_dst_n_     = thrust::device_vector<ElemType>(width_ * height_ * 3);
        d_blurred_   = thrust::device_vector<float>(width_ * height_ * 3);
        d_magnitude_ = thrust::device_vector<float>(width_ * height_);
        d_rtv_       = thrust::device_vector<float>(width_ * height_);
        d_guide_     = thrust::device_vector<ElemType>(width_ * height_ * 3);
    }

    ~Impl() {}

    void execute(const ElemType* const d_src, ElemType* const d_dst) {
        thrust::copy(d_src, d_src + width_ * height_ * 3, d_dst_n_.begin());

        for (int itr = 0; itr < nitr_; itr++) {
            thrust::copy(d_dst_n_.begin(), d_dst_n_.end(), d_src_n_.begin());
            compute_magnitude(d_src_n_, d_magnitude_);
            compute_blur_and_rtv(d_src_n_, d_magnitude_, d_blurred_, d_rtv_);
            compute_guide(d_blurred_, d_rtv_, d_guide_);
            joint_bilateral_filter(d_src_n_, d_guide_, d_dst_n_, 2 * ksize_ - 1, sigma_space_, sigma_color_);
        }

        cudaDeviceSynchronize();
        thrust::copy(d_dst_n_.begin(), d_dst_n_.end(), d_dst);
    }

private:
    template <typename MagnitudeType>
    void compute_magnitude(const thrust::device_vector<ElemType>& d_image,
                           thrust::device_vector<MagnitudeType>& d_magnitude) {
        const dim3 grid_dim{static_cast<::std::uint32_t>(height_)};
        const dim3 block_dim{static_cast<::std::uint32_t>(width_)};
        compute_magnitude_kernel<<<grid_dim, block_dim>>>(
            d_image.data().get(), d_magnitude.data().get(), width_, height_);
        CUDASafeCall();
    }

    template <typename MagnitudeType, typename BlurredType, typename RTVType>
    void compute_blur_and_rtv(const thrust::device_vector<ElemType>& d_image,
                              const thrust::device_vector<MagnitudeType>& d_magnitude,
                              thrust::device_vector<BlurredType>& d_blurred, thrust::device_vector<RTVType>& d_rtv) {
        const dim3 grid_dim{static_cast<::std::uint32_t>(height_)};
        const dim3 block_dim{static_cast<::std::uint32_t>(width_)};
        compute_blur_and_rtv_kernel<<<grid_dim, block_dim>>>(
            d_image.data().get(), d_magnitude.data().get(), d_blurred.data().get(), d_rtv.data().get(), ksize_,
            width_, height_);
        CUDASafeCall();
    }

    template <typename BlurredType, typename RTVType, typename GuideType>
    void compute_guide(const thrust::device_vector<BlurredType>& d_blurred, const thrust::device_vector<RTVType>& d_rtv,
                       thrust::device_vector<GuideType>& d_guide) {
        const dim3 grid_dim{static_cast<::std::uint32_t>(height_)};
        const dim3 block_dim{static_cast<::std::uint32_t>(width_)};
        compute_guide_kernel<<<grid_dim, block_dim>>>(
            d_blurred.data().get(), d_rtv.data().get(), d_guide.data().get(), ksize_, width_, height_);
        CUDASafeCall();
    }

    template <typename GuideType>
    void joint_bilateral_filter(const thrust::device_vector<ElemType>& d_src,
                                const thrust::device_vector<GuideType>& d_guide,
                                thrust::device_vector<ElemType>& d_dst,
                                const int ksize, const float sigma_space, const float sigma_color) {
        const auto gauss_color_coeff = -1.f / (2 * sigma_color * sigma_color);
        const auto gauss_space_coeff = -1.f / (2 * sigma_space * sigma_space);
        const auto radius  = ksize / 2;

        thrust::host_vector<float> h_kernel_space(ksize * ksize, 0.f);
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                const auto kidx = (ky + radius) * ksize + (kx + radius);
                const auto r2 = kx * kx + ky * ky;
                if (r2 > radius * radius) {
                    continue;
                }
                h_kernel_space[kidx] = ::std::exp(r2 * gauss_space_coeff);
            }
        }
        thrust::device_vector<float> kernel_space = h_kernel_space;

        thrust::host_vector<float> h_kernel_color_table(256 * 3);
        for (int i = 0; i < h_kernel_color_table.size(); i++) {
            h_kernel_color_table[i] = ::std::exp((i * i) * gauss_color_coeff);
        }
        thrust::device_vector<float> kernel_color_table = h_kernel_color_table;

        const dim3 grid_dim{static_cast<::std::uint32_t>(height_)};
        const dim3 block_dim{static_cast<::std::uint32_t>(width_)};
        joint_bilateral_filter_kernel<<<grid_dim, block_dim>>>(
            d_src.data().get(), d_guide.data().get(), d_dst.data().get(), ksize, kernel_space.data().get(),
            kernel_color_table.data().get(), width_, height_);
        CUDASafeCall();
    }

private:
    // static constexpr auto jbf_sigma_color = 0.05f * 1.73205080757f; // 0.05 * sqrt(3)
    static constexpr auto jbf_sigma_color = 1.73205080757f; // sqrt(3)

    const int   width_;
    const int   height_;
    const int   ksize_;
    const int   nitr_;
    const float sigma_space_;
    const float sigma_color_;

    thrust::device_vector<ElemType> d_src_n_;
    thrust::device_vector<ElemType> d_dst_n_;
    thrust::device_vector<float>    d_blurred_;
    thrust::device_vector<float>    d_magnitude_;
    thrust::device_vector<float>    d_rtv_;
    thrust::device_vector<ElemType> d_guide_;
};

BilateralTextureFilter::BilateralTextureFilter(const int width, const int height, const int ksize, const int nitr) {
    impl_ = new BilateralTextureFilter::Impl(width, height, ksize, nitr);
}

BilateralTextureFilter::~BilateralTextureFilter() {
    delete impl_;
}

void BilateralTextureFilter::execute(const ElemType* const d_src, ElemType* const d_dst) {
    impl_->execute(d_src, d_dst);
    cudaDeviceSynchronize();
}

} // namespace cuda
