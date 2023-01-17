#include <thrust/host_vector.h>

#include "bilateral_filter_impl.cuh"
#include "device_utilities.cuh"
#include "host_utilities.hpp"

__global__ void bilateral_filter_kernel(
    const std::uint8_t* const src,
    std::uint8_t* const       dst,
    const int                 ksize,
    const float* const        kernel_space,
    const float* const        kernel_color_table,
    const int                 width,
    const int                 height
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride_3ch = width * 3;
    const int x = idx % width;
    const int y = idx / width;
    const auto radius  = ksize / 2;

    extern __shared__ float s_kernel_buffer[];
    auto s_kernel_space       = &s_kernel_buffer[0];
    auto s_kernel_color_table = &s_kernel_buffer[ksize * ksize];

    for (int i = threadIdx.x; i < ksize * ksize; i += blockDim.x) {
        s_kernel_space[i] = kernel_space[i];
    }
    for (int i = threadIdx.x; i < 256 * 3; i += blockDim.x) {
        s_kernel_color_table[i] = kernel_color_table[i];
    }
    __syncthreads();

    const auto get_kernel_space = [ksize, radius, s_kernel_space](const int kx, const int ky) {
        return s_kernel_space[(ky + radius) * ksize + (kx + radius)];
    };

    const auto get_kernel_color = [s_kernel_color_table](const auto a, const auto b) {
        const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
        const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
        const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
        const auto color_distance = abs(diff0) + abs(diff1) + abs(diff2);
        return s_kernel_color_table[color_distance];
    };

    const auto src_center_pix = src + stride_3ch * y + x * 3;
    auto sum0 = 0.f;
    auto sum1 = 0.f;
    auto sum2 = 0.f;
    auto sum_k = 0.f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto x_clamped = clamp(x + kx, 0, width - 1);
            const auto y_clamped = clamp(y + ky, 0, height - 1);
            const auto pix       = src + stride_3ch * y_clamped + x_clamped * 3;
            const auto src_pix = src + stride_3ch * y_clamped + x_clamped * 3;
            const auto kernel    = get_kernel_space(kx, ky) * get_kernel_color(src_center_pix, src_pix);

            sum0 += pix[0] * kernel;
            sum1 += pix[1] * kernel;
            sum2 += pix[2] * kernel;
            sum_k += kernel;
        }
    }

    dst[stride_3ch * y + x * 3 + 0] = static_cast<std::uint8_t>(sum0 / sum_k);
    dst[stride_3ch * y + x * 3 + 1] = static_cast<std::uint8_t>(sum1 / sum_k);
    dst[stride_3ch * y + x * 3 + 2] = static_cast<std::uint8_t>(sum2 / sum_k);
}

__global__ void joint_bilateral_filter_kernel(
    const std::uint8_t* const src,
    const std::uint8_t* const guide,
    std::uint8_t* const       dst,
    const int                 ksize,
    const float* const        kernel_space,
    const float* const        kernel_color_table,
    const int                 width,
    const int                 height
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride_3ch = width * 3;
    const int x = idx % width;
    const int y = idx / width;
    const auto radius  = ksize / 2;

    extern __shared__ float s_kernel_buffer[];
    auto s_kernel_space       = &s_kernel_buffer[0];
    auto s_kernel_color_table = &s_kernel_buffer[ksize * ksize];

    for (int i = threadIdx.x; i < ksize * ksize; i += blockDim.x) {
        s_kernel_space[i] = kernel_space[i];
    }
    for (int i = threadIdx.x; i < 256 * 3; i += blockDim.x) {
        s_kernel_color_table[i] = kernel_color_table[i];
    }
    __syncthreads();

    const auto get_kernel_space = [ksize, radius, s_kernel_space](const int kx, const int ky) {
        return s_kernel_space[(ky + radius) * ksize + (kx + radius)];
    };

    const auto get_kernel_color = [s_kernel_color_table](const auto a, const auto b) {
        const auto diff0 = static_cast<int>(a[0]) - static_cast<int>(b[0]);
        const auto diff1 = static_cast<int>(a[1]) - static_cast<int>(b[1]);
        const auto diff2 = static_cast<int>(a[2]) - static_cast<int>(b[2]);
        const auto color_distance = abs(diff0) + abs(diff1) + abs(diff2);
        return s_kernel_color_table[color_distance];
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

    dst[stride_3ch * y + x * 3 + 0] = static_cast<std::uint8_t>(sum0 / sum_k);
    dst[stride_3ch * y + x * 3 + 1] = static_cast<std::uint8_t>(sum1 / sum_k);
    dst[stride_3ch * y + x * 3 + 2] = static_cast<std::uint8_t>(sum2 / sum_k);
}

CudaBilateralFilter::Impl::Impl(
    const int   width,
    const int   height,
    const int   ksize,
    const float sigma_space,
    const float sigma_color)
: width_(width),
  height_(height),
  ksize_(ksize),
  sigma_space_(sigma_space),
  sigma_color_(sigma_color) {
    const auto gauss_color_coeff = -1.f / (2 * sigma_color * sigma_color);
    const auto gauss_space_coeff = -1.f / (2 * sigma_space * sigma_space);
    const auto radius  = ksize / 2;

    thrust::host_vector<float> h_kernel_space(ksize_ * ksize_);
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto kidx = (ky + radius) * ksize_ + (kx + radius);
            const auto r2 = kx * kx + ky * ky;
            if (r2 > radius * radius) {
                continue;
            }
            h_kernel_space[kidx] = std::exp(r2 * gauss_space_coeff);
        }
    }
    d_kernel_space_ = h_kernel_space;

    thrust::host_vector<float> h_kernel_color_table(256 * 3);
    for (int i = 0; i < h_kernel_color_table.size(); i++) {
        h_kernel_color_table[i] = std::exp((i * i) * gauss_color_coeff);
    }
    d_kernel_color_table_ = h_kernel_color_table;
}

CudaBilateralFilter::Impl::~Impl() {}

void CudaBilateralFilter::Impl::bilateral_filter(
    const std::uint8_t* const d_src,
    std::uint8_t* const       d_dst
) const {
    const dim3 grid_dim{static_cast<std::uint32_t>(height_)};
    const dim3 block_dim{static_cast<std::uint32_t>(width_)};
    const std::uint32_t smem_size = (d_kernel_space_.size() + d_kernel_color_table_.size()) * sizeof(float);
    bilateral_filter_kernel<<<grid_dim, block_dim, smem_size>>>(
        d_src, d_dst, ksize_, d_kernel_space_.data().get(), d_kernel_color_table_.data().get(), width_, height_);
    CUDASafeCall();
}

void CudaBilateralFilter::Impl::joint_bilateral_filter(
    const std::uint8_t* const d_src,
    const std::uint8_t* const d_guide,
    std::uint8_t* const       d_dst
) const {
    const dim3 grid_dim{static_cast<std::uint32_t>(height_)};
    const dim3 block_dim{static_cast<std::uint32_t>(width_)};
    const std::uint32_t smem_size = (d_kernel_space_.size() + d_kernel_color_table_.size()) * sizeof(float);
    joint_bilateral_filter_kernel<<<grid_dim, block_dim, smem_size>>>(
        d_src, d_guide, d_dst, ksize_, d_kernel_space_.data().get(), d_kernel_color_table_.data().get(),
        width_, height_);
    CUDASafeCall();
}

CudaBilateralFilter::CudaBilateralFilter(
    const int   width,
    const int   height,
    const int   ksize,
    const float sigma_space,
    const float sigma_color
) {
    impl_ = new CudaBilateralFilter::Impl(width, height, ksize, sigma_space, sigma_color);
}

CudaBilateralFilter::~CudaBilateralFilter() {
    delete impl_;
}

void CudaBilateralFilter::bilateral_filter(
    const std::uint8_t* const d_src,
    std::uint8_t* const d_dst
) const {
    impl_->bilateral_filter(d_src, d_dst);
    cudaDeviceSynchronize();
}


void CudaBilateralFilter::joint_bilateral_filter(
    const std::uint8_t* const d_src,
    const std::uint8_t* const d_guide,
    std::uint8_t* const d_dst
) const {
    impl_->joint_bilateral_filter(d_src, d_guide, d_dst);
    cudaDeviceSynchronize();
}
