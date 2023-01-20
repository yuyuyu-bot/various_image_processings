#include "bilateral_filter_impl.cuh"
#include "device_utilities.cuh"
#include "host_utilities.hpp"

constexpr auto kernel_color_table_len = 256 * 3;

__global__ void bilateral_filter_kernel(
    const std::uint8_t* const src,
    std::uint8_t* const       dst,
    const int                 ksize,
    const float* const        kernel_space,
    const float* const        kernel_color_table,
    const int                 width,
    const int                 height
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int stride_3ch = width * 3;
    const int radius  = ksize / 2;

    extern __shared__ float s_bilateral_filter_buffer[];
    auto s_kernel_space       = &s_bilateral_filter_buffer[0];
    auto s_kernel_color_table = &s_bilateral_filter_buffer[ksize * ksize];

    const int smem_width    = blockDim.x + ksize - 1;
    const int smem_height   = blockDim.y + ksize - 1;
    const int smem_stride   = smem_width * 3;
    const int smem_origin_x = x - tx - radius;
    const int smem_origin_y = y - ty - radius;
    auto s_src              = reinterpret_cast<std::uint8_t*>(&s_kernel_color_table[kernel_color_table_len]);

    for (int i = blockDim.x * threadIdx.y + threadIdx.x; i < ksize * ksize; i += blockDim.x * blockDim.y) {
        s_kernel_space[i] = kernel_space[i];
    }
    for (int i = blockDim.x * threadIdx.y + threadIdx.x; i < kernel_color_table_len; i += blockDim.x * blockDim.y) {
        s_kernel_color_table[i] = kernel_color_table[i];
    }

    const auto get_s_src_ptr = [s_src, smem_stride, smem_origin_x, smem_origin_y](const int src_x, const int src_y) {
        const auto s_src_x = src_x - smem_origin_x;
        const auto s_src_y = src_y - smem_origin_y;
        return &s_src[smem_stride * s_src_y + s_src_x * 3];
    };

    for (int y_offset = ty; y_offset < smem_height; y_offset += blockDim.y) {
        for (int x_offset = tx; x_offset < smem_width; x_offset += blockDim.x) {
            auto* const s_src_ptr = get_s_src_ptr(smem_origin_x + x_offset, smem_origin_y + y_offset);
            const auto x_clamped = clamp(smem_origin_x + x_offset, 0, width - 1);
            const auto y_clamped = clamp(smem_origin_y + y_offset, 0, height - 1);
            s_src_ptr[0] = src[stride_3ch * y_clamped + x_clamped * 3 + 0];
            s_src_ptr[1] = src[stride_3ch * y_clamped + x_clamped * 3 + 1];
            s_src_ptr[2] = src[stride_3ch * y_clamped + x_clamped * 3 + 2];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

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
    auto sumk = 0.f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto src_pix   = get_s_src_ptr(x + kx, y + ky);
            const auto kernel    = get_kernel_space(kx, ky) * get_kernel_color(src_center_pix, src_pix);

            sum0 += src_pix[0] * kernel;
            sum1 += src_pix[1] * kernel;
            sum2 += src_pix[2] * kernel;
            sumk += kernel;
        }
    }

    dst[stride_3ch * y + x * 3 + 0] = static_cast<std::uint8_t>(sum0 / sumk + 0.5f);
    dst[stride_3ch * y + x * 3 + 1] = static_cast<std::uint8_t>(sum1 / sumk + 0.5f);
    dst[stride_3ch * y + x * 3 + 2] = static_cast<std::uint8_t>(sum2 / sumk + 0.5f);
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
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int stride_3ch = width * 3;
    const int radius  = ksize / 2;

    extern __shared__ float s_joint_bilateral_filter_buffer[];
    auto s_kernel_space       = &s_joint_bilateral_filter_buffer[0];
    auto s_kernel_color_table = &s_joint_bilateral_filter_buffer[ksize * ksize];

    const int smem_width    = blockDim.x + ksize - 1;
    const int smem_height   = blockDim.y + ksize - 1;
    const int smem_stride = smem_width * 3;
    const int smem_origin_x = x - tx - radius;
    const int smem_origin_y = y - ty - radius;
    auto s_src              = reinterpret_cast<std::uint8_t*>(&s_kernel_color_table[kernel_color_table_len]);
    auto s_guide            = &s_src[smem_stride * smem_height];

    for (int i = blockDim.x * threadIdx.y + threadIdx.x; i < ksize * ksize; i += blockDim.x * blockDim.y) {
        s_kernel_space[i] = kernel_space[i];
    }
    for (int i = blockDim.x * threadIdx.y + threadIdx.x; i < kernel_color_table_len; i += blockDim.x * blockDim.y) {
        s_kernel_color_table[i] = kernel_color_table[i];
    }

    const auto get_s_src_ptr = [s_src, smem_stride, smem_origin_x, smem_origin_y](const int src_x, const int src_y) {
        const auto s_src_x = src_x - smem_origin_x;
        const auto s_src_y = src_y - smem_origin_y;
        return &s_src[smem_stride * s_src_y + s_src_x * 3];
    };

    const auto get_s_guide_ptr = [s_guide, smem_stride, smem_origin_x, smem_origin_y](const int guide_x, const int guide_y) {
        const auto s_guide_x = guide_x - smem_origin_x;
        const auto s_guide_y = guide_y - smem_origin_y;
        return &s_guide[smem_stride * s_guide_y + s_guide_x * 3];
    };

    for (int y_offset = ty; y_offset < smem_height; y_offset += blockDim.y) {
        for (int x_offset = tx; x_offset < smem_width; x_offset += blockDim.x) {
            auto* const s_src_ptr   = get_s_src_ptr(smem_origin_x + x_offset, smem_origin_y + y_offset);
            auto* const s_guide_ptr = get_s_guide_ptr(smem_origin_x + x_offset, smem_origin_y + y_offset);
            const auto x_clamped    = clamp(smem_origin_x + x_offset, 0, width - 1);
            const auto y_clamped    = clamp(smem_origin_y + y_offset, 0, height - 1);
            s_src_ptr[0]   = src[stride_3ch * y_clamped + x_clamped * 3 + 0];
            s_src_ptr[1]   = src[stride_3ch * y_clamped + x_clamped * 3 + 1];
            s_src_ptr[2]   = src[stride_3ch * y_clamped + x_clamped * 3 + 2];
            s_guide_ptr[0] = guide[stride_3ch * y_clamped + x_clamped * 3 + 0];
            s_guide_ptr[1] = guide[stride_3ch * y_clamped + x_clamped * 3 + 1];
            s_guide_ptr[2] = guide[stride_3ch * y_clamped + x_clamped * 3 + 2];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

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
    auto sumk = 0.f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto x_clamped = clamp(x + kx, 0, width - 1);
            const auto y_clamped = clamp(y + ky, 0, height - 1);
            const auto src_pix   = get_s_src_ptr(x + kx, y + ky);
            const auto guide_pix = get_s_guide_ptr(x + kx, y + ky);
            const auto kernel    = get_kernel_space(kx, ky) * get_kernel_color(guide_center_pix, guide_pix);

            sum0 += src_pix[0] * kernel;
            sum1 += src_pix[1] * kernel;
            sum2 += src_pix[2] * kernel;
            sumk += kernel;
        }
    }

    dst[stride_3ch * y + x * 3 + 0] = static_cast<std::uint8_t>(sum0 / sumk + 0.5f);
    dst[stride_3ch * y + x * 3 + 1] = static_cast<std::uint8_t>(sum1 / sumk + 0.5f);
    dst[stride_3ch * y + x * 3 + 2] = static_cast<std::uint8_t>(sum2 / sumk + 0.5f);
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
  sigma_color_(sigma_color),
  d_kernel_space_(ksize * ksize),
  d_kernel_color_table_(kernel_color_table_len) {
    const auto gauss_color_coeff = -1.f / (2 * sigma_color * sigma_color);
    const auto gauss_space_coeff = -1.f / (2 * sigma_space * sigma_space);
    const auto radius  = ksize / 2;

    std::vector<float> h_kernel_space(ksize_ * ksize_);
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
    thrust::copy(h_kernel_space.begin(), h_kernel_space.end(), d_kernel_space_.begin());

    std::vector<float> h_kernel_color_table(kernel_color_table_len);
    for (int i = 0; i < h_kernel_color_table.size(); i++) {
        h_kernel_color_table[i] = std::exp((i * i) * gauss_color_coeff);
    }
    thrust::copy(h_kernel_color_table.begin(), h_kernel_color_table.end(), d_kernel_color_table_.begin());
}

void CudaBilateralFilter::Impl::bilateral_filter(
    const std::uint8_t* const d_src,
    std::uint8_t* const       d_dst
) const {
    const std::uint32_t block_width  = 32u;
    const std::uint32_t block_height = 32u;
    const std::uint32_t grid_width   = (width_  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height_ + block_height - 1) / block_height;

    const dim3 grid_dim (grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    const std::uint32_t smem_size =
        (d_kernel_space_.size() + d_kernel_color_table_.size()) * sizeof(float) +
        (block_width + ksize_ - 1) * (block_height + ksize_ - 1) * 3 * sizeof(std::uint8_t);
    bilateral_filter_kernel<<<grid_dim, block_dim, smem_size>>>(
        d_src, d_dst, ksize_, d_kernel_space_.data().get(), d_kernel_color_table_.data().get(), width_, height_);
    CUDASafeCall();
}

void CudaBilateralFilter::Impl::joint_bilateral_filter(
    const std::uint8_t* const d_src,
    const std::uint8_t* const d_guide,
    std::uint8_t* const       d_dst
) const {
    const std::uint32_t block_width  = 32u;
    const std::uint32_t block_height = 32u;
    const std::uint32_t grid_width   = (width_  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height_ + block_height - 1) / block_height;

    const dim3 grid_dim (grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    const std::uint32_t smem_size =
        (d_kernel_space_.size() + d_kernel_color_table_.size()) * sizeof(float) +
        (block_width + ksize_ - 1) * (block_height + ksize_ - 1) * 3 * sizeof(std::uint8_t) +
        (block_width + ksize_ - 1) * (block_height + ksize_ - 1) * 3 * sizeof(std::uint8_t);
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
