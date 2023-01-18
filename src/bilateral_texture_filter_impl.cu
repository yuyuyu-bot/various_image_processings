#include <thrust/host_vector.h>

#include "bilateral_texture_filter_impl.cuh"
#include "device_utilities.cuh"
#include "host_utilities.hpp"

static constexpr auto epsilon = 1e-9;

__global__ void compute_magnitude_kernel(
    const std::uint8_t* const image,
    float* const magnitude,
    const int width,
    const int height
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int stride_3ch = width * 3;
    const int stride = width;

    extern __shared__ std::uint8_t s_image_buffer[];
    const int smem_width  = blockDim.x + 2;
    const int smem_height = blockDim.y + 2;
    const int smem_stride = smem_width * 3;
    const int smem_origin_x = x - tx - 1;
    const int smem_origin_y = y - ty - 1;

    const auto get_s_img_ptr = [smem_stride, smem_origin_x, smem_origin_y](const int img_x, const int img_y) {
        const auto s_img_x = img_x - smem_origin_x;
        const auto s_img_y = img_y - smem_origin_y;
        return &s_image_buffer[smem_stride * s_img_y + s_img_x * 3];
    };

    for (int y_offset = ty; y_offset < smem_height; y_offset += blockDim.y) {
        for (int x_offset = tx; x_offset < smem_width; x_offset += blockDim.x) {
            auto* const s_img_ptr = get_s_img_ptr(smem_origin_x + x_offset, smem_origin_y + y_offset);
            const auto x_clamped = clamp(smem_origin_x + x_offset, 0, width - 1);
            const auto y_clamped = clamp(smem_origin_y + y_offset, 0, height - 1);
            s_img_ptr[0] = image[stride_3ch * y_clamped + x_clamped * 3 + 0];
            s_img_ptr[1] = image[stride_3ch * y_clamped + x_clamped * 3 + 1];
            s_img_ptr[2] = image[stride_3ch * y_clamped + x_clamped * 3 + 2];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    const auto compute_del = [&get_s_img_ptr, stride_3ch](const int x0, const int y0, const int x1, const int y1) {
        const auto* const s_img_ptr0 = get_s_img_ptr(x0, y0);
        const auto* const s_img_ptr1 = get_s_img_ptr(x1, y1);
        const auto diff0 = s_img_ptr0[0] - s_img_ptr1[0];
        const auto diff1 = s_img_ptr0[1] - s_img_ptr1[1];
        const auto diff2 = s_img_ptr0[2] - s_img_ptr1[2];
        return diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
    };

    const auto del_x = compute_del(x - 1, y, x + 1, y);
    const auto del_y = compute_del(x, y - 1, x, y + 1);
    magnitude[stride * y + x] = sqrtf(del_x + del_y);
}

__global__ void compute_blur_and_rtv_kernel(
    const std::uint8_t* const image,
    const float* const magnitude,
    float* const blurred,
    float* const rtv,
    const int ksize,
    const int width,
    const int height
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int stride_3ch = width * 3;
    const int stride = width;
    const int radius = ksize / 2;

    // config of shared memory
    const int smem_width    = blockDim.x + ksize - 1;
    const int smem_height   = blockDim.y + ksize - 1;
    const int smem_origin_x = x - tx - radius;
    const int smem_origin_y = y - ty - radius;

    // shared image
    extern __shared__ std::uint8_t s_image_magnitude_buffer[];
    auto s_img = &s_image_magnitude_buffer[0];
    const auto s_img_stride = smem_width * 3;
    const auto get_s_img_ptr = [s_img, s_img_stride, smem_origin_x, smem_origin_y](const int img_x, const int img_y) {
        const auto s_img_x = img_x - smem_origin_x;
        const auto s_img_y = img_y - smem_origin_y;
        return &s_img[s_img_stride * s_img_y + s_img_x * 3];
    };

    // shared magnitude
    auto s_mag = reinterpret_cast<float*>(&s_img[smem_width * smem_height * 3]);
    const auto s_mag_stride = smem_width;
    const auto get_s_mag_ptr = [s_mag, s_mag_stride, smem_origin_x, smem_origin_y](const int mag_x, const int mag_y) {
        const auto s_mag_x = mag_x - smem_origin_x;
        const auto s_mag_y = mag_y - smem_origin_y;
        return &s_mag[s_mag_stride * s_mag_y + s_mag_x];
    };

    // copy global data to shared memory
    for (int y_offset = ty; y_offset < smem_height; y_offset += blockDim.y) {
        for (int x_offset = tx; x_offset < smem_width; x_offset += blockDim.x) {
            auto* const s_img_ptr = get_s_img_ptr(smem_origin_x + x_offset, smem_origin_y + y_offset);
            auto* const s_mag_ptr = get_s_mag_ptr(smem_origin_x + x_offset, smem_origin_y + y_offset);
            const auto x_clamped = clamp(smem_origin_x + x_offset, 0, width - 1);
            const auto y_clamped = clamp(smem_origin_y + y_offset, 0, height - 1);
            s_img_ptr[0] = image[stride_3ch * y_clamped + x_clamped * 3 + 0];
            s_img_ptr[1] = image[stride_3ch * y_clamped + x_clamped * 3 + 1];
            s_img_ptr[2] = image[stride_3ch * y_clamped + x_clamped * 3 + 2];
            *s_mag_ptr   = magnitude[stride * y_clamped + x_clamped];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    auto sum0 = 0.f;
    auto sum1 = 0.f;
    auto sum2 = 0.f;

    auto intensity_max = 0.f;
    auto intensity_min = 256.f;
    auto magnitude_max = 0.f;
    auto magnitude_sum = 0.f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto x_clamped = clamp(x + kx, 0, width - 1);
            const auto y_clamped = clamp(y + ky, 0, height - 1);

            const auto s_img_ptr = get_s_img_ptr(x + kx, y + ky);
            sum0 += s_img_ptr[0];
            sum1 += s_img_ptr[1];
            sum2 += s_img_ptr[2];

            const auto intensity = (s_img_ptr[0] + s_img_ptr[1] + s_img_ptr[2]) / 3.f;
            intensity_max  = max(intensity_max, intensity);
            intensity_min  = min(intensity_min, intensity);

            const auto mag = *get_s_mag_ptr(x + kx, y + ky);
            magnitude_max  = max(magnitude_max, mag);
            magnitude_sum += mag;
        }
    }

    blurred[stride_3ch * y + x * 3 + 0] = sum0 / (ksize * ksize);
    blurred[stride_3ch * y + x * 3 + 1] = sum1 / (ksize * ksize);
    blurred[stride_3ch * y + x * 3 + 2] = sum2 / (ksize * ksize);
    rtv[stride * y + x] = (intensity_max - intensity_min) * magnitude_max / (magnitude_sum + epsilon);
}

__global__ void compute_guide_kernel(
    const float* const blurred,
    const float* const rtv,
    std::uint8_t* const guide,
    const int ksize,
    const int width,
    const int height
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int stride_3ch = width * 3;
    const int stride = width;
    const auto radius  = ksize / 2;
    const auto sigma_alpha = 1.f / (5 * ksize);

    // config of shared memory
    const int smem_width    = blockDim.x + ksize - 1;
    const int smem_height   = blockDim.y + ksize - 1;
    const int smem_origin_x = x - tx - radius;
    const int smem_origin_y = y - ty - radius;

    extern __shared__ float s_rtv_buffer[];
    auto s_rtv = &s_rtv_buffer[0];
    const auto get_s_rtv_ptr = [s_rtv, smem_width, smem_origin_x, smem_origin_y](const int rtv_x, const int rtv_y) {
        const auto s_rtv_x = rtv_x - smem_origin_x;
        const auto s_rtv_y = rtv_y - smem_origin_y;
        return &s_rtv[smem_width * s_rtv_y + s_rtv_x];
    };

    // copy global data to shared memory
    for (int y_offset = ty; y_offset < smem_height; y_offset += blockDim.y) {
        for (int x_offset = tx; x_offset < smem_width; x_offset += blockDim.x) {
            auto* const s_rtv_ptr  = get_s_rtv_ptr(smem_origin_x + x_offset, smem_origin_y + y_offset);
            const auto x_clamped = clamp(smem_origin_x + x_offset, 0, width - 1);
            const auto y_clamped = clamp(smem_origin_y + y_offset, 0, height - 1);
            *s_rtv_ptr = rtv[stride * y_clamped + x_clamped];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    auto rtv_min = 1e10f;
    auto rtv_min_x = 0;
    auto rtv_min_y = 0;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            const auto rtv_value = *get_s_rtv_ptr(x + kx, y + ky);
            if (rtv_min > rtv_value) {
                rtv_min = rtv_value;
                rtv_min_x = clamp(x + kx, 0, width - 1);
                rtv_min_y = clamp(y + ky, 0, height - 1);
            }
        }
    }

    const auto alpha = 2 / (1 + exp(sigma_alpha * (*get_s_rtv_ptr(x, y) - *get_s_rtv_ptr(rtv_min_x, rtv_min_y)))) - 1.f;
    guide[stride_3ch * y + x * 3 + 0] =      alpha  * blurred[stride_3ch * rtv_min_y + rtv_min_x * 3 + 0] +
                                        (1 - alpha) * blurred[stride_3ch * y + x * 3 + 0];
    guide[stride_3ch * y + x * 3 + 1] =      alpha  * blurred[stride_3ch * rtv_min_y + rtv_min_x * 3 + 1] +
                                        (1 - alpha) * blurred[stride_3ch * y + x * 3 + 1];
    guide[stride_3ch * y + x * 3 + 2] =      alpha  * blurred[stride_3ch * rtv_min_y + rtv_min_x * 3 + 2] +
                                        (1 - alpha) * blurred[stride_3ch * y + x * 3 + 2];
}

CudaBilateralTextureFilter::Impl::Impl(
    const int width,
    const int height,
    const int ksize,
    const int nitr)
: width_(width),
  height_(height),
  ksize_(ksize),
  nitr_(nitr),
  jbf_executor_(width, height, 2 * ksize - 1, ksize - 1, jbf_sigma_color) {
    d_src_n_     = thrust::device_vector<std::uint8_t>(width_ * height_ * 3);
    d_dst_n_     = thrust::device_vector<std::uint8_t>(width_ * height_ * 3);
    d_blurred_   = thrust::device_vector<float>(width_ * height_ * 3);
    d_magnitude_ = thrust::device_vector<float>(width_ * height_);
    d_rtv_       = thrust::device_vector<float>(width_ * height_);
    d_guide_     = thrust::device_vector<std::uint8_t>(width_ * height_ * 3);
}

CudaBilateralTextureFilter::Impl::~Impl() {}

void CudaBilateralTextureFilter::Impl::execute(
    const std::uint8_t* const d_src,
    std::uint8_t* const d_dst
) {
    thrust::copy(d_src, d_src + width_ * height_ * 3, d_dst_n_.begin());

    for (int itr = 0; itr < nitr_; itr++) {
        thrust::copy(d_dst_n_.begin(), d_dst_n_.end(), d_src_n_.begin());
        compute_magnitude(d_src_n_, d_magnitude_);
        compute_blur_and_rtv(d_src_n_, d_magnitude_, d_blurred_, d_rtv_);
        compute_guide(d_blurred_, d_rtv_, d_guide_);
        jbf_executor_.joint_bilateral_filter(d_src_n_.data().get(), d_guide_.data().get(), d_dst_n_.data().get());
    }

    thrust::copy(d_dst_n_.begin(), d_dst_n_.end(), d_dst);
}

void CudaBilateralTextureFilter::Impl::compute_magnitude(
    const thrust::device_vector<std::uint8_t>& d_image,
    thrust::device_vector<float>& d_magnitude
) {
    const std::uint32_t block_width  = 16u;
    const std::uint32_t block_height = 16u;
    const std::uint32_t grid_width   = (width_  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height_ + block_height - 1) / block_height;

    const dim3 grid_dim (grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    const std::uint32_t smem_size = (block_width + 2) * (block_height + 2) * 3 * sizeof(std::uint8_t);

    compute_magnitude_kernel<<<grid_dim, block_dim, smem_size>>>(
        d_image.data().get(), d_magnitude.data().get(), width_, height_);
    CUDASafeCall();
}

void CudaBilateralTextureFilter::Impl::compute_blur_and_rtv(
    const thrust::device_vector<std::uint8_t>& d_image,
    const thrust::device_vector<float>& d_magnitude,
    thrust::device_vector<float>& d_blurred,
    thrust::device_vector<float>& d_rtv
) {
    const std::uint32_t block_width  = 16u;
    const std::uint32_t block_height = 16u;
    const std::uint32_t grid_width   = (width_  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height_ + block_height - 1) / block_height;

    const dim3 grid_dim (grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    const std::uint32_t smem_size =
        (block_width + ksize_ - 1) * (block_height + ksize_ - 1) * 3 * sizeof(std::uint8_t) +
        (block_width + ksize_ - 1) * (block_height + ksize_ - 1) * sizeof(float);

    compute_blur_and_rtv_kernel<<<grid_dim, block_dim, smem_size>>>(
        d_image.data().get(), d_magnitude.data().get(), d_blurred.data().get(), d_rtv.data().get(), ksize_,
        width_, height_);
    CUDASafeCall();
}

void CudaBilateralTextureFilter::Impl::compute_guide(
    const thrust::device_vector<float>& d_blurred,
    const thrust::device_vector<float>& d_rtv,
    thrust::device_vector<std::uint8_t>& d_guide
) {
    const std::uint32_t block_width  = 16u;
    const std::uint32_t block_height = 16u;
    const std::uint32_t grid_width   = (width_  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height_ + block_height - 1) / block_height;

    const dim3 grid_dim (grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    const std::uint32_t smem_size = (block_width + ksize_ - 1) * (block_height + ksize_ - 1) * sizeof(float);

    compute_guide_kernel<<<grid_dim, block_dim, smem_size>>>(
        d_blurred.data().get(), d_rtv.data().get(), d_guide.data().get(), ksize_, width_, height_);
    CUDASafeCall();
}

CudaBilateralTextureFilter::CudaBilateralTextureFilter(
    const int width,
    const int height,
    const int ksize,
    const int nitr
) {
    impl_ = new CudaBilateralTextureFilter::Impl(width, height, ksize, nitr);
}

CudaBilateralTextureFilter::~CudaBilateralTextureFilter() {
    delete impl_;
}

void CudaBilateralTextureFilter::execute(
    const std::uint8_t* const d_src,
    std::uint8_t* const d_dst
) {
    impl_->execute(d_src, d_dst);
    cudaDeviceSynchronize();
}
