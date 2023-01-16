#include <thrust/host_vector.h>

#include "bilateral_texture_filter_impl.cuh"
#include "device_utilities.cuh"
#include "host_utilities.hpp"

static constexpr auto epsilon = 1e-9;

template <typename ImageType>
__global__ void compute_magnitude_kernel(
    const ImageType* const image,
    float* const magnitude,
    const int width,
    const int height)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int stride_3ch = width * 3;
    const int stride = width;

    extern __shared__ ImageType s_image_buffer[];
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

template <typename ImageType, typename BlurredType, typename RTVType>
__global__ void compute_blur_and_rtv_kernel(
    const ImageType* const image,
    const float* const magnitude,
    BlurredType* const blurred,
    RTVType* const rtv,
    const int ksize,
    const int width,
    const int height)
{
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
    extern __shared__ ImageType s_image_magnitude_buffer[];
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

    auto sum0 = 0;
    auto sum1 = 0;
    auto sum2 = 0;

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

    blurred[stride_3ch * y + x * 3 + 0] = static_cast<BlurredType>(sum0 / (ksize * ksize));
    blurred[stride_3ch * y + x * 3 + 1] = static_cast<BlurredType>(sum1 / (ksize * ksize));
    blurred[stride_3ch * y + x * 3 + 2] = static_cast<BlurredType>(sum2 / (ksize * ksize));
    rtv[stride * y + x] = (intensity_max - intensity_min) * magnitude_max / (magnitude_sum + epsilon);
}

template <typename BlurredType, typename RTVType, typename GuideType>
__global__ void compute_guide_kernel(
    const BlurredType* const blurred,
    const RTVType* const rtv,
    GuideType* const guide,
    const int ksize,
    const int width,
    const int height)
{
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
__global__ void joint_bilateral_filter_kernel(
    const ImageType* const src,
    const GuideType* const guide,
    ImageType* const dst,
    const int ksize,
    const float* const kernel_space,
    const float* const kernel_color_table,
    const int width,
    const int height)
{
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

    dst[stride_3ch * y + x * 3 + 0] = static_cast<ImageType>(sum0 / sum_k);
    dst[stride_3ch * y + x * 3 + 1] = static_cast<ImageType>(sum1 / sum_k);
    dst[stride_3ch * y + x * 3 + 2] = static_cast<ImageType>(sum2 / sum_k);
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
  sigma_space_(ksize - 1),
  sigma_color_(jbf_sigma_color)
{
    d_src_n_     = thrust::device_vector<ElemType>(width_ * height_ * 3);
    d_dst_n_     = thrust::device_vector<ElemType>(width_ * height_ * 3);
    d_blurred_   = thrust::device_vector<float>(width_ * height_ * 3);
    d_magnitude_ = thrust::device_vector<float>(width_ * height_);
    d_rtv_       = thrust::device_vector<float>(width_ * height_);
    d_guide_     = thrust::device_vector<ElemType>(width_ * height_ * 3);
}

CudaBilateralTextureFilter::Impl::~Impl() {}

void CudaBilateralTextureFilter::Impl::execute(
    const ElemType* const d_src,
    ElemType* const d_dst)
{
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

void CudaBilateralTextureFilter::Impl::compute_magnitude(
    const thrust::device_vector<ElemType>& d_image,
    thrust::device_vector<float>& d_magnitude)
{
    const std::uint32_t block_width  = 32u;
    const std::uint32_t block_height = 32u;
    const std::uint32_t grid_width   = (width_  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height_ + block_height - 1) / block_height;

    const dim3 grid_dim (grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    const std::uint32_t smem_size = (block_width + 2) * (block_height + 2) * 3 * sizeof(ElemType);

    compute_magnitude_kernel<<<grid_dim, block_dim, smem_size>>>(
        d_image.data().get(), d_magnitude.data().get(), width_, height_);
    CUDASafeCall();
}

template <typename BlurredType, typename RTVType>
void CudaBilateralTextureFilter::Impl::compute_blur_and_rtv(
    const thrust::device_vector<ElemType>& d_image,
    const thrust::device_vector<float>& d_magnitude,
    thrust::device_vector<BlurredType>& d_blurred,
    thrust::device_vector<RTVType>& d_rtv)
{
    const std::uint32_t block_width  = 16u;
    const std::uint32_t block_height = 8u;
    const std::uint32_t grid_width   = (width_  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height_ + block_height - 1) / block_height;

    const dim3 grid_dim (grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    const std::uint32_t smem_size =
        (block_width + ksize_ - 1) * (block_height + ksize_ - 1) * 3 * sizeof(ElemType) +
        (block_width + ksize_ - 1) * (block_height + ksize_ - 1) * sizeof(float);

    compute_blur_and_rtv_kernel<<<grid_dim, block_dim, smem_size>>>(
        d_image.data().get(), d_magnitude.data().get(), d_blurred.data().get(), d_rtv.data().get(), ksize_,
        width_, height_);
    CUDASafeCall();
}

template <typename BlurredType, typename RTVType, typename GuideType>
void CudaBilateralTextureFilter::Impl::compute_guide(
    const thrust::device_vector<BlurredType>& d_blurred,
    const thrust::device_vector<RTVType>& d_rtv,
    thrust::device_vector<GuideType>& d_guide)
{
    const dim3 grid_dim{static_cast<std::uint32_t>(height_)};
    const dim3 block_dim{static_cast<std::uint32_t>(width_)};
    compute_guide_kernel<<<grid_dim, block_dim>>>(
        d_blurred.data().get(), d_rtv.data().get(), d_guide.data().get(), ksize_, width_, height_);
    CUDASafeCall();
}

template <typename GuideType>
void CudaBilateralTextureFilter::Impl::joint_bilateral_filter(
    const thrust::device_vector<ElemType>& d_src,
    const thrust::device_vector<GuideType>& d_guide,
    thrust::device_vector<ElemType>& d_dst,
    const int ksize,
    const float sigma_space,
    const float sigma_color)
{
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
            h_kernel_space[kidx] = std::exp(r2 * gauss_space_coeff);
        }
    }
    thrust::device_vector<float> kernel_space = h_kernel_space;

    thrust::host_vector<float> h_kernel_color_table(256 * 3);
    for (int i = 0; i < h_kernel_color_table.size(); i++) {
        h_kernel_color_table[i] = std::exp((i * i) * gauss_color_coeff);
    }
    thrust::device_vector<float> kernel_color_table = h_kernel_color_table;

    const dim3 grid_dim{static_cast<std::uint32_t>(height_)};
    const dim3 block_dim{static_cast<std::uint32_t>(width_)};
    const std::uint32_t smem_size = (kernel_space.size() + kernel_color_table.size()) * sizeof(float);
    joint_bilateral_filter_kernel<<<grid_dim, block_dim, smem_size>>>(
        d_src.data().get(), d_guide.data().get(), d_dst.data().get(), ksize, kernel_space.data().get(),
        kernel_color_table.data().get(), width_, height_);
    CUDASafeCall();
}

CudaBilateralTextureFilter::CudaBilateralTextureFilter(
    const int width,
    const int height,
    const int ksize,
    const int nitr)
{
    impl_ = new CudaBilateralTextureFilter::Impl(width, height, ksize, nitr);
}

CudaBilateralTextureFilter::~CudaBilateralTextureFilter() {
    delete impl_;
}

void CudaBilateralTextureFilter::execute(
    const ElemType* const d_src,
    ElemType* const d_dst)
{
    impl_->execute(d_src, d_dst);
    cudaDeviceSynchronize();
}
