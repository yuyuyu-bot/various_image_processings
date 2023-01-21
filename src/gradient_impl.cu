#include <cstdint>
#include "device_utilities.cuh"
#include "host_utilities.hpp"

#include "cuda/gradient.hpp"

template <typename SrcType>
__device__ void compute_gradient_kernel_core(
    const SrcType* const src,
    SrcType* const shared_buffer,
    float* const dst,
    const int width,
    const int height,
    const int src_ch
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int src_stride = width * src_ch;
    const int dst_stride = width;

    const int smem_width  = blockDim.x + 2;
    const int smem_height = blockDim.y + 2;
    const int smem_stride = smem_width * src_ch;
    const int smem_origin_x = x - tx - 1;
    const int smem_origin_y = y - ty - 1;

    const auto get_s_src_ptr =
        [shared_buffer, src_ch, smem_stride, smem_origin_x, smem_origin_y](const int src_x, const int src_y) {
            const auto s_src_x = src_x - smem_origin_x;
            const auto s_src_y = src_y - smem_origin_y;
            return &shared_buffer[smem_stride * s_src_y + s_src_x * src_ch];
        };

    for (int y_offset = ty; y_offset < smem_height; y_offset += blockDim.y) {
        for (int x_offset = tx; x_offset < smem_width; x_offset += blockDim.x) {
            auto* const s_src_ptr = get_s_src_ptr(smem_origin_x + x_offset, smem_origin_y + y_offset);
            const auto x_clamped = clamp(smem_origin_x + x_offset, 0, width - 1);
            const auto y_clamped = clamp(smem_origin_y + y_offset, 0, height - 1);
            for (int ch = 0; ch < src_ch; ch++) {
                s_src_ptr[ch] = src[src_stride * y_clamped + x_clamped * src_ch + ch];
            }
        }
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    const auto compute_del =
        [src_ch, &get_s_src_ptr, src_stride](const int x0, const int y0, const int x1, const int y1) {
            const auto* const s_img_ptr0 = get_s_src_ptr(x0, y0);
            const auto* const s_img_ptr1 = get_s_src_ptr(x1, y1);
            auto diff = 0.f;
            for (int ch = 0; ch < src_ch; ch++) {
                diff += (s_img_ptr1[ch] - s_img_ptr0[ch]) * (s_img_ptr1[ch] - s_img_ptr0[ch]);
            }
            return diff;
        };

    const auto del_x = compute_del(x - 1, y, x + 1, y);
    const auto del_y = compute_del(x, y - 1, x, y + 1);
    dst[dst_stride * y + x] = sqrtf(del_x + del_y);
}

__global__ void compute_gradient_kernel(
    const std::uint8_t* const src,
    float* const dst,
    const int width,
    const int height,
    const int src_ch
) {
    extern __shared__ std::uint8_t s_src_buffer_u8[];
    compute_gradient_kernel_core(src, s_src_buffer_u8, dst, width, height, src_ch);
}

__global__ void compute_gradient_kernel(
    const float* const src,
    float* const dst,
    const int width,
    const int height,
    const int src_ch
) {
    extern __shared__ float s_src_buffer_f32[];
    compute_gradient_kernel_core(src, s_src_buffer_f32, dst, width, height, src_ch);
}

template <typename SrcType>
void cuda_gradient_impl(
    const SrcType* const d_src,
    float* const d_dst,
    const int width,
    const int height,
    const int src_ch
) {
    const std::uint32_t block_width  = 32u;
    const std::uint32_t block_height = 16u;
    const std::uint32_t grid_width   = (width  + block_width  - 1) / block_width;
    const std::uint32_t grid_height  = (height + block_height - 1) / block_height;

    const dim3 grid_dim (grid_width, grid_height);
    const dim3 block_dim(block_width, block_height);
    const std::uint32_t smem_size = (block_width + 2) * (block_height + 2) * src_ch * sizeof(SrcType);

    compute_gradient_kernel<<<grid_dim, block_dim, smem_size>>>(d_src, d_dst, width, height, src_ch);
    CUDASafeCall();
}

template void cuda_gradient_impl<std::uint8_t>(const std::uint8_t*, float*, int, int, int);
template void cuda_gradient_impl<float>(const float*, float*, int, int, int);
