#ifndef CUDA_GRADIENT_HPP
#define CUDA_GRADIENT_HPP

template <typename SrcType>
void cuda_gradient_impl(
    const SrcType* const d_src,
    float* const d_dst,
    const int width,
    const int height,
    const int src_ch
);

template <typename SrcType>
void cuda_gradient(
    const SrcType* const d_src,
    float* const d_dst,
    const int width,
    const int height,
    const int src_ch = 1
) {
    static_assert(std::is_same_v<SrcType, std::uint8_t> || std::is_same_v<SrcType, float>);
    cuda_gradient_impl(d_src, d_dst, width, height, src_ch);
}

#endif // CUDA_GRADIENT_HPP
