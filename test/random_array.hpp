#ifndef TEST_RANDOM_ARRAY_HPP
#define TEST_RANDOM_ARRAY_HPP

#include <memory>
#include <random>

namespace {

template <typename ElemType>
inline auto random_array(const std::size_t len, const ElemType max = 255) {
    std::mt19937 rand_gen(42);

    auto array = std::make_unique<ElemType[]>(len);
    for (int i = 0; i < len; i++) {
        array[i] = rand_gen() % max;
    }

    return array;
}

template <>
inline auto random_array<float>(const std::size_t len, const float max) {
    std::mt19937 rand_gen(42);

    auto array = std::make_unique<float[]>(len);
    for (int i = 0; i < len; i++) {
        array[i] = max * static_cast<float>(rand_gen()) / std::numeric_limits<std::uint32_t>::max();
    }

    return array;
}

} // anonymous namespace

#endif // TEST_RANDOM_ARRAY_HPP
