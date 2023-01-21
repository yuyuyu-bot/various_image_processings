#ifndef TEST_RANDOM_ARRAY_HPP
#define TEST_RANDOM_ARRAY_HPP

#include <memory>
#include <random>

namespace {

template <typename ElemType>
inline auto random_array(const std::size_t len) {
    std::random_device seed_gen;
    std::mt19937 rand_gen(seed_gen());

    auto array = std::make_unique<ElemType[]>(len);
    for (int i = 0; i < len; i++) {
        array[i] = rand_gen() % std::numeric_limits<ElemType>::max();
    }

    return array;
}

template <>
inline auto random_array<float>(const std::size_t len) {
    std::random_device seed_gen;
    std::mt19937 rand_gen(seed_gen());

    auto array = std::make_unique<float[]>(len);
    for (int i = 0; i < len; i++) {
        array[i] = static_cast<float>(rand_gen()) / std::numeric_limits<std::uint32_t>::max();
    }

    return array;
}

} // anonymous namespace

#endif // TEST_RANDOM_ARRAY_HPP
