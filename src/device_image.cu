#include <cstdint>
#include <thrust/device_vector.h>
#include "cuda/device_image.hpp"

template <typename ElemType>
class DeviceImage<ElemType>::Impl {
public:
    Impl(const std::size_t len) : data_(len) {}

    void upload(const ElemType* const data) {
        thrust::copy(data, data + data_.size(), data_.begin());
    }

    void download(ElemType* const data) {
        thrust::copy(data_.begin(), data_.end(), data);
    }

    ElemType* get() {
        return data_.data().get();
    }

private:
    thrust::device_vector<ElemType> data_;
};

template <typename ElemType>
DeviceImage<ElemType>::DeviceImage(const int width, const int height, const int channels) {
    impl_ = new DeviceImage<ElemType>::Impl(width * height * channels);
}

template <typename ElemType>
DeviceImage<ElemType>::~DeviceImage() {
    delete impl_;
}

template <typename ElemType>
void DeviceImage<ElemType>::upload(const ElemType* const data) {
    impl_->upload(data);
}

template <typename ElemType>
void DeviceImage<ElemType>::download(ElemType* const data) {
    impl_->download(data);
}

template <typename ElemType>
ElemType* DeviceImage<ElemType>::get() {
    return impl_->get();
}

template class DeviceImage<std::uint8_t>;
template class DeviceImage<float>;
