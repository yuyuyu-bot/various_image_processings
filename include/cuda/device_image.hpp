#ifndef CUDA_DEVICE_IMAGE_HPP
#define CUDA_DEVICE_IMAGE_HPP

template <typename ElemType>
class DeviceImage {
public:
    DeviceImage(const int width, const int height, const int channels = 1);
    ~DeviceImage();
    void upload(const ElemType* const data);
    void download(ElemType* const data);
    ElemType* get();

private:
    class Impl;
    Impl* impl_;
};

#endif // CUDA_DEVICE_IMAGE_HPP
