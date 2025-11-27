#ifndef CLS_HPP__
#define CLS_HPP__

#include "NvInferVersion.h"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "trt/infer.hpp"
#include "kernels/kernel_warp.hpp"
#include <memory>
#include "common/trt_tensor.hpp"


#if NV_TENSORRT_MAJOR >= 10
#include "common/tensorrt.hpp"
namespace TensorRT = TensorRT10;
#else
#include "common/tensorrt8.hpp"
namespace TensorRT = TensorRT8;
#endif


namespace Cls
{

class ClsModelImpl : public InferBase
{

public:
    ClsModelImpl() = default;
    // virtual ~ClsModelImpl() = default;
    std::vector<std::shared_ptr<tensor::Memory<unsigned char>>> preprocess_buffers_;
    std::vector<std::shared_ptr<tensor::Memory<float>>> affine_matrixs_;
    tensor::Memory<float> input_buffer_, output_array_;
    tensor::Memory<int> classes_indices_;
    int num_classes_ = 0;

    // std::vector<std::string> class_names_;
    std::shared_ptr<TensorRT::Engine> trt_;

    int network_input_width_, network_input_height_;
    norm_image::Norm normalize_;

    bool isdynamic_model_ = false;
    int max_batch_size_ = 1;
    // int device_id_  = 0;
    float confidence_threshold_;

    bool load(const std::string &engine_file,
        int gpu_id,
        int max_batch_size);

    void adjust_memory(int batch_size)
    {
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        input_buffer_.cpu(batch_size * input_numel);
        output_array_.gpu(batch_size * num_classes_);
        output_array_.cpu(batch_size * num_classes_);
        classes_indices_.cpu(batch_size);
        classes_indices_.gpu(batch_size);
        if ((int)preprocess_buffers_.size() < (size_t)batch_size) {
            for (int i=preprocess_buffers_.size(); i< (size_t)batch_size; ++i){
                preprocess_buffers_.push_back(std::make_shared<tensor::Memory<unsigned char>>());
                affine_matrixs_.push_back(std::make_shared<tensor::Memory<float>>());
                // 分配仿射矩阵苏需要的空间
                affine_matrixs_[i]->gpu(6);
                affine_matrixs_[i]->cpu(6);
            }
        }
    }

    void preprocess(int ibatch, const tensor::Image &image,
        std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer_,
        affine::CropResizeMatrix &affine,
        void *stream = nullptr)
    {
        int crop_size = std::min(image.width, image.height);
        int start_x   = (image.width  - crop_size) / 2;
        int start_y   = (image.height - crop_size) / 2;
        affine.compute(std::make_tuple(crop_size, crop_size),
                       std::make_tuple(network_input_width_, network_input_height_),
                       std::make_tuple(start_x, start_y)); 
        size_t input_numel = network_input_height_ * network_input_width_ * 3;
        float *input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image = image.width * image.height * 3;
        uint8_t *image_device= preprocess_buffer_->gpu(size_image);
        uint8_t *image_host  = preprocess_buffer_->cpu(size_image);

        float *affine_metrix_device = affine_matrixs_[ibatch]->gpu();
        float *affine_metrix_host   = affine_matrixs_[ibatch]->cpu();

        cudaStream_t stream_ = this->get_stream((cudaStream_t)stream);
        memcpy(image_host, image.bgrptr, size_image);
        // memcpy(image_cpu, image.bgrptr, size_image);
        memcpy(affine_metrix_host, affine.d2i, sizeof(affine.d2i));

        checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
        checkRuntime(cudaMemcpyAsync(affine_metrix_device, affine_metrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));
        warp_affine_bilinear_and_normalize_plane(
            image_device,
            image.width * 3,
            image.width,
            image.height,
            input_device,
            network_input_width_,
            network_input_height_,
            affine_metrix_device,
            114,
            normalize_,
            stream_
        );
    }

    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr);
};

std::shared_ptr<InferBase> load_cls(const std::string &engine_file,
                                    int gpu_id,
                                    int max_batch_size);
};
#endif