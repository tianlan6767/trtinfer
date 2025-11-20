#include "common/affine.hpp"
#include "common/check.hpp"
#include "kernels/kernel_warp.hpp"
#include "trt/cls/cls.hpp"


namespace Cls
{

// static void classifer_softmax(float *predict, int length, int *max_index, cudaStream_t stream)
// {
//     int block_size = 256;
//     checkKernel(softmax<<<1, block_size, block_size * sizeof(float), stream>>>(predict, length, max_index));
// }

bool ClsModelImpl::load(const std::string &engine_file,
                    int gpu_id,
                    int max_batch_size)
{
    trt_ = TensorRT::load(engine_file);
    device_id_ = gpu_id;
    if (trt_ == nullptr)
    {
        std::cerr << "Error load TensorRT engine: " << engine_file << std::endl;
        return false;
    }
    trt_->print("ClsModel");
    isdynamic_model_ = trt_->has_dynamic_dim();
    max_batch_size_ = max_batch_size;
    auto input_dims = trt_->static_dims(0);
    network_input_height_ = input_dims[2];
    network_input_width_  = input_dims[3];
    normalize_ = norm_image::Norm::alpha_beta(1 / 255.0f, 0.0f, norm_image::ChannelType::SwapRB);
    num_classes_ = trt_->static_dims(1)[1];
    return true;
}

InferResult ClsModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
{
    // 推理图片的数量
    int num_image = inputs.size();
    assert(num_image < max_batch_size_);
    auto input_dims = trt_->static_dims(0);
    int infer_batch_size = input_dims[0];
    if (infer_batch_size !=num_image)
    {
        if (isdynamic_model_)
        {
            infer_batch_size = num_image;
            input_dims[0] = num_image;
            if(!trt_->set_run_dims(0, input_dims)){
                printf("Error set dynamic input dims for cls model\n");
                return {};
            }
        }
        else
        {
            if (infer_batch_size < num_image)
            {
                printf("Error cls model batch size not enough, model batch size is %d, but infer batch size is %d\n",
                       infer_batch_size, num_image);
                return {};
            }
        }
    }
    // 分配存储空间
    adjust_memory(infer_batch_size);

    std::vector<affine::ResizeMatrix> affine_matrixs(infer_batch_size);

    // 预处理
    cudaStream_t stream_ = (cudaStream_t)stream;
    for (int i=0; i < num_image; ++i)
    {
        preprocess(i,
                   tensor::Image(inputs[i].data, inputs[i].cols, inputs[i].rows),
                   preprocess_buffers_[i],
                   affine_matrixs[i],
                   stream);
    }

    float *output_array_device = output_array_.gpu();

#if NV_TENSORRT_MAJOR >= 10
    std::unordered_map<std::string, const void *> bindings = {
        {"images", input_buffer_.gpu()},
        {"output0", output_array_device}
    };
    if(!trt_->forward(bindings, stream_)){
        std::cerr << "Error forward for cls model\n";
        return {};
    }
#else
    std::vector<void *> bindings(input_buffer_.gpu(), output_array_device);
    if(!trt_->forward(bindings, stream_)){
        std::cerr << "Error forward for cls model\n";
        return {};
    }
#endif
    for (int ib = 0; ib < infer_batch_size; ++ib)
    {
        float *output_array_device = output_array_.gpu() + ib * num_classes_;
        int *classes_indices_device = classes_indices_.gpu() + ib;
        classifer_softmax(output_array_device,
                          num_classes_,
                          classes_indices_device,
                          stream_);
    }

    checkRuntime(cudaMemcpyAsync(output_array_.cpu(),
                                 output_array_.gpu(),
                                 output_array_.gpu_bytes(),
                                 cudaMemcpyDeviceToHost,
                                 stream_));

    checkRuntime(cudaMemcpyAsync(classes_indices_.cpu(),
                                 classes_indices_.gpu(),
                                 classes_indices_.gpu_bytes(),
                                 cudaMemcpyDeviceToHost,
                                 stream_));
    checkRuntime(cudaStreamSynchronize(stream_));
    std::vector<object::ClsAttribute> arrout;
    arrout.reserve(num_image);  // 预分配空间，不改变 size
    for(int ib=0; ib < num_image; ++ib)
    {
        float *output_array_cpu = output_array_.cpu() + ib * num_classes_;
        int *max_index = classes_indices_.cpu() + ib;
        int index = *max_index;
        float max_score = output_array_cpu[index];
        // arrout[ib] = object::ClsAttribute(max_score, index);
        arrout.emplace_back(max_score, index);
    }
    return arrout;
}

static InferBase *loadraw(const std::string &engine_file,
                          int gpu_id,
                          int max_batch_size)
{
    ClsModelImpl *impl = new ClsModelImpl();
    if(!impl->load(engine_file, gpu_id, max_batch_size)){
        delete impl;
        impl =nullptr;
    }
    return impl;
}

std::shared_ptr<InferBase> load_cls(const std::string &engine_file,
                                    int gpu_id,
                                    int max_batch_size)
{
    try
    {
        checkRuntime(cudaSetDevice(gpu_id));
        return std::shared_ptr<ClsModelImpl>((
            ClsModelImpl *)loadraw(engine_file, gpu_id, max_batch_size));
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error load cls model:" << ex.what() << std::endl;
        return nullptr;
    }
}


} // end namespace Cls