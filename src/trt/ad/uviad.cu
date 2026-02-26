#include "common/affine.hpp"
#include "common/check.hpp"
#include "common/image.hpp"
#include "common/object.hpp"
#include "kernels/kernel_warp.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/opencv.hpp"
#include "trt/ad/uviad.hpp"
#include <memory>
// #include "common/trt_tensor.hpp"


static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

namespace AD
{

void UviadModelImpl::adjust_memory(int batch_size)
{
    size_t input_numel = network_input_height_ * network_input_width_ * 3;
    input_buffer_.gpu(batch_size * input_numel);
    input_buffer_.cpu(batch_size * input_numel);
    segment_predict_.gpu(batch_size * segment_head_dims_[1] * segment_head_dims_[2] * segment_head_dims_[3]);
    output_boxarray_.gpu(batch_size * max_image_boxes_ * num_box_element_);
    output_boxarray_.cpu(batch_size * max_image_boxes_ * num_box_element_);

    mask_affine_matrix_.gpu(batch_size * 6);
    mask_affine_matrix_.cpu(batch_size * 6);
}
bool UviadModelImpl::load(const std::string &engine_file,
                            // const std::vector<std::string> &names,
                            float confidence_threshold,
                            int gpu_id,
                            int max_batch_size)
{
    device_id_ = gpu_id;
    auto device_guard = this->get_device(); //保证device正确切换
    trt_ = TensorRT::load(engine_file);
    if (trt_ == nullptr)
    {
        std::cerr << "Failed to load TensorRT engine from file: " << engine_file << std::endl;
        return false;
    }

    trt_->print("UVIAD Model");

    this->confidence_threshold_ = confidence_threshold;
    this->max_batch_size_      = max_batch_size;
    auto input_dims = trt_->static_dims(0); 
    segment_head_dims_ = trt_->static_dims(1);
    network_input_width_  = input_dims[3];
    network_input_height_ = input_dims[2];
    isdynamic_model_ = trt_->has_dynamic_dim();
    normalize_ = norm_image::Norm::alpha_beta(1 / 255.0f, 0.0f, norm_image::ChannelType::SwapRB);
    return true;
}
static InferBase *loadraw(const std::string &engine_file,
                          float confidence_threshold,
                          int gpu_id,
                          int max_batch_size)
{
    UviadModelImpl *impl = new UviadModelImpl();
    if (!impl->load(engine_file, confidence_threshold, gpu_id, max_batch_size))
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}


std::shared_ptr<InferBase> load_uviad(const std::string &engine_file,
                                      int gpu_id = 0,
                                      float confidence_threshold = 0.5f,
                                      int max_batch_size = 1)
{
    try
    {
        return std::shared_ptr<UviadModelImpl>(
            (UviadModelImpl *)
                loadraw(engine_file, confidence_threshold, gpu_id, max_batch_size));
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error loading UVIAD model: " << ex.what() << std::endl;
        return nullptr;
    }
}


void UviadModelImpl::preprocess(int ibatch,
                                const tensor::Image &image,
                                std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer,
                                affine::ResizeMatrix &affine,
                                void *stream)
{
    affine.compute(
        std::make_tuple(image.width, image.height),
        std::make_tuple(network_input_width_, network_input_height_)
    );
    size_t input_numel = network_input_height_ * network_input_width_ * 3;
    float *input_device = input_buffer_.gpu() + ibatch *input_numel;
    size_t size_image = image.width * image.height * 3;

    uint8_t *image_device = preprocess_buffer->gpu(size_image);
    uint8_t *image_host = preprocess_buffer->cpu(size_image);

    float *affine_matrix_device = affine_matrixs_[ibatch]->gpu();
    float *affine_matrix_host = affine_matrixs_[ibatch]->cpu();

    float *inverse_affine_matrix_device = inverse_affine_matrixs_[ibatch]->gpu();
    float *inverse_affine_matrix_host = inverse_affine_matrixs_[ibatch]->cpu();

    cudaStream_t stream_ = this->get_stream((cudaStream_t)stream);
    memcpy(image_host, image.bgrptr, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    memcpy(inverse_affine_matrix_host, affine.i2d, sizeof(affine.i2d));
    checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
    checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));
    checkRuntime(cudaMemcpyAsync(inverse_affine_matrix_device, inverse_affine_matrix_host, sizeof(affine.i2d), cudaMemcpyHostToDevice, stream_));

    warp_affine_bilinear_and_normalize_plane(
        image_device, 
        image.width * 3, 
        image.width, 
        image.height,
        input_device, 
        network_input_width_, 
        network_input_height_,
        affine_matrix_device,
        114,
        normalize_,
        stream_);
}

InferResult UviadModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
{
    auto device_guard = this->get_device(); //保证device正确切换
    int num_image = static_cast<int>(inputs.size());
    assert(num_image <= max_batch_size_);
    // 输入的维度 batch X 3 X H X W
    auto input_dims = trt_->static_dims(0);
    int infer_batch_size = input_dims[0];
    if (infer_batch_size != num_image)
    {
        if (isdynamic_model_)
        {
            infer_batch_size = num_image;
            input_dims[0] = num_image;
            if (!trt_->set_run_dims(0, input_dims))
            {
                printf("Fail to set run dims\n");
                return {};
            }
        }
        else
        {
            printf("When using static shape model, number of images[%d] must be "
                   "equal to the batch size[%d] defined in the engine.\n",
                   num_image, infer_batch_size);
            return {};
        }
    }

    //分配存储空间
    adjust_memory(infer_batch_size);
    std::vector<affine::ResizeMatrix> affine_matrices(infer_batch_size);
    cudaStream_t stream_ = this->get_stream((cudaStream_t)stream);
    for (int i=0; i<num_image; ++i)
    {
        preprocess(i,
                  tensor::Image(inputs[i].data, inputs[i].cols, inputs[i].rows),
                  preprocess_buffers_[i],
                  affine_matrices[i],
                  stream_);
    }
    float *segment_output_device = segment_predict_.gpu();

#if NV_TENSORRT_MAJOR >= 10

    std::unordered_map<std::string, const void *> bindings = {
        {"images", input_buffer_.gpu()},
        {"output0", segment_output_device},
    };
    if (!trt_->forward(bindings, stream_))
    {
        printf("Failed to run inference.\n");
        return {};
    }
#else
    std::vector<void *> bindings{
        input_buffer_.gpu(),
        segment_output_device,
    };
    if (!trt_->forward(bindings, stream_))
    {
        printf("Failed to run inference.\n");
        return {};
    }
#endif
    
    std::vector<object::SegmentationResultArray> arrout(num_image);
    for (int ib=0; ib<num_image; ++ib)
    {   
        float *parray = output_boxarray_.cpu() + ib * max_image_boxes_ * num_box_element_;
        auto input = inputs[ib];
        auto &output = arrout[ib];
        decode_segment(ib, parray, input, output, stream_);
        
    }

    return arrout;
}



void UviadModelImpl::decode_segment(int ib, float *parray, const cv::Mat &input, object::SegmentationResultArray &output, cudaStream_t stream)
{

    cudaStream_t stream_ = this->get_stream((cudaStream_t)stream);

    // int original_width = image.cols;
    // int original_height = image.rows;

    // 整图小图的segment预测结果
    float *mask_head_predict = segment_predict_.gpu();
    float *mask_head_predict_host = segment_predict_.cpu();

    float *mask_head_predict_ib = mask_head_predict + ib * segment_head_dims_[1] * segment_head_dims_[2] * segment_head_dims_[3];
    float *mask_head_predict_host_ib = mask_head_predict_host + ib * segment_head_dims_[1] * segment_head_dims_[2] * segment_head_dims_[3];

    // 整图大图的segment输出结果
    unsigned char *original_mask_out_device = original_segment_cache_.gpu();


    float *i2d = inverse_affine_matrixs_[ib]->cpu();
    
    int bytes_of_mask_out = segment_head_dims_[1] * segment_head_dims_[2] * segment_head_dims_[3];
    
    // 预测的mask_predict经过sigmoid和阈值处理后会写入mask_out_device
    float *mask_out_device = segment_cache_.gpu(bytes_of_mask_out);
    auto mask_out_width = segment_head_dims_[3];
    auto mask_out_height = segment_head_dims_[2];

    cv::Mat mask_head_predict_mat(segment_head_dims_[2], segment_head_dims_[3], CV_32FC1, mask_head_predict_host_ib);
    // 对mask进行归一化和阈值处理
    normalize_and_thres_mask(mask_head_predict_ib, mask_out_device, segment_head_dims_[1] * segment_head_dims_[2] * segment_head_dims_[3], confidence_threshold_, stream_);

    // 计算mask_out的轮廓点
    float *mask_out_host = segment_cache_.cpu(bytes_of_mask_out);
    cv::Mat mask_out_mat(mask_out_height, mask_out_width, CV_8UC1, mask_out_host);  
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_out_mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    int box_idx = 0;

    for (const auto &contour : contours)
    {   
        std::shared_ptr<object::SegmentMap> seg = nullptr;
        if(contour.size() < 3) continue;
        cv::Rect bbox = cv::boundingRect(contour);
        auto roi = mask_head_predict_mat(bbox);
        if (cv::countNonZero(roi) == 0 || bbox.width <= 3 || bbox.height <= 3) continue; // 如果ROI内没有有效的mask，则跳过该框
        cv::Point minloc, maxloc;
        double minval, maxval;
        auto meanVal = cv::mean(roi, roi!=0);
        if (float(meanVal.val[0] / 255.0) < confidence_threshold_) continue; // 如果ROI内的平均置信度低于阈值，则跳过该框
        cv::minMaxLoc(roi, &minval, &maxval, &minloc, &maxloc);
        // 将检测框信息写入输出数组

        
        float *pbox = parray + box_idx * num_box_element_;
        float left, top, right, bottom;
        affine_project(i2d, bbox.x, bbox.y, &left, &top);
        affine_project(i2d, bbox.x + bbox.width, bbox.y + bbox.height, &right, &bottom);

        pbox[0] = left;
        pbox[1] = top;
        pbox[2] = right;
        pbox[3] = bottom;
        pbox[4] = float(maxval / 255.0); // score
        pbox[5] = 0;    // class_id
        pbox[6] = 1.0f; // keep_prob
        pbox[7] = box_idx;
        pbox[8] = ib;
        box_idx++;   
        
        float box_width = bbox.width;
        float box_height = bbox.height;
        float original_box_width = right - left;
        float original_box_height = bottom - top; 
        box_segment_cache_.gpu(bbox.width * bbox.height);    
        original_box_segment_cache_.gpu(original_box_width * original_box_height);
        float *box_segment_device = box_segment_cache_.gpu();
        unsigned char *original_box_mask_out_device = original_box_segment_cache_.gpu();
        // 将mask从网络输入尺寸仿射变换回原图尺寸
        warp_affine_bilinear_single_channel_mask_plane(box_segment_device,
                                                        box_width,
                                                        box_width,
                                                        box_height,
                                                        original_box_mask_out_device,
                                                        original_box_width,
                                                        original_box_height,
                                                        inverse_affine_matrixs_[ib]->gpu(),
                                                        0,
                                                        stream_);
        seg = std::make_shared<object::SegmentMap>(original_box_width, original_box_height);
        unsigned char *original_mask_out_host = seg->data;
        checkRuntime(cudaMemcpyAsync(
            original_mask_out_host,
            original_mask_out_device,
            original_segment_cache_.gpu_bytes(),
            cudaMemcpyDeviceToHost,
            stream_));
        object::Box seg_box(left, top, right, bottom, float(maxval / 255.0), 0);
        object::SegmentationInstance result_object_box(seg_box, seg);
        output.push_back(result_object_box);
    }
}

} // namespace AD