#include "trt/deeplab/deeplabv3.hpp"

#include "common/affine.hpp"
#include "common/check.hpp"
#include "common/image.hpp"
#include "common/object.hpp"
#include "kernels/kernel_warp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace deeplab
{

void set_integer_letterbox(affine::LetterBoxMatrix &affine, int src_w, int src_h, int dst_w, int dst_h)
{
    const float scale = std::min(dst_w / (float)src_w, dst_h / (float)src_h);
    const int content_w = std::max(1, (int)(src_w * scale));
    const int content_h = std::max(1, (int)(src_h * scale));
    const int pad_x     = (dst_w - content_w) / 2;
    const int pad_y     = (dst_h - content_h) / 2;
    const float sx      = src_w / (float)content_w;
    const float sy      = src_h / (float)content_h;
    affine.d2i[0] = sx;
    affine.d2i[1] = 0.f;
    affine.d2i[2] = -pad_x * sx;
    affine.d2i[3] = 0.f;
    affine.d2i[4] = sy;
    affine.d2i[5] = -pad_y * sy;
    affine.i2d[0] = 1.f / sx;
    affine.i2d[1] = 0.f;
    affine.i2d[2] = (float)pad_x;
    affine.i2d[3] = 0.f;
    affine.i2d[4] = 1.f / sy;
    affine.i2d[5] = (float)pad_y;
}

cv::Rect integer_letterbox_roi(int src_w, int src_h, int net_w, int net_h)
{
    const float scale = std::min(net_w / (float)src_w, net_h / (float)src_h);
    const int content_w = std::max(1, (int)(src_w * scale));
    const int content_h = std::max(1, (int)(src_h * scale));
    const int pad_x     = (net_w - content_w) / 2;
    const int pad_y     = (net_h - content_h) / 2;
    cv::Rect roi(pad_x, pad_y, content_w, content_h);
    return roi & cv::Rect(0, 0, net_w, net_h);
}

void decode_class_map(const cv::Mat &class_img, const cv::Mat &score_img, int num_classes, float confidence_threshold,
                      const std::vector<std::string> &class_names, SegOutput seg_output,
                      std::vector<object::SegmentationInstance> &output)
{
    if (seg_output == SegOutput::CLASS_MAP)
    {
        cv::Mat src = class_img.isContinuous() ? class_img : class_img.clone();
        auto seg    = std::make_shared<object::SegmentMap>(src.cols, src.rows);
        memcpy(seg->data, src.data, (size_t)src.cols * src.rows);
        object::Box box(0.f, 0.f, (float)src.cols, (float)src.rows, 1.f, -1, "class_map");
        output.emplace_back(box, std::move(seg));
        return;
    }

    for (int c = 1; c < num_classes; ++c)
    {
        cv::Mat bin      = (class_img == c);
        cv::Mat bin_keep = bin.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        const std::string &name = class_names[(size_t)c];
        for (const auto &contour : contours)
        {
            if (contour.size() < 3)
                continue;
            cv::Rect bbox = cv::boundingRect(contour);
            if (bbox.width < 3 || bbox.height < 3)
                continue;
            cv::Mat roi_bin   = bin_keep(bbox);
            cv::Mat roi_score = score_img(bbox);
            double mean_score = cv::mean(roi_score, roi_bin)[0];
            if (mean_score < confidence_threshold)
                continue;
            if (!roi_bin.isContinuous())
                roi_bin = roi_bin.clone();
            auto seg = std::make_shared<object::SegmentMap>(bbox.width, bbox.height);
            memcpy(seg->data, roi_bin.data, (size_t)bbox.width * bbox.height);
            object::Box box((float)bbox.x, (float)bbox.y, (float)(bbox.x + bbox.width), (float)(bbox.y + bbox.height),
                            (float)mean_score, c, name);
            output.emplace_back(box, std::move(seg));
        }
    }
}

void DeeplabV3ModelImpl::adjust_memory(int batch_size)
{
    const size_t input_numel = (size_t)network_input_height_ * network_input_width_ * 3;
    const size_t spatial     = (size_t)network_input_height_ * network_input_width_;
    input_buffer_.gpu(batch_size * input_numel);
    logits_.gpu(batch_size * (size_t)num_classes_ * spatial);
    class_ids_.gpu(batch_size * spatial);
    class_ids_.cpu(batch_size * spatial);
    scores_.gpu(batch_size * spatial);
    scores_.cpu(batch_size * spatial);

    if ((int)preprocess_buffers_.size() < batch_size)
    {
        for (int i = (int)preprocess_buffers_.size(); i < batch_size; ++i)
        {
            preprocess_buffers_.push_back(std::make_shared<tensor::Memory<unsigned char>>());
            affine_matrixs_.push_back(std::make_shared<tensor::Memory<float>>());
            inverse_affine_matrixs_.push_back(std::make_shared<tensor::Memory<float>>());
            affine_matrixs_[i]->gpu(6);
            affine_matrixs_[i]->cpu(6);
            inverse_affine_matrixs_[i]->gpu(6);
            inverse_affine_matrixs_[i]->cpu(6);
        }
    }
}

bool DeeplabV3ModelImpl::load(const std::string &engine_file, const std::vector<std::string> &names,
                              float confidence_threshold, int gpu_id, int max_batch_size, SegOutput seg_output)
{
    device_id_ = gpu_id;
    auto device_guard = this->get_device();
    trt_ = TensorRT::load(engine_file);
    if (trt_ == nullptr)
    {
        std::cerr << "Failed to load TensorRT engine: " << engine_file << std::endl;
        return false;
    }
    trt_->print("DeepLabV3");

    class_names_            = names;
    confidence_threshold_   = confidence_threshold;
    max_batch_size_         = max_batch_size;
    seg_output_             = seg_output;
    isdynamic_model_        = trt_->has_dynamic_dim();

    int input_index  = 0;
    int output_index = 1;
    for (int i = 0; i < trt_->num_bindings(); ++i)
    {
        if (trt_->is_input(i))
            input_index = i;
        else
            output_index = i;
    }
    auto input_dims        = trt_->static_dims(input_index);
    output_dims_           = trt_->static_dims(output_index);
    network_input_height_  = input_dims.size() >= 4 ? input_dims[2] : 0;
    network_input_width_   = input_dims.size() >= 4 ? input_dims[3] : 0;
    num_classes_           = output_dims_.size() >= 4 ? output_dims_[1] : 0;
    if (network_input_width_ <= 0 || network_input_height_ <= 0 || num_classes_ <= 0 || num_classes_ > 255)
    {
        std::cerr << "Unexpected DeepLabV3 IO dims\n";
        return false;
    }
    while ((int)class_names_.size() < num_classes_)
        class_names_.push_back("class_" + std::to_string((int)class_names_.size()));

    // bubbliiiing DeepLab：BGR->RGB，像素 /255，letterbox 填 128
    normalize_ = norm_image::Norm::alpha_beta(1 / 255.0f, 0.0f, norm_image::ChannelType::SwapRB);
    return true;
}

void DeeplabV3ModelImpl::preprocess(int ibatch, const tensor::Image &image,
                                    std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer,
                                    affine::LetterBoxMatrix &affine, void *stream)
{
    set_integer_letterbox(affine, image.width, image.height, network_input_width_, network_input_height_);
    const size_t input_numel = (size_t)network_input_height_ * network_input_width_ * 3;
    float *input_device      = input_buffer_.gpu() + ibatch * input_numel;
    const size_t size_image  = (size_t)image.width * image.height * 3;

    uint8_t *image_device = preprocess_buffer->gpu(size_image);
    uint8_t *image_host   = preprocess_buffer->cpu(size_image);

    float *affine_matrix_device = affine_matrixs_[ibatch]->gpu();
    float *affine_matrix_host   = affine_matrixs_[ibatch]->cpu();
    float *i2d_device           = inverse_affine_matrixs_[ibatch]->gpu();
    float *i2d_host             = inverse_affine_matrixs_[ibatch]->cpu();

    cudaStream_t stream_ = this->get_stream((cudaStream_t)stream);
    memcpy(image_host, image.bgrptr, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    memcpy(i2d_host, affine.i2d, sizeof(affine.i2d));
    checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
    checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice,
                                 stream_));
    checkRuntime(cudaMemcpyAsync(i2d_device, i2d_host, sizeof(affine.i2d), cudaMemcpyHostToDevice, stream_));

    warp_affine_bilinear_and_normalize_plane(image_device, image.width * 3, image.width, image.height, input_device,
                                             network_input_width_, network_input_height_, affine_matrix_device, 128,
                                             normalize_, stream_);
}

void DeeplabV3ModelImpl::decode_segment(int ib, const cv::Mat &input,
                                        std::vector<object::SegmentationInstance> &output)
{
    const int nw = network_input_width_;
    const int nh = network_input_height_;
    const int ow = input.cols;
    const int oh = input.rows;
    const size_t spatial = (size_t)nw * nh;
    unsigned char *class_host = class_ids_.cpu() + ib * spatial;
    float *score_host         = scores_.cpu() + ib * spatial;

    cv::Mat class_net(nh, nw, CV_8UC1, class_host);
    cv::Mat score_net(nh, nw, CV_32FC1, score_host);
    const cv::Rect roi = integer_letterbox_roi(ow, oh, nw, nh);
    cv::Mat class_img;
    cv::resize(class_net(roi), class_img, input.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat score_img;
    if (seg_output_ != SegOutput::CLASS_MAP)
        cv::resize(score_net(roi), score_img, input.size(), 0, 0, cv::INTER_LINEAR);
    decode_class_map(class_img, score_img, num_classes_, (float)confidence_threshold_, class_names_, seg_output_,
                     output);
}

InferResult DeeplabV3ModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
{
    auto device_guard = this->get_device();
    const int num_image = (int)inputs.size();
    if (num_image <= 0 || num_image > max_batch_size_)
        return {};

    auto input_dims      = trt_->static_dims(0);
    int infer_batch_size = input_dims[0];
    if (infer_batch_size != num_image)
    {
        if (isdynamic_model_)
        {
            infer_batch_size = num_image;
            input_dims[0]    = num_image;
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

    adjust_memory(infer_batch_size);
    std::vector<affine::LetterBoxMatrix> affine_matrices(infer_batch_size);
    cudaStream_t stream_ = this->get_stream((cudaStream_t)stream);
    for (int i = 0; i < num_image; ++i)
    {
        preprocess(i, tensor::Image(inputs[i].data, inputs[i].cols, inputs[i].rows), preprocess_buffers_[i],
                   affine_matrices[i], stream_);
    }

    float *logits_device = logits_.gpu();
#if NV_TENSORRT_MAJOR >= 10
    std::unordered_map<std::string, const void *> bindings = {
        {input_name_, input_buffer_.gpu()},
        {output_name_, logits_device},
    };
    if (!trt_->forward(bindings, stream_))
    {
        printf("Failed to run DeepLabV3 inference.\n");
        return {};
    }
#else
    std::vector<void *> bindings{input_buffer_.gpu(), logits_device};
    if (!trt_->forward(bindings, stream_))
    {
        printf("Failed to run DeepLabV3 inference.\n");
        return {};
    }
#endif

    semantic_argmax(logits_device, class_ids_.gpu(), scores_.gpu(), infer_batch_size, num_classes_,
                    network_input_height_, network_input_width_, stream_);
    checkRuntime(cudaMemcpyAsync(class_ids_.cpu(), class_ids_.gpu(), class_ids_.gpu_bytes(), cudaMemcpyDeviceToHost,
                                 stream_));
    if (seg_output_ != SegOutput::CLASS_MAP)
        checkRuntime(cudaMemcpyAsync(scores_.cpu(), scores_.gpu(), scores_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
    checkRuntime(cudaStreamSynchronize(stream_));

    std::vector<object::SegmentationResultArray> arrout(num_image);
    for (int ib = 0; ib < num_image; ++ib)
        decode_segment(ib, inputs[ib], arrout[ib]);
    return arrout;
}

static InferBase *loadraw(const std::string &engine_file, const std::vector<std::string> &names,
                          float confidence_threshold, int gpu_id, int max_batch_size, SegOutput seg_output)
{
    auto *impl = new DeeplabV3ModelImpl();
    if (!impl->load(engine_file, names, confidence_threshold, gpu_id, max_batch_size, seg_output))
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

std::shared_ptr<InferBase> load_deeplabv3(const std::string &engine_file, const std::vector<std::string> &names,
                                          int gpu_id, float confidence_threshold, int max_batch_size,
                                          SegOutput seg_output)
{
    try
    {
        return std::shared_ptr<DeeplabV3ModelImpl>(
            (DeeplabV3ModelImpl *)loadraw(engine_file, names, confidence_threshold, gpu_id, max_batch_size, seg_output));
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error loading DeepLabV3: " << ex.what() << std::endl;
        return nullptr;
    }
}

} // namespace deeplab
