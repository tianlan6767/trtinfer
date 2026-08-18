#include "trt/deeplab/deeplabv3_sahi.hpp"

#include "common/check.hpp"
#include "common/object.hpp"
#include "kernels/kernel_warp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace deeplab
{
namespace
{

void softmax_letterbox_to_slice(const float *logits, int num_classes, int net_h, int net_w, int slice_h, int slice_w,
                                std::vector<cv::Mat> &out)
{
    const int spatial = net_h * net_w;
    cv::Rect roi      = integer_letterbox_roi(slice_w, slice_h, net_w, net_h);
    std::vector<cv::Mat> sm(num_classes);
    for (int c = 0; c < num_classes; ++c)
        sm[c] = cv::Mat(net_h, net_w, CV_32FC1);

    for (int y = 0; y < net_h; ++y)
    {
        for (int x = 0; x < net_w; ++x)
        {
            const int s = y * net_w + x;
            float maxv  = logits[s];
            for (int c = 1; c < num_classes; ++c)
                maxv = std::max(maxv, logits[c * spatial + s]);
            float sum = 0.f;
            for (int c = 0; c < num_classes; ++c)
            {
                const float e = std::exp(logits[c * spatial + s] - maxv);
                sm[c].at<float>(y, x) = e;
                sum += e;
            }
            const float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
            for (int c = 0; c < num_classes; ++c)
                sm[c].at<float>(y, x) *= inv;
        }
    }

    out.resize(num_classes);
    const cv::Size dst(slice_w, slice_h);
    for (int c = 0; c < num_classes; ++c)
        cv::resize(sm[c](roi), out[c], dst, 0, 0, cv::INTER_LINEAR);
}

} // namespace

void DeeplabV3SahiModelImpl::adjust_memory(int batch_size)
{
    const size_t input_numel = (size_t)network_input_height_ * network_input_width_ * 3;
    const size_t spatial     = (size_t)network_input_height_ * network_input_width_;
    input_buffer_.gpu(batch_size * input_numel);
    logits_.gpu(batch_size * (size_t)num_classes_ * spatial);
    logits_.cpu(batch_size * (size_t)num_classes_ * spatial);
    affine_matrix_.gpu(6);
    affine_matrix_.cpu(6);
}

bool DeeplabV3SahiModelImpl::load(const std::string &engine_file, const std::vector<std::string> &names,
                                  float confidence_threshold, int gpu_id, int max_batch_size, bool auto_slice,
                                  int slice_width, int slice_height, double slice_horizontal_ratio,
                                  double slice_vertical_ratio, SegOutput seg_output)
{
    device_id_ = gpu_id;
    auto device_guard = this->get_device();
    trt_ = TensorRT::load(engine_file);
    if (trt_ == nullptr)
    {
        std::cerr << "Failed to load TensorRT engine: " << engine_file << std::endl;
        return false;
    }
    trt_->print("DeepLabV3-SAHI");

    slice_                    = std::make_shared<slice::SliceImage>();
    class_names_              = names;
    confidence_threshold_     = confidence_threshold;
    max_batch_size_           = std::max(1, max_batch_size);
    auto_slice_               = auto_slice;
    slice_width_              = slice_width;
    slice_height_             = slice_height;
    slice_horizontal_ratio_   = slice_horizontal_ratio;
    slice_vertical_ratio_     = slice_vertical_ratio;
    seg_output_               = seg_output;
    isdynamic_model_          = trt_->has_dynamic_dim();

    int input_index = 0, output_index = 1;
    for (int i = 0; i < trt_->num_bindings(); ++i)
    {
        if (trt_->is_input(i))
            input_index = i;
        else
            output_index = i;
    }
    auto input_dims       = trt_->static_dims(input_index);
    auto output_dims      = trt_->static_dims(output_index);
    network_input_height_ = input_dims.size() >= 4 ? input_dims[2] : 0;
    network_input_width_  = input_dims.size() >= 4 ? input_dims[3] : 0;
    num_classes_          = output_dims.size() >= 4 ? output_dims[1] : 0;
    if (network_input_width_ <= 0 || network_input_height_ <= 0 || num_classes_ <= 0)
        return false;
    while ((int)class_names_.size() < num_classes_)
        class_names_.push_back("class_" + std::to_string((int)class_names_.size()));

    normalize_ = norm_image::Norm::alpha_beta(1 / 255.0f, 0.0f, norm_image::ChannelType::SwapRB);
    return true;
}

void DeeplabV3SahiModelImpl::preprocess_slice(int ibatch, int slice_index, const affine::LetterBoxMatrix &affine,
                                              void *stream)
{
    const size_t input_numel = (size_t)network_input_height_ * network_input_width_ * 3;
    const size_t size_image  = (size_t)slice_->slice_width_ * slice_->slice_height_ * 3;
    float *input_device      = input_buffer_.gpu() + ibatch * input_numel;
    uint8_t *image_device    = slice_->output_images_.gpu() + slice_index * size_image;
    cudaStream_t stream_     = this->get_stream((cudaStream_t)stream);

    float *affine_host   = affine_matrix_.cpu();
    float *affine_device = affine_matrix_.gpu();
    memcpy(affine_host, affine.d2i, sizeof(affine.d2i));
    checkRuntime(cudaMemcpyAsync(affine_device, affine_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream_));

    warp_affine_bilinear_and_normalize_plane(image_device, slice_->slice_width_ * 3, slice_->slice_width_,
                                             slice_->slice_height_, input_device, network_input_width_,
                                             network_input_height_, affine_device, 128, normalize_, stream_);
}

InferResult DeeplabV3SahiModelImpl::forwards(const std::vector<cv::Mat> &inputs, void *stream)
{
    if (inputs.size() != 1)
    {
        printf("DeepLabV3-SAHI only supports batch=1 full image.\n");
        return {};
    }
    auto device_guard    = this->get_device();
    cudaStream_t stream_ = this->get_stream((cudaStream_t)stream);
    const cv::Mat &image = inputs[0];

    if (auto_slice_)
        slice_->autoSlice(tensor::Image(image.data, image.cols, image.rows), stream_);
    else
        slice_->slice(tensor::Image(image.data, image.cols, image.rows), slice_width_, slice_height_,
                      slice_horizontal_ratio_, slice_vertical_ratio_, stream_);

    const int num_slices = slice_->slice_num_h_ * slice_->slice_num_v_;
    if (num_slices <= 0)
        return {};

    auto input_dims  = trt_->static_dims(0);
    int engine_batch = input_dims[0];
    int chunk        = 1;
    if (isdynamic_model_)
    {
        chunk = std::min(num_slices, max_batch_size_);
        if (engine_batch > 0)
            chunk = std::min(chunk, engine_batch);
    }
    else
        chunk = std::max(1, engine_batch);
    adjust_memory(chunk);

    affine::LetterBoxMatrix affine;
    set_integer_letterbox(affine, slice_->slice_width_, slice_->slice_height_, network_input_width_,
                          network_input_height_);

    const int H = image.rows, W = image.cols, C = num_classes_;
    std::vector<float> acc((size_t)C * H * W, 0.f);
    std::vector<float> cnt((size_t)H * W, 0.f);
    const size_t spatial_net = (size_t)network_input_height_ * network_input_width_;

    for (int start = 0; start < num_slices; start += chunk)
    {
        const int n = std::min(chunk, num_slices - start);
        if (isdynamic_model_)
        {
            input_dims[0] = n;
            if (!trt_->set_run_dims(0, input_dims))
            {
                printf("Fail to set run dims\n");
                return {};
            }
        }
        const int run_n = isdynamic_model_ ? n : chunk;
        for (int i = 0; i < run_n; ++i)
        {
            const int si = (start + i < num_slices) ? (start + i) : (num_slices - 1);
            preprocess_slice(i, si, affine, stream_);
        }

        float *logits_device = logits_.gpu();
#if NV_TENSORRT_MAJOR >= 10
        std::unordered_map<std::string, const void *> bindings = {
            {input_name_, input_buffer_.gpu()},
            {output_name_, logits_device},
        };
        if (!trt_->forward(bindings, stream_))
        {
            printf("Failed to run DeepLabV3-SAHI inference.\n");
            return {};
        }
#else
        std::vector<void *> bindings{input_buffer_.gpu(), logits_device};
        if (!trt_->forward(bindings, stream_))
        {
            printf("Failed to run DeepLabV3-SAHI inference.\n");
            return {};
        }
#endif
        checkRuntime(cudaMemcpyAsync(logits_.cpu(), logits_.gpu(), run_n * C * spatial_net * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        const int *starts = slice_->slice_start_point_.cpu();
        for (int i = 0; i < n; ++i)
        {
            const int si = start + i;
            const int sx = starts[si * 2];
            const int sy = starts[si * 2 + 1];
            std::vector<cv::Mat> slice_prob;
            softmax_letterbox_to_slice(logits_.cpu() + i * C * spatial_net, C, network_input_height_,
                                       network_input_width_, slice_->slice_height_, slice_->slice_width_, slice_prob);
            const int sh = slice_->slice_height_;
            const int sw = slice_->slice_width_;
            for (int y = 0; y < sh; ++y)
            {
                const int gy = sy + y;
                if (gy < 0 || gy >= H)
                    continue;
                for (int x = 0; x < sw; ++x)
                {
                    const int gx = sx + x;
                    if (gx < 0 || gx >= W)
                        continue;
                    const int g = gy * W + gx;
                    cnt[g] += 1.f;
                    for (int c = 0; c < C; ++c)
                        acc[(size_t)c * H * W + g] += slice_prob[c].at<float>(y, x);
                }
            }
        }
    }

    cv::Mat class_img(H, W, CV_8UC1);
    cv::Mat score_img(H, W, CV_32FC1);
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            const int g  = y * W + x;
            const float n = std::max(cnt[g], 1e-6f);
            int best      = 0;
            float bestv   = acc[g] / n;
            for (int c = 1; c < C; ++c)
            {
                const float v = acc[(size_t)c * H * W + g] / n;
                if (v > bestv)
                {
                    bestv = v;
                    best  = c;
                }
            }
            class_img.at<uchar>(y, x)  = (uchar)best;
            score_img.at<float>(y, x)  = bestv;
        }
    }

    std::vector<object::SegmentationResultArray> arrout(1);
    decode_class_map(class_img, score_img, num_classes_, confidence_threshold_, class_names_, seg_output_, arrout[0]);
    return arrout;
}

static InferBase *loadraw(const std::string &engine_file, const std::vector<std::string> &names,
                          float confidence_threshold, int gpu_id, int max_batch_size, bool auto_slice, int slice_width,
                          int slice_height, double slice_horizontal_ratio, double slice_vertical_ratio,
                          SegOutput seg_output)
{
    auto *impl = new DeeplabV3SahiModelImpl();
    if (!impl->load(engine_file, names, confidence_threshold, gpu_id, max_batch_size, auto_slice, slice_width,
                    slice_height, slice_horizontal_ratio, slice_vertical_ratio, seg_output))
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

std::shared_ptr<InferBase> load_deeplabv3_sahi(const std::string &engine_file, const std::vector<std::string> &names,
                                               int gpu_id, float confidence_threshold, int max_batch_size,
                                               bool auto_slice, int slice_width, int slice_height,
                                               double slice_horizontal_ratio, double slice_vertical_ratio,
                                               SegOutput seg_output)
{
    try
    {
        return std::shared_ptr<DeeplabV3SahiModelImpl>((DeeplabV3SahiModelImpl *)loadraw(
            engine_file, names, confidence_threshold, gpu_id, max_batch_size, auto_slice, slice_width, slice_height,
            slice_horizontal_ratio, slice_vertical_ratio, seg_output));
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error loading DeepLabV3-SAHI: " << ex.what() << std::endl;
        return nullptr;
    }
}

} // namespace deeplab
