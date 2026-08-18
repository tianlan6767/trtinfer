#ifndef DEEPLABV3_HPP__
#define DEEPLABV3_HPP__

#include "NvInferVersion.h"
#include "common/affine.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "kernels/kernel_warp.hpp"
#include "common/object.hpp"
#include "opencv2/core.hpp"
#include "trt/infer.hpp"
#include <memory>
#include <string>
#include <vector>

#if NV_TENSORRT_MAJOR >= 10
#include "common/tensorrt.hpp"
namespace TensorRT = TensorRT10;
#else
#include "common/tensorrt8.hpp"
namespace TensorRT = TensorRT8;
#endif

namespace deeplab
{

void set_integer_letterbox(affine::LetterBoxMatrix &affine, int src_w, int src_h, int dst_w, int dst_h);
cv::Rect integer_letterbox_roi(int src_w, int src_h, int net_w, int net_h);
void decode_class_map(const cv::Mat &class_img, const cv::Mat &score_img, int num_classes, float confidence_threshold,
                      const std::vector<std::string> &class_names, SegOutput seg_output,
                      std::vector<object::SegmentationInstance> &output);

class DeeplabV3ModelImpl : public InferBase
{
  public:
    tensor::Memory<float> input_buffer_;
    tensor::Memory<float> logits_;
    tensor::Memory<unsigned char> class_ids_;
    tensor::Memory<float> scores_;
    std::vector<std::shared_ptr<tensor::Memory<unsigned char>>> preprocess_buffers_;
    std::vector<std::shared_ptr<tensor::Memory<float>>> affine_matrixs_;
    std::vector<std::shared_ptr<tensor::Memory<float>>> inverse_affine_matrixs_;

    std::shared_ptr<TensorRT::Engine> trt_;
    std::vector<std::string> class_names_;
    std::vector<int> output_dims_;
    std::string input_name_  = "images";
    std::string output_name_ = "output";

    int network_input_width_  = 0;
    int network_input_height_ = 0;
    int num_classes_          = 0;
    int max_batch_size_       = 1;
    float confidence_threshold_ = 0.5f;
    SegOutput seg_output_       = SegOutput::INSTANCES;
    bool isdynamic_model_     = false;
    norm_image::Norm normalize_;

    bool load(const std::string &engine_file, const std::vector<std::string> &names, float confidence_threshold,
              int gpu_id, int max_batch_size, SegOutput seg_output);
    void adjust_memory(int batch_size);
    void preprocess(int ibatch, const tensor::Image &image,
                    std::shared_ptr<tensor::Memory<unsigned char>> preprocess_buffer, affine::LetterBoxMatrix &affine,
                    void *stream = nullptr);

    InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr) override;

  private:
    void decode_segment(int ib, const cv::Mat &input, std::vector<object::SegmentationInstance> &output);
};

std::shared_ptr<InferBase> load_deeplabv3(const std::string &engine_file, const std::vector<std::string> &names,
                                          int gpu_id, float confidence_threshold, int max_batch_size,
                                          SegOutput seg_output = SegOutput::INSTANCES);

} // namespace deeplab

#endif
