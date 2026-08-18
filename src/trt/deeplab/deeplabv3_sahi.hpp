#ifndef DEEPLABV3_SAHI_HPP__
#define DEEPLABV3_SAHI_HPP__

#include "trt/deeplab/deeplabv3.hpp"
#include "trt/slice/slice.hpp"

namespace deeplab
{

class DeeplabV3SahiModelImpl : public InferBase
{
  public:
    tensor::Memory<float> input_buffer_;
    tensor::Memory<float> logits_;
    tensor::Memory<float> affine_matrix_;
    std::shared_ptr<slice::SliceImage> slice_;
    std::shared_ptr<TensorRT::Engine> trt_;
    std::vector<std::string> class_names_;
    std::string input_name_  = "images";
    std::string output_name_ = "output";
    norm_image::Norm normalize_;

    bool auto_slice_ = false;
    int slice_width_  = 960;
    int slice_height_ = 960;
    double slice_horizontal_ratio_ = 0.1;
    double slice_vertical_ratio_   = 0.1;

    int network_input_width_  = 0;
    int network_input_height_ = 0;
    int num_classes_          = 0;
    int max_batch_size_       = 1;
    float confidence_threshold_ = 0.5f;
    SegOutput seg_output_       = SegOutput::INSTANCES;
    bool isdynamic_model_     = false;

    bool load(const std::string &engine_file, const std::vector<std::string> &names, float confidence_threshold,
              int gpu_id, int max_batch_size, bool auto_slice, int slice_width, int slice_height,
              double slice_horizontal_ratio, double slice_vertical_ratio, SegOutput seg_output);
    void adjust_memory(int batch_size);
    void preprocess_slice(int ibatch, int slice_index, const affine::LetterBoxMatrix &affine, void *stream);
    InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr) override;
};

std::shared_ptr<InferBase> load_deeplabv3_sahi(const std::string &engine_file, const std::vector<std::string> &names,
                                               int gpu_id, float confidence_threshold, int max_batch_size,
                                               bool auto_slice, int slice_width, int slice_height,
                                               double slice_horizontal_ratio, double slice_vertical_ratio,
                                               SegOutput seg_output = SegOutput::INSTANCES);

} // namespace deeplab

#endif
