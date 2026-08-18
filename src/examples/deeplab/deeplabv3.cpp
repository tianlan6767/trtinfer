#include "common/ilogger.hpp"
#include "common/object.hpp"
#include "osd/osd.hpp"
#include "trt/infer.hpp"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

static std::vector<std::string> classes_names = {"_background_", "胶路", "基准面"};

void run_single_deeplabv3(const std::string &filename, cv::Mat &image, std::shared_ptr<InferBase> model_)
{
    std::vector<cv::Mat> inputs = {image};
    auto begin_timer            = iLogger::timestamp_now_float();
    auto det                    = model_->forwards(inputs);
    auto end_timer              = iLogger::timestamp_now_float();
    printf("Inference time: %.2f ms\n", (end_timer - begin_timer));

    auto output_path = "result/deeplabv3-" + std::filesystem::path(filename).stem().string() + ".jpg";
    std::visit(
        [&inputs, &output_path](auto &&result)
        {
            int batch_size = (int)inputs.size();
            using T        = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<T, std::vector<object::SegmentationResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    printf("Batch %d: instances=%zu\n", i, result[i].size());
                    osd_segmentation(inputs[i], result[i]);
                    cv::imwrite(output_path, inputs[i]);
                }
            }
        },
        det);
}

void run_deeplabv3()
{
    const std::string engine =
        "/home/ps/workspace/trt/cvter/workspace/deepv3/logs_tgwy_v3/best_epoch_weights.trtmodel";
    const std::string image_path =
        "/home/ps/workspace/trt/cvter/workspace/deepv3/datasets/tgwy/VOC2007/JPEGImages/101_PMST_反射率图_crop.jpg";

    std::shared_ptr<InferBase> model_ =
        load(engine, ModelType::DEEPLABV3, classes_names, 0, 0.35f, 0.0f, 1, false, 0, 0, 0.0, 0.0,
             SegOutput::CLASS_MAP);
    if (!model_)
    {
        printf("Failed to load DeepLabV3 engine: %s\n", engine.c_str());
        return;
    }
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        printf("Failed to read image: %s\n", image_path.c_str());
        return;
    }
    run_single_deeplabv3(image_path, image, model_);
}

void run_deeplabv3_sahi()
{
    const std::string engine =
        "/home/ps/workspace/trt/cvter/workspace/deepv3/logs_tgwy_v3/best_epoch_weights.trtmodel";
    const std::string image_path =
        "/home/ps/workspace/trt/cvter/workspace/deepv3/datasets/tgwy/VOC2007/JPEGImages/101_PMST_反射率图_crop.jpg";

    std::shared_ptr<InferBase> model_ =
        load(engine, ModelType::DEEPLABV3SAHI, classes_names, 0, 0.35f, 0.0f, 1, false, 960, 960, 0.1, 0.1,
             SegOutput::CLASS_MAP);
    if (!model_)
    {
        printf("Failed to load DeepLabV3-SAHI engine: %s\n", engine.c_str());
        return;
    }
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        printf("Failed to read image: %s\n", image_path.c_str());
        return;
    }
    run_single_deeplabv3(image_path, image, model_);
}
