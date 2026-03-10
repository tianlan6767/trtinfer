#include "opencv2/core/mat.hpp"
#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"
#include <filesystem>
#include "common/ilogger.hpp"

static std::vector<std::string> classes_names = {};

void run_single_uviad(const std::string &filename, cv::Mat &image, std::shared_ptr<InferBase> model_)
{       
        std::vector<cv::Mat> inputs = {image};
        auto begin_timer = iLogger::timestamp_now_float();
        auto det = model_->forwards(inputs);
        auto end_timer = iLogger::timestamp_now_float();
        printf("Inference time: %.2f ms\n", (end_timer - begin_timer));

        auto output_path = "result/uviad-seg " + std::filesystem::path(filename).stem().string() + ".jpg";
        std::visit(
            [&inputs, &output_path](auto &&result)
            {
                int batch_size = inputs.size();
                using T        = std::decay_t<decltype(result)>;
                if constexpr (std::is_same_v<T, std::vector<object::SegmentationResultArray>>)
                {
                    for (int i = 0; i < batch_size; i++)
                    {
                        printf("Batch %d: size : %zu\n", i, result[i].size());
                        osd_segmentation(inputs[i], result[i]);
                        cv::imwrite(output_path, inputs[i]);
                    }
                    
                }
            },
            det);

}

void run_uviad()
{   
    std::string test_dir = "/home/ps/workspace/trt/trt-sahi-yolo/workspace/ad/test";
    std::shared_ptr<InferBase> model_ = load("/home/ps/workspace/trt/trt-sahi-yolo/workspace/ad/ad2/net_100.trtmodel",
        ModelType::UVIAD,
        classes_names,
        0,  
        0.45f,
        0.0f,
        1,
        false,
        0,
        0,
        0.0,
        0.0);
    
    std::vector<std::string> images_path;
    cv::glob(test_dir + "/*.jpg", images_path);
    for (const auto &image_path : images_path)
    {
        cv::Mat image = cv::imread(image_path);
        run_single_uviad(image_path, image, model_);
    }

    // // std::exit(0);
    // cv::Mat image = cv::imread("/home/ps/workspace/trt/trt-sahi-yolo/workspace/ad/test/ng_loss_cnc_0729_small_wf_silver_taotu_51-95-1_1_42_2.jpg");
    // std::vector<cv::Mat> images = {image};
    // auto det = model_->forwards(images);
    // printf("Batch size : %zu\n", images.size());    
    // std::visit(
    //     [&images](auto &&result)
    //     {
    //         int batch_size = images.size();
    //         using T        = std::decay_t<decltype(result)>;
    //         if constexpr (std::is_same_v<T, std::vector<object::SegmentationResultArray>>)
    //         {
    //             for (int i = 0; i < batch_size; i++)
    //             {
    //                 printf("Batch %d: size : %zu\n", i, result[i].size());
    //                 osd_segmentation(images[i], result[i]);
    //                 cv::imwrite("result/uviad-seg " + std::to_string(i) + ".jpg", images[i]);
    //             }
    //         }
    //     },
    //     det);
}