#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"

static std::vector<std::string> classes_names = {};

void run_uviad()
{
    std::shared_ptr<InferBase> model_ = load("/home/ps/workspace/trt/trt-sahi-yolo/workspace/ad/anomaly.trtmodel",
        ModelType::UVIAD,
        classes_names,
        0,  
        0.5f,
        0.45f,
        1,
        false,
        0,
        0,
        0.0,
        0.0);
    std::exit(0);
    cv::Mat image = cv::imread("/home/ps/workspace/trt/trt-sahi-yolo/workspace/ad/test/cnc_0804_ngs_small_wifi_54-20-2_1_2.jpg");
    std::vector<cv::Mat> images = {image};
    auto det = model_->forwards(images);
    std::visit(
        [&images](auto &&result)
        {
            int batch_size = images.size();
            using T        = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<T, std::vector<object::SegmentationResultArray>>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    printf("Batch %d: size : %zu\n", i, result[i].size());
                    osd_segmentation(images[i], result[i]);
                    cv::imwrite("result/uviad-seg.jpg", images[i]);
                }
                
            }
        },
        det);
}