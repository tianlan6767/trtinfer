#include "trt/infer.hpp"
#include "common/timer.hpp"
#include "common/object.hpp"

void run_cls()
{
    std::shared_ptr<InferBase> model_ = load("/home/ps/workspace/trt/cvter/workspace/pretrain/yolo11s-cls-dy.engine",
        ModelType::CLS,
        std::vector<std::string>{},
        0,
        0.0f,
        0.0f,
        16,
        false,
        0,
        0,
        0.0,
        0.0);
    cv::Mat image = cv::imread("cat_dog/cat.0.jpg");
    std::vector<cv::Mat> images = {image};
    nv::EventTimer timer;
    for (int i = 0; i< 20; i++){
        model_->forwards(images);
    }
    for (int i = 0; i< 100; i++){
        timer.start();
        auto clsResults = model_->forwards(images);
        timer.stop();
    }
    auto clsResults = model_->forwards(images);
    // std::cout << "Inference done! Time elapsed: " << timer.elapsed() << " ms" << std::endl;
    std::visit(
        [&](auto &&result)
        {
            int batch_size = images.size();
            using T        = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<T, object::ClsResultArray>)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    printf("Batch %d: size : %d\n", i, result.size());
                    for (const auto &cls : result)
                    {
                        printf("  Class: %d, Score: %.4f\n", cls.id, cls.score);
                    }
                }
                
            }
        },
        clsResults);
}