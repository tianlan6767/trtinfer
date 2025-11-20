#include "trt/infer.hpp"
#include "common/timer.hpp"


void run_cls()
{
    std::shared_ptr<InferBase> model_ = load("/home/ps/workspace/trt/trt-sahi-yolo/workspace/pretrain/yolo11s-cls-dy.engine",
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
    for (int i = 0; i< 20; i++)
        auto clsResults = model_->forwards(images);
    timer.start();
    for (int i = 0; i< 1000; i++)
        auto clsResults = model_->forwards(images);
    timer.stop();
}