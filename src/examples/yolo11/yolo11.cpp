#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"
#include <filesystem>
#include <chrono>
#include "common/timer.hpp"

namespace fs = std::filesystem;

static std::vector<std::string> classes_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

auto hasVaildExtension = [](const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower); // 转小写
    return (ext == ".jpg" || ext == ".jpeg" || ext == ",png" || ext == ".bmp");
};

void run_yolo11()
{
    std::shared_ptr<InferBase> model_ = load("/home/ps/workspace/trt/trt-sahi-yolo/td_0816.transd-dy.trtmodel",
        ModelType::YOLO11,
        classes_names,
        0,
        0.25f,
        0.45f,
        18,
        false,
        0,
        0,
        0.0,
        0.0);
    // cv::Mat image = cv::imread("/home/ps/workspace/trt/trt-sahi-yolo/workspace/infer1/LK_0515_0208_00_orig_data_0025_408-104-2_1_18-APS.jpg");
    std::string img_path = "/home/ps/workspace/trt/trt-sahi-yolo/workspace/src1";
    std::string savePath = "/home/ps/workspace/trt/trt-sahi-yolo/workspace/src1_result";
    fs::create_directories(savePath);

    std::vector<cv::String> image_paths_;
    image_paths_.reserve(10000);
    cv::glob(img_path, image_paths_, true);
    std::vector<cv::String> images_path(image_paths_.begin(), image_paths_.end());
    for(int i=0; i< 10; i++){
        cv::Mat m(3072, 4096, CV_8UC3, cv::Scalar(0,0,0));
        model_->forwards({m});
    }
    cudaDeviceSynchronize();
    // for(auto &path : images_path)
    nv::EventTimer timer;
    for (int i=0; i<images_path.size(); i++)
    {
        auto path = images_path[i];
        if(!hasVaildExtension(path))
            continue;
        cv::Mat image = cv::imread(path);
        // 打印文件名 + 尺寸
        fs::path imgFile(path);
        printf("Processing: %s | Size: %dx%d\n",
            imgFile.filename().string().c_str(),
            image.cols, image.rows);
        std::vector<cv::Mat> images = { image };
        // timer.start();
        auto start = std::chrono::high_resolution_clock::now();
        auto det = model_->forwards(images);
        // timer.stop();
        cudaDeviceSynchronize();  // 确保完成
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "耗时：" << elapsed << "ms" << std::endl;
        std::visit(
            [&](auto &&result)
            {
                int batch_size = images.size();
                using T        = std::decay_t<decltype(result)>;
                if constexpr (std::is_same_v<T, std::vector<object::DetectionResultArray>>)
                {
                    for (int i = 0; i < batch_size; i++)
                    {
                        printf("Batch %d: size : %d\n", i, result[i].size());
                        osd_detection(images[i], result[i]);
                        fs::path imageSave(path);
                        fs::path imgDstFile = fs::path(savePath) / (imageSave.stem().string() + "_det.jpg");
                        cv::imwrite(imgDstFile.string(), images[i]);
                    }

                }
            },
            det);
        }
}


void run_yolo11_sahi()
{
    std::shared_ptr<InferBase> model_ = load("/home/ps/workspace/trt/trt-sahi-yolo/td_0816.transd-dy.trtmodel",
        ModelType::YOLO11SAHI,
        classes_names,
        0,
        0.5f,
        0.45f,
        18,
        false,
        1280,
        1280,
        0.2,
        0.2);
    // cv::Mat image = cv::imread("/home/ps/workspace/trt/trt-sahi-yolo/workspace/infer1/LK_0515_0208_00_orig_data_0025_408-104-2_1_18-APS.jpg");
    std::string img_path = "/home/ps/workspace/trt/trt-sahi-yolo/workspace/test2";
    std::string savePath = "/home/ps/workspace/trt/trt-sahi-yolo/workspace/test2_result";
    fs::create_directories(savePath);

    std::vector<cv::String> image_paths_;
    image_paths_.reserve(10000);
    cv::glob(img_path, image_paths_, true);
    std::vector<cv::String> images_path(image_paths_.begin(), image_paths_.end());

    // for(auto &path : images_path)
    for(int i=0; i < 2000; i++)
    {
        if(!hasVaildExtension(images_path[0]))
            continue;
        auto path = images_path[0];
        cv::Mat image = cv::imread(path);
        // 打印文件名 + 尺寸
        fs::path imgFile(path);
        printf("Processing: %s | Size: %dx%d\n",
            imgFile.filename().string().c_str(),
            image.cols, image.rows);
        std::vector<cv::Mat> images = { image };
        auto det = model_->forwards(images);

        std::visit(
            [&](auto &&result)
            {
                int batch_size = images.size();
                using T        = std::decay_t<decltype(result)>;
                if constexpr (std::is_same_v<T, std::vector<object::DetectionResultArray>>)
                {
                    for (int i = 0; i < batch_size; i++)
                    {
                        printf("Batch %d: size : %d\n", i, result[i].size());
                        osd_detection(images[i], result[i]);
                        fs::path imageSave(path);
                        fs::path imgDstFile = fs::path(savePath) / (imageSave.stem().string() + "_det.jpg");
                        cv::imwrite(imgDstFile.string(), images[i]);
                    }

                }
            },
            det);
        }
}