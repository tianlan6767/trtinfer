#include <sstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "opencv2/opencv.hpp"
#include "trt/infer.hpp"
#include "cv_match/matcher.h"
#include "cv_caliper/caliper.h"
#include "cv_caliper/metrology.h"
#include "cv_blob/blob.h"

#define UNUSED(expr) do { (void)(expr); } while (0)

using namespace std;

namespace py=pybind11;

#include "pybind_cvmat.hpp"
#include "bind_trad.h"

struct DetResult
{
    object::Box box;
    std::vector<object::KeyPoint> keypoints;
    cv::Mat seg;
    object::OBBox obb;
    object::ClsAttribute cls;

};

class TrtInfer{
public:
    TrtInfer(
        const std::string &model_path,
        ModelType model_type,
        const std::vector<std::string> &names,
        int gpu_id                    = 0,
        float confidence_threshold    = 0.5f,
        float nms_threshold           = 0.45f,
        int max_batch_size            = 32,
        bool auto_slice               = true,
        int slice_width               = 640,
        int slice_height              = 640,
        double slice_horizontal_ratio = 0.3,
        double slice_vertical_ratio   = 0.3,
        SegOutput seg_output          = SegOutput::INSTANCES)
    {
        instance_ = load(model_path, model_type, names, gpu_id, confidence_threshold, nms_threshold, max_batch_size, auto_slice, slice_width, slice_height, slice_horizontal_ratio, slice_vertical_ratio, seg_output);
    }

    std::vector<std::vector<DetResult>> forwards(const std::vector<cv::Mat>& images)
    {
        InferResult variant_batch_results = instance_->forwards(images);

        std::vector<std::vector<DetResult>> final_det_results;

        std::visit(
            // Lambda to process the actual data from the variant
            [&images, &final_det_results](auto&& arg_batch_results) 
            {
                // arg_batch_results is one of the types in InferResult, e.g., std::vector<object::DetectionObbResultArray>&
                int num_images_in_batch = images.size();
                
                // Resize the outer vector to hold results for each image
                final_det_results.resize(num_images_in_batch);

                // Type of the current batch result (e.g., std::vector<object::PoseResultArray>)
                using CurrentBatchResultType = std::decay_t<decltype(arg_batch_results)>;

                if constexpr (std::is_same_v<CurrentBatchResultType, std::vector<object::PoseResultArray>>)
                {
                    // arg_batch_results is std::vector<object::PoseResultArray>
                    // Each element arg_batch_results[i] is an object::PoseResultArray (i.e., std::vector<object::PoseInstance>)
                    for (int i = 0; i < num_images_in_batch; ++i)
                    {
                        if (i >= arg_batch_results.size()) continue; // Safety check

                        for (const auto& pose_instance : arg_batch_results[i]) // pose_instance is object::PoseInstance
                        {
                            DetResult det_item;
                            det_item.box = pose_instance.box;
                            det_item.keypoints = pose_instance.keypoints;
                            final_det_results[i].push_back(det_item);
                        }
                    }
                }
                else if constexpr (std::is_same_v<CurrentBatchResultType, std::vector<object::DetectionResultArray>>)
                {
                    // arg_batch_results is std::vector<object::DetectionResultArray>
                    // arg_batch_results[i] is object::DetectionResultArray (i.e., std::vector<object::Box>)
                    for (int i = 0; i < num_images_in_batch; ++i)
                    {
                        if (i >= arg_batch_results.size()) continue;

                        for (const auto& box_instance : arg_batch_results[i]) // box_instance is object::Box
                        {
                            DetResult det_item;
                            det_item.box = box_instance;
                            final_det_results[i].push_back(det_item);
                        }
                    }
                }
                else if constexpr (std::is_same_v<CurrentBatchResultType, std::vector<object::DetectionObbResultArray>>)
                {
                    // arg_batch_results is std::vector<object::DetectionObbResultArray>
                    // arg_batch_results[i] is object::DetectionObbResultArray (i.e., std::vector<object::OBBox>)
                    for (int i = 0; i < num_images_in_batch; ++i)
                    {
                        if (i >= arg_batch_results.size()) continue;

                       
                        for (const auto& obb_instance : arg_batch_results[i]) // obb_instance is object::OBBox
                        {
                            DetResult det_item;
                            det_item.obb = obb_instance;
                            final_det_results[i].push_back(det_item);
                        }
                    }
                }
                else if constexpr (std::is_same_v<CurrentBatchResultType, std::vector<object::SegmentationResultArray>>)
                {
                    // arg_batch_results is std::vector<object::SegmentationResultArray>
                    // arg_batch_results[i] is object::SegmentationResultArray (i.e., std::vector<object::SegmentationInstance>)
                    for (int i = 0; i < num_images_in_batch; ++i)
                    {
                        if (i >= arg_batch_results.size()) continue;

                        for (const auto& seg_instance : arg_batch_results[i]) // seg_instance is object::SegmentationInstance
                        {
                            DetResult det_item;
                            det_item.box = seg_instance.box; // Populate bounding box

                            // Convert object::SegmentMap to cv::Mat for DetResult::seg
                            if (seg_instance.seg && seg_instance.seg->data && seg_instance.seg->width > 0 && seg_instance.seg->height > 0)
                            {
                                // Create a cv::Mat wrapper. Assuming CV_8UC1 for mask data.
                                cv::Mat mask_wrapper_mat(seg_instance.seg->height,
                                                         seg_instance.seg->width,
                                                         CV_8UC1, // unsigned char, 1 channel
                                                         seg_instance.seg->data);
                                
                                // IMPORTANT: Clone the data. The mask_wrapper_mat only points to
                                // SegmentMap's data. Cloning creates a new cv::Mat with its own copy
                                // of the data, ensuring its lifetime is independent of SegmentMap.
                                det_item.seg = mask_wrapper_mat.clone();
                            }
                            // else: det_item.seg will be an empty cv::Mat by default.
                            final_det_results[i].push_back(det_item);
                        }
                    }
                }
                // else { /* Handle other types or std::monostate if present in InferResult */ }
                else if constexpr (std::is_same_v<CurrentBatchResultType, object::ClsResultArray>)
                {   
                    // cout << "Processing ClsResultArray for batch of size: " << arg_batch_results.size() << endl;
                    // arg_batch_results is object::ClsResultArray (i.e., std::vector<object::ClsAttribute>)
                    for (int i = 0; i < num_images_in_batch; ++i)
                    {
                        if (i >= arg_batch_results.size()) continue;

                        DetResult cls_item;
                        cls_item.cls = arg_batch_results[i];  // 每张图片一个 ClsAttribute
                        final_det_results[i].push_back(cls_item);
                    }
                }
            },
            variant_batch_results
        );

        return final_det_results;
    }

    bool valid()
    {
        return instance_ != nullptr;
    }

private:
    std::shared_ptr<InferBase> instance_;

};


PYBIND11_MODULE(cvter, m){
    py::enum_<ModelType>(m, "ModelType")
        .value("YOLOV5", ModelType::YOLOV5)
        .value("YOLO11", ModelType::YOLO11)
        .value("YOLO11POSE", ModelType::YOLO11POSE)
        .value("YOLO11SEG", ModelType::YOLO11SEG)
        .value("YOLO11OBB", ModelType::YOLO11OBB)
        .value("YOLOV5SAHI", ModelType::YOLOV5SAHI)
        .value("YOLO11SAHI", ModelType::YOLO11SAHI)
        .value("YOLO11POSESAHI", ModelType::YOLO11POSESAHI)
        .value("YOLO11SEGSAHI", ModelType::YOLO11SEGSAHI)
        .value("YOLO11OBBSAHI", ModelType::YOLO11OBBSAHI)
        .value("DFINE", ModelType::DFINE)
        .value("DFINESAHI", ModelType::DFINESAHI)
        .value("CLS", ModelType::CLS)
        .value("UVIAD", ModelType::UVIAD)
        .value("DEEPLABV3", ModelType::DEEPLABV3)
        .value("DEEPLABV3SAHI", ModelType::DEEPLABV3SAHI)
        .export_values();

    py::enum_<SegOutput>(m, "SegOutput")
        .value("INSTANCES", SegOutput::INSTANCES)
        .value("CLASS_MAP", SegOutput::CLASS_MAP)
        .export_values();

    py::class_<object::Box>(m, "Box")
        .def_readwrite("left", &object::Box::left)
        .def_readwrite("top", &object::Box::top)
        .def_readwrite("right", &object::Box::right)
        .def_readwrite("bottom", &object::Box::bottom)
        .def_readwrite("score", &object::Box::score)
        .def_readwrite("class_id", &object::Box::class_id)
        .def_readwrite("class_name", &object::Box::class_name)
        .def("__repr__", [](const object::Box &box) {
            std::ostringstream oss;
            oss << "Box(left: " << box.left
                << ", top: " << box.top
                << ", right: " << box.right
                << ", bottom: " << box.bottom
                << ", score: " << box.score
                << ", class_id: " << box.class_id
                << ", class_name: " << box.class_name
                << ")";
            return oss.str();
        });
    py::class_<object::KeyPoint>(m, "KeyPoint")
        .def_readwrite("x", &object::KeyPoint::x)
        .def_readwrite("y", &object::KeyPoint::y)
        .def_readwrite("vis", &object::KeyPoint::score)
        .def("__repr__", [](const object::KeyPoint &Key_point) {
            std::ostringstream oss;
            oss << "KeyPoint(x: " << Key_point.x
                << ", y: " << Key_point.y
                << ", vis: " << Key_point.score
                << ")";
            return oss.str();
        });

    py::class_<object::OBBox>(m, "OBBox")
        .def_readwrite("cx", &object::OBBox::cx)
        .def_readwrite("cy", &object::OBBox::cy)
        .def_readwrite("width", &object::OBBox::width)
        .def_readwrite("height", &object::OBBox::height)
        .def_readwrite("angle", &object::OBBox::angle)
        .def_readwrite("score", &object::OBBox::score)
        .def_readwrite("class_name", &object::OBBox::class_name)
        .def("__repr__", [](const object::OBBox &obbox) {
            std::ostringstream oss;
            oss << "OBBox(cx: " << obbox.cx
                << ", cy: " << obbox.cy
                << ", width: " << obbox.width
                << ", height: " << obbox.height
                << ", angle: " << obbox.angle
                << ", score: " << obbox.score
                << ", class_name: " << obbox.class_name
                << ")";
            return oss.str();
        });
    
    py::class_<object::ClsAttribute>(m, "ClsAttribute")
        .def_readwrite("class_id", &object::ClsAttribute::id)
        .def_readwrite("score", &object::ClsAttribute::score)
        .def("__repr__", [](const object::ClsAttribute &cls_attr) {
            std::ostringstream oss;
            oss << "ClsAttribute(class_id: " << cls_attr.id
                << ", score: " << cls_attr.score
                << ")";
            return oss.str();
        });
    
    py::class_<DetResult>(m, "DetResult")
        .def_readwrite("box", &DetResult::box)
        .def_readwrite("keypoints", &DetResult::keypoints)
        .def_readwrite("seg", &DetResult::seg)
        .def_readwrite("obb", &DetResult::obb)
        .def_readwrite("cls", &DetResult::cls);
    
    
    py::class_<TrtInfer>(m, "TrtInfer")
        .def(py::init<string, ModelType, vector<string>, int, float, float, int, bool, int, int, double, double, SegOutput>(),
            py::arg("model_path"),
            py::arg("model_type"),
            py::arg("names"),
            py::arg("gpu_id") = 0,
            py::arg("confidence_threshold") = 0.5f,
            py::arg("nms_threshold") = 0.45f,
            py::arg("max_batch_size") = 32,
            py::arg("auto_slice") = true,
            py::arg("slice_width") = 640,
            py::arg("slice_height") = 640,
            py::arg("slice_horizontal_ratio") = 0.3,
            py::arg("slice_vertical_ratio") = 0.3,
            py::arg("seg_output") = SegOutput::INSTANCES)
    .def_property_readonly("valid", &TrtInfer::valid)
    .def("forwards", &TrtInfer::forwards, py::arg("images"));

    bindTrad(m);
};
