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

#define UNUSED(expr) do { (void)(expr); } while (0)

using namespace std;

namespace py=pybind11;

namespace pybind11 { namespace detail {
template<>
struct type_caster<cv::Mat>
{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    //! 1. cast numpy.ndarray to cv::Mat
    bool load(handle obj, bool)
    {
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        //const int ndims = (int)info.ndim;
        int nh = 1;
        int nw = 1;
        int nc = 1;
        int ndims = info.ndim;
        if(ndims == 2)
        {
            nh = info.shape[0];
            nw = info.shape[1];
        } 
        else if(ndims == 3)
        {
            nh = info.shape[0];
            nw = info.shape[1];
            nc = info.shape[2];
        }
        else
        {
            char msg[64];
            std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
            throw std::logic_error(msg);
            return false;
        }

        int dtype;
        if(info.format == format_descriptor<unsigned char>::format())
        {
            dtype = CV_8UC(nc);
        }
        else if (info.format == format_descriptor<int>::format())
        {
            dtype = CV_32SC(nc);
        }
        else if (info.format == format_descriptor<float>::format())
        {
            dtype = CV_32FC(nc);
        }
        else
        {
            throw std::logic_error("Unsupported type, only support uchar, int32, float");
            return false;
        }
        value = cv::Mat(nh, nw, dtype, info.ptr).clone();
        return true;
    }

    //! 2. cast cv::Mat to numpy.ndarray
    static handle cast(const cv::Mat& mat, return_value_policy, handle defval){
        UNUSED(defval);

        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type)? 2 : 3;

        if(depth == CV_8U)
        {
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }
        else if(depth == CV_32S)
        {
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }
        else if(depth == CV_32F)
        {
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }
        else
        {
            throw std::logic_error("Unsupported type, only support uchar, int32, float");
        }
        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) 
        {
            bufferdim = {(size_t) nh, (size_t) nw};
            strides = {elemsize * (size_t) nw, elemsize};
        } 
        else if (dim == 3) 
        {
                bufferdim = {(size_t) nh, (size_t) nw, (size_t) nc};
                strides = {(size_t) elemsize * nw * nc, (size_t) elemsize * nc, (size_t) elemsize};
        }
        return array(buffer_info( mat.data,  elemsize,  format, dim, bufferdim, strides )).release();
    }
};

}}//! end namespace pybind11::detail




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

class MatcherWrapper {
public:
    std::unique_ptr<template_matching::Matcher> matcher;

    MatcherWrapper(const template_matching::MatcherParam& param)
    {
        // 创建 matcher 对象
        matcher = template_matching::GetMatcher(param);
        if (!matcher) {
            throw std::runtime_error("GetMatcher 返回空指针");
        }
    }

    void setTemplate(const cv::Mat& templ, const cv::Mat& mask = cv::Mat()) {
        matcher->setTemplate(templ, mask);
    }

    py::list match(const cv::Mat& img) {
        try {
            std::vector<template_matching::MatchResult> cpp_results;
            int n = matcher->match(img, cpp_results);
            UNUSED(n);

            py::list py_results;
            for (auto& r : cpp_results)
                py_results.append(r);
            return py_results;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception in match: " << e.what() << std::endl;
            throw py::value_error(e.what());
        }
    }
};

class CaliperWrapper {
public:
    caliper::CaliperParam param;

    CaliperWrapper() = default;
    explicit CaliperWrapper(const caliper::CaliperParam& p) : param(p) {}

    caliper::LineResult find_line(const cv::Mat& img, double x0, double y0, double x1, double y1,
                                  bool search_horizontal) const
    {
        return caliper::findLineRect(img, x0, y0, x1, y1, search_horizontal, param);
    }

    caliper::LineResult find_line_oriented(const cv::Mat& img, double cx, double cy, double phi_deg,
                                           double length1, double length2) const
    {
        return caliper::findLineOriented(img, cv::Point2d(cx, cy), phi_deg * CV_PI / 180.0,
                                         length1, length2, param);
    }

    caliper::CircleResult find_circle(const cv::Mat& img, double cx, double cy, double radius,
                                      double search, double start_deg, double end_deg) const
    {
        return caliper::findCircle(img, cv::Point2d(cx, cy), radius, search, start_deg, end_deg, param);
    }
};

PYBIND11_MODULE(cvter, m){
    py::class_<cv::Point2d>(m, "Point2d")
    .def(py::init<>())
    .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
    .def_readwrite("x", &cv::Point2d::x)
    .def_readwrite("y", &cv::Point2d::y)
    .def("__repr__", [](const cv::Point2d &p){
        return "<Point2d x=" + std::to_string(p.x) + " y=" + std::to_string(p.y) + ">";
    });

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

    // -----------------------------
    // MatcherType 枚举
    // -----------------------------
    py::enum_<template_matching::MatcherType>(m, "MatcherType")
        .value("PATTERN", template_matching::MatcherType::PATTERN)
        .value("SHAPE", template_matching::MatcherType::SHAPE)
        .export_values();
    
    // MatcherParam 结构体
    py::class_<template_matching::MatcherParam>(m, "MatcherParam")
        .def(py::init<>())
        .def_readwrite("matcherType", &template_matching::MatcherParam::matcherType)
        .def_readwrite("maxCount", &template_matching::MatcherParam::maxCount)
        .def_readwrite("scoreThreshold", &template_matching::MatcherParam::scoreThreshold)
        .def_readwrite("iouThreshold", &template_matching::MatcherParam::iouThreshold)
        .def_readwrite("angle", &template_matching::MatcherParam::angle)
        .def_readwrite("minArea", &template_matching::MatcherParam::minArea)
        .def_readwrite("meanBorder", &template_matching::MatcherParam::meanBorder)
        .def_readwrite("stopLayer", &template_matching::MatcherParam::stopLayer)
        .def_readwrite("edgeMinMag", &template_matching::MatcherParam::edgeMinMag)
        .def_readwrite("maxEdgePoints", &template_matching::MatcherParam::maxEdgePoints)
        .def_readwrite("usePolarity", &template_matching::MatcherParam::usePolarity)
        .def_readwrite("greediness", &template_matching::MatcherParam::greediness)
        .def("__repr__", [](const template_matching::MatcherParam &param) {
            std::ostringstream oss;
            oss << "MatcherParam(matcherType: " << static_cast<int>(param.matcherType)
                << ", maxCount: " << param.maxCount
                << ", scoreThreshold: " << param.scoreThreshold
                << ", iouThreshold: " << param.iouThreshold
                << ", angle: " << param.angle
                << ", minArea: " << param.minArea
                << ", meanBorder: " << (param.meanBorder ? "true" : "false")
                << ", stopLayer: " << param.stopLayer
                << ", edgeMinMag: " << param.edgeMinMag
                << ", maxEdgePoints: " << param.maxEdgePoints
                << ", usePolarity: " << (param.usePolarity ? "true" : "false")
                << ", greediness: " << param.greediness
                << ")";
            return oss.str();
        });

    // -----------------------------
    // MatchResult 结构体
    // -----------------------------
    py::class_<template_matching::MatchResult>(m, "MatchResult")
        .def_readwrite("LeftTop", &template_matching::MatchResult::LeftTop)
        .def_readwrite("LeftBottom", &template_matching::MatchResult::LeftBottom)
        .def_readwrite("RightTop", &template_matching::MatchResult::RightTop)
        .def_readwrite("RightBottom", &template_matching::MatchResult::RightBottom)
        .def_readwrite("Center", &template_matching::MatchResult::Center)
        .def_readwrite("Angle", &template_matching::MatchResult::Angle)
        .def_readwrite("Score", &template_matching::MatchResult::Score)
        .def("__repr__", [](const template_matching::MatchResult &r){
            std::ostringstream oss;
            oss << "MatchResult(LeftTop=(" << r.LeftTop.x << "," << r.LeftTop.y << ")"
                << ", LeftBottom=(" << r.LeftBottom.x << "," << r.LeftBottom.y << ")"
                << ", RightTop=(" << r.RightTop.x << "," << r.RightTop.y << ")"
                << ", RightBottom=(" << r.RightBottom.x << "," << r.RightBottom.y << ")"
                << ", Center=(" << r.Center.x << "," << r.Center.y << ")"
                << ", Angle=" << r.Angle
                << ", Score=" << r.Score
                << ")";
            return oss.str();
        });


    py::class_<MatcherWrapper>(m, "MatcherWrapper")
    .def(py::init<const template_matching::MatcherParam&>())
    .def("setTemplate", &MatcherWrapper::setTemplate, py::arg("templ"), py::arg("mask") = cv::Mat())
    .def("match", &MatcherWrapper::match);

    // -----------------------------
    // 卡尺：矩形找直线 / 环形找圆
    // -----------------------------
    py::enum_<caliper::Polarity>(m, "CaliperPolarity")
        .value("DarkToLight", caliper::Polarity::DarkToLight)
        .value("LightToDark", caliper::Polarity::LightToDark)
        .value("Both", caliper::Polarity::Both)
        .export_values();

    py::enum_<caliper::EdgeSelect>(m, "CaliperSelect")
        .value("First", caliper::EdgeSelect::First)
        .value("Last", caliper::EdgeSelect::Last)
        .value("Strongest", caliper::EdgeSelect::Strongest)
        .export_values();

    py::class_<caliper::CaliperParam>(m, "CaliperParam")
        .def(py::init<>())
        .def_readwrite("polarity", &caliper::CaliperParam::polarity)
        .def_readwrite("select", &caliper::CaliperParam::select)
        .def_readwrite("projection", &caliper::CaliperParam::projection)
        .def_readwrite("numCalipers", &caliper::CaliperParam::numCalipers)
        .def_readwrite("stride", &caliper::CaliperParam::stride)
        .def_readwrite("contrast", &caliper::CaliperParam::contrast)
        .def_readwrite("sigma", &caliper::CaliperParam::sigma)
        .def_readwrite("outlierRatio", &caliper::CaliperParam::outlierRatio)
        .def_readwrite("minPoints", &caliper::CaliperParam::minPoints)
        .def_readwrite("radialInward", &caliper::CaliperParam::radialInward)
        .def("__repr__", [](const caliper::CaliperParam& p) {
            std::ostringstream oss;
            oss << "CaliperParam(polarity=" << static_cast<int>(p.polarity)
                << ", select=" << static_cast<int>(p.select)
                << ", projection=" << p.projection
                << ", numCalipers=" << p.numCalipers
                << ", stride=" << p.stride
                << ", contrast=" << p.contrast
                << ", sigma=" << p.sigma
                << ", outlierRatio=" << p.outlierRatio
                << ", minPoints=" << p.minPoints
                << ", radialInward=" << (p.radialInward ? "true" : "false")
                << ")";
            return oss.str();
        });

    py::class_<caliper::LineResult>(m, "LineResult")
        .def_readwrite("found", &caliper::LineResult::found)
        .def_readwrite("p1", &caliper::LineResult::p1)
        .def_readwrite("p2", &caliper::LineResult::p2)
        .def_readwrite("center", &caliper::LineResult::center)
        .def_readwrite("angle", &caliper::LineResult::angle)
        .def_readwrite("rms", &caliper::LineResult::rms)
        .def_readwrite("length", &caliper::LineResult::length)
        .def_readwrite("numPoints", &caliper::LineResult::numPoints)
        .def_readwrite("numInliers", &caliper::LineResult::numInliers)
        .def_readwrite("points", &caliper::LineResult::points)
        .def_readwrite("inliers", &caliper::LineResult::inliers)
        .def("__repr__", [](const caliper::LineResult& r) {
            std::ostringstream oss;
            oss << "LineResult(found=" << (r.found ? "true" : "false")
                << ", p1=(" << r.p1.x << "," << r.p1.y << ")"
                << ", p2=(" << r.p2.x << "," << r.p2.y << ")"
                << ", angle=" << r.angle
                << ", rms=" << r.rms
                << ", length=" << r.length
                << ", n=" << r.numInliers << "/" << r.numPoints
                << ")";
            return oss.str();
        });

    py::class_<caliper::CircleResult>(m, "CircleResult")
        .def_readwrite("found", &caliper::CircleResult::found)
        .def_readwrite("center", &caliper::CircleResult::center)
        .def_readwrite("radius", &caliper::CircleResult::radius)
        .def_readwrite("rms", &caliper::CircleResult::rms)
        .def_readwrite("numPoints", &caliper::CircleResult::numPoints)
        .def_readwrite("numInliers", &caliper::CircleResult::numInliers)
        .def_readwrite("points", &caliper::CircleResult::points)
        .def_readwrite("inliers", &caliper::CircleResult::inliers)
        .def("__repr__", [](const caliper::CircleResult& r) {
            std::ostringstream oss;
            oss << "CircleResult(found=" << (r.found ? "true" : "false")
                << ", center=(" << r.center.x << "," << r.center.y << ")"
                << ", r=" << r.radius
                << ", rms=" << r.rms
                << ", n=" << r.numInliers << "/" << r.numPoints
                << ")";
            return oss.str();
        });

    py::class_<CaliperWrapper>(m, "CaliperWrapper")
        .def(py::init<>())
        .def(py::init<const caliper::CaliperParam&>(), py::arg("param"))
        .def_readwrite("param", &CaliperWrapper::param)
        .def("find_line", &CaliperWrapper::find_line,
             py::arg("image"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
             py::arg("search_horizontal") = true)
        .def("find_line_oriented", &CaliperWrapper::find_line_oriented,
             py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("phi_deg"),
             py::arg("length1"), py::arg("length2"))
        .def("find_circle", &CaliperWrapper::find_circle,
             py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("radius"),
             py::arg("search") = 15.0, py::arg("start_deg") = 0.0, py::arg("end_deg") = 360.0);

    m.def("find_line",
          [](const cv::Mat& img, double x0, double y0, double x1, double y1,
             bool search_horizontal, const caliper::CaliperParam& param) {
              return caliper::findLineRect(img, x0, y0, x1, y1, search_horizontal, param);
          },
          py::arg("image"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
          py::arg("search_horizontal") = true, py::arg("param") = caliper::CaliperParam());

    m.def("find_line_oriented",
          [](const cv::Mat& img, double cx, double cy, double phi_deg, double length1, double length2,
             const caliper::CaliperParam& param) {
              return caliper::findLineOriented(img, cv::Point2d(cx, cy), phi_deg * CV_PI / 180.0,
                                               length1, length2, param);
          },
          py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("phi_deg"),
          py::arg("length1"), py::arg("length2"), py::arg("param") = caliper::CaliperParam());

    m.def("find_circle",
          [](const cv::Mat& img, double cx, double cy, double radius, double search,
             double start_deg, double end_deg, const caliper::CaliperParam& param) {
              return caliper::findCircle(img, cv::Point2d(cx, cy), radius, search, start_deg, end_deg, param);
          },
          py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("radius"),
          py::arg("search") = 15.0, py::arg("start_deg") = 0.0, py::arg("end_deg") = 360.0,
          py::arg("param") = caliper::CaliperParam());
};
