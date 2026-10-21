#include "bind_trad.h"
#include "bind_extra.h"
#include "pybind_cvmat.hpp"

#include "cv_blob/blob.h"
#include "cv_caliper/caliper.h"
#include "cv_caliper/metrology.h"
#include "cv_match/matcher.h"

#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

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
            (void)n;

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

    caliper::RectResult find_rectangle(const cv::Mat& img, double cx, double cy, double width, double height,
                                       double angle_deg, double search) const
    {
        return caliper::findRectangle(img, cv::Point2d(cx, cy), width, height, angle_deg * CV_PI / 180.0,
                                      search, param);
    }

    caliper::RectResult find_rectangle_box(const cv::Mat& img, double x0, double y0, double x1, double y1,
                                           double search) const
    {
        return caliper::findRectangleBox(img, x0, y0, x1, y1, search, param);
    }

    caliper::MultiLineResult find_edges(const cv::Mat& img, double x0, double y0, double x1, double y1,
                                        bool search_horizontal) const
    {
        return caliper::findEdgesRect(img, x0, y0, x1, y1, search_horizontal, param);
    }

    caliper::MultiLineResult find_edges_oriented(const cv::Mat& img, double cx, double cy, double phi_deg,
                                                 double length1, double length2) const
    {
        return caliper::findEdgesOriented(img, cv::Point2d(cx, cy), phi_deg * CV_PI / 180.0,
                                          length1, length2, param);
    }

    caliper::MultiCircleResult find_circle_edges(const cv::Mat& img, double cx, double cy, double radius,
                                                 double search, double start_deg, double end_deg) const
    {
        return caliper::findCircleEdges(img, cv::Point2d(cx, cy), radius, search, start_deg, end_deg, param);
    }

    caliper::ArcResult find_arc(const cv::Mat& img, double cx, double cy, double radius, double search,
                                double start_deg, double end_deg) const
    {
        return caliper::findArc(img, cv::Point2d(cx, cy), radius, search, start_deg, end_deg, param);
    }

    caliper::DualWidthResult measure_dual_width(const cv::Mat& img, double cx, double cy, double phi_deg,
                                                double length1, double length2) const
    {
        return caliper::measureDualWidth(img, cv::Point2d(cx, cy), phi_deg * CV_PI / 180.0,
                                         length1, length2, param);
    }

    caliper::DualWidthResult measure_radial_width(const cv::Mat& img, double cx, double cy, double radius,
                                                  double search, double start_deg, double end_deg) const
    {
        return caliper::measureRadialWidth(img, cv::Point2d(cx, cy), radius, search, start_deg, end_deg, param);
    }

    caliper::GapResult measure_gap(const cv::Mat& img, double cx, double cy, double phi_deg, double length1,
                                   double half_gap, double search) const
    {
        return caliper::measureGap(img, cv::Point2d(cx, cy), phi_deg * CV_PI / 180.0,
                                   length1, half_gap, search, param);
    }
};


void bindTrad(py::module_& m)
{
    py::class_<cv::Point2d>(m, "Point2d")
    .def(py::init<>())
    .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
    .def_readwrite("x", &cv::Point2d::x)
    .def_readwrite("y", &cv::Point2d::y)
    .def("__repr__", [](const cv::Point2d &p){
        return "<Point2d x=" + std::to_string(p.x) + " y=" + std::to_string(p.y) + ">";
    });

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
        .def_readwrite("minScale", &template_matching::MatcherParam::minScale)
        .def_readwrite("maxScale", &template_matching::MatcherParam::maxScale)
        .def_readwrite("scaleStep", &template_matching::MatcherParam::scaleStep)
        .def_readwrite("numScales", &template_matching::MatcherParam::numScales)
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
                << ", minScale: " << param.minScale
                << ", maxScale: " << param.maxScale
                << ", scaleStep: " << param.scaleStep
                << ", numScales: " << param.numScales
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
        .def_readwrite("Scale", &template_matching::MatchResult::Scale)
        .def("__repr__", [](const template_matching::MatchResult &r){
            std::ostringstream oss;
            oss << "MatchResult(LeftTop=(" << r.LeftTop.x << "," << r.LeftTop.y << ")"
                << ", LeftBottom=(" << r.LeftBottom.x << "," << r.LeftBottom.y << ")"
                << ", RightTop=(" << r.RightTop.x << "," << r.RightTop.y << ")"
                << ", RightBottom=(" << r.RightBottom.x << "," << r.RightBottom.y << ")"
                << ", Center=(" << r.Center.x << "," << r.Center.y << ")"
                << ", Angle=" << r.Angle
                << ", Score=" << r.Score
                << ", Scale=" << r.Scale
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
        .value("All", caliper::EdgeSelect::All)
        .value("Nth", caliper::EdgeSelect::Nth)
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
        .def_readwrite("edgeIndex", &caliper::CaliperParam::edgeIndex)
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
                << ", edgeIndex=" << p.edgeIndex
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
             py::arg("search") = 15.0, py::arg("start_deg") = 0.0, py::arg("end_deg") = 360.0)
        .def("find_rectangle", &CaliperWrapper::find_rectangle,
             py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("width"), py::arg("height"),
             py::arg("angle_deg") = 0.0, py::arg("search") = 12.0)
        .def("find_rectangle_box", &CaliperWrapper::find_rectangle_box,
             py::arg("image"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
             py::arg("search") = 12.0)
        .def("find_edges", &CaliperWrapper::find_edges,
             py::arg("image"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
             py::arg("search_horizontal") = true)
        .def("find_edges_oriented", &CaliperWrapper::find_edges_oriented,
             py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("phi_deg"),
             py::arg("length1"), py::arg("length2"))
        .def("find_circle_edges", &CaliperWrapper::find_circle_edges,
             py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("radius"),
             py::arg("search") = 15.0, py::arg("start_deg") = 0.0, py::arg("end_deg") = 360.0)
        .def("find_arc", &CaliperWrapper::find_arc,
             py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("radius"),
             py::arg("search") = 15.0, py::arg("start_deg") = 0.0, py::arg("end_deg") = 360.0)
        .def("measure_dual_width", &CaliperWrapper::measure_dual_width,
             py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("phi_deg"),
             py::arg("length1"), py::arg("length2"))
        .def("measure_radial_width", &CaliperWrapper::measure_radial_width,
             py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("radius"),
             py::arg("search") = 15.0, py::arg("start_deg") = 0.0, py::arg("end_deg") = 360.0)
        .def("measure_gap", &CaliperWrapper::measure_gap,
             py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("phi_deg"),
             py::arg("length1"), py::arg("half_gap"), py::arg("search"));

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

    py::class_<caliper::RectResult>(m, "RectResult")
        .def(py::init<>())
        .def_readwrite("found", &caliper::RectResult::found)
        .def_readwrite("center", &caliper::RectResult::center)
        .def_readwrite("width", &caliper::RectResult::width)
        .def_readwrite("height", &caliper::RectResult::height)
        .def_readwrite("angle", &caliper::RectResult::angle)
        .def_readwrite("rms", &caliper::RectResult::rms)
        .def_readwrite("topLeft", &caliper::RectResult::topLeft)
        .def_readwrite("topRight", &caliper::RectResult::topRight)
        .def_readwrite("bottomRight", &caliper::RectResult::bottomRight)
        .def_readwrite("bottomLeft", &caliper::RectResult::bottomLeft)
        .def_readwrite("top", &caliper::RectResult::top)
        .def_readwrite("right", &caliper::RectResult::right)
        .def_readwrite("bottom", &caliper::RectResult::bottom)
        .def_readwrite("left", &caliper::RectResult::left)
        .def("__repr__", [](const caliper::RectResult& r) {
            std::ostringstream oss;
            oss << "RectResult(found=" << (r.found ? "true" : "false")
                << ", center=(" << r.center.x << "," << r.center.y << ")"
                << ", width=" << r.width
                << ", height=" << r.height
                << ", angle=" << r.angle
                << ", rms=" << r.rms
                << ")";
            return oss.str();
        });

    m.def("find_rectangle",
          [](const cv::Mat& img, double cx, double cy, double width, double height, double angle_deg,
             double search, const caliper::CaliperParam& param) {
              return caliper::findRectangle(img, cv::Point2d(cx, cy), width, height,
                                            angle_deg * CV_PI / 180.0, search, param);
          },
          py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("width"), py::arg("height"),
          py::arg("angle_deg") = 0.0, py::arg("search") = 12.0,
          py::arg("param") = caliper::CaliperParam());

    m.def("find_rectangle_box",
          [](const cv::Mat& img, double x0, double y0, double x1, double y1, double search,
             const caliper::CaliperParam& param) {
              return caliper::findRectangleBox(img, x0, y0, x1, y1, search, param);
          },
          py::arg("image"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
          py::arg("search") = 12.0, py::arg("param") = caliper::CaliperParam());

    py::class_<metrology::LineFit>(m, "LineFit")
        .def(py::init<>())
        .def_readwrite("found", &metrology::LineFit::found)
        .def_readwrite("p1", &metrology::LineFit::p1)
        .def_readwrite("p2", &metrology::LineFit::p2)
        .def_readwrite("center", &metrology::LineFit::center)
        .def_readwrite("direction", &metrology::LineFit::direction)
        .def_readwrite("angle", &metrology::LineFit::angle)
        .def_readwrite("rms", &metrology::LineFit::rms)
        .def_readwrite("length", &metrology::LineFit::length)
        .def_readwrite("numPoints", &metrology::LineFit::numPoints)
        .def_readwrite("numInliers", &metrology::LineFit::numInliers)
        .def_readwrite("inliers", &metrology::LineFit::inliers);

    py::class_<metrology::SizeSummary>(m, "SizeSummary")
        .def(py::init<>())
        .def_readwrite("found", &metrology::SizeSummary::found)
        .def_readwrite("center", &metrology::SizeSummary::center)
        .def_readwrite("width", &metrology::SizeSummary::width)
        .def_readwrite("height", &metrology::SizeSummary::height)
        .def_readwrite("angle", &metrology::SizeSummary::angle)
        .def_readwrite("diameter", &metrology::SizeSummary::diameter);

    m.def("distance_points",
          [](double x0, double y0, double x1, double y1) {
              return metrology::distancePoints(cv::Point2d(x0, y0), cv::Point2d(x1, y1));
          },
          py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"));
    m.def("distance_point_line",
          [](double px, double py, double x0, double y0, double x1, double y1) {
              return metrology::distancePointToLine(cv::Point2d(px, py), cv::Point2d(x0, y0), cv::Point2d(x1, y1));
          },
          py::arg("px"), py::arg("py"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"));
    m.def("distance_lines",
          [](double ax0, double ay0, double ax1, double ay1, double bx0, double by0, double bx1, double by1) {
              return metrology::distanceBetweenLines(cv::Point2d(ax0, ay0), cv::Point2d(ax1, ay1),
                                                     cv::Point2d(bx0, by0), cv::Point2d(bx1, by1));
          },
          py::arg("ax0"), py::arg("ay0"), py::arg("ax1"), py::arg("ay1"),
          py::arg("bx0"), py::arg("by0"), py::arg("bx1"), py::arg("by1"));
    m.def("angle_lines",
          [](double ax0, double ay0, double ax1, double ay1, double bx0, double by0, double bx1, double by1) {
              return metrology::angleBetweenLines(cv::Point2d(ax0, ay0), cv::Point2d(ax1, ay1),
                                                  cv::Point2d(bx0, by0), cv::Point2d(bx1, by1));
          },
          py::arg("ax0"), py::arg("ay0"), py::arg("ax1"), py::arg("ay1"),
          py::arg("bx0"), py::arg("by0"), py::arg("bx1"), py::arg("by1"));
    m.def("summarize_rectangle",
          [](double tlx, double tly, double trx, double try_, double brx, double bry, double blx, double bly) {
              return metrology::summarizeRectangle(cv::Point2d(tlx, tly), cv::Point2d(trx, try_),
                                                   cv::Point2d(brx, bry), cv::Point2d(blx, bly));
          },
          py::arg("tlx"), py::arg("tly"), py::arg("trx"), py::arg("tr_y"),
          py::arg("brx"), py::arg("bry"), py::arg("blx"), py::arg("bly"));
    m.def("summarize_circle",
          [](double cx, double cy, double radius) {
              return metrology::summarizeCircle(cv::Point2d(cx, cy), radius);
          },
          py::arg("cx"), py::arg("cy"), py::arg("radius"));

    // -----------------------------
    // 预处理 + Blob
    // -----------------------------
    py::enum_<blob::ThresholdType>(m, "ThresholdType")
        .value("Fixed", blob::ThresholdType::Fixed)
        .value("Otsu", blob::ThresholdType::Otsu)
        .export_values();

    py::enum_<blob::MorphOp>(m, "MorphOp")
        .value("Erode", blob::MorphOp::Erode)
        .value("Dilate", blob::MorphOp::Dilate)
        .value("Open", blob::MorphOp::Open)
        .value("Close", blob::MorphOp::Close)
        .export_values();

    py::class_<blob::ThresholdParam>(m, "ThresholdParam")
        .def(py::init<>())
        .def_readwrite("type", &blob::ThresholdParam::type)
        .def_readwrite("value", &blob::ThresholdParam::value)
        .def_readwrite("maxValue", &blob::ThresholdParam::maxValue)
        .def_readwrite("invert", &blob::ThresholdParam::invert);

    py::class_<blob::MorphParam>(m, "MorphParam")
        .def(py::init<>())
        .def_readwrite("op", &blob::MorphParam::op)
        .def_readwrite("ksize", &blob::MorphParam::ksize)
        .def_readwrite("iterations", &blob::MorphParam::iterations);

    py::class_<blob::BlobFilter>(m, "BlobFilter")
        .def(py::init<>())
        .def_readwrite("minArea", &blob::BlobFilter::minArea)
        .def_readwrite("maxArea", &blob::BlobFilter::maxArea)
        .def_readwrite("minCircularity", &blob::BlobFilter::minCircularity)
        .def_readwrite("maxCircularity", &blob::BlobFilter::maxCircularity)
        .def_readwrite("minAspect", &blob::BlobFilter::minAspect)
        .def_readwrite("maxAspect", &blob::BlobFilter::maxAspect)
        .def_readwrite("maxCount", &blob::BlobFilter::maxCount);

    py::class_<blob::ContourStats>(m, "ContourStats")
        .def(py::init<>())
        .def_readwrite("area", &blob::ContourStats::area)
        .def_readwrite("perimeter", &blob::ContourStats::perimeter)
        .def_readwrite("circularity", &blob::ContourStats::circularity)
        .def_readwrite("aspect", &blob::ContourStats::aspect)
        .def_readwrite("angle", &blob::ContourStats::angle)
        .def_readwrite("center", &blob::ContourStats::center)
        .def_readwrite("left", &blob::ContourStats::left)
        .def_readwrite("top", &blob::ContourStats::top)
        .def_readwrite("width", &blob::ContourStats::width)
        .def_readwrite("height", &blob::ContourStats::height);

    py::class_<blob::Blob>(m, "Blob")
        .def(py::init<>())
        .def_readwrite("label", &blob::Blob::label)
        .def_readwrite("area", &blob::Blob::area)
        .def_readwrite("perimeter", &blob::Blob::perimeter)
        .def_readwrite("circularity", &blob::Blob::circularity)
        .def_readwrite("aspect", &blob::Blob::aspect)
        .def_readwrite("angle", &blob::Blob::angle)
        .def_readwrite("center", &blob::Blob::center)
        .def_readwrite("left", &blob::Blob::left)
        .def_readwrite("top", &blob::Blob::top)
        .def_readwrite("width", &blob::Blob::width)
        .def_readwrite("height", &blob::Blob::height);

    m.def("crop_roi", &blob::cropRoi, py::arg("image"), py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"));
    m.def("gaussian_blur", &blob::gaussianBlur, py::arg("image"), py::arg("ksize") = 5, py::arg("sigma") = 0.0);
    m.def("median_blur", &blob::medianBlur, py::arg("image"), py::arg("ksize") = 5);
    m.def("threshold", &blob::threshold, py::arg("image"), py::arg("param") = blob::ThresholdParam());
    m.def("morphology", &blob::morphology, py::arg("image"), py::arg("param") = blob::MorphParam());
    m.def("contour_stats", &blob::contourStats, py::arg("binary"));
    m.def("find_blobs", &blob::findBlobs, py::arg("binary"), py::arg("filter") = blob::BlobFilter(),
          py::arg("roi_x") = 0, py::arg("roi_y") = 0, py::arg("roi_w") = 0, py::arg("roi_h") = 0);

    bindExtra(m);
}
