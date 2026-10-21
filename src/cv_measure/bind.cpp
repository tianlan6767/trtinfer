#include "cv_measure/bind.h"

#include "pybind_cvmat.hpp"

#include "cv_caliper/caliper.h"
#include "cv_caliper/metrology.h"
#include "cv_measure/measure.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace
{

std::vector<cv::Point2d> pointsFromPy(const py::handle& h)
{
	if (py::isinstance<py::sequence>(h) && !py::isinstance<py::array>(h)) {
		py::sequence seq = py::reinterpret_borrow<py::sequence>(h);
		if (seq.size() == 0) {
			return {};
		}
		if (py::isinstance<cv::Point2d>(seq[0])) {
			return h.cast<std::vector<cv::Point2d>>();
		}
	}
	auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(h);
	if (!arr || arr.ndim() != 2 || arr.shape(1) != 2) {
		throw std::invalid_argument("points 需要 (N,2) 数组，或 Point2d 列表");
	}
	std::vector<cv::Point2d> pts;
	pts.reserve(static_cast<size_t>(arr.shape(0)));
	auto view = arr.unchecked<2>();
	for (py::ssize_t i = 0; i < view.shape(0); ++i) {
		pts.emplace_back(view(i, 0), view(i, 1));
	}
	return pts;
}

}  // namespace

void bindMeasure(py::module_& m)
{
	py::class_<metrology::PointHit>(m, "PointHit")
		.def(py::init<>())
		.def_readwrite("found", &metrology::PointHit::found)
		.def_readwrite("point", &metrology::PointHit::point)
		.def("__repr__", [](const metrology::PointHit& h) {
			std::ostringstream oss;
			oss << "PointHit(found=" << (h.found ? "true" : "false")
				<< ", point=(" << h.point.x << "," << h.point.y << "))";
			return oss.str();
		});

	m.def("segment_length",
		  [](double x0, double y0, double x1, double y1) {
			  return metrology::segmentLength({x0, y0}, {x1, y1});
		  },
		  py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"));
	m.def("midpoint",
		  [](double x0, double y0, double x1, double y1) {
			  return metrology::midpoint({x0, y0}, {x1, y1});
		  },
		  py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"));
	m.def("foot_of_perpendicular",
		  [](double px, double py, double x0, double y0, double x1, double y1, bool clamp) {
			  return metrology::footOfPerpendicular({px, py}, {x0, y0}, {x1, y1}, clamp);
		  },
		  py::arg("px"), py::arg("py"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
		  py::arg("clamp") = false);
	m.def("intersect_lines",
		  [](double ax0, double ay0, double ax1, double ay1, double bx0, double by0, double bx1, double by1) {
			  return metrology::intersectLinesAt({ax0, ay0}, {ax1, ay1}, {bx0, by0}, {bx1, by1});
		  },
		  py::arg("ax0"), py::arg("ay0"), py::arg("ax1"), py::arg("ay1"),
		  py::arg("bx0"), py::arg("by0"), py::arg("bx1"), py::arg("by1"));
	m.def("distance_parallel_lines",
		  [](double ax0, double ay0, double ax1, double ay1, double bx0, double by0, double bx1, double by1,
			 double angle_tol_deg) {
			  return metrology::distanceParallelLines({ax0, ay0}, {ax1, ay1}, {bx0, by0}, {bx1, by1}, angle_tol_deg);
		  },
		  py::arg("ax0"), py::arg("ay0"), py::arg("ax1"), py::arg("ay1"),
		  py::arg("bx0"), py::arg("by0"), py::arg("bx1"), py::arg("by1"),
		  py::arg("angle_tol_deg") = 5.0);
	m.def("distance_circle_centers",
		  [](double x0, double y0, double x1, double y1) {
			  return metrology::distanceCircleCenters({x0, y0}, {x1, y1});
		  },
		  py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"));
	m.def("distance_point_circle",
		  [](double px, double py, double cx, double cy, double radius) {
			  return metrology::distancePointToCircle({px, py}, {cx, cy}, radius);
		  },
		  py::arg("px"), py::arg("py"), py::arg("cx"), py::arg("cy"), py::arg("radius"));
	m.def("signed_distance_point_circle",
		  [](double px, double py, double cx, double cy, double radius) {
			  return metrology::signedDistancePointToCircle({px, py}, {cx, cy}, radius);
		  },
		  py::arg("px"), py::arg("py"), py::arg("cx"), py::arg("cy"), py::arg("radius"));
	m.def("distance_circle_line",
		  [](double cx, double cy, double radius, double x0, double y0, double x1, double y1) {
			  return metrology::distanceCircleToLine({cx, cy}, radius, {x0, y0}, {x1, y1});
		  },
		  py::arg("cx"), py::arg("cy"), py::arg("radius"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"));
	m.def("signed_distance_circle_line",
		  [](double cx, double cy, double radius, double x0, double y0, double x1, double y1) {
			  return metrology::signedDistanceCircleToLine({cx, cy}, radius, {x0, y0}, {x1, y1});
		  },
		  py::arg("cx"), py::arg("cy"), py::arg("radius"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"));

	py::class_<caliper::IndexedLine>(m, "IndexedLine")
		.def(py::init<>())
		.def_readwrite("index", &caliper::IndexedLine::index)
		.def_readwrite("line", &caliper::IndexedLine::line);
	py::class_<caliper::IndexedCircle>(m, "IndexedCircle")
		.def(py::init<>())
		.def_readwrite("index", &caliper::IndexedCircle::index)
		.def_readwrite("circle", &caliper::IndexedCircle::circle);
	py::class_<caliper::MultiLineResult>(m, "MultiLineResult")
		.def(py::init<>())
		.def_readwrite("found", &caliper::MultiLineResult::found)
		.def_readwrite("edges", &caliper::MultiLineResult::edges);
	py::class_<caliper::MultiCircleResult>(m, "MultiCircleResult")
		.def(py::init<>())
		.def_readwrite("found", &caliper::MultiCircleResult::found)
		.def_readwrite("edges", &caliper::MultiCircleResult::edges);
	py::class_<caliper::ArcResult>(m, "ArcResult")
		.def(py::init<>())
		.def_readwrite("found", &caliper::ArcResult::found)
		.def_readwrite("center", &caliper::ArcResult::center)
		.def_readwrite("radius", &caliper::ArcResult::radius)
		.def_readwrite("rms", &caliper::ArcResult::rms)
		.def_readwrite("startDeg", &caliper::ArcResult::startDeg)
		.def_readwrite("endDeg", &caliper::ArcResult::endDeg)
		.def_readwrite("spanDeg", &caliper::ArcResult::spanDeg)
		.def_readwrite("arcLength", &caliper::ArcResult::arcLength)
		.def_readwrite("chord", &caliper::ArcResult::chord)
		.def_readwrite("chordStart", &caliper::ArcResult::chordStart)
		.def_readwrite("chordEnd", &caliper::ArcResult::chordEnd)
		.def_readwrite("numPoints", &caliper::ArcResult::numPoints)
		.def_readwrite("numInliers", &caliper::ArcResult::numInliers)
		.def_readwrite("points", &caliper::ArcResult::points)
		.def_readwrite("inliers", &caliper::ArcResult::inliers);
	py::class_<caliper::DualWidthResult>(m, "DualWidthResult")
		.def(py::init<>())
		.def_readwrite("found", &caliper::DualWidthResult::found)
		.def_readwrite("radial", &caliper::DualWidthResult::radial)
		.def_readwrite("width", &caliper::DualWidthResult::width)
		.def_readwrite("rms", &caliper::DualWidthResult::rms)
		.def_readwrite("angle", &caliper::DualWidthResult::angle)
		.def_readwrite("numSamples", &caliper::DualWidthResult::numSamples)
		.def_readwrite("edge0", &caliper::DualWidthResult::edge0)
		.def_readwrite("edge1", &caliper::DualWidthResult::edge1)
		.def_readwrite("circle0", &caliper::DualWidthResult::circle0)
		.def_readwrite("circle1", &caliper::DualWidthResult::circle1)
		.def_readwrite("samples", &caliper::DualWidthResult::samples);
	py::class_<caliper::GapResult>(m, "GapResult")
		.def(py::init<>())
		.def_readwrite("found", &caliper::GapResult::found)
		.def_readwrite("parallel", &caliper::GapResult::parallel)
		.def_readwrite("width", &caliper::GapResult::width)
		.def_readwrite("angle", &caliper::GapResult::angle)
		.def_readwrite("rms", &caliper::GapResult::rms)
		.def_readwrite("first", &caliper::GapResult::first)
		.def_readwrite("second", &caliper::GapResult::second)
		.def_readwrite("p1", &caliper::GapResult::p1)
		.def_readwrite("p2", &caliper::GapResult::p2);

	m.def("find_edges",
		  [](const cv::Mat& img, double x0, double y0, double x1, double y1, bool search_horizontal,
			 const caliper::CaliperParam& param) {
			  return caliper::findEdgesRect(img, x0, y0, x1, y1, search_horizontal, param);
		  },
		  py::arg("image"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
		  py::arg("search_horizontal") = true, py::arg("param") = caliper::CaliperParam());
	m.def("find_edges_oriented",
		  [](const cv::Mat& img, double cx, double cy, double phi_deg, double length1, double length2,
			 const caliper::CaliperParam& param) {
			  return caliper::findEdgesOriented(img, {cx, cy}, phi_deg * CV_PI / 180.0, length1, length2, param);
		  },
		  py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("phi_deg"), py::arg("length1"), py::arg("length2"),
		  py::arg("param") = caliper::CaliperParam());
	m.def("find_circle_edges",
		  [](const cv::Mat& img, double cx, double cy, double radius, double search, double start_deg, double end_deg,
			 const caliper::CaliperParam& param) {
			  return caliper::findCircleEdges(img, {cx, cy}, radius, search, start_deg, end_deg, param);
		  },
		  py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("radius"), py::arg("search") = 15.0,
		  py::arg("start_deg") = 0.0, py::arg("end_deg") = 360.0, py::arg("param") = caliper::CaliperParam());
	m.def("find_arc",
		  [](const cv::Mat& img, double cx, double cy, double radius, double search, double start_deg, double end_deg,
			 const caliper::CaliperParam& param) {
			  return caliper::findArc(img, {cx, cy}, radius, search, start_deg, end_deg, param);
		  },
		  py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("radius"), py::arg("search") = 15.0,
		  py::arg("start_deg") = 0.0, py::arg("end_deg") = 360.0, py::arg("param") = caliper::CaliperParam());
	m.def("measure_dual_width",
		  [](const cv::Mat& img, double cx, double cy, double phi_deg, double length1, double length2,
			 const caliper::CaliperParam& param) {
			  return caliper::measureDualWidth(img, {cx, cy}, phi_deg * CV_PI / 180.0, length1, length2, param);
		  },
		  py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("phi_deg"), py::arg("length1"), py::arg("length2"),
		  py::arg("param") = caliper::CaliperParam());
	m.def("measure_radial_width",
		  [](const cv::Mat& img, double cx, double cy, double radius, double search, double start_deg, double end_deg,
			 const caliper::CaliperParam& param) {
			  return caliper::measureRadialWidth(img, {cx, cy}, radius, search, start_deg, end_deg, param);
		  },
		  py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("radius"), py::arg("search") = 15.0,
		  py::arg("start_deg") = 0.0, py::arg("end_deg") = 360.0, py::arg("param") = caliper::CaliperParam());
	m.def("measure_gap",
		  [](const cv::Mat& img, double cx, double cy, double phi_deg, double length1, double half_gap, double search,
			 const caliper::CaliperParam& param) {
			  return caliper::measureGap(img, {cx, cy}, phi_deg * CV_PI / 180.0, length1, half_gap, search, param);
		  },
		  py::arg("image"), py::arg("cx"), py::arg("cy"), py::arg("phi_deg"), py::arg("length1"), py::arg("half_gap"),
		  py::arg("search"), py::arg("param") = caliper::CaliperParam());
	m.def("gap_between", &caliper::gapBetween, py::arg("first"), py::arg("second"));

	py::enum_<measure::FitMethod>(m, "FitMethod")
		.value("LeastSquares", measure::FitMethod::LeastSquares)
		.value("Ransac", measure::FitMethod::Ransac)
		.export_values();
	py::class_<measure::FitParam>(m, "FitParam")
		.def(py::init<>())
		.def_readwrite("method", &measure::FitParam::method)
		.def_readwrite("threshold", &measure::FitParam::threshold)
		.def_readwrite("iterations", &measure::FitParam::iterations)
		.def_readwrite("seed", &measure::FitParam::seed);

	py::class_<measure::CircleFit>(m, "CircleFit")
		.def(py::init<>())
		.def_readwrite("found", &measure::CircleFit::found)
		.def_readwrite("center", &measure::CircleFit::center)
		.def_readwrite("radius", &measure::CircleFit::radius)
		.def_readwrite("rms", &measure::CircleFit::rms)
		.def_readwrite("numPoints", &measure::CircleFit::numPoints)
		.def_readwrite("numInliers", &measure::CircleFit::numInliers)
		.def_readwrite("inliers", &measure::CircleFit::inliers);
	py::class_<measure::EllipseFit>(m, "EllipseFit")
		.def(py::init<>())
		.def_readwrite("found", &measure::EllipseFit::found)
		.def_readwrite("center", &measure::EllipseFit::center)
		.def_readwrite("a", &measure::EllipseFit::a)
		.def_readwrite("b", &measure::EllipseFit::b)
		.def_readwrite("angle", &measure::EllipseFit::angle)
		.def_readwrite("rms", &measure::EllipseFit::rms)
		.def_readwrite("numPoints", &measure::EllipseFit::numPoints)
		.def_readwrite("numInliers", &measure::EllipseFit::numInliers)
		.def_readwrite("inliers", &measure::EllipseFit::inliers);
	py::class_<measure::BoxSize>(m, "BoxSize")
		.def(py::init<>())
		.def_readwrite("found", &measure::BoxSize::found)
		.def_readwrite("center", &measure::BoxSize::center)
		.def_readwrite("width", &measure::BoxSize::width)
		.def_readwrite("height", &measure::BoxSize::height)
		.def_readwrite("angle", &measure::BoxSize::angle)
		.def_readwrite("c0", &measure::BoxSize::c0)
		.def_readwrite("c1", &measure::BoxSize::c1)
		.def_readwrite("c2", &measure::BoxSize::c2)
		.def_readwrite("c3", &measure::BoxSize::c3);
	py::class_<measure::QuadFit>(m, "QuadFit")
		.def(py::init<>())
		.def_readwrite("found", &measure::QuadFit::found)
		.def_readwrite("rectangle", &measure::QuadFit::rectangle)
		.def_readwrite("center", &measure::QuadFit::center)
		.def_readwrite("width", &measure::QuadFit::width)
		.def_readwrite("height", &measure::QuadFit::height)
		.def_readwrite("angle", &measure::QuadFit::angle)
		.def_readwrite("angleError", &measure::QuadFit::angleError)
		.def_readwrite("rms", &measure::QuadFit::rms)
		.def_readwrite("c0", &measure::QuadFit::c0)
		.def_readwrite("c1", &measure::QuadFit::c1)
		.def_readwrite("c2", &measure::QuadFit::c2)
		.def_readwrite("c3", &measure::QuadFit::c3);
	py::class_<measure::FormResult>(m, "FormResult")
		.def(py::init<>())
		.def_readwrite("found", &measure::FormResult::found)
		.def_readwrite("ok", &measure::FormResult::pass)
		.def_readwrite("value", &measure::FormResult::value)
		.def_readwrite("tolerance", &measure::FormResult::tolerance)
		.def_readwrite("margin", &measure::FormResult::margin)
		.def_readwrite("rms", &measure::FormResult::rms);
	py::class_<measure::CircleSize>(m, "CircleSize")
		.def(py::init<>())
		.def_readwrite("found", &measure::CircleSize::found)
		.def_readwrite("center", &measure::CircleSize::center)
		.def_readwrite("radius", &measure::CircleSize::radius)
		.def_readwrite("diameter", &measure::CircleSize::diameter);
	py::class_<measure::WidthFunction>(m, "WidthFunction")
		.def(py::init<>())
		.def_readwrite("found", &measure::WidthFunction::found)
		.def_readwrite("minWidth", &measure::WidthFunction::minWidth)
		.def_readwrite("maxWidth", &measure::WidthFunction::maxWidth)
		.def_readwrite("minAngle", &measure::WidthFunction::minAngle)
		.def_readwrite("maxAngle", &measure::WidthFunction::maxAngle)
		.def_readwrite("minA", &measure::WidthFunction::minA)
		.def_readwrite("minB", &measure::WidthFunction::minB)
		.def_readwrite("maxA", &measure::WidthFunction::maxA)
		.def_readwrite("maxB", &measure::WidthFunction::maxB);
	py::class_<measure::GageResult>(m, "GageResult")
		.def(py::init<>())
		.def_readwrite("found", &measure::GageResult::found)
		.def_readwrite("value", &measure::GageResult::value)
		.def_readwrite("centerDistance", &measure::GageResult::centerDistance)
		.def_readwrite("angle", &measure::GageResult::angle)
		.def_readwrite("a", &measure::GageResult::a)
		.def_readwrite("b", &measure::GageResult::b);
	py::class_<measure::PixelScale>(m, "PixelScale")
		.def(py::init<>())
		.def_readwrite("found", &measure::PixelScale::found)
		.def_readwrite("mmPerPx", &measure::PixelScale::mmPerPx)
		.def_readwrite("mmPerPxX", &measure::PixelScale::mmPerPxX)
		.def_readwrite("mmPerPxY", &measure::PixelScale::mmPerPxY)
		.def_readwrite("source", &measure::PixelScale::source);
	py::class_<measure::FixtureFrame>(m, "FixtureFrame")
		.def(py::init<>())
		.def_readwrite("origin", &measure::FixtureFrame::origin)
		.def_readwrite("angleDeg", &measure::FixtureFrame::angleDeg);
	py::class_<measure::DimValue>(m, "DimValue")
		.def(py::init<>())
		.def_readwrite("found", &measure::DimValue::found)
		.def_readwrite("scaled", &measure::DimValue::scaled)
		.def_readwrite("pixels", &measure::DimValue::pixels)
		.def_readwrite("physical", &measure::DimValue::physical)
		.def_readwrite("fromPx", &measure::DimValue::fromPx)
		.def_readwrite("toPx", &measure::DimValue::toPx)
		.def_readwrite("fromFrame", &measure::DimValue::fromFrame)
		.def_readwrite("toFrame", &measure::DimValue::toFrame);

	m.def("fit_line",
		  [](const py::handle& points, const measure::FitParam& param) {
			  return measure::fitLine(pointsFromPy(points), param);
		  },
		  py::arg("points"), py::arg("param") = measure::FitParam());
	m.def("fit_circle",
		  [](const py::handle& points, const measure::FitParam& param) {
			  return measure::fitCircle(pointsFromPy(points), param);
		  },
		  py::arg("points"), py::arg("param") = measure::FitParam());
	m.def("fit_ellipse",
		  [](const py::handle& points, const measure::FitParam& param) {
			  return measure::fitEllipse(pointsFromPy(points), param);
		  },
		  py::arg("points"), py::arg("param") = measure::FitParam());
	m.def("fit_parallelogram",
		  [](const py::handle& points) { return measure::fitParallelogram(pointsFromPy(points)); },
		  py::arg("points"));
	m.def("size_aabb",
		  [](const py::handle& points) { return measure::axisAlignedBox(pointsFromPy(points)); },
		  py::arg("points"));
	m.def("size_min_area",
		  [](const py::handle& points) { return measure::minAreaBox(pointsFromPy(points)); },
		  py::arg("points"));
	m.def("contour_points", &measure::contourPoints, py::arg("image"));
	m.def("size_min_area_contour", &measure::minAreaBoxContour, py::arg("image"));
	m.def("fit_parallelogram_contour", &measure::fitParallelogramContour, py::arg("image"));

	m.def("check_parallelism",
		  [](double ax0, double ay0, double ax1, double ay1, double bx0, double by0, double bx1, double by1, double tol) {
			  return measure::checkParallelism({ax0, ay0}, {ax1, ay1}, {bx0, by0}, {bx1, by1}, tol);
		  },
		  py::arg("ax0"), py::arg("ay0"), py::arg("ax1"), py::arg("ay1"),
		  py::arg("bx0"), py::arg("by0"), py::arg("bx1"), py::arg("by1"), py::arg("tol_deg"));
	m.def("check_perpendicularity",
		  [](double ax0, double ay0, double ax1, double ay1, double bx0, double by0, double bx1, double by1, double tol) {
			  return measure::checkPerpendicularity({ax0, ay0}, {ax1, ay1}, {bx0, by0}, {bx1, by1}, tol);
		  },
		  py::arg("ax0"), py::arg("ay0"), py::arg("ax1"), py::arg("ay1"),
		  py::arg("bx0"), py::arg("by0"), py::arg("bx1"), py::arg("by1"), py::arg("tol_deg"));
	m.def("check_straightness",
		  [](const py::handle& points, double tol) { return measure::checkStraightness(pointsFromPy(points), tol); },
		  py::arg("points"), py::arg("tol_px"));
	m.def("check_roundness",
		  [](const py::handle& points, double tol) { return measure::checkRoundness(pointsFromPy(points), tol); },
		  py::arg("points"), py::arg("tol_px"));

	m.def("circle_inscribed",
		  [](const py::handle& points) { return measure::inscribedCircle(pointsFromPy(points)); },
		  py::arg("points"));
	m.def("circle_inscribed_mask", &measure::inscribedCircleMask, py::arg("image"));
	m.def("circle_circumscribed",
		  [](const py::handle& points) { return measure::circumscribedCircle(pointsFromPy(points)); },
		  py::arg("points"));
	m.def("width_function",
		  [](const py::handle& points, double step) { return measure::widthFunction(pointsFromPy(points), step); },
		  py::arg("points"), py::arg("step_deg") = 1.0);

	m.def("hole_to_edge",
		  [](double cx, double cy, double radius, double x0, double y0, double x1, double y1) {
			  return measure::holeToEdge({cx, cy}, radius, {x0, y0}, {x1, y1});
		  },
		  py::arg("cx"), py::arg("cy"), py::arg("radius"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"));
	m.def("hole_to_hole",
		  [](double x0, double y0, double r0, double x1, double y1, double r1) {
			  return measure::holeToHole({x0, y0}, r0, {x1, y1}, r1);
		  },
		  py::arg("x0"), py::arg("y0"), py::arg("r0"), py::arg("x1"), py::arg("y1"), py::arg("r1"));
	m.def("edge_to_edge",
		  [](double ax0, double ay0, double ax1, double ay1, double bx0, double by0, double bx1, double by1, double tol) {
			  return measure::edgeToEdge({ax0, ay0}, {ax1, ay1}, {bx0, by0}, {bx1, by1}, tol);
		  },
		  py::arg("ax0"), py::arg("ay0"), py::arg("ax1"), py::arg("ay1"),
		  py::arg("bx0"), py::arg("by0"), py::arg("bx1"), py::arg("by1"), py::arg("angle_tol_deg") = 15.0);

	m.def("scale_mm_per_px", &measure::scaleMmPerPx, py::arg("mm_per_px"));
	m.def("scale_mm_per_px_xy", &measure::scaleMmPerPxXY, py::arg("mm_per_px_x"), py::arg("mm_per_px_y"));
	m.def("scale_from_camera", &measure::scaleFromCameraMatrix, py::arg("camera_matrix"), py::arg("z_mm"));
	m.def("scale_from_yaml", &measure::scaleFromYaml, py::arg("path"), py::arg("z_mm") = 0.0);
	m.def("to_fixture",
		  [](double x, double y, const measure::PixelScale& scale, const measure::FixtureFrame& frame) {
			  return measure::toFixture({x, y}, scale, frame);
		  },
		  py::arg("x"), py::arg("y"), py::arg("scale") = measure::PixelScale(),
		  py::arg("frame") = measure::FixtureFrame());
	m.def("dimension_points",
		  [](double x0, double y0, double x1, double y1, const measure::PixelScale& scale,
			 const measure::FixtureFrame& frame) {
			  return measure::dimensionPoints({x0, y0}, {x1, y1}, scale, frame);
		  },
		  py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
		  py::arg("scale") = measure::PixelScale(), py::arg("frame") = measure::FixtureFrame());
	m.def("dimension_point_line",
		  [](double px, double py, double x0, double y0, double x1, double y1, const measure::PixelScale& scale,
			 const measure::FixtureFrame& frame) {
			  return measure::dimensionPointLine({px, py}, {x0, y0}, {x1, y1}, scale, frame);
		  },
		  py::arg("px"), py::arg("py"), py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
		  py::arg("scale") = measure::PixelScale(), py::arg("frame") = measure::FixtureFrame());
	m.def("dimension_lines",
		  [](double ax0, double ay0, double ax1, double ay1, double bx0, double by0, double bx1, double by1,
			 const measure::PixelScale& scale, const measure::FixtureFrame& frame, double tol) {
			  return measure::dimensionLines({ax0, ay0}, {ax1, ay1}, {bx0, by0}, {bx1, by1}, scale, frame, tol);
		  },
		  py::arg("ax0"), py::arg("ay0"), py::arg("ax1"), py::arg("ay1"),
		  py::arg("bx0"), py::arg("by0"), py::arg("bx1"), py::arg("by1"),
		  py::arg("scale") = measure::PixelScale(), py::arg("frame") = measure::FixtureFrame(),
		  py::arg("angle_tol_deg") = 15.0);
	m.def("scale_gage", &measure::scaleGage, py::arg("gage"), py::arg("scale") = measure::PixelScale(),
		  py::arg("frame") = measure::FixtureFrame());
}
