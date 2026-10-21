// 尺寸测量。不依赖 TensorRT。编译见 Makefile 的 test-measure。
#include "cv_caliper/caliper.h"
#include "cv_caliper/metrology.h"
#include "cv_measure/measure.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{

int g_fails = 0;

void expect(bool cond, const std::string& msg)
{
	if (cond) {
		std::cout << "OK  " << msg << std::endl;
		return;
	}
	std::cerr << "FAIL " << msg << std::endl;
	++g_fails;
}

void expectNear(double value, double target, double tol, const std::string& msg)
{
	expect(std::isfinite(value) && std::abs(value - target) <= tol,
		   msg + " got " + std::to_string(value) + " want " + std::to_string(target));
}

double angDist(double a, double b)
{
	double d = std::fmod(a - b, 180.0);
	if (d > 90.0) {
		d -= 180.0;
	}
	if (d < -90.0) {
		d += 180.0;
	}
	return std::abs(d);
}

caliper::CaliperParam bothParam()
{
	caliper::CaliperParam p;
	p.polarity = caliper::Polarity::Both;
	p.contrast = 12;
	p.stride = 3;
	p.projection = 3;
	p.sigma = 0.6;
	p.minPoints = 6;
	p.outlierRatio = 0.2;
	return p;
}

void testGeometry()
{
	expectNear(metrology::segmentLength({0, 0}, {3, 4}), 5.0, 1e-9, "segment length");
	const cv::Point2d mid = metrology::midpoint({0, 0}, {4, 2});
	expectNear(mid.x, 2.0, 1e-9, "midpoint x");
	expectNear(mid.y, 1.0, 1e-9, "midpoint y");

	const auto foot = metrology::footOfPerpendicular({3, 5}, {0, 0}, {10, 0}, false);
	expect(foot.found, "foot found");
	expectNear(foot.point.x, 3.0, 1e-9, "foot x");
	expectNear(foot.point.y, 0.0, 1e-9, "foot y");
	const auto clamped = metrology::footOfPerpendicular({-2, 5}, {0, 0}, {10, 0}, true);
	expect(clamped.found && std::abs(clamped.point.x) < 1e-9 && std::abs(clamped.point.y) < 1e-9, "clamped foot");

	const auto hit = metrology::intersectLinesAt({0, 0}, {10, 0}, {0, 0}, {0, 10});
	expect(hit.found && std::abs(hit.point.x) < 1e-6 && std::abs(hit.point.y) < 1e-6, "intersect axes");
	expect(!metrology::intersectLinesAt({0, 0}, {10, 0}, {0, 1}, {10, 1}).found, "parallel no intersection");

	expectNear(metrology::distanceParallelLines({0, 0}, {10, 0}, {1, 4}, {9, 4}, 5.0), 4.0, 1e-6, "parallel distance");
	expect(!std::isfinite(metrology::distanceParallelLines({0, 0}, {10, 0}, {0, 0}, {0, 10}, 5.0)),
		   "perpendicular is not a parallel distance");

	expectNear(metrology::distanceCircleCenters({0, 0}, {3, 4}), 5.0, 1e-9, "circle centers");
	expectNear(metrology::signedDistancePointToCircle({13, 0}, {0, 0}, 10), 3.0, 1e-9, "point outside circle");
	expectNear(metrology::signedDistancePointToCircle({6, 0}, {0, 0}, 10), -4.0, 1e-9, "point inside circle");
	expectNear(metrology::distancePointToCircle({6, 0}, {0, 0}, 10), 4.0, 1e-9, "point to circumference");
	expectNear(metrology::signedDistanceCircleToLine({0, 0}, 10, {0, 12}, {5, 12}), 2.0, 1e-6, "circle line gap signed");
	expectNear(metrology::distanceCircleToLine({0, 0}, 10, {0, 4}, {5, 4}), 0.0, 1e-6, "intersecting circle line gap");
	expect(!std::isfinite(metrology::distancePointToCircle({0, 0}, {0, 0}, -1)), "negative radius");
}

void testFitAndForm()
{
	std::vector<cv::Point2d> line;
	for (int i = 0; i < 30; ++i) {
		line.emplace_back(i, 0.2 * i + 1.0);
	}
	line.emplace_back(5, 40);
	line.emplace_back(15, -30);
	line.emplace_back(25, 35);
	const auto ls = measure::fitLine(line);
	expect(ls.found && ls.rms > 2.0, "least squares line feels outliers");
	measure::FitParam ransac;
	ransac.method = measure::FitMethod::Ransac;
	ransac.threshold = 1.5;
	const auto robust = measure::fitLine(line, ransac);
	expect(robust.found && robust.rms < 0.2 && robust.numInliers >= 28, "ransac line drops outliers");

	std::vector<cv::Point2d> circle;
	for (int i = 0; i < 36; ++i) {
		const double t = i * 2.0 * CV_PI / 36.0;
		circle.emplace_back(20.0 + 15.0 * std::cos(t), 10.0 + 15.0 * std::sin(t));
	}
	circle.emplace_back(20, 40);
	circle.emplace_back(-10, 10);
	const auto circleFit = measure::fitCircle(circle, ransac);
	expect(circleFit.found, "ransac circle found");
	expectNear(circleFit.radius, 15.0, 0.4, "ransac circle radius");
	expect(std::hypot(circleFit.center.x - 20, circleFit.center.y - 10) < 0.4, "ransac circle center");

	std::vector<cv::Point2d> ellipse;
	const double ang = 25.0 * CV_PI / 180.0;
	const double ca = std::cos(ang);
	const double sa = std::sin(ang);
	for (int i = 0; i < 48; ++i) {
		const double t = i * 2.0 * CV_PI / 48.0;
		const double x = 50.0 * std::cos(t);
		const double y = 22.0 * std::sin(t);
		ellipse.emplace_back(80.0 + x * ca - y * sa, 70.0 + x * sa + y * ca);
	}
	const auto ell = measure::fitEllipse(ellipse);
	expect(ell.found, "ellipse fit");
	expectNear(ell.a, 50.0, 2.0, "ellipse major");
	expectNear(ell.b, 22.0, 2.0, "ellipse minor");
	expect(std::hypot(ell.center.x - 80, ell.center.y - 70) < 2.0, "ellipse center");
	expect(angDist(ell.angle, 25.0) < 5.0, "ellipse angle got " + std::to_string(ell.angle));

	const cv::Point2d q0(30, 40), q1(150, 55), q2(130, 120), q3(10, 105);
	std::vector<cv::Point2d> quad = {q0, q1, q2, q3};
	for (int i = 1; i < 8; ++i) {
		const double t = i / 8.0;
		quad.push_back(q0 * (1 - t) + q1 * t);
		quad.push_back(q1 * (1 - t) + q2 * t);
		quad.push_back(q2 * (1 - t) + q3 * t);
		quad.push_back(q3 * (1 - t) + q0 * t);
	}
	const auto para = measure::fitParallelogram(quad);
	expect(para.found && !para.rectangle, "parallelogram is not a rectangle");
	expect(std::hypot(para.center.x - 80, para.center.y - 80) < 1.0, "parallelogram center");
	expectNear(std::max(para.width, para.height), std::hypot(120, 15), 1.0, "parallelogram long side");
	expectNear(std::min(para.width, para.height), std::hypot(20, 65), 1.0, "parallelogram short side");

	std::vector<cv::Point2d> rect = {{10, 10}, {110, 10}, {110, 50}, {10, 50}};
	const auto asRect = measure::fitParallelogram(rect);
	expect(asRect.found && asRect.rectangle && asRect.angleError < 2.0, "rectangle flag");
	expectNear(asRect.width, 100.0, 1.0, "rectangle width");
	expectNear(asRect.height, 40.0, 1.0, "rectangle height");

	std::vector<cv::Point2d> straight;
	for (int i = 0; i < 21; ++i) {
		straight.emplace_back(i, 0);
	}
	const auto okLine = measure::checkStraightness(straight, 0.1);
	expect(okLine.found && okLine.pass && okLine.value < 1e-6, "straightness pass");
	straight.emplace_back(10, 3);
	const auto badLine = measure::checkStraightness(straight, 1.0);
	expect(badLine.found && !badLine.pass && badLine.value >= 3.0 - 1e-6, "straightness fail");

	std::vector<cv::Point2d> round;
	for (int i = 0; i < 24; ++i) {
		const double t = i * 2.0 * CV_PI / 24.0;
		round.emplace_back(30.0 * std::cos(t), 30.0 * std::sin(t));
	}
	const auto okRound = measure::checkRoundness(round, 0.5);
	expect(okRound.found && okRound.pass && okRound.value < 0.2, "roundness pass got " + std::to_string(okRound.value));
	round.emplace_back(38, 0);
	const auto badRound = measure::checkRoundness(round, 1.0);
	expect(badRound.found && !badRound.pass && badRound.value >= 6.0, "roundness fail");

	const auto parallel = measure::checkParallelism({0, 0}, {10, 0}, {0, 4}, {8, 4}, 1.0);
	expect(parallel.found && parallel.pass && parallel.value < 1e-6, "parallelism");
	const auto perp = measure::checkPerpendicularity({0, 0}, {10, 0}, {0, 0}, {0, 10}, 1.0);
	expect(perp.found && perp.pass && perp.value < 1e-6, "perpendicularity");
	const auto skew = measure::checkPerpendicularity({0, 0}, {10, 0}, {0, 0}, {1, 5}, 1.0);
	expect(skew.found && !skew.pass && skew.value > 5.0, "perpendicularity fail");
}

void testSizeAndGage()
{
	const std::vector<cv::Point2d> box = {{0, 0}, {100, 0}, {100, 40}, {0, 40}};
	const auto aabb = measure::axisAlignedBox(box);
	expect(aabb.found, "aabb found");
	expectNear(aabb.width, 100, 1e-6, "aabb width");
	expectNear(aabb.height, 40, 1e-6, "aabb height");
	const auto rot = measure::minAreaBox(box);
	expect(rot.found, "min area found");
	expectNear(std::max(rot.width, rot.height), 100, 1.0, "min area long");
	expectNear(std::min(rot.width, rot.height), 40, 1.0, "min area short");
	const auto circ = measure::circumscribedCircle(box);
	expect(circ.found, "circumscribed");
	expectNear(circ.diameter, std::hypot(100, 40), 0.3, "circumscribed diameter");
	const auto inside = measure::inscribedCircle(box);
	expect(inside.found, "inscribed");
	expectNear(inside.diameter, 40, 3.0, "inscribed diameter");
	const auto width = measure::widthFunction(box, 1.0);
	expect(width.found, "width function");
	expectNear(width.minWidth, 40, 0.2, "min width");
	expectNear(width.maxWidth, std::hypot(100, 40), 0.5, "max width");
	expect(angDist(width.maxAngle, std::atan2(40.0, 100.0) * 180.0 / CV_PI) < 2.0,
		   "max width angle " + std::to_string(width.maxAngle));
	expect(angDist(width.minAngle, 90.0) < 2.0, "min width angle " + std::to_string(width.minAngle));

	const auto holeEdge = measure::holeToEdge({0, 0}, 10, {30, -5}, {30, 8});
	expect(holeEdge.found, "hole to edge");
	expectNear(holeEdge.value, 20, 1e-6, "hole clearance");
	expectNear(holeEdge.centerDistance, 30, 1e-6, "hole center distance");
	const auto holes = measure::holeToHole({0, 0}, 15, {80, 0}, 20);
	expect(holes.found, "hole to hole");
	expectNear(holes.centerDistance, 80, 1e-9, "hole center span");
	expectNear(holes.value, 45, 1e-6, "hole rim gap");
	const auto edges = measure::edgeToEdge({0, 0}, {20, 0}, {0, 9}, {15, 9}, 15);
	expect(edges.found, "edge to edge");
	expectNear(edges.value, 9, 1e-6, "edge gap");
	expect(!measure::edgeToEdge({0, 0}, {10, 0}, {0, 0}, {0, 10}, 15).found, "skew edges rejected");

	cv::Mat mask(100, 140, CV_8U, cv::Scalar(0));
	cv::rectangle(mask, cv::Rect(10, 10, 120, 80), cv::Scalar(255), cv::FILLED);
	cv::rectangle(mask, cv::Rect(80, 35, 50, 30), cv::Scalar(0), cv::FILLED);
	const auto notch = measure::inscribedCircleMask(mask);
	expect(notch.found, "concave inscribed");
	expect(notch.diameter > 60.0 && notch.diameter < 78.0,
		   "concave inscribed diameter " + std::to_string(notch.diameter));
}

void testScale()
{
	const auto iso = measure::scaleMmPerPx(0.1);
	expect(iso.found && std::abs(iso.mmPerPx - 0.1) < 1e-12, "explicit scale");
	measure::FixtureFrame frame;
	frame.origin = {100, 50};
	frame.angleDeg = 90;
	const cv::Point2d local = measure::toFixture({100, 60}, iso, frame);
	expectNear(local.x, 1.0, 1e-6, "fixture x");
	expectNear(local.y, 0.0, 1e-6, "fixture y");
	const auto dim = measure::dimensionPoints({100, 50}, {100, 60}, iso, frame);
	expect(dim.found && dim.scaled, "dimension scaled");
	expectNear(dim.pixels, 10, 1e-9, "dimension pixels");
	expectNear(dim.physical, 1, 1e-6, "dimension mm");
	expectNear(dim.toFrame.x, 1, 1e-6, "dimension frame x");

	const auto xy = measure::scaleMmPerPxXY(0.05, 0.0625);
	const auto diag = measure::dimensionPoints({0, 0}, {100, 100}, xy, {});
	expectNear(diag.physical, std::hypot(5.0, 6.25), 1e-6, "anisotropic length");
	const auto toLine = measure::dimensionPointLine({0, 5}, {0, 0}, {10, 0}, measure::scaleMmPerPx(0.2), {});
	expectNear(toLine.pixels, 5, 1e-6, "point line pixels");
	expectNear(toLine.physical, 1, 1e-6, "point line mm");

	cv::Mat K = (cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 800, 240, 0, 0, 1);
	const auto fromK = measure::scaleFromCameraMatrix(K, 50);
	expect(fromK.found, "camera scale");
	expectNear(fromK.mmPerPxX, 0.05, 1e-9, "fx scale");
	expectNear(fromK.mmPerPxY, 0.0625, 1e-9, "fy scale");

	const std::string path = "/tmp/cvter_measure_scale.yml";
	{
		cv::FileStorage fs(path, cv::FileStorage::WRITE);
		fs << "found" << 1;
		fs << "rms" << 0.2;
		fs << "imageWidth" << 640;
		fs << "imageHeight" << 480;
		fs << "cameraMatrix" << K;
		fs << "distCoeffs" << cv::Mat::zeros(1, 5, CV_64F);
		fs << "nviews" << 0;
	}
	const auto loaded = measure::scaleFromYaml(path, 50);
	expect(loaded.found && loaded.source.find("cameraMatrix") != std::string::npos, "yaml camera scale");
	expectNear(loaded.mmPerPxX, 0.05, 1e-9, "yaml fx");
	{
		cv::FileStorage fs("/tmp/cvter_measure_mmpx.yml", cv::FileStorage::WRITE);
		fs << "mmPerPx" << 0.02;
	}
	const auto direct = measure::scaleFromYaml("/tmp/cvter_measure_mmpx.yml", 0);
	expect(direct.found, "yaml mmPerPx");
	expectNear(direct.mmPerPx, 0.02, 1e-12, "yaml mm value");

	const auto gage = measure::holeToHole({0, 0}, 10, {40, 0}, 10);
	const auto scaled = measure::scaleGage(gage, measure::scaleMmPerPx(0.25), {});
	expectNear(gage.value, 20, 1e-6, "gage pixels");
	expectNear(scaled.physical, 5, 1e-6, "gage mm");
}

void testCalipers()
{
	cv::Mat edges(180, 220, CV_8U, cv::Scalar(20));
	edges.colRange(40, 100).setTo(200);
	edges.colRange(100, 150).setTo(80);
	edges.colRange(150, 220).setTo(230);
	cv::GaussianBlur(edges, edges, cv::Size(5, 5), 0.8);
	auto param = bothParam();
	param.select = caliper::EdgeSelect::All;
	const auto all = caliper::findEdgesRect(edges, 10, 20, 190, 160, true, param);
	expect(all.found && all.edges.size() >= 3, "three edges " + std::to_string(all.edges.size()));
	auto xOf = [](const caliper::MultiLineResult& m, int index) {
		for (const auto& e : m.edges) {
			if (e.index == index && e.line.found) {
				return e.line.center.x;
			}
		}
		return -1.0;
	};
	expectNear(xOf(all, 0), 40, 1.8, "edge 0");
	expectNear(xOf(all, 1), 100, 1.8, "edge 1");
	expectNear(xOf(all, 2), 150, 1.8, "edge 2");

	param.select = caliper::EdgeSelect::Nth;
	param.edgeIndex = 1;
	const auto nth = caliper::findLineRect(edges, 10, 20, 190, 160, true, param);
	expect(nth.found, "nth edge");
	expectNear(nth.center.x, 100, 2.0, "nth x");
	param.select = caliper::EdgeSelect::First;
	const auto first = caliper::findLineRect(edges, 10, 20, 190, 160, true, param);
	expect(first.found, "first edge");
	expectNear(first.center.x, 40, 4.0, "first x");
	param.select = caliper::EdgeSelect::Last;
	const auto last = caliper::findLineRect(edges, 10, 20, 190, 160, true, param);
	expect(last.found, "last edge");
	expectNear(last.center.x, 150, 4.0, "last x");

	cv::Mat bar(160, 200, CV_8U, cv::Scalar(30));
	bar.colRange(70, 130).setTo(210);
	cv::GaussianBlur(bar, bar, cv::Size(5, 5), 0.8);
	const auto wall = caliper::measureDualWidth(bar, {100, 80}, CV_PI / 2.0, 55, 45, bothParam());
	expect(wall.found, "wall width found");
	expectNear(wall.width, 60, 2.0, "wall width");

	cv::Mat slot(160, 220, CV_8U, cv::Scalar(210));
	slot.colRange(80, 145).setTo(30);
	cv::GaussianBlur(slot, slot, cv::Size(5, 5), 0.8);
	const auto groove = caliper::measureDualWidth(slot, {112.5, 80}, CV_PI / 2.0, 55, 50, bothParam());
	expect(groove.found, "slot width found");
	expectNear(groove.width, 65, 2.0, "slot width");

	cv::Mat gap(180, 250, CV_8U, cv::Scalar(40));
	gap.colRange(30, 90).setTo(210);
	gap.colRange(150, 220).setTo(210);
	cv::GaussianBlur(gap, gap, cv::Size(5, 5), 0.8);
	caliper::CaliperParam inward = bothParam();
	inward.polarity = caliper::Polarity::LightToDark;
	const auto opening = caliper::measureGap(gap, {120, 90}, CV_PI / 2.0, 55, 30, 16, inward);
	expect(opening.found && opening.parallel, "gap found");
	expectNear(opening.width, 60, 2.5, "gap width");

	cv::Mat disk(280, 280, CV_8U, cv::Scalar(200));
	cv::circle(disk, {140, 140}, 60, cv::Scalar(25), cv::FILLED);
	cv::GaussianBlur(disk, disk, cv::Size(5, 5), 0.8);
	caliper::CaliperParam arcParam = bothParam();
	arcParam.polarity = caliper::Polarity::DarkToLight;
	const auto arc = caliper::findArc(disk, {140, 140}, 60, 16, -30, 110, arcParam);
	expect(arc.found, "arc found");
	expectNear(arc.radius, 60, 2.0, "arc radius");
	expect(arc.spanDeg > 90.0 && arc.spanDeg < 150.0, "arc span " + std::to_string(arc.spanDeg));
	expectNear(arc.arcLength, arc.radius * arc.spanDeg * CV_PI / 180.0, 1.0, "arc length");
	expect(arc.chord > 40.0 && arc.chord < arc.radius * 2.0 + 1.0, "arc chord");

	cv::Mat ring(280, 280, CV_8U, cv::Scalar(25));
	cv::circle(ring, {140, 140}, 70, cv::Scalar(200), cv::FILLED);
	cv::circle(ring, {140, 140}, 45, cv::Scalar(25), cv::FILLED);
	cv::GaussianBlur(ring, ring, cv::Size(5, 5), 1.0);
	const auto radial = caliper::measureRadialWidth(ring, {140, 140}, 58, 22, 0, 360, bothParam());
	expect(radial.found && radial.radial, "radial width found");
	expectNear(radial.width, 25, 2.5, "radial width");
	expect(radial.circle0.found && radial.circle1.found, "both ring edges");
}

}  // namespace

int main()
{
	testGeometry();
	testFitAndForm();
	testSizeAndGage();
	testScale();
	testCalipers();
	if (g_fails) {
		std::cerr << g_fails << " check(s) failed" << std::endl;
		return 1;
	}
	std::cout << "all measure checks passed" << std::endl;
	return 0;
}
