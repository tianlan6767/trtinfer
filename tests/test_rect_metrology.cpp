// 矩形卡尺 + 测量辅助。不依赖 TensorRT。编译见 Makefile 的 test-rect。
#include "cv_caliper/caliper.h"
#include "cv_caliper/metrology.h"

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

void paintRect(cv::Mat& img, cv::Point2d c, double w, double h, double phiDeg, uchar value)
{
	const double phi = phiDeg * CV_PI / 180.0;
	const cv::Point2d ux(std::cos(phi), std::sin(phi));
	const cv::Point2d uy(-std::sin(phi), std::cos(phi));
	const double hw = 0.5 * w;
	const double hh = 0.5 * h;
	std::vector<cv::Point> pts = {
		cv::Point(cvRound(c.x - ux.x * hw - uy.x * hh), cvRound(c.y - ux.y * hw - uy.y * hh)),
		cv::Point(cvRound(c.x + ux.x * hw - uy.x * hh), cvRound(c.y + ux.y * hw - uy.y * hh)),
		cv::Point(cvRound(c.x + ux.x * hw + uy.x * hh), cvRound(c.y + ux.y * hw + uy.y * hh)),
		cv::Point(cvRound(c.x - ux.x * hw + uy.x * hh), cvRound(c.y - ux.y * hw + uy.y * hh)),
	};
	cv::fillConvexPoly(img, pts, cv::Scalar(value));
}

caliper::CaliperParam brightParam()
{
	caliper::CaliperParam p;
	p.polarity = caliper::Polarity::DarkToLight;
	p.select = caliper::EdgeSelect::Strongest;
	p.contrast = 12;
	p.stride = 2;
	p.projection = 3;
	p.sigma = 0.8;
	p.minPoints = 6;
	return p;
}

void testMetrology()
{
	expect(std::abs(metrology::distancePoints({0, 0}, {3, 4}) - 5.0) < 1e-9, "point distance 3-4-5");
	expect(std::abs(metrology::distancePointToLine({0, 1}, {0, 0}, {10, 0}) - 1.0) < 1e-9, "point to horizontal line");
	const double gap = metrology::distanceBetweenLines({0, 0}, {10, 0}, {0, 4}, {8, 4});
	expect(std::abs(gap - 4.0) < 1e-6, "parallel line distance");
	const double ang = metrology::angleBetweenLines({0, 0}, {10, 0}, {0, 0}, {0, 10});
	expect(std::abs(ang - 90.0) < 1e-6, "perpendicular angle");

	std::vector<cv::Point2d> pts;
	for (int i = 0; i < 8; ++i) {
		pts.emplace_back(i, 2.0 * i + 1.0);
	}
	const metrology::LineFit fit = metrology::fitLine(pts);
	expect(fit.found && fit.rms < 1e-6, "fit line rms");
	expect(std::abs(fit.angle - std::atan2(2.0, 1.0) * 180.0 / CV_PI) < 1e-4, "fit line angle");

	const metrology::SizeSummary box = metrology::summarizeRectangle({0, 0}, {30, 0}, {30, 10}, {0, 10});
	expect(box.found && std::abs(box.width - 30) < 1e-6 && std::abs(box.height - 10) < 1e-6, "rect summary size");
	expect(std::abs(box.center.x - 15) < 1e-6 && std::abs(box.center.y - 5) < 1e-6, "rect summary center");
	const metrology::SizeSummary circ = metrology::summarizeCircle({4, 5}, 7);
	expect(circ.found && std::abs(circ.diameter - 14) < 1e-9, "circle summary diameter");
}

void testFindRectangle()
{
	const cv::Point2d truth(160, 110);
	const double width = 150;
	const double height = 90;
	const double angle = 7.0;
	cv::Mat img(240, 320, CV_8U, cv::Scalar(30));
	paintRect(img, truth, width, height, angle, 210);
	cv::GaussianBlur(img, img, cv::Size(5, 5), 1.0);

	const auto param = brightParam();
	const caliper::RectResult hit = caliper::findRectangle(
		img, cv::Point2d(truth.x + 3.0, truth.y - 2.0), width, height, angle * CV_PI / 180.0, 16.0, param);
	std::cout << "    rect found=" << hit.found << " center=(" << hit.center.x << "," << hit.center.y
			  << ") size=" << hit.width << "x" << hit.height << " angle=" << hit.angle
			  << " rms=" << hit.rms
			  << " sides=" << hit.top.found << hit.right.found << hit.bottom.found << hit.left.found
			  << std::endl;
	expect(hit.found, "rotated rectangle found");
	expect(std::hypot(hit.center.x - truth.x, hit.center.y - truth.y) < 2.5, "rectangle center");
	expect(std::abs(hit.width - width) < 3.0 && std::abs(hit.height - height) < 3.0, "rectangle size");
	expect(std::abs(hit.angle - angle) < 1.5, "rectangle angle");

	cv::Mat axis(200, 260, CV_8U, cv::Scalar(25));
	cv::rectangle(axis, cv::Rect(40, 30, 120, 70), cv::Scalar(200), cv::FILLED);
	cv::GaussianBlur(axis, axis, cv::Size(3, 3), 0.8);
	const caliper::RectResult box = caliper::findRectangleBox(axis, 40, 30, 160, 100, 12.0, param);
	std::cout << "    box found=" << box.found << " center=(" << box.center.x << "," << box.center.y
			  << ") size=" << box.width << "x" << box.height << " angle=" << box.angle << std::endl;
	expect(box.found, "axis-aligned rectangle found");
	expect(std::hypot(box.center.x - 100, box.center.y - 65) < 2.5, "axis-aligned center");
	expect(std::abs(box.width - 120) < 3.0 && std::abs(box.height - 70) < 3.0, "axis-aligned size");

	cv::Mat edge(80, 80, CV_8U, cv::Scalar(40));
	edge.colRange(0, 36).setTo(180);
	caliper::CaliperParam lineParam = param;
	lineParam.polarity = caliper::Polarity::LightToDark;
	const caliper::LineResult line = caliper::findLineRect(edge, 20, 8, 52, 72, true, lineParam);
	expect(line.found && std::abs(line.center.x - 36) < 2.0, "existing line caliper still finds an edge");
}

}  // namespace

int main()
{
	testMetrology();
	testFindRectangle();
	if (g_fails) {
		std::cerr << g_fails << " check(s) failed" << std::endl;
		return 1;
	}
	std::cout << "all rectangle metrology checks passed" << std::endl;
	return 0;
}
