// 尺度搜索：不依赖 TensorRT。编译见 Makefile 的 test-scale。
#include "cv_match/matcher.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int g_fails = 0;

void expect(bool cond, const std::string& msg)
{
	if (cond)
	{
		std::cout << "OK  " << msg << std::endl;
		return;
	}
	std::cerr << "FAIL " << msg << std::endl;
	++g_fails;
}

cv::Mat asymmetricPatch(int h = 56, int w = 88)
{
	cv::Mat img(h, w, CV_8U, cv::Scalar(40));
	img(cv::Rect(8, 8, w - 16, h - 16)).setTo(210);
	img(cv::Rect(8, 8, 20, 14)).setTo(40);
	img(cv::Rect(w - 36, h - 20, 28, 12)).setTo(90);
	return img;
}

cv::Mat lShape(int h = 64, int w = 80)
{
	cv::Mat img(h, w, CV_8U, cv::Scalar(30));
	img(cv::Rect(8, 8, 16, h - 16)).setTo(220);
	img(cv::Rect(8, h - 24, w - 16, 16)).setTo(220);
	return img;
}

void paste(cv::Mat& dst, const cv::Mat& patch, int x, int y)
{
	patch.copyTo(dst(cv::Rect(x, y, patch.cols, patch.rows)));
}

const template_matching::MatchResult* best(const std::vector<template_matching::MatchResult>& hits)
{
	if (hits.empty())
		return nullptr;
	const template_matching::MatchResult* r = &hits[0];
	for (const auto& h : hits)
		if (h.Score > r->Score)
			r = &h;
	return r;
}

void testPatternIdentity()
{
	using namespace template_matching;
	cv::Mat patch = asymmetricPatch();
	cv::Mat canvas(220, 280, CV_8U, cv::Scalar(40));
	const int x0 = 70, y0 = 55;
	paste(canvas, patch, x0, y0);
	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 0;
	param.scoreThreshold = 0.7;
	param.maxCount = 1;
	param.iouThreshold = 0.3;
	auto matcher = GetMatcher(param);
	expect(matcher != nullptr, "pattern matcher created");
	expect(matcher->setTemplate(patch) == 0, "pattern setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	const MatchResult* r = best(hits);
	expect(r != nullptr, "pattern identity found");
	if (!r)
		return;
	expect(r->Score > 0.9, "pattern identity score");
	expect(std::fabs(r->Scale - 1.0) < 1e-6, "pattern identity scale is 1");
	expect(std::fabs(r->LeftTop.x - x0) < 2.0 && std::fabs(r->LeftTop.y - y0) < 2.0, "pattern identity origin");
}

void testPatternScaled()
{
	using namespace template_matching;
	cv::Mat patch = asymmetricPatch();
	const double scale = 1.25;
	cv::Mat scaled;
	cv::resize(patch, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
	cv::Mat canvas(460, 560, CV_8U, cv::Scalar(40));
	const int x0 = 150, y0 = 120;
	paste(canvas, scaled, x0, y0);
	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 0;
	param.scoreThreshold = 0.55;
	param.maxCount = 1;
	param.iouThreshold = 0.3;
	param.minScale = 0.9;
	param.maxScale = 1.5;
	param.scaleStep = 0.1;
	auto matcher = GetMatcher(param);
	expect(matcher->setTemplate(patch) == 0, "pattern scaled setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	const MatchResult* r = best(hits);
	expect(r != nullptr, "pattern scaled found");
	if (!r)
		return;
	std::cout << "    pattern scale=" << r->Scale << " score=" << r->Score
			  << " center=(" << r->Center.x << "," << r->Center.y << ")" << std::endl;
	expect(r->Score > 0.7, "pattern scaled score");
	expect(std::fabs(r->Scale - scale) < 0.11, "pattern scaled scale");
	const double cx = x0 + scaled.cols / 2.0;
	const double cy = y0 + scaled.rows / 2.0;
	expect(std::hypot(r->Center.x - cx, r->Center.y - cy) < 12.0, "pattern scaled center");
}

void testPatternScaledMask()
{
	using namespace template_matching;
	cv::Mat patch = asymmetricPatch();
	const double scale = 1.2;
	cv::Mat scaled;
	cv::resize(patch, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
	cv::Mat canvas(460, 560, CV_8U, cv::Scalar(40));
	const int x0 = 140, y0 = 100;
	paste(canvas, scaled, x0, y0);
	cv::Mat mask = cv::Mat::zeros(patch.size(), CV_8U);
	const int h = patch.rows, w = patch.cols;
	mask(cv::Rect(6, 6, 10, h - 12)).setTo(255);
	mask(cv::Rect(w - 16, 6, 10, h - 12)).setTo(255);
	mask(cv::Rect(6, 6, w - 12, 10)).setTo(255);
	mask(cv::Rect(6, h - 16, w - 12, 10)).setTo(255);

	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 0;
	param.scoreThreshold = 0.45;
	param.maxCount = 1;
	param.iouThreshold = 0.3;
	param.minScale = 0.9;
	param.maxScale = 1.45;
	param.scaleStep = 0.1;
	auto matcher = GetMatcher(param);
	expect(matcher->setTemplate(patch, mask) == 0, "pattern mask setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	const MatchResult* r = best(hits);
	expect(r != nullptr, "pattern scaled+mask found");
	if (!r)
		return;
	std::cout << "    pattern+mask scale=" << r->Scale << " score=" << r->Score
			  << " center=(" << r->Center.x << "," << r->Center.y << ")" << std::endl;
	expect(r->Score > 0.45, "pattern scaled+mask score");
	expect(std::fabs(r->Scale - scale) < 0.11, "pattern scaled+mask scale");
	const double cx = x0 + scaled.cols / 2.0;
	const double cy = y0 + scaled.rows / 2.0;
	expect(std::hypot(r->Center.x - cx, r->Center.y - cy) < 14.0, "pattern scaled+mask center");
}

void testShapeIdentity()
{
	using namespace template_matching;
	cv::Mat patch = lShape();
	cv::Mat canvas(240, 300, CV_8U, cv::Scalar(30));
	const int x0 = 90, y0 = 70;
	paste(canvas, patch, x0, y0);
	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 0;
	param.scoreThreshold = 0.4;
	param.maxCount = 1;
	param.minArea = 256;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 280;
	param.greediness = 0.85;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(patch) == 0, "shape identity setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	const MatchResult* r = best(hits);
	expect(r != nullptr, "shape identity found");
	if (!r)
		return;
	expect(std::fabs(r->Scale - 1.0) < 1e-6, "shape identity scale is 1");
	expect(r->Score > 0.55, "shape identity score");
	const double cx = x0 + patch.cols / 2.0;
	const double cy = y0 + patch.rows / 2.0;
	expect(std::hypot(r->Center.x - cx, r->Center.y - cy) < 4.0, "shape identity center");
}

void testShapeScaled()
{
	using namespace template_matching;
	cv::Mat patch = lShape();
	const double scale = 1.25;
	cv::Mat scaled;
	cv::resize(patch, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
	cv::Mat canvas(420, 480, CV_8U, cv::Scalar(30));
	const int x0 = 140, y0 = 110;
	paste(canvas, scaled, x0, y0);
	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 0;
	param.scoreThreshold = 0.4;
	param.maxCount = 1;
	param.minArea = 256;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 280;
	param.usePolarity = true;
	param.greediness = 0.85;
	param.minScale = 0.9;
	param.maxScale = 1.5;
	param.scaleStep = 0.1;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(patch) == 0, "shape scaled setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	const MatchResult* r = best(hits);
	expect(r != nullptr, "shape scaled found");
	if (!r)
		return;
	std::cout << "    shape scale=" << r->Scale << " score=" << r->Score
			  << " center=(" << r->Center.x << "," << r->Center.y << ")" << std::endl;
	expect(r->Score > 0.45, "shape scaled score");
	expect(std::fabs(r->Scale - scale) < 0.12, "shape scaled scale");
	const double cx = x0 + scaled.cols / 2.0;
	const double cy = y0 + scaled.rows / 2.0;
	expect(std::hypot(r->Center.x - cx, r->Center.y - cy) < 12.0, "shape scaled center");
}

void testShapeNumScalesDown()
{
	using namespace template_matching;
	cv::Mat patch = lShape();
	const double scale = 0.8;
	cv::Mat scaled;
	cv::resize(patch, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
	cv::Mat canvas(360, 420, CV_8U, cv::Scalar(30));
	const int x0 = 120, y0 = 90;
	paste(canvas, scaled, x0, y0);
	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 0;
	param.scoreThreshold = 0.4;
	param.maxCount = 1;
	param.minArea = 256;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 280;
	param.greediness = 0.85;
	param.minScale = 0.7;
	param.maxScale = 1.1;
	param.numScales = 5;  // 0.7, 0.8, 0.9, 1.0, 1.1
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(patch) == 0, "shape numScales setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	const MatchResult* r = best(hits);
	expect(r != nullptr, "shape downscale found");
	if (!r)
		return;
	std::cout << "    shape down scale=" << r->Scale << " score=" << r->Score
			  << " center=(" << r->Center.x << "," << r->Center.y << ")" << std::endl;
	expect(std::fabs(r->Scale - scale) < 0.08, "shape downscale scale");
	const double cx = x0 + scaled.cols / 2.0;
	const double cy = y0 + scaled.rows / 2.0;
	expect(std::hypot(r->Center.x - cx, r->Center.y - cy) < 12.0, "shape downscale center");
}

// 合成图复现 Halcon Matching 例图上的失败模式。官方图是 MVTec 版权，不进仓库。
// green_dots：三个不同尺度的圆点，其中一个更暗。
// rings：多个圆环 + 角度搜索，旧 SHAPE 会在同一中心留一串角度重复。
// board_01：单个非对称件，位姿应对齐。
// cap_exposure / face_mask：PATTERN 在整体变亮后仍应是高分 NCC，只出一个命中。

cv::Mat filledDisk(int d, int bg, int fg)
{
	cv::Mat img(d, d, CV_8U, cv::Scalar(bg));
	cv::circle(img, cv::Point(d / 2, d / 2), std::max(2, d / 2 - 2), cv::Scalar(fg), cv::FILLED, cv::LINE_8);
	return img;
}

cv::Mat ringPatch(int d, int thick, int bg, int fg)
{
	cv::Mat img(d, d, CV_8U, cv::Scalar(bg));
	cv::circle(img, cv::Point(d / 2, d / 2), std::max(4, d / 2 - 4), cv::Scalar(fg), thick, cv::LINE_8);
	return img;
}

int countNear(const std::vector<template_matching::MatchResult>& hits, cv::Point2d c, double tol)
{
	int n = 0;
	for (const auto& h : hits)
		if (std::hypot(h.Center.x - c.x, h.Center.y - c.y) <= tol)
			++n;
	return n;
}

const template_matching::MatchResult* nearestHit(const std::vector<template_matching::MatchResult>& hits, cv::Point2d c)
{
	const template_matching::MatchResult* r = nullptr;
	double best = 1e9;
	for (const auto& h : hits)
	{
		const double d = std::hypot(h.Center.x - c.x, h.Center.y - c.y);
		if (d < best)
		{
			best = d;
			r = &h;
		}
	}
	return r;
}

void testGreenDotsMultiScale()
{
	using namespace template_matching;
	const int bg = 36;
	const int templD = 56;
	cv::Mat templ = filledDisk(templD, bg, 220);
	struct Inst { double scale; int fg; int x; int y; };
	const Inst insts[] = {
		{1.0, 220, 28, 26},
		{0.80, 200, 210, 36},
		{1.45, 130, 70, 200},
	};
	cv::Mat canvas(460, 520, CV_8U, cv::Scalar(bg));
	std::vector<cv::Point2d> truth;
	std::vector<double> scales;
	for (const Inst& in : insts)
	{
		const int d = std::max(8, (int)std::lround(templD * in.scale));
		cv::Mat p = filledDisk(d, bg, in.fg);
		paste(canvas, p, in.x, in.y);
		truth.emplace_back(in.x + d / 2.0, in.y + d / 2.0);
		scales.push_back(in.scale);
	}

	const double searchAngle = 12.0;
	for (MatcherType type : {MatcherType::PATTERN, MatcherType::SHAPE})
	{
		const char* name = type == MatcherType::PATTERN ? "pattern" : "shape";
		MatcherParam param;
		param.matcherType = type;
		param.angle = searchAngle;
		param.scoreThreshold = 0.5;
		param.maxCount = 8;
		param.iouThreshold = 0.5;
		param.minArea = 256;
		param.edgeMinMag = 10;
		param.maxEdgePoints = 320;
		param.usePolarity = true;
		param.greediness = 0.8;
		param.minScale = 0.6;
		param.maxScale = 1.6;
		param.scaleStep = 0.1;
		auto matcher = GetMatcher(param);
		expect(matcher && matcher->setTemplate(templ) == 0, std::string(name) + " green-dots setTemplate");
		std::vector<MatchResult> hits;
		matcher->match(canvas, hits);
		std::cout << "    " << name << " green-dots n=" << hits.size() << std::endl;
		for (const auto& h : hits)
			std::cout << "      sc=" << h.Score << " scale=" << h.Scale << " ang=" << h.Angle
					  << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
		expect((int)hits.size() == 3, std::string(name) + " green-dots hit count is 3");
		for (size_t i = 0; i < truth.size(); ++i)
		{
			expect(countNear(hits, truth[i], 12.0) == 1, std::string(name) + " green-dots one hit per dot");
			const MatchResult* h = nearestHit(hits, truth[i]);
			expect(h != nullptr, std::string(name) + " green-dots nearest");
			if (!h)
				continue;
			expect(std::fabs(h->Scale - scales[i]) < 0.16, std::string(name) + " green-dots scale");
			expect(std::fabs(h->Angle) <= searchAngle + 1.0, std::string(name) + " green-dots angle inside search");
		}
	}

	// maxCount 刚好等于实例数时，角度重复不能把最弱的圆点挤掉。
	MatcherParam tight;
	tight.matcherType = MatcherType::SHAPE;
	tight.angle = searchAngle;
	tight.scoreThreshold = 0.5;
	tight.maxCount = 3;
	tight.iouThreshold = 0.5;
	tight.minArea = 256;
	tight.edgeMinMag = 10;
	tight.maxEdgePoints = 320;
	tight.greediness = 0.8;
	tight.minScale = 0.6;
	tight.maxScale = 1.6;
	tight.scaleStep = 0.1;
	auto shape = GetMatcher(tight);
	expect(shape && shape->setTemplate(templ) == 0, "shape green-dots tight setTemplate");
	std::vector<MatchResult> tightHits;
	shape->match(canvas, tightHits);
	expect((int)tightHits.size() == 3, "shape green-dots maxCount=3 keeps all three");
	for (const auto& c : truth)
		expect(countNear(tightHits, c, 12.0) == 1, "shape green-dots tight one per center");
}

void testRingsNoAngleDuplicates()
{
	using namespace template_matching;
	cv::Mat templ = ringPatch(80, 7, 30, 220);
	const cv::Point origins[] = {{30, 30}, {180, 40}, {40, 200}, {200, 190}};
	cv::Mat canvas(360, 360, CV_8U, cv::Scalar(30));
	std::vector<cv::Point2d> truth;
	for (cv::Point o : origins)
	{
		paste(canvas, templ, o.x, o.y);
		truth.emplace_back(o.x + templ.cols / 2.0, o.y + templ.rows / 2.0);
	}
	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 15;
	param.scoreThreshold = 0.5;
	param.maxCount = 20;
	param.iouThreshold = 0.5;
	param.minArea = 256;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 320;
	param.greediness = 0.8;
	param.minScale = 0.9;
	param.maxScale = 1.1;
	param.scaleStep = 0.05;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "shape rings setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	std::cout << "    shape rings n=" << hits.size() << std::endl;
	expect((int)hits.size() == 4, "shape rings hit count is 4");
	for (const auto& c : truth)
		expect(countNear(hits, c, 8.0) == 1, "shape rings one hit per ring");

	param.matcherType = MatcherType::PATTERN;
	param.angle = 0;
	param.minScale = 1;
	param.maxScale = 1;
	auto pattern = GetMatcher(param);
	expect(pattern && pattern->setTemplate(templ) == 0, "pattern rings setTemplate");
	hits.clear();
	pattern->match(canvas, hits);
	expect((int)hits.size() == 4, "pattern rings unchanged at 4");
	for (const auto& c : truth)
		expect(countNear(hits, c, 6.0) == 1, "pattern rings one hit per ring");
}

void testRingClutterNotDuplicated()
{
	using namespace template_matching;
	cv::Mat templ = ringPatch(80, 7, 30, 220);
	cv::Mat canvas(320, 320, CV_8U, cv::Scalar(30));
	const int x0 = 120, y0 = 110;
	paste(canvas, templ, x0, y0);
	cv::ellipse(canvas, cv::Point(60, 60), cv::Size(40, 40), 0, 20, 200, cv::Scalar(210), 6, cv::LINE_8);
	cv::ellipse(canvas, cv::Point(260, 240), cv::Size(50, 36), 30, 0, 260, cv::Scalar(200), 5, cv::LINE_8);
	const cv::Point2d truth(x0 + templ.cols / 2.0, y0 + templ.rows / 2.0);

	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 20;
	param.scoreThreshold = 0.5;
	param.maxCount = 15;
	param.iouThreshold = 0.5;
	param.minArea = 256;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 320;
	param.greediness = 0.8;
	param.minScale = 0.85;
	param.maxScale = 1.2;
	param.scaleStep = 0.1;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "shape clutter setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	std::cout << "    shape clutter n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " scale=" << h.Scale << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect(countNear(hits, truth, 8.0) == 1, "shape clutter one hit on the true ring");
	expect((int)hits.size() <= 2, "shape clutter does not multiply arc false positives");
	const MatchResult* top = best(hits);
	expect(top && std::hypot(top->Center.x - truth.x, top->Center.y - truth.y) < 8.0, "shape clutter best is the ring");
}

void testBoardPoseUnchanged()
{
	using namespace template_matching;
	cv::Mat templ = asymmetricPatch(72, 110);
	cv::Mat canvas(280, 360, CV_8U, cv::Scalar(40));
	const int x0 = 90, y0 = 70;
	paste(canvas, templ, x0, y0);
	const double cx = x0 + templ.cols / 2.0;
	const double cy = y0 + templ.rows / 2.0;
	for (MatcherType type : {MatcherType::PATTERN, MatcherType::SHAPE})
	{
		const char* name = type == MatcherType::PATTERN ? "pattern" : "shape";
		MatcherParam param;
		param.matcherType = type;
		param.angle = 10;
		param.scoreThreshold = 0.5;
		param.maxCount = 5;
		param.iouThreshold = 0.3;
		param.minArea = 256;
		param.edgeMinMag = 12;
		param.maxEdgePoints = 280;
		param.greediness = 0.85;
		auto matcher = GetMatcher(param);
		expect(matcher && matcher->setTemplate(templ) == 0, std::string(name) + " board setTemplate");
		std::vector<MatchResult> hits;
		matcher->match(canvas, hits);
		expect((int)hits.size() == 1, std::string(name) + " board single hit");
		const MatchResult* r = best(hits);
		expect(r != nullptr, std::string(name) + " board found");
		if (!r)
			continue;
		expect(std::hypot(r->Center.x - cx, r->Center.y - cy) < 4.0, std::string(name) + " board center");
		expect(std::fabs(r->Angle) < 2.5, std::string(name) + " board angle");
		expect(std::fabs(r->Scale - 1.0) < 1e-6, std::string(name) + " board scale is 1");
	}
}

void testRotatedAsymmetric()
{
	using namespace template_matching;
	cv::Mat patch = asymmetricPatch();
	const int h = patch.rows, w = patch.cols;
	cv::Mat canvas(320, 360, CV_8U, cv::Scalar(40));
	const double cx = 180.0, cy = 160.0;
	paste(canvas, patch, (int)(cx - w / 2.0), (int)(cy - h / 2.0));
	const double rot = 8.0;
	cv::Mat scene;
	cv::warpAffine(canvas, scene, cv::getRotationMatrix2D(cv::Point2f((float)cx, (float)cy), rot, 1.0),
				   canvas.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(40));
	for (MatcherType type : {MatcherType::PATTERN, MatcherType::SHAPE})
	{
		const char* name = type == MatcherType::PATTERN ? "pattern" : "shape";
		MatcherParam param;
		param.matcherType = type;
		param.angle = 12;
		param.scoreThreshold = 0.5;
		param.maxCount = 3;
		param.iouThreshold = 0.3;
		param.minArea = 256;
		param.edgeMinMag = 12;
		param.maxEdgePoints = 280;
		param.greediness = 0.85;
		auto matcher = GetMatcher(param);
		expect(matcher && matcher->setTemplate(patch) == 0, std::string(name) + " rotated setTemplate");
		std::vector<MatchResult> hits;
		matcher->match(scene, hits);
		expect((int)hits.size() == 1, std::string(name) + " rotated single hit");
		const MatchResult* r = best(hits);
		expect(r != nullptr, std::string(name) + " rotated found");
		if (!r)
			continue;
		std::cout << "    " << name << " rotated ang=" << r->Angle << " c=(" << r->Center.x << "," << r->Center.y << ")" << std::endl;
		expect(std::fabs(r->Angle - rot) < 2.5, std::string(name) + " rotated angle");
		expect(std::hypot(r->Center.x - cx, r->Center.y - cy) < 8.0, std::string(name) + " rotated center");
	}
}

void testPatternBrightnessLikeExposure()
{
	using namespace template_matching;
	cv::Mat patch = asymmetricPatch();
	cv::Mat canvas(220, 280, CV_8U, cv::Scalar(40));
	const int x0 = 70, y0 = 55;
	paste(canvas, patch, x0, y0);
	canvas += cv::Scalar(50);
	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 0;
	param.scoreThreshold = 0.7;
	param.maxCount = 5;
	param.iouThreshold = 0.3;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(patch) == 0, "pattern exposure setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	expect((int)hits.size() == 1, "pattern exposure single hit");
	const MatchResult* r = best(hits);
	expect(r && r->Score > 0.9, "pattern exposure score stays high");
	if (!r)
		return;
	expect(std::fabs(r->LeftTop.x - x0) < 2.0 && std::fabs(r->LeftTop.y - y0) < 2.0, "pattern exposure origin");
}

}  // namespace

int main()
{
	testPatternIdentity();
	testPatternScaled();
	testPatternScaledMask();
	testShapeIdentity();
	testShapeScaled();
	testShapeNumScalesDown();
	testGreenDotsMultiScale();
	testRingsNoAngleDuplicates();
	testRingClutterNotDuplicated();
	testBoardPoseUnchanged();
	testRotatedAsymmetric();
	testPatternBrightnessLikeExposure();
	if (g_fails)
	{
		std::cerr << g_fails << " check(s) failed" << std::endl;
		return 1;
	}
	std::cout << "all scale-search checks passed" << std::endl;
	return 0;
}
