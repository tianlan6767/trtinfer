// 尺度搜索：不依赖 TensorRT。编译见 Makefile 的 test-scale。
#include "cv_match/matcher.h"
#include "cv_match/base_matcher.h"

#include <algorithm>
#include <chrono>
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

	// 整圈在角度搜索下四个象限都对齐，不能被局部弧的过滤删掉。
	param.angle = 20;
	auto patternAng = GetMatcher(param);
	expect(patternAng && patternAng->setTemplate(templ) == 0, "pattern rings angle setTemplate");
	hits.clear();
	patternAng->match(canvas, hits);
	std::cout << "    pattern rings ang20 n=" << hits.size() << std::endl;
	expect((int)hits.size() == 4, "pattern rings with angle stay at 4");
	for (const auto& c : truth)
		expect(countNear(hits, c, 8.0) == 1, "pattern rings angle one hit per ring");
}

// 一个整圈加远处的椭圆弧。全局 NCC 能让弧过 0.5，但象限对不齐，不应留成额外实例。
void testPatternRingArcRejected()
{
	using namespace template_matching;
	cv::Mat templ = ringPatch(72, 6, 30, 220);
	cv::Mat canvas(360, 420, CV_8U, cv::Scalar(30));
	const int x0 = 150, y0 = 130;
	paste(canvas, templ, x0, y0);
	cv::ellipse(canvas, cv::Point(70, 70), cv::Size(48, 48), 0, 10, 220, cv::Scalar(210), 6, cv::LINE_8);
	cv::ellipse(canvas, cv::Point(330, 80), cv::Size(40, 55), 20, 0, 300, cv::Scalar(200), 5, cv::LINE_8);
	cv::ellipse(canvas, cv::Point(80, 280), cv::Size(55, 36), -15, 30, 250, cv::Scalar(215), 6, cv::LINE_8);
	const cv::Point2d truth(x0 + templ.cols / 2.0, y0 + templ.rows / 2.0);

	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 20;
	param.scoreThreshold = 0.5;
	param.maxCount = 15;
	param.iouThreshold = 0.5;
	param.minArea = 256;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "pattern arc setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	std::cout << "    pattern one+arcs n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect((int)hits.size() == 1, "pattern arc scene is only the full ring");
	expect(countNear(hits, truth, 8.0) == 1, "pattern arc scene hit is on the ring");

	param.matcherType = MatcherType::SHAPE;
	param.angle = 15;
	param.minScale = 0.85;
	param.maxScale = 1.15;
	param.scaleStep = 0.1;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 320;
	param.greediness = 0.8;
	auto shape = GetMatcher(param);
	expect(shape && shape->setTemplate(templ) == 0, "shape arc setTemplate");
	hits.clear();
	shape->match(canvas, hits);
	std::cout << "    shape one+arcs n=" << hits.size() << std::endl;
	expect(countNear(hits, truth, 8.0) == 1, "shape arc scene still one hit on the ring");
	expect((int)hits.size() <= 2, "shape arc scene does not grow");
}

// 大模板只有中心一小块有对比。旁边较小的实例落在大框里，但中心是分开的，两个都要留下。
void testPatternKeepsNeighborInMargin()
{
	using namespace template_matching;
	const int box = 200;
	const int rad = 22;
	cv::Mat templ(box, box, CV_8U, cv::Scalar(180));
	cv::circle(templ, cv::Point(box / 2, box / 2), rad, cv::Scalar(40), cv::FILLED, cv::LINE_8);
	const int dist = 72;
	const cv::Point2d a(220, 220);
	const cv::Point2d b(220 + dist, 220);
	const cv::Point2d c(220, 520);
	cv::Mat canvas(700, 760, CV_8U, cv::Scalar(180));
	cv::circle(canvas, a, rad, cv::Scalar(40), cv::FILLED, cv::LINE_8);
	cv::circle(canvas, b, (int)std::lround(rad * 0.75), cv::Scalar(40), cv::FILLED, cv::LINE_8);
	cv::circle(canvas, c, rad, cv::Scalar(40), cv::FILLED, cv::LINE_8);
	const cv::Point2d truth[] = {a, b, c};

	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 0;
	param.scoreThreshold = 0.5;
	param.maxCount = 8;
	param.iouThreshold = 0.5;
	param.minArea = 64;
	param.minScale = 0.75;
	param.maxScale = 1.0;
	param.scaleStep = 0.25;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "pattern margin setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	std::cout << "    pattern margin n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " scale=" << h.Scale << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect((int)hits.size() == 3, "pattern margin keeps the neighbor inside the large box");
	for (const auto& p : truth)
		expect(countNear(hits, p, 14.0) == 1, "pattern margin one hit per dot");

	param.matcherType = MatcherType::SHAPE;
	param.edgeMinMag = 10;
	param.maxEdgePoints = 320;
	param.greediness = 0.8;
	auto shape = GetMatcher(param);
	expect(shape && shape->setTemplate(templ) == 0, "shape margin setTemplate");
	hits.clear();
	shape->match(canvas, hits);
	std::cout << "    shape margin n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " scale=" << h.Scale << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	// 这块大空白模板上，SHAPE 的边缘模型锁不住旁边那个更小的圆，本来就只有两个大圆。
	// 这里只守住不要多出一个贴在真目标上的错尺度旁瓣，也不要把已经找到的大圆并掉。
	expect(countNear(hits, a, 14.0) == 1, "shape margin keeps the large dot");
	expect(countNear(hits, c, 14.0) == 1, "shape margin keeps the far dot");
	expect((int)hits.size() <= 3, "shape margin does not add a scale sidelobe");
}

template_matching::MatchResult axisBox(double cx, double cy, double side, double score, double scale)
{
	template_matching::MatchResult r;
	const double h = side * 0.5;
	r.LeftTop = cv::Point2d(cx - h, cy - h);
	r.RightTop = cv::Point2d(cx + h, cy - h);
	r.RightBottom = cv::Point2d(cx + h, cy + h);
	r.LeftBottom = cv::Point2d(cx - h, cy + h);
	r.Center = cv::Point2d(cx, cy);
	r.Score = score;
	r.Scale = scale;
	return r;
}

// 尺寸差开、中心也分开时，高分邻点要留下；贴在旁边的低分错尺度锁要去掉。同尺度、中心已经分开的不删。
void testOverlapKeepsDifferentSizeNeighbor()
{
	using namespace template_matching;
	{
		std::vector<MatchResult> hits = {
			axisBox(0, 0, 100, 0.90, 1.0),
			axisBox(33, 0, 70, 0.85, 0.70),
		};
		BaseMatcher::suppressMatchOverlap(hits, 0.5, 8, true);
		expect((int)hits.size() == 2, "different-size high scores both stay");
	}
	{
		std::vector<MatchResult> hits = {
			axisBox(0, 0, 100, 0.90, 1.0),
			axisBox(33, 0, 70, 0.40, 0.70),
		};
		BaseMatcher::suppressMatchOverlap(hits, 0.5, 8, true);
		expect((int)hits.size() == 1, "weak different-scale sidelobe is dropped");
		expect(hits.size() == 1 && hits[0].Score > 0.8, "the stronger hit remains");
	}
	{
		std::vector<MatchResult> hits = {
			axisBox(0, 0, 80, 0.90, 1.0),
			axisBox(72, 0, 80, 0.55, 1.0),
		};
		BaseMatcher::suppressMatchOverlap(hits, 0.5, 8, true);
		expect((int)hits.size() == 2, "same-scale separated hits both stay");
	}
	{
		std::vector<MatchResult> hits = {
			axisBox(0, 0, 80, 0.90, 1.0),
			axisBox(10, 0, 80, 0.80, 1.0),
		};
		BaseMatcher::suppressMatchOverlap(hits, 0.5, 8, true);
		expect((int)hits.size() == 1, "same center still collapses to one");
	}
	{
		// 同尺度、中心距落在 0.4 和 0.65 倍边长之间、低分贴在高分旁边。旧规则留着它。
		std::vector<MatchResult> hits = {
			axisBox(0, 0, 100, 0.95, 1.0),
			axisBox(55, 0, 100, 0.65, 1.0),
		};
		BaseMatcher::suppressMatchOverlap(hits, 0.5, 8, true);
		expect((int)hits.size() == 1, "near same-scale weak sidelobe is dropped");
		expect(hits.size() == 1 && hits[0].Score > 0.9, "near sidelobe keeps the stronger hit");
	}
}

// 只挡掉一个象限时，其余象限仍对齐，不能当成半圈弧删掉。
// 模板空白边很大，第三颗较小的点周围背景和模板对不上，全局 NCC 落在 0.5 下面。物体本身仍对齐，应补回来。
void testPatternWeakThirdOnNoise()
{
	using namespace template_matching;
	const int box = 200;
	const int rad = 28;
	cv::Mat templ(box, box, CV_8U, cv::Scalar(180));
	cv::circle(templ, cv::Point(box / 2, box / 2), rad, cv::Scalar(40), cv::FILLED, cv::LINE_8);
	cv::circle(templ, cv::Point(box / 2 - 8, box / 2 - 6), 5, cv::Scalar(230), cv::FILLED, cv::LINE_8);
	cv::Mat canvas(640, 720, CV_8U);
	cv::randu(canvas, 20, 235);
	auto paint = [&](double scale, int cx, int cy, bool full) {
		cv::Mat scaled;
		cv::resize(templ, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
		const int x = cx - scaled.cols / 2;
		const int y = cy - scaled.rows / 2;
		if (full)
		{
			scaled.copyTo(canvas(cv::Rect(x, y, scaled.cols, scaled.rows)));
			return;
		}
		for (int r = 0; r < scaled.rows; ++r)
		{
			for (int c = 0; c < scaled.cols; ++c)
			{
				if (std::abs((int)scaled.at<uchar>(r, c) - 180) > 25)
					canvas.at<uchar>(y + r, x + c) = scaled.at<uchar>(r, c);
			}
		}
	};
	const cv::Point2d a(180, 180);
	const cv::Point2d b(480, 200);
	const cv::Point2d c(250, 480);
	paint(1.0, (int)a.x, (int)a.y, true);
	paint(0.9, (int)b.x, (int)b.y, true);
	paint(0.8, (int)c.x, (int)c.y, false);

	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 5;
	param.scoreThreshold = 0.5;
	param.maxCount = 6;
	param.iouThreshold = 0.5;
	param.minArea = 64;
	param.minScale = 0.75;
	param.maxScale = 1.0;
	param.scaleStep = 0.05;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "pattern weak-third setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	std::cout << "    pattern weak-third n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " scale=" << h.Scale << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect((int)hits.size() == 3, "pattern weak-third keeps the sub-threshold dot");
	expect(countNear(hits, a, 12.0) == 1, "pattern weak-third dot A");
	expect(countNear(hits, b, 12.0) == 1, "pattern weak-third dot B");
	expect(countNear(hits, c, 12.0) == 1, "pattern weak-third dot C");
	const MatchResult* weak = nearestHit(hits, c);
	expect(weak && weak->Score >= 0.5, "pattern weak-third reported score passes the threshold");
}

// 缺一口的圆环全局分可以到 0.85 以上，但最差象限明显低于其余象限。只留整圈。
void testPatternAlmostRingRejected()
{
	using namespace template_matching;
	cv::Mat templ = ringPatch(96, 5, 40, 210);
	cv::Mat canvas(360, 640, CV_8U, cv::Scalar(40));
	const int x0 = 40, y0 = 40;
	paste(canvas, templ, x0, y0);
	cv::ellipse(canvas, cv::Point(260, 90), cv::Size(36, 36), 0, 15, 270, cv::Scalar(210), 5, cv::LINE_8);
	cv::ellipse(canvas, cv::Point(480, 90), cv::Size(36, 36), 0, 0, 300, cv::Scalar(210), 5, cv::LINE_8);
	const cv::Point2d truth(x0 + templ.cols / 2.0, y0 + templ.rows / 2.0);

	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 20;
	param.scoreThreshold = 0.5;
	param.maxCount = 10;
	param.iouThreshold = 0.5;
	param.minArea = 256;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "pattern almost-ring setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	std::cout << "    pattern almost-ring n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect((int)hits.size() == 1, "pattern almost-ring drops the open arcs");
	expect(countNear(hits, truth, 8.0) == 1, "pattern almost-ring keeps the full ring");
}

// 圆点旁边的低分错位锁。真目标分很高，旁瓣大约 0.65，中心距超过 0.4 倍边长，以前会留下。
void testShapeDotSidelobeDropped()
{
	using namespace template_matching;
	const int box = 140;
	const int rad = 46;
	cv::Mat templ(box, box, CV_8U, cv::Scalar(190));
	cv::Point c(box / 2, box / 2);
	for (int y = 0; y < box; ++y)
	{
		for (int x = 0; x < box; ++x)
		{
			const double d = std::hypot(x - c.x, y - c.y);
			if (d <= rad)
				templ.at<uchar>(y, x) = (uchar)std::lround(25 + 55 * d / rad);
		}
	}
	cv::circle(templ, c + cv::Point(-rad / 5, -rad / 5), std::max(2, rad / 8), cv::Scalar(210), cv::FILLED, cv::LINE_8);
	cv::Mat canvas(640, 780, CV_8U, cv::Scalar(190));
	const cv::Point2d truth[] = {{200, 180}, {460, 200}, {250, 460}};
	const double scales[] = {1.0, 0.90, 0.80};
	for (int i = 0; i < 3; ++i)
	{
		cv::Mat scaled;
		cv::resize(templ, scaled, cv::Size(), scales[i], scales[i], cv::INTER_AREA);
		const int x = (int)truth[i].x - scaled.cols / 2;
		const int y = (int)truth[i].y - scaled.rows / 2;
		for (int r = 0; r < scaled.rows; ++r)
			for (int c = 0; c < scaled.cols; ++c)
				if (scaled.at<uchar>(r, c) < 170)
					canvas.at<uchar>(y + r, x + c) = scaled.at<uchar>(r, c);
	}
	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 20;
	param.scoreThreshold = 0.5;
	param.maxCount = 8;
	param.iouThreshold = 0.5;
	param.minArea = 256;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 400;
	param.usePolarity = false;
	param.greediness = 0.8;
	param.minScale = 0.8;
	param.maxScale = 1.0;
	param.scaleStep = 0.05;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "shape sidelobe setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	std::cout << "    shape sidelobe n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " scale=" << h.Scale << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect((int)hits.size() == 3, "shape sidelobe drops the extra lock");
	for (const auto& p : truth)
		expect(countNear(hits, p, 14.0) == 1, "shape sidelobe one hit per dot");
}

void testPatternOneQuadrantOcclusion()
{
	using namespace template_matching;
	cv::Mat patch = asymmetricPatch();
	cv::Mat canvas(240, 320, CV_8U, cv::Scalar(40));
	const int x0 = 80, y0 = 70;
	paste(canvas, patch, x0, y0);
	canvas(cv::Rect(x0 + patch.cols / 2, y0 + patch.rows / 2, patch.cols - patch.cols / 2, patch.rows - patch.rows / 2)).setTo(15);
	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 0;
	param.scoreThreshold = 0.45;
	param.maxCount = 3;
	param.iouThreshold = 0.3;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(patch) == 0, "pattern occlusion setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(canvas, hits);
	std::cout << "    pattern one-quadrant n=" << hits.size();
	if (!hits.empty())
		std::cout << " sc=" << best(hits)->Score;
	std::cout << std::endl;
	expect((int)hits.size() == 1, "pattern one-quadrant occlusion stays one hit");
	const MatchResult* r = best(hits);
	expect(r && std::fabs(r->LeftTop.x - x0) < 3.0 && std::fabs(r->LeftTop.y - y0) < 3.0, "pattern one-quadrant origin");
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

// 欠曝时一块象限混进噪声，灰度 NCC 会掉进旧的「缺一口」区间。整块瓶盖仍要留下。
void testPatternExposureQuadrantKept()
{
	using namespace template_matching;
	cv::Mat cap(160, 160, CV_8U, cv::Scalar(70));
	cv::circle(cap, {80, 80}, 58, cv::Scalar(160), cv::FILLED);
	cv::circle(cap, {58, 60}, 14, cv::Scalar(230), cv::FILLED);
	cv::rectangle(cap, {36, 98}, {124, 120}, cv::Scalar(25), cv::FILLED);
	cv::putText(cap, "C", {68, 92}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(40), 2);
	cv::setRNGSeed(7);
	cv::Mat noise(160, 160, CV_8U);
	cv::randu(noise, 0, 255);
	cv::GaussianBlur(noise, noise, cv::Size(0, 0), 1.5);
	const double mix = 0.8;
	cv::Mat dark = cap.clone();
	for (int y = 0; y < dark.rows; ++y)
	{
		for (int x = 0; x < dark.cols; ++x)
		{
			double v = cap.at<uchar>(y, x) * 0.5 + 8.0;
			if (y < 80 && x >= 80)
				v = (1.0 - mix) * v + mix * noise.at<uchar>(y, x) * 0.4;
			dark.at<uchar>(y, x) = (uchar)std::max(0.0, std::min(255.0, v));
		}
	}
	cv::Mat scene(360, 400, CV_8U, cv::Scalar(12));
	dark.copyTo(scene(cv::Rect(60, 50, 160, 160)));
	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 0;
	param.scoreThreshold = 0.5;
	param.maxCount = 2;
	param.iouThreshold = 0.5;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(cap) == 0, "pattern exposure-quadrant setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(scene, hits);
	std::cout << "    pattern exposure-quadrant n=" << hits.size();
	if (!hits.empty())
		std::cout << " sc=" << hits[0].Score;
	std::cout << std::endl;
	expect((int)hits.size() == 1, "pattern exposure-quadrant keeps the cap");
	const MatchResult* r = best(hits);
	expect(r && r->Score >= 0.5, "pattern exposure-quadrant score");
	expect(r && std::hypot(r->Center.x - 140.0, r->Center.y - 130.0) < 8.0, "pattern exposure-quadrant center");
}

// 整圈加一块高光仍是整圈。缺了一大段的弧要丢掉，哪怕全局分在 0.65–0.80。
void testPatternSpecularRingDropsOpenArc()
{
	using namespace template_matching;
	const int d = 110;
	cv::Mat ring(d, d, CV_8U, cv::Scalar(50));
	cv::circle(ring, {d / 2, d / 2}, 40, cv::Scalar(200), 7);
	cv::Mat scene(280, 820, CV_8U, cv::Scalar(50));
	const cv::Point2d fullC(20 + d / 2.0, 40 + d / 2.0);
	const cv::Point2d specC(180 + d / 2.0, 40 + d / 2.0);
	ring.copyTo(scene(cv::Rect(20, 40, d, d)));
	cv::Mat spec = ring.clone();
	cv::circle(spec, {d / 2 + 28, d / 2 - 22}, 14, cv::Scalar(250), cv::FILLED);
	spec.copyTo(scene(cv::Rect(180, 40, d, d)));
	cv::ellipse(scene, {415, 95}, {40, 40}, 0, 10, 250, cv::Scalar(200), 7);
	cv::ellipse(scene, {595, 95}, {40, 40}, 0, 20, 200, cv::Scalar(200), 7);

	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 12;
	param.scoreThreshold = 0.5;
	param.maxCount = 8;
	param.iouThreshold = 0.5;
	param.minArea = 256;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(ring) == 0, "pattern specular-ring setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(scene, hits);
	std::cout << "    pattern specular-ring n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect((int)hits.size() == 2, "pattern specular-ring keeps two full rings");
	expect(countNear(hits, fullC, 10.0) == 1, "pattern specular-ring full copy");
	expect(countNear(hits, specC, 10.0) == 1, "pattern specular-ring highlighted copy");
}

// 花纹上的弱梯度以前能锁成第四个圆。真圆点边缘很强，仍应是 3 个。
void testShapeMottleLockDropped()
{
	using namespace template_matching;
	auto disk = [](int d, int bg, int fg) {
		cv::Mat img(d, d, CV_8U, cv::Scalar(bg));
		cv::circle(img, {d / 2, d / 2}, d / 2 - 3, cv::Scalar(fg), cv::FILLED);
		cv::circle(img, {d / 2 + d / 6, d / 2 - d / 8}, std::max(3, d / 10), cv::Scalar(std::min(255, bg + 80)), cv::FILLED);
		return img;
	};
	cv::Mat templ = disk(120, 140, 40);
	cv::setRNGSeed(3);
	cv::Mat scene(560, 720, CV_8U);
	cv::randu(scene, 110, 170);
	cv::GaussianBlur(scene, scene, cv::Size(0, 0), 1.2);
	auto paint = [&](const cv::Mat& src, int cx, int cy) {
		const int x = cx - src.cols / 2;
		const int y = cy - src.rows / 2;
		src.copyTo(scene(cv::Rect(x, y, src.cols, src.rows)));
	};
	const cv::Point2d truth[] = {{180, 160}, {480, 180}, {300, 420}};
	paint(templ, 180, 160);
	cv::Mat mid;
	cv::resize(disk(120, 140, 55), mid, cv::Size(), 0.9, 0.9, cv::INTER_AREA);
	paint(mid, 480, 180);
	cv::Mat small;
	cv::resize(disk(120, 140, 70), small, cv::Size(), 0.8, 0.8, cv::INTER_AREA);
	cv::GaussianBlur(small, small, cv::Size(0, 0), 1.4);
	const int x = 300 - small.cols / 2;
	const int y = 420 - small.rows / 2;
	for (int r = 0; r < small.rows; ++r)
		for (int c = 0; c < small.cols; ++c)
			if (small.at<uchar>(r, c) < 120)
				scene.at<uchar>(y + r, x + c) = small.at<uchar>(r, c);

	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 25;
	param.scoreThreshold = 0.5;
	param.maxCount = 8;
	param.iouThreshold = 0.5;
	param.minArea = 256;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 400;
	param.usePolarity = false;
	param.greediness = 0.8;
	param.minScale = 0.8;
	param.maxScale = 1.0;
	param.scaleStep = 0.05;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "shape mottle setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(scene, hits);
	std::cout << "    shape mottle n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " scale=" << h.Scale << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect((int)hits.size() == 3, "shape mottle drops texture locks");
	for (const auto& p : truth)
		expect(countNear(hits, p, 16.0) == 1, "shape mottle one hit per dot");
}

// 欠曝把右半边压平。两块象限几乎没对比，但左半边仍然对齐，整颗瓶盖要留下。
void testPatternCrushedHalfCapKept()
{
	using namespace template_matching;
	cv::Mat cap(160, 160, CV_8U, cv::Scalar(70));
	cv::circle(cap, {80, 80}, 58, cv::Scalar(160), cv::FILLED);
	cv::circle(cap, {58, 60}, 14, cv::Scalar(230), cv::FILLED);
	cv::rectangle(cap, {36, 98}, {124, 120}, cv::Scalar(25), cv::FILLED);
	cv::putText(cap, "C", {68, 92}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(40), 2);
	cv::Mat dark(cap.size(), CV_8U);
	for (int y = 0; y < cap.rows; ++y)
	{
		for (int x = 0; x < cap.cols; ++x)
		{
			int v = (int)std::lround(cap.at<uchar>(y, x) * 0.45 + 6);
			if (x >= 80)
				v = 8 + (cap.at<uchar>(y, x) & 3);
			dark.at<uchar>(y, x) = (uchar)std::min(255, std::max(0, v));
		}
	}
	cv::Mat scene(360, 420, CV_8U, cv::Scalar(10));
	dark.copyTo(scene(cv::Rect(70, 60, 160, 160)));
	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 0;
	param.scoreThreshold = 0.5;
	param.maxCount = 2;
	param.iouThreshold = 0.5;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(cap) == 0, "pattern crushed-cap setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(scene, hits);
	std::cout << "    pattern crushed-cap n=" << hits.size();
	if (!hits.empty())
		std::cout << " sc=" << hits[0].Score << " c=(" << hits[0].Center.x << "," << hits[0].Center.y << ")";
	std::cout << std::endl;
	expect((int)hits.size() == 1, "pattern crushed-cap keeps the cap");
	const MatchResult* r = best(hits);
	expect(r && r->Score >= 0.5, "pattern crushed-cap score");
	// 右半被压平后峰会往还剩结构的一侧偏，大约二三十像素，仍在瓶盖圆里。
	expect(r && std::hypot(r->Center.x - 150.0, r->Center.y - 140.0) < 32.0, "pattern crushed-cap center");
}

// 刚过 0.5 的残弧不要留。整圈和高光整圈还在。
void testPatternFaintRingArcDropped()
{
	using namespace template_matching;
	const int d = 110;
	cv::Mat ring(d, d, CV_8U, cv::Scalar(50));
	cv::circle(ring, {d / 2, d / 2}, 40, cv::Scalar(200), 7);
	cv::Mat scene(260, 520, CV_8U, cv::Scalar(50));
	const cv::Point2d fullC(30 + d / 2.0, 40 + d / 2.0);
	ring.copyTo(scene(cv::Rect(30, 40, d, d)));
	cv::ellipse(scene, {340, 95}, {40, 40}, 0, 30, 190, cv::Scalar(120), 7);
	MatcherParam param;
	param.matcherType = MatcherType::PATTERN;
	param.angle = 10;
	param.scoreThreshold = 0.5;
	param.maxCount = 6;
	param.iouThreshold = 0.5;
	param.minArea = 256;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(ring) == 0, "pattern faint-arc setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(scene, hits);
	std::cout << "    pattern faint-arc n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect(countNear(hits, fullC, 10.0) == 1, "pattern faint-arc keeps the full ring");
	expect((int)hits.size() == 1, "pattern faint-arc drops the weak open arc");
}

// 托盘：边缘 mask、内部孔、一颗缺左边。定尺度下完整件高分，残缺件过 0.5，框落在零件上而不是孔上。
void testShapeTrayMaskAndChip()
{
	using namespace template_matching;
	const int box = 104;
	cv::Point tc(48, 52);
	cv::Mat templ(box, box, CV_8U, cv::Scalar(32));
	cv::circle(templ, tc, 34, cv::Scalar(210), cv::FILLED);
	cv::rectangle(templ, {tc.x + 20, tc.y - 6}, {tc.x + 38, tc.y + 6}, cv::Scalar(210), cv::FILLED);
	cv::circle(templ, tc, 11, cv::Scalar(32), cv::FILLED);
	cv::Mat mask = cv::Mat::zeros(templ.size(), CV_8U);
	cv::circle(mask, tc, 38, cv::Scalar(255), cv::FILLED);
	cv::rectangle(mask, {tc.x + 16, tc.y - 10}, {tc.x + 44, tc.y + 10}, cv::Scalar(255), cv::FILLED);

	auto stamp = [&](cv::Mat& dst, cv::Point c, bool chip) {
		cv::Mat tile = cv::Mat(dst, cv::Rect(c.x - tc.x, c.y - tc.y, box, box));
		templ.copyTo(tile);
		if (!chip)
			return;
		// 只咬掉左侧大约 70° 的外圈，孔和凸耳还在。缺一段应仍高于 0.5，并低于完整件。
		constexpr double kPi = 3.14159265358979323846;
		for (int y = 0; y < tile.rows; ++y)
		{
			for (int x = 0; x < tile.cols; ++x)
			{
				const float dx = (float)x - tc.x;
				const float dy = (float)y - tc.y;
				const float r = std::hypot(dx, dy);
				if (r < 16.f || r > 40.f)
					continue;
				const double ang = std::atan2((double)dy, (double)dx);
				const double d = std::fabs(std::atan2(std::sin(ang - kPi), std::cos(ang - kPi)));
				if (d < 35.0 * kPi / 180.0)
					tile.at<uchar>(y, x) = 32;
			}
		}
	};

	cv::Mat scene(520, 640, CV_8U, cv::Scalar(32));
	const cv::Point places[] = {{90, 90}, {250, 100}, {410, 90}, {100, 280}, {270, 290}, {430, 270}};
	const int chipIdx = 4;
	std::vector<cv::Point2d> truth;
	for (int i = 0; i < 6; ++i)
	{
		stamp(scene, places[i], i == chipIdx);
		truth.emplace_back(places[i].x, places[i].y);
	}
	const cv::Point stray(560, 420);
	cv::circle(scene, stray, 18, cv::Scalar(210), 4);
	cv::circle(scene, stray, 11, cv::Scalar(32), cv::FILLED);

	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 12;
	param.scoreThreshold = 0.5;
	param.maxCount = 10;
	param.iouThreshold = 0.1;
	param.minArea = 256;
	param.edgeMinMag = 22;
	param.edgeContrast = 80;
	param.maxEdgePoints = 220;
	param.usePolarity = true;
	param.greediness = 0.9;
	param.oriBins = 16;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ, mask) == 0, "shape tray setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(scene, hits);
	std::cout << "    shape tray n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect((int)hits.size() == 6, "shape tray finds every part and not the stray hole");
	for (int i = 0; i < 6; ++i)
		expect(countNear(hits, truth[i], 10.0) == 1, "shape tray one hit per part");
	const MatchResult* chip = nearestHit(hits, truth[chipIdx]);
	const MatchResult* full = nearestHit(hits, truth[0]);
	expect(chip && chip->Score >= 0.5, "shape tray chipped part passes 0.5");
	// 和全程浮点同一套点之后，完整件大约 0.89，不再是亚像素挪点时的大约 0.95。
	expect(full && full->Score >= 0.85, "shape tray complete part stays high");
	expect(chip && full && chip->Score < full->Score - 0.05, "shape tray chip scores below a complete part");
	expect(countNear(hits, cv::Point2d(stray.x, stray.y), 18.0) == 0, "shape tray does not lock the stray hole");
}

// 对比度设得很高时，分数不能再按 2.5 倍门槛把正常边缘乘掉。
void testShapeHighContrastStillFinds()
{
	using namespace template_matching;
	cv::Mat templ(88, 88, CV_8U, cv::Scalar(40));
	cv::rectangle(templ, {16, 18}, {70, 68}, cv::Scalar(220), cv::FILLED);
	cv::rectangle(templ, {28, 30}, {46, 48}, cv::Scalar(40), cv::FILLED);
	cv::Mat scene(240, 280, CV_8U, cv::Scalar(40));
	const int x0 = 80, y0 = 70;
	paste(scene, templ, x0, y0);
	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 8;
	param.scoreThreshold = 0.5;
	param.maxCount = 3;
	param.iouThreshold = 0.1;
	param.minArea = 256;
	param.edgeMinMag = 100;
	param.edgeContrast = 0;
	param.maxEdgePoints = 200;
	param.greediness = 0.9;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "shape high-contrast setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(scene, hits);
	std::cout << "    shape high-contrast n=" << hits.size();
	if (!hits.empty())
		std::cout << " sc=" << hits[0].Score;
	std::cout << std::endl;
	expect((int)hits.size() == 1, "shape high-contrast still one hit");
	const MatchResult* r = best(hits);
	expect(r && r->Score >= 0.75, "shape high-contrast score stays high");
	const double cx = x0 + templ.cols / 2.0;
	const double cy = y0 + templ.rows / 2.0;
	expect(r && std::hypot(r->Center.x - cx, r->Center.y - cy) < 6.0, "shape high-contrast center");
}

// 左边缘只露出大约七成的圆。中心在模型半径里面，旧的顶层 margin 根本搜不到。
// 出界点要跳过再归一化，不能当成 0 把可见的那段乘到阈值下。画面中间的整圆也要在。
void testShapeBorderPartial()
{
	using namespace template_matching;
	const int box = 160;
	const int rad = 50;
	cv::Mat templ(box, box, CV_8U, cv::Scalar(32));
	cv::circle(templ, {box / 2, box / 2}, rad, cv::Scalar(220), cv::FILLED);
	cv::circle(templ, {box / 2, box / 2}, 14, cv::Scalar(32), cv::FILLED);
	cv::Mat mask = cv::Mat::zeros(templ.size(), CV_8U);
	cv::circle(mask, {box / 2, box / 2}, rad + 4, cv::Scalar(255), cv::FILLED);

	const cv::Point2d edgeC(30, 220);
	const cv::Point2d fullC(360, 220);
	cv::Mat scene(460, 560, CV_8U, cv::Scalar(32));
	cv::circle(scene, edgeC, rad, cv::Scalar(220), cv::FILLED);
	cv::circle(scene, edgeC, 14, cv::Scalar(32), cv::FILLED);
	cv::circle(scene, fullC, rad, cv::Scalar(220), cv::FILLED);
	cv::circle(scene, fullC, 14, cv::Scalar(32), cv::FILLED);

	for (int bins : {0, 16})
	{
		MatcherParam param;
		param.matcherType = MatcherType::SHAPE;
		param.angle = 8;
		param.scoreThreshold = 0.5;
		param.maxCount = 6;
		param.iouThreshold = 0.4;
		param.minArea = 256;
		param.edgeMinMag = 20;
		param.edgeContrast = 80;
		param.maxEdgePoints = 240;
		param.usePolarity = true;
		param.greediness = 0.9;
		param.oriBins = bins;
		auto matcher = GetMatcher(param);
		expect(matcher && matcher->setTemplate(templ, mask) == 0, "shape border setTemplate");
		std::vector<MatchResult> hits;
		matcher->match(scene, hits);
		std::cout << "    shape border oriBins=" << bins << " n=" << hits.size() << std::endl;
		for (const auto& h : hits)
			std::cout << "      sc=" << h.Score << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
		expect(countNear(hits, edgeC, 14.0) == 1, "shape border finds the clipped part");
		expect(countNear(hits, fullC, 12.0) == 1, "shape border keeps the interior part");
		expect((int)hits.size() == 2, "shape border has no extra hit");
		const MatchResult* edge = nearestHit(hits, edgeC);
		expect(edge && edge->Score >= 0.5, "shape border clipped score passes 0.5");
	}
}

// 左边缘大约 68% 可见。oriBins=16 只改顶层粗搜，模型点必须和全程浮点是同一套，
// 否则末层分会岔开（233 上残件从大约 0.53 掉到 0.36）。两条路径的分数要贴在一起。
void testShapeOri16PartialMatchesFloat()
{
	using namespace template_matching;
	const int box = 160;
	const int rad = 54;
	cv::Mat templ(box, box, CV_8U, cv::Scalar(36));
	cv::circle(templ, {box / 2, box / 2}, rad, cv::Scalar(210), cv::FILLED);
	cv::circle(templ, {box / 2, box / 2}, 14, cv::Scalar(36), cv::FILLED);
	cv::rectangle(templ, {box / 2 + rad / 3, box / 2 - 8}, {box / 2 + rad - 4, box / 2 + 8}, cv::Scalar(210), cv::FILLED);
	cv::Mat mask = cv::Mat::zeros(templ.size(), CV_8U);
	cv::circle(mask, {box / 2, box / 2}, rad + 6, cv::Scalar(255), cv::FILLED);

	const cv::Point2d partial(28, 170);
	const cv::Point2d interior(460, 420);
	cv::Mat scene(860, 980, CV_8U, cv::Scalar(36));
	auto paint = [&](cv::Point2d c) {
		const int x0 = (int)std::lround(c.x - (box - 1) / 2.0);
		const int y0 = (int)std::lround(c.y - (box - 1) / 2.0);
		for (int y = 0; y < box; ++y)
		{
			const int sy = y0 + y;
			if (sy < 0 || sy >= scene.rows)
				continue;
			for (int x = 0; x < box; ++x)
			{
				const int sx = x0 + x;
				if (sx < 0 || sx >= scene.cols)
					continue;
				scene.at<uchar>(sy, sx) = templ.at<uchar>(y, x);
			}
		}
	};
	paint(partial);
	paint(interior);

	double partialScore[2] = {0, 0};
	double interiorScore[2] = {0, 0};
	int idx = 0;
	for (int bins : {0, 16})
	{
		MatcherParam param;
		param.matcherType = MatcherType::SHAPE;
		param.angle = 8;
		param.scoreThreshold = 0.5;
		param.maxCount = 4;
		param.iouThreshold = 0.4;
		param.minArea = 256;
		param.edgeMinMag = 29;
		param.edgeContrast = 119;
		param.maxEdgePoints = 240;
		param.usePolarity = true;
		param.greediness = 0.9;
		param.oriBins = bins;
		auto matcher = GetMatcher(param);
		expect(matcher && matcher->setTemplate(templ, mask) == 0, "shape ori16-partial setTemplate");
		std::vector<MatchResult> hits;
		matcher->match(scene, hits);
		std::cout << "    shape ori16-partial bins=" << bins << " n=" << hits.size() << std::endl;
		for (const auto& h : hits)
			std::cout << "      sc=" << h.Score << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
		expect((int)hits.size() == 2, "shape ori16-partial no extra hit");
		const MatchResult* edge = nearestHit(hits, partial);
		const MatchResult* full = nearestHit(hits, interior);
		expect(edge && std::hypot(edge->Center.x - partial.x, edge->Center.y - partial.y) < 8.0,
			   "shape ori16-partial finds the clipped part");
		expect(full && std::hypot(full->Center.x - interior.x, full->Center.y - interior.y) < 8.0,
			   "shape ori16-partial keeps the interior part");
		expect(edge && edge->Score >= 0.85, "shape ori16-partial clipped score stays high");
		expect(full && full->Score >= 0.85, "shape ori16-partial interior score stays high");
		if (edge)
			partialScore[idx] = edge->Score;
		if (full)
			interiorScore[idx] = full->Score;
		++idx;
	}
	expect(std::fabs(partialScore[0] - partialScore[1]) < 0.03, "shape ori16-partial clipped score matches float");
	expect(std::fabs(interiorScore[0] - interiorScore[1]) < 0.03, "shape ori16-partial interior score matches float");
}

// 第三只圆环对比度低、边缘发虚。方向贴得上，不能再被 mag/32 乘到 0.5 下面。
void testShapeFaintRingThird()
{
	using namespace template_matching;
	const int d = 96;
	cv::Mat templ = ringPatch(d, 6, 40, 210);
	cv::Mat scene(360, 640, CV_8U, cv::Scalar(40));
	const cv::Point origins[] = {{24, 40}, {230, 36}, {430, 48}};
	std::vector<cv::Point2d> truth;
	for (int i = 0; i < 3; ++i)
	{
		cv::Mat tile = templ.clone();
		if (i == 2)
		{
			tile.setTo(40);
			cv::circle(tile, {d / 2, d / 2}, d / 2 - 4, cv::Scalar(58), 6);
			cv::GaussianBlur(tile, tile, cv::Size(0, 0), 1.1);
		}
		paste(scene, tile, origins[i].x, origins[i].y);
		truth.emplace_back(origins[i].x + d / 2.0, origins[i].y + d / 2.0);
	}
	// 缺一大段的弧，不能跟着第三只真圆环一起留下来。
	cv::ellipse(scene, {120, 280}, {36, 36}, 0, 20, 190, cv::Scalar(200), 6);

	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 12;
	param.scoreThreshold = 0.5;
	param.maxCount = 8;
	param.iouThreshold = 0.5;
	param.minArea = 256;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 320;
	param.usePolarity = true;
	param.greediness = 0.85;
	param.minScale = 0.9;
	param.maxScale = 1.1;
	param.scaleStep = 0.05;
	param.oriBins = 0;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ) == 0, "shape faint-ring setTemplate");
	std::vector<MatchResult> hits;
	matcher->match(scene, hits);
	std::cout << "    shape faint-ring n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " c=(" << h.Center.x << "," << h.Center.y << ")" << std::endl;
	expect((int)hits.size() == 3, "shape faint-ring finds the dim ring and drops the arc");
	for (const auto& c : truth)
		expect(countNear(hits, c, 10.0) == 1, "shape faint-ring one hit per ring");
	const MatchResult* dim = nearestHit(hits, truth[2]);
	expect(dim && dim->Score >= 0.5, "shape faint-ring dim score passes 0.5");
}

// ±180°、oriBins=16：粗搜走方向响应图，正放和转过约 25° 的 L 都要留下。
// 末层仍是浮点分。不在这里跑 oriBins=0 的 ±180（那条路径没有响应图）。
void testShapeOri16ResponseMapWideAngle()
{
	using namespace template_matching;
	cv::Mat patch = lShape();
	cv::Mat padded(patch.rows + 40, patch.cols + 40, CV_8U, cv::Scalar(30));
	paste(padded, patch, 20, 20);
	cv::Mat rot;
	const cv::Point2f pivot((padded.cols - 1) * 0.5f, (padded.rows - 1) * 0.5f);
	const cv::Mat M = cv::getRotationMatrix2D(pivot, 25.0, 1.0);
	cv::warpAffine(padded, rot, M, padded.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(30));

	cv::Mat scene(460, 640, CV_8U, cv::Scalar(30));
	const cv::Point uprightAt(36, 48);
	const cv::Point turnedAt(360, 210);
	paste(scene, padded, uprightAt.x, uprightAt.y);
	paste(scene, rot, turnedAt.x, turnedAt.y);
	const cv::Point2d uprightC(uprightAt.x + (padded.cols - 1) / 2.0, uprightAt.y + (padded.rows - 1) / 2.0);
	const cv::Point2d turnedC(turnedAt.x + (padded.cols - 1) / 2.0, turnedAt.y + (padded.rows - 1) / 2.0);

	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 180;
	param.scoreThreshold = 0.5;
	param.maxCount = 4;
	param.iouThreshold = 0.4;
	param.minArea = 256;
	param.edgeMinMag = 12;
	param.maxEdgePoints = 280;
	param.usePolarity = true;
	param.greediness = 0.9;
	param.oriBins = 16;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(padded) == 0, "shape ori16-map setTemplate");
	std::vector<MatchResult> hits;
	const auto t0 = std::chrono::steady_clock::now();
	matcher->match(scene, hits);
	const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::cout << "    shape ori16-map ±180 find " << ms << " ms n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " ang=" << h.Angle << " c=(" << h.Center.x << "," << h.Center.y << ")"
				  << std::endl;
	expect(countNear(hits, uprightC, 8.0) == 1, "shape ori16-map finds the upright L");
	expect(countNear(hits, turnedC, 10.0) == 1, "shape ori16-map finds the rotated L");
	const MatchResult* upright = nearestHit(hits, uprightC);
	const MatchResult* turned = nearestHit(hits, turnedC);
	expect(upright && upright->Score >= 0.55, "shape ori16-map upright score");
	expect(turned && turned->Score >= 0.45, "shape ori16-map rotated score");
	expect(upright && std::fabs(upright->Angle) <= 8.0, "shape ori16-map upright angle");
	if (turned)
	{
		const double d25 = std::min(std::fabs(turned->Angle - 25.0), std::fabs(turned->Angle + 25.0));
		expect(d25 <= 8.0, "shape ori16-map rotated angle near ±25");
	}
	// 粗搜按角度下标拼回、去重按全序比较。再跑一次，命中个数和位姿必须一样。
	std::vector<MatchResult> again;
	matcher->match(scene, again);
	expect((int)again.size() == (int)hits.size(), "shape ori16-map repeat keeps the same count");
	if (again.size() == hits.size())
	{
		for (size_t i = 0; i < hits.size(); ++i)
		{
			expect(std::fabs(again[i].Score - hits[i].Score) < 1e-9, "shape ori16-map repeat score");
			expect(std::hypot(again[i].Center.x - hits[i].Center.x, again[i].Center.y - hits[i].Center.y) < 1e-6,
				   "shape ori16-map repeat center");
			expect(std::fabs(again[i].Angle - hits[i].Angle) < 1e-6, "shape ori16-map repeat angle");
		}
	}
}

// 步长 4、±180° 之后仍要留下转过的件和贴左边的件。
// 不是 233 原图。桌面机用 233 复测 n 和 find。
void testShapeOri16Stride4KeepsWideHits()
{
	using namespace template_matching;
	const int tw = 96, th = 128;
	cv::Mat templ(th, tw, CV_8U, cv::Scalar(36));
	cv::rectangle(templ, {8, 10}, {tw - 10, th - 12}, cv::Scalar(210), 4);
	cv::rectangle(templ, {18, 24}, {tw / 2, th / 2}, cv::Scalar(210), cv::FILLED);
	cv::circle(templ, {tw - 28, th - 32}, 12, cv::Scalar(40), cv::FILLED);
	cv::Mat mask = cv::Mat::zeros(templ.size(), CV_8U);
	cv::rectangle(mask, {4, 6}, {tw - 6, th - 8}, cv::Scalar(255), cv::FILLED);

	auto paint = [&](cv::Mat& scene, cv::Point2d c, double angDeg) {
		cv::Mat rot;
		const cv::Point2f pivot((templ.cols - 1) * 0.5f, (templ.rows - 1) * 0.5f);
		const cv::Mat M = cv::getRotationMatrix2D(pivot, angDeg, 1.0);
		cv::warpAffine(templ, rot, M, templ.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(36));
		const int x0 = (int)std::lround(c.x - (rot.cols - 1) / 2.0);
		const int y0 = (int)std::lround(c.y - (rot.rows - 1) / 2.0);
		for (int y = 0; y < rot.rows; ++y)
		{
			const int sy = y0 + y;
			if (sy < 0 || sy >= scene.rows)
				continue;
			for (int x = 0; x < rot.cols; ++x)
			{
				const int sx = x0 + x;
				if (sx < 0 || sx >= scene.cols)
					continue;
				const uchar v = rot.at<uchar>(y, x);
				if (v != 36)
					scene.at<uchar>(sy, sx) = v;
			}
		}
	};

	cv::Mat scene(480, 720, CV_8U, cv::Scalar(36));
	const struct
	{
		cv::Point2d c;
		double ang;
	} inst[] = {
		{{180, 120}, 0},
		{{420, 110}, 25},
		{{600, 160}, -40},
		{{200, 320}, 140},
		{{480, 340}, 15},
		{{36, 240}, 0},
	};
	for (const auto& it : inst)
		paint(scene, it.c, it.ang);

	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 180;
	param.scoreThreshold = 0.5;
	param.maxCount = 12;
	param.iouThreshold = 0.4;
	param.minArea = 256;
	param.edgeMinMag = 29;
	param.edgeContrast = 119;
	param.maxEdgePoints = 280;
	param.usePolarity = true;
	param.greediness = 0.9;
	param.oriBins = 16;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ, mask) == 0, "shape ori16-stride4 setTemplate");
	std::vector<MatchResult> hits;
	const auto t0 = std::chrono::steady_clock::now();
	matcher->match(scene, hits);
	const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::cout << "    shape ori16-stride4 ±180 find " << ms << " ms n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " ang=" << h.Angle << " c=(" << h.Center.x << "," << h.Center.y << ")"
				  << std::endl;
	expect((int)hits.size() == (int)(sizeof(inst) / sizeof(inst[0])), "shape ori16-stride4 keeps every part");
	for (const auto& it : inst)
	{
		expect(countNear(hits, it.c, 12.0) == 1, "shape ori16-stride4 one hit per part");
		const MatchResult* h = nearestHit(hits, it.c);
		expect(h && h->Score >= 0.5, "shape ori16-stride4 score passes 0.5");
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

// 左边缘大约只露出 2/3（中心 x=26.3，模板宽 144，和 233 上 T11 同一类几何）。
// ±180°、oriBins=16、对比度 29/119。顶层角度步长如果再乘 2，15° 和 140° 这两件会从粗搜里消失。
// 不是 233 原图。
void testShapeOri16WideLeftEdgePartial()
{
	using namespace template_matching;
	const int tw = 144, th = 190;
	cv::Mat templ(th, tw, CV_8U, cv::Scalar(36));
	cv::rectangle(templ, {12, 14}, {tw - 14, th - 16}, cv::Scalar(210), 5);
	cv::rectangle(templ, {28, 30}, {tw / 2, th / 2}, cv::Scalar(210), cv::FILLED);
	cv::circle(templ, {tw - 36, 48}, 16, cv::Scalar(210), cv::FILLED);
	cv::circle(templ, {tw - 40, th - 40}, 10, cv::Scalar(36), cv::FILLED);
	cv::line(templ, {24, th - 36}, {tw - 24, th - 36}, cv::Scalar(210), 4);
	cv::Mat mask = cv::Mat::zeros(templ.size(), CV_8U);
	cv::rectangle(mask, {6, 8}, {tw - 8, th - 10}, cv::Scalar(255), cv::FILLED);

	auto paint = [&](cv::Mat& scene, cv::Point2d c, double angDeg) {
		cv::Mat rot;
		const cv::Point2f pivot((templ.cols - 1) * 0.5f, (templ.rows - 1) * 0.5f);
		const cv::Mat M = cv::getRotationMatrix2D(pivot, angDeg, 1.0);
		cv::warpAffine(templ, rot, M, templ.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(36));
		const int x0 = (int)std::lround(c.x - (rot.cols - 1) / 2.0);
		const int y0 = (int)std::lround(c.y - (rot.rows - 1) / 2.0);
		for (int y = 0; y < rot.rows; ++y)
		{
			const int sy = y0 + y;
			if (sy < 0 || sy >= scene.rows)
				continue;
			for (int x = 0; x < rot.cols; ++x)
			{
				const int sx = x0 + x;
				if (sx < 0 || sx >= scene.cols)
					continue;
				const uchar v = rot.at<uchar>(y, x);
				if (v != 36)
					scene.at<uchar>(sy, sx) = v;
			}
		}
	};

	cv::Mat scene(700, 820, CV_8U, cv::Scalar(36));
	const struct
	{
		cv::Point2d c;
		double ang;
	} inst[] = {
		{{26.3, 160}, 15},
		{{26.3, 380}, 140},
		{{26.3, 590}, 40},
		{{520, 220}, 25},
	};
	for (const auto& it : inst)
		paint(scene, it.c, it.ang);

	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 180;
	param.scoreThreshold = 0.5;
	param.maxCount = 200;
	param.iouThreshold = 0.4;
	param.minArea = 256;
	param.edgeMinMag = 29;
	param.edgeContrast = 119;
	param.maxEdgePoints = 320;
	param.usePolarity = true;
	param.greediness = 0.9;
	param.oriBins = 16;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ, mask) == 0, "shape ori16-left-edge setTemplate");
	std::vector<MatchResult> hits;
	const auto t0 = std::chrono::steady_clock::now();
	matcher->match(scene, hits);
	const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::cout << "    shape ori16-left-edge ±180 find " << ms << " ms n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " ang=" << h.Angle << " c=(" << h.Center.x << "," << h.Center.y << ")"
				  << std::endl;
	expect((int)hits.size() == (int)(sizeof(inst) / sizeof(inst[0])), "shape ori16-left-edge keeps every part");
	for (const auto& it : inst)
	{
		expect(countNear(hits, it.c, 12.0) == 1, "shape ori16-left-edge one hit per part");
		const MatchResult* h = nearestHit(hits, it.c);
		expect(h && h->Score >= 0.5, "shape ori16-left-edge score passes 0.5");
		if (!h)
			continue;
		double dAng = std::fabs(h->Angle - it.ang);
		if (dAng > 180.0)
			dAng = 360.0 - dAng;
		expect(dAng <= 8.0, "shape ori16-left-edge angle stays on the part");
	}
}

// 233 同尺寸：1288×964，模板 144×190，左缘中心 (26.3, 156)，大约只露出 2/3。
// oriBins=16、±180°、29/119 必须留下这一件；oriBins=0 在较窄角度里也要留下。
// 不是 233 原图。干净高对比的贴边件在响应图上也会是满分；这个用例锁的是
// 同一几何在工具同款参数下不能丢，桌面机上的纹理图还要靠贴边浮点粗分。
void testShapeOri16T11SizedLeftEdge()
{
	using namespace template_matching;
	const int tw = 144, th = 190;
	cv::Mat templ(th, tw, CV_8U, cv::Scalar(36));
	cv::rectangle(templ, {12, 14}, {tw - 14, th - 16}, cv::Scalar(200), 4);
	cv::rectangle(templ, {30, 36}, {tw / 2 + 4, th / 2}, cv::Scalar(210), cv::FILLED);
	cv::circle(templ, {tw - 34, 52}, 14, cv::Scalar(180), cv::FILLED);
	cv::circle(templ, {tw - 42, th - 44}, 9, cv::Scalar(40), cv::FILLED);
	cv::line(templ, {22, th - 40}, {tw - 20, th - 28}, cv::Scalar(220), 3);
	cv::Mat mask = cv::Mat::zeros(templ.size(), CV_8U);
	cv::rectangle(mask, {6, 8}, {tw - 8, th - 10}, cv::Scalar(255), cv::FILLED);

	auto paint = [&](cv::Mat& scene, cv::Point2d c, double angDeg) {
		cv::Mat rot;
		const cv::Point2f pivot((templ.cols - 1) * 0.5f, (templ.rows - 1) * 0.5f);
		const cv::Mat M = cv::getRotationMatrix2D(pivot, angDeg, 1.0);
		cv::warpAffine(templ, rot, M, templ.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(36));
		const int x0 = (int)std::lround(c.x - (rot.cols - 1) / 2.0);
		const int y0 = (int)std::lround(c.y - (rot.rows - 1) / 2.0);
		for (int y = 0; y < rot.rows; ++y)
		{
			const int sy = y0 + y;
			if (sy < 0 || sy >= scene.rows)
				continue;
			for (int x = 0; x < rot.cols; ++x)
			{
				const int sx = x0 + x;
				if (sx < 0 || sx >= scene.cols)
					continue;
				const uchar v = rot.at<uchar>(y, x);
				if (v != 36)
					scene.at<uchar>(sy, sx) = v;
			}
		}
	};

	cv::Mat scene(964, 1288, CV_8U, cv::Scalar(36));
	const cv::Point2d partial(26.3, 156);
	const cv::Point2d interior(640, 420);
	paint(scene, partial, 0.0);
	paint(scene, interior, 18.0);

	auto run = [&](int bins, double angle, const char* tag) {
		MatcherParam param;
		param.matcherType = MatcherType::SHAPE;
		param.angle = angle;
		param.scoreThreshold = 0.5;
		param.maxCount = 200;
		param.iouThreshold = 0.4;
		param.minArea = 256;
		param.edgeMinMag = 29;
		param.edgeContrast = 119;
		param.maxEdgePoints = 320;
		param.usePolarity = true;
		param.greediness = 0.9;
		param.oriBins = bins;
		auto matcher = GetMatcher(param);
		expect(matcher && matcher->setTemplate(templ, mask) == 0, std::string("shape t11-size setTemplate ") + tag);
		std::vector<MatchResult> hits;
		const auto t0 = std::chrono::steady_clock::now();
		matcher->match(scene, hits);
		const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		std::cout << "    shape t11-size " << tag << " find " << ms << " ms n=" << hits.size() << std::endl;
		for (const auto& h : hits)
			std::cout << "      sc=" << h.Score << " ang=" << h.Angle << " c=(" << h.Center.x << "," << h.Center.y << ")"
					  << std::endl;
		expect(countNear(hits, partial, 12.0) == 1, std::string("shape t11-size finds left edge ") + tag);
		const MatchResult* edge = nearestHit(hits, partial);
		expect(edge && edge->Score >= 0.5, std::string("shape t11-size left edge score ") + tag);
		if (bins == 16)
		{
			expect(countNear(hits, interior, 12.0) == 1, "shape t11-size keeps interior");
			if (edge)
			{
				double dAng = std::fabs(edge->Angle);
				if (dAng > 180.0)
					dAng = 360.0 - dAng;
				expect(dAng <= 8.0, "shape t11-size left edge angle");
			}
		}
	};
	run(16, 180, "ori16");
	run(0, 20, "ori0");
}

// 压过对比度的左缘残件，背景是确定性细边，不是空白底上的高对比件。
// oriBins=16、±180°、29/119 必须留下 x≈26 这一件。不是 233 原图。
void testShapeOri16TexturedLeftEdge()
{
	using namespace template_matching;
	const int tw = 144, th = 190;
	cv::Mat templ(th, tw, CV_8U, cv::Scalar(36));
	cv::rectangle(templ, {12, 14}, {tw - 14, th - 16}, cv::Scalar(200), 4);
	cv::rectangle(templ, {30, 36}, {tw / 2 + 4, th / 2}, cv::Scalar(210), cv::FILLED);
	cv::circle(templ, {tw - 34, 52}, 14, cv::Scalar(180), cv::FILLED);
	cv::circle(templ, {tw - 42, th - 44}, 9, cv::Scalar(40), cv::FILLED);
	cv::line(templ, {22, th - 40}, {tw - 20, th - 28}, cv::Scalar(220), 3);
	cv::Mat mask = cv::Mat::zeros(templ.size(), CV_8U);
	cv::rectangle(mask, {6, 8}, {tw - 8, th - 10}, cv::Scalar(255), cv::FILLED);

	auto paint = [&](cv::Mat& scene, cv::Point2d c, double angDeg) {
		cv::Mat rot;
		const cv::Point2f pivot((templ.cols - 1) * 0.5f, (templ.rows - 1) * 0.5f);
		const cv::Mat M = cv::getRotationMatrix2D(pivot, angDeg, 1.0);
		cv::warpAffine(templ, rot, M, templ.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(36));
		const int x0 = (int)std::lround(c.x - (rot.cols - 1) / 2.0);
		const int y0 = (int)std::lround(c.y - (rot.rows - 1) / 2.0);
		for (int y = 0; y < rot.rows; ++y)
		{
			const int sy = y0 + y;
			if (sy < 0 || sy >= scene.rows)
				continue;
			for (int x = 0; x < rot.cols; ++x)
			{
				const int sx = x0 + x;
				if (sx < 0 || sx >= scene.cols)
					continue;
				const uchar v = rot.at<uchar>(y, x);
				if (v != 36)
					scene.at<uchar>(sy, sx) = v;
			}
		}
	};

	cv::Mat scene(964, 1288, CV_8U, cv::Scalar(42));
	for (int y = 20; y < scene.rows; y += 36)
	{
		for (int x = 24; x < scene.cols; x += 44)
		{
			const int shade = 90 + ((x * 3 + y * 5) % 60);
			cv::line(scene, {x, y}, {x + 12, y + ((x / 44) % 5) - 2}, cv::Scalar(shade), 1);
		}
	}
	const cv::Point2d partial(26.3, 156);
	const cv::Point2d interior(700, 480);
	paint(scene, partial, 0.0);
	paint(scene, interior, 12.0);
	// 左缘再压一截对比度，并叠一点固定噪声。
	for (int y = 0; y < scene.rows; ++y)
	{
		uchar* row = scene.ptr<uchar>(y);
		for (int x = 0; x < 120 && x < scene.cols; ++x)
		{
			const int n = ((x * 17 + y * 29) % 11) - 5;
			int v = (int)std::lround((row[x] - 42) * 0.45 + 42 + n);
			if (v < 0)
				v = 0;
			if (v > 255)
				v = 255;
			row[x] = (uchar)v;
		}
	}
	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 180;
	param.scoreThreshold = 0.5;
	param.maxCount = 200;
	param.iouThreshold = 0.4;
	param.minArea = 256;
	param.edgeMinMag = 29;
	param.edgeContrast = 119;
	param.maxEdgePoints = 320;
	param.usePolarity = true;
	param.greediness = 0.9;
	param.oriBins = 16;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ, mask) == 0, "shape ori16-texture setTemplate");
	std::vector<MatchResult> hits;
	const auto t0 = std::chrono::steady_clock::now();
	matcher->match(scene, hits);
	const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::cout << "    shape ori16-texture ±180 find " << ms << " ms n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " ang=" << h.Angle << " c=(" << h.Center.x << "," << h.Center.y << ")"
				  << std::endl;
	expect(countNear(hits, partial, 12.0) == 1, "shape ori16-texture finds left edge");
	const MatchResult* edge = nearestHit(hits, partial);
	expect(edge && edge->Score >= 0.5, "shape ori16-texture left edge score");
	if (edge)
	{
		double dAng = std::fabs(edge->Angle);
		if (dAng > 180.0)
			dAng = 360.0 - dAng;
		expect(dAng <= 8.0, "shape ori16-texture left edge angle");
	}
	expect(countNear(hits, interior, 12.0) == 1, "shape ori16-texture keeps interior");

	param.oriBins = 0;
	param.angle = 20;
	auto floatMatcher = GetMatcher(param);
	expect(floatMatcher && floatMatcher->setTemplate(templ, mask) == 0, "shape ori0-texture setTemplate");
	std::vector<MatchResult> floatHits;
	floatMatcher->match(scene, floatHits);
	std::cout << "    shape ori0-texture n=" << floatHits.size() << std::endl;
	expect(countNear(floatHits, partial, 12.0) == 1, "shape ori0-texture finds left edge");
}

// 贴边弱残件右侧有只在较细一层才明显的强边。第一层若用 ±10 px，峰会漂到大约 +16 px、
// 分数掉到 0.3 下面；±4 px 应留在真中心附近并且分数过 0.5。不是 233 原图。
void testShapeOri16BorderRefineDoesNotDrift()
{
	using namespace template_matching;
	const int tw = 144, th = 190;
	cv::Mat templ(th, tw, CV_8U, cv::Scalar(36));
	cv::rectangle(templ, {12, 14}, {tw - 14, th - 16}, cv::Scalar(200), 4);
	cv::rectangle(templ, {30, 36}, {tw / 2 + 4, th / 2}, cv::Scalar(210), cv::FILLED);
	cv::circle(templ, {tw - 34, 52}, 14, cv::Scalar(180), cv::FILLED);
	cv::circle(templ, {tw - 42, th - 44}, 9, cv::Scalar(40), cv::FILLED);
	cv::line(templ, {22, th - 40}, {tw - 20, th - 28}, cv::Scalar(220), 3);
	cv::Mat mask = cv::Mat::zeros(templ.size(), CV_8U);
	cv::rectangle(mask, {6, 8}, {tw - 8, th - 10}, cv::Scalar(255), cv::FILLED);

	auto paint = [&](cv::Mat& dst, cv::Point2d c, double angDeg, double gain) {
		cv::Mat rot;
		const cv::Point2f pivot((templ.cols - 1) * 0.5f, (templ.rows - 1) * 0.5f);
		const cv::Mat M = cv::getRotationMatrix2D(pivot, angDeg, 1.0);
		cv::warpAffine(templ, rot, M, templ.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(36));
		const int x0 = (int)std::lround(c.x - (rot.cols - 1) / 2.0);
		const int y0 = (int)std::lround(c.y - (rot.rows - 1) / 2.0);
		for (int y = 0; y < rot.rows; ++y)
		{
			const int sy = y0 + y;
			if (sy < 0 || sy >= dst.rows)
				continue;
			for (int x = 0; x < rot.cols; ++x)
			{
				const int sx = x0 + x;
				if (sx < 0 || sx >= dst.cols)
					continue;
				const int src = rot.at<uchar>(y, x);
				if (src == 36)
					continue;
				int v = 36 + (int)std::lround((src - 36) * gain);
				if (v < 0)
					v = 0;
				if (v > 255)
					v = 255;
				dst.at<uchar>(sy, sx) = (uchar)v;
			}
		}
	};

	cv::Mat scene(420, 640, CV_8U, cv::Scalar(36));
	for (int y = 0; y < scene.rows; y += 4)
	{
		uchar* row = scene.ptr<uchar>(y);
		for (int x = 0; x < scene.cols; x += 5)
			row[x] = (uchar)(30 + ((x * 13 + y * 7) % 17));
	}
	const cv::Point2d partial(26.3, 180);
	paint(scene, partial, 0.0, 0.55);
	// 右移 16px 的细轮廓。顶层几乎看不见，第一层 ±10 px 够得着，±4 px 够不着。
	cv::Mat lure(scene.size(), CV_8U, cv::Scalar(36));
	paint(lure, cv::Point2d(partial.x + 16.0, partial.y), 0.0, 1.0);
	for (int y = 0; y < scene.rows; ++y)
	{
		for (int x = 0; x < scene.cols; ++x)
		{
			const int lv = lure.at<uchar>(y, x);
			if (lv < 80)
				continue;
			const int sv = scene.at<uchar>(y, x);
			if (std::abs(sv - 36) > 18)
				continue;
			scene.at<uchar>(y, x) = (uchar)lv;
		}
	}

	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 180;
	param.scoreThreshold = 0.5;
	param.maxCount = 50;
	param.iouThreshold = 0.4;
	param.minArea = 256;
	param.edgeMinMag = 29;
	param.edgeContrast = 119;
	param.maxEdgePoints = 320;
	param.usePolarity = true;
	param.greediness = 0.9;
	param.oriBins = 16;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ, mask) == 0, "shape ori16-drift setTemplate");
	std::vector<MatchResult> hits;
	const auto t0 = std::chrono::steady_clock::now();
	matcher->match(scene, hits);
	const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::cout << "    shape ori16-drift ±180 find " << ms << " ms n=" << hits.size() << std::endl;
	for (const auto& h : hits)
		std::cout << "      sc=" << h.Score << " ang=" << h.Angle << " c=(" << h.Center.x << "," << h.Center.y << ")"
				  << std::endl;
	expect(countNear(hits, partial, 10.0) == 1, "shape ori16-drift stays on the left edge");
	const MatchResult* edge = nearestHit(hits, partial);
	expect(edge && edge->Score >= 0.5, "shape ori16-drift score passes 0.5");
	if (edge)
		expect(edge->Center.x < partial.x + 10.0, "shape ori16-drift does not walk right");

	param.oriBins = 0;
	param.angle = 30;
	auto floatMatcher = GetMatcher(param);
	expect(floatMatcher && floatMatcher->setTemplate(templ, mask) == 0, "shape ori0-drift setTemplate");
	std::vector<MatchResult> floatHits;
	floatMatcher->match(scene, floatHits);
	std::cout << "    shape ori0-drift n=" << floatHits.size() << std::endl;
	for (const auto& h : floatHits)
		std::cout << "      sc=" << h.Score << " ang=" << h.Angle << " c=(" << h.Center.x << "," << h.Center.y << ")"
				  << std::endl;
	expect(countNear(floatHits, partial, 10.0) == 1, "shape ori0-drift finds left edge");
}

// 10px 缺角在第一层只比 180° 高出大约 0.025，±2、步长 1、门槛 0.04 留在 -180°。
// 下一层分差大约 0.06，同一门槛要把它带回大约 0°。7.6° 那件不能被翻到对面。
// 完全对称的矩形转 180° 框不变，不在这里要求它回到 0°。
void testShapeOri16NotchDoesNotFlip()
{
	using namespace template_matching;
	const int tw = 143, th = 183;
	cv::Mat templ(th, tw, CV_8U, cv::Scalar(40));
	cv::rectangle(templ, {16, 14}, {tw - 18, th - 16}, cv::Scalar(190), cv::FILLED);
	cv::rectangle(templ, {16, 14}, {26, 24}, cv::Scalar(40), cv::FILLED);
	cv::Mat mask = cv::Mat::zeros(templ.size(), CV_8U);
	cv::rectangle(mask, {6, 6}, {tw - 8, th - 8}, cv::Scalar(255), cv::FILLED);

	cv::Mat scene(964, 1288, CV_8U, cv::Scalar(40));
	for (int y = 1; y < scene.rows; y += 3)
	{
		uchar* row = scene.ptr<uchar>(y);
		for (int x = 0; x < scene.cols; ++x)
			row[x] = (uchar)(40 + ((x * 5 + y * 3) % 50));
	}
	const cv::Point2d truth(364, 490);
	const double ang = 0.37;
	const cv::Point2d truthB(670.3, 845.3);
	const double angB = 7.64;
	cv::Mat rot;
	const cv::Point2f pivot((templ.cols - 1) * 0.5f, (templ.rows - 1) * 0.5f);
	cv::warpAffine(templ, rot, cv::getRotationMatrix2D(pivot, ang, 1.0), templ.size(), cv::INTER_LINEAR,
				   cv::BORDER_CONSTANT, cv::Scalar(40));
	const int x0 = (int)std::lround(truth.x - (rot.cols - 1) / 2.0);
	const int y0 = (int)std::lround(truth.y - (rot.rows - 1) / 2.0);
	for (int y = 0; y < rot.rows; ++y)
	{
		for (int x = 0; x < rot.cols; ++x)
		{
			const int sx = x0 + x, sy = y0 + y;
			if (sx < 0 || sy < 0 || sx >= scene.cols || sy >= scene.rows)
				continue;
			const uchar v = rot.at<uchar>(y, x);
			if (v != 40)
				scene.at<uchar>(sy, sx) = v;
		}
	}
	cv::Mat rotB;
	cv::warpAffine(templ, rotB, cv::getRotationMatrix2D(pivot, angB, 1.0), templ.size(), cv::INTER_LINEAR,
				   cv::BORDER_CONSTANT, cv::Scalar(40));
	const int x0b = (int)std::lround(truthB.x - (rotB.cols - 1) / 2.0);
	const int y0b = (int)std::lround(truthB.y - (rotB.rows - 1) / 2.0);
	for (int y = 0; y < rotB.rows; ++y)
	{
		for (int x = 0; x < rotB.cols; ++x)
		{
			const int sx = x0b + x, sy = y0b + y;
			if (sx < 0 || sy < 0 || sx >= scene.cols || sy >= scene.rows)
				continue;
			const uchar v = rotB.at<uchar>(y, x);
			if (v != 40)
				scene.at<uchar>(sy, sx) = v;
		}
	}

	MatcherParam param;
	param.matcherType = MatcherType::SHAPE;
	param.angle = 180;
	param.scoreThreshold = 0.5;
	param.maxCount = 3000;
	param.iouThreshold = 0.1;
	param.minArea = 64;
	param.edgeMinMag = 29;
	param.edgeContrast = 119;
	param.maxEdgePoints = 320;
	param.usePolarity = true;
	param.greediness = 0.9;
	param.oriBins = 16;
	auto matcher = GetMatcher(param);
	expect(matcher && matcher->setTemplate(templ, mask) == 0, "shape ori16-notch setTemplate");
	std::vector<MatchResult> hits;
	const auto t0 = std::chrono::steady_clock::now();
	matcher->match(scene, hits);
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	const MatchResult* h = nearestHit(hits, truth);
	double d = 1e9;
	double dAng = 180;
	if (h)
	{
		d = std::hypot(h->Center.x - truth.x, h->Center.y - truth.y);
		dAng = std::fabs(h->Angle - ang);
		if (dAng > 180.0)
			dAng = 360.0 - dAng;
	}
	std::cout << "    shape ori16-notch ±180 find " << ms << " ms n=" << hits.size();
	if (h)
		std::cout << " sc=" << h->Score << " ang=" << h->Angle << " d=" << d << " dang=" << dAng;
	std::cout << std::endl;
	expect(h && d <= 8.0, "shape ori16-notch stays on the painted center");
	expect(h && h->Score >= 0.5, "shape ori16-notch score passes 0.5");
	expect(h && dAng <= 15.0, "shape ori16-notch does not flip 180");
	const MatchResult* b = nearestHit(hits, truthB);
	double dB = 1e9;
	double dAngB = 180;
	if (b)
	{
		dB = std::hypot(b->Center.x - truthB.x, b->Center.y - truthB.y);
		dAngB = std::fabs(b->Angle - angB);
		if (dAngB > 180.0)
			dAngB = 360.0 - dAngB;
		std::cout << "    shape ori16-notch B sc=" << b->Score << " ang=" << b->Angle << " d=" << dB
				  << " dang=" << dAngB << std::endl;
	}
	expect(b && dB <= 8.0, "shape ori16-notch B stays on the painted center");
	expect(b && b->Score >= 0.5, "shape ori16-notch B score passes 0.5");
	expect(b && dAngB <= 15.0, "shape ori16-notch B does not flip 180");
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
	testPatternRingArcRejected();
	testPatternKeepsNeighborInMargin();
	testOverlapKeepsDifferentSizeNeighbor();
	testPatternWeakThirdOnNoise();
	testPatternAlmostRingRejected();
	testShapeDotSidelobeDropped();
	testPatternExposureQuadrantKept();
	testPatternCrushedHalfCapKept();
	testPatternFaintRingArcDropped();
	testPatternSpecularRingDropsOpenArc();
	testShapeMottleLockDropped();
	testShapeTrayMaskAndChip();
	testShapeHighContrastStillFinds();
	testShapeBorderPartial();
	testShapeOri16PartialMatchesFloat();
	testShapeOri16ResponseMapWideAngle();
	testShapeOri16Stride4KeepsWideHits();
	testShapeOri16WideLeftEdgePartial();
	testShapeOri16T11SizedLeftEdge();
	testShapeOri16TexturedLeftEdge();
	testShapeOri16BorderRefineDoesNotDrift();
	testShapeOri16NotchDoesNotFlip();
	testShapeFaintRingThird();
	testPatternOneQuadrantOcclusion();
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
