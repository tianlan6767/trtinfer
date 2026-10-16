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

}  // namespace

int main()
{
	testPatternIdentity();
	testPatternScaled();
	testPatternScaledMask();
	testShapeIdentity();
	testShapeScaled();
	testShapeNumScalesDown();
	if (g_fails)
	{
		std::cerr << g_fails << " check(s) failed" << std::endl;
		return 1;
	}
	std::cout << "all scale-search checks passed" << std::endl;
	return 0;
}
