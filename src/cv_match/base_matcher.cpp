#include "base_matcher.h"

#include <algorithm>
#include <cmath>

namespace template_matching
{

	BaseMatcher::BaseMatcher()
	{
		logger_ = spdlog::get("template_matching");
		logger_->set_level(spdlog::level::info);
		
	}

	BaseMatcher::~BaseMatcher()
	{
		// std::cout << "~BaseMatcher()" << std::endl;
	}

	bool BaseMatcher::isInited()
	{
		return initFinishedFlag_;
	}

	bool BaseMatcher::initMatcher(const template_matching::MatcherParam& param)
	{
		matchParam_ = param;

		return true;
	}

	void BaseMatcher::setMetricsTime(const bool& enabled)
	{
		metricsTime_ = enabled;
	}

	bool BaseMatcher::getMetricsTime()
	{
		return metricsTime_;
	}

	void BaseMatcher::drawResult(const cv::Mat& frame, std::vector<template_matching::MatchResult> matchResults)
	{
		if (frame.empty())
		{
			logger_->warn("image empty.");
			return;
		}

		cv::Mat drawFrame = frame.clone();
		if (drawFrame.channels() == 1)
		{
			cv::cvtColor(drawFrame, drawFrame, cv::COLOR_GRAY2BGR);
		}
		
		for (int i = 0; i < matchResults.size(); i++)
		{
			cv::Point2i temp;
			std::vector<cv::Point2i> pts;
			temp.x = std::round(matchResults[i].LeftTop.x);
			temp.y = std::round(matchResults[i].LeftTop.y);
			pts.push_back(temp);
			temp.x = std::round(matchResults[i].RightTop.x);
			temp.y = std::round(matchResults[i].RightTop.y);
			pts.push_back(temp);
			temp.x = std::round(matchResults[i].RightBottom.x);
			temp.y = std::round(matchResults[i].RightBottom.y);
			pts.push_back(temp);
			temp.x = std::round(matchResults[i].LeftBottom.x);
			temp.y = std::round(matchResults[i].LeftBottom.y);
			pts.push_back(temp);

			cv::polylines(drawFrame, pts, true, cv::Scalar(0, 255, 0), 1, cv::LINE_8);
		}

		cv::imwrite("demo.png", drawFrame);
		cv::imshow("demo", drawFrame);
		cv::waitKey();
	}

	namespace
	{
		constexpr int kMaxMatchScales = 48;
		constexpr double kScaleIdentityEps = 1e-4;

		double snapMatchScale(double s)
		{
			if (std::fabs(s - 1.0) < 1e-3)
				return 1.0;
			return s;
		}

		std::vector<double> uniqueScales(std::vector<double> scales)
		{
			std::vector<double> uniq;
			for (double s : scales)
			{
				if (!std::isfinite(s) || s <= 0.0)
					continue;
				bool dup = false;
				for (double u : uniq)
				{
					if (std::fabs(u - s) < 1e-4)
					{
						dup = true;
						break;
					}
				}
				if (!dup)
					uniq.push_back(s);
				if ((int)uniq.size() >= kMaxMatchScales)
					break;
			}
			if (uniq.empty())
				uniq.push_back(1.0);
			return uniq;
		}
	}

	std::vector<double> BaseMatcher::sampleMatchScales(const MatcherParam& param)
	{
		double lo = param.minScale;
		double hi = param.maxScale;
		if (!(lo > 0.0) || !(hi > 0.0) || !std::isfinite(lo) || !std::isfinite(hi))
			return {1.0};
		if (lo > hi)
			std::swap(lo, hi);
		if (std::fabs(lo - 1.0) < kScaleIdentityEps && std::fabs(hi - 1.0) < kScaleIdentityEps)
			return {1.0};

		std::vector<double> scales;
		if (std::fabs(hi - lo) < kScaleIdentityEps)
		{
			scales.push_back(snapMatchScale(lo));
			return scales;
		}
		if (param.numScales > 1)
		{
			const int n = std::min(param.numScales, kMaxMatchScales);
			for (int i = 0; i < n; ++i)
			{
				const double t = (double)i / (double)(n - 1);
				scales.push_back(snapMatchScale(lo + (hi - lo) * t));
			}
			return uniqueScales(std::move(scales));
		}

		double step = param.scaleStep;
		if (!(step > 0.0) || !std::isfinite(step))
			step = 0.05;
		if (step < 1e-3)
			step = 1e-3;
		for (double s = lo; (int)scales.size() < kMaxMatchScales; s += step)
		{
			double v = s;
			if (v > hi)
				v = hi;
			scales.push_back(snapMatchScale(v));
			if (v >= hi - 1e-9)
				break;
		}
		if (scales.empty() || scales.back() < hi - 1e-6)
			scales.push_back(snapMatchScale(hi));
		return uniqueScales(std::move(scales));
	}

	bool BaseMatcher::scaleSearchEnabled(const MatcherParam& param)
	{
		const std::vector<double> scales = sampleMatchScales(param);
		return !(scales.size() == 1 && std::fabs(scales[0] - 1.0) < kScaleIdentityEps);
	}

	namespace
	{
		double matchBoxSide(const MatchResult& r)
		{
			const double w = std::hypot(r.RightTop.x - r.LeftTop.x, r.RightTop.y - r.LeftTop.y);
			const double h = std::hypot(r.LeftBottom.x - r.LeftTop.x, r.LeftBottom.y - r.LeftTop.y);
			return std::max(1.0, std::min(w, h));
		}

		// 较小框被另一个框盖住的比例。完全包含时为 1。
		double smallerCoveredFraction(const MatchResult& a, const MatchResult& b)
		{
			std::vector<cv::Point2f> pa{
				cv::Point2f((float)a.LeftTop.x, (float)a.LeftTop.y),
				cv::Point2f((float)a.RightTop.x, (float)a.RightTop.y),
				cv::Point2f((float)a.RightBottom.x, (float)a.RightBottom.y),
				cv::Point2f((float)a.LeftBottom.x, (float)a.LeftBottom.y)};
			std::vector<cv::Point2f> pb{
				cv::Point2f((float)b.LeftTop.x, (float)b.LeftTop.y),
				cv::Point2f((float)b.RightTop.x, (float)b.RightTop.y),
				cv::Point2f((float)b.RightBottom.x, (float)b.RightBottom.y),
				cv::Point2f((float)b.LeftBottom.x, (float)b.LeftBottom.y)};
			if (pa.size() < 3 || pb.size() < 3)
				return 0.0;
			const cv::RotatedRect ra = cv::minAreaRect(pa);
			const cv::RotatedRect rb = cv::minAreaRect(pb);
			const double areaA = std::max(1.0, (double)ra.size.area());
			const double areaB = std::max(1.0, (double)rb.size.area());
			std::vector<cv::Point2f> inter;
			const int typ = cv::rotatedRectangleIntersection(ra, rb, inter);
			if (typ == cv::INTERSECT_NONE)
				return 0.0;
			double interArea = 0.0;
			if (typ == cv::INTERSECT_FULL)
				interArea = std::min(areaA, areaB);
			else if (inter.size() >= 3)
			{
				std::vector<cv::Point2f> hull;
				cv::convexHull(inter, hull, true);
				interArea = std::max(0.0, cv::contourArea(hull));
			}
			return interArea / std::min(areaA, areaB);
		}
	}

	void BaseMatcher::suppressMatchOverlap(std::vector<MatchResult>& results, double iouThreshold, int maxCount)
	{
		if (maxCount < 1)
			maxCount = 1;
		std::sort(results.begin(), results.end(), [](const MatchResult& a, const MatchResult& b) {
			return a.Score > b.Score;
		});
		// 中心落在较小目标内部，就是同一实例的另一个角度或相邻尺度，跟夹角无关。
		const double centerFrac = 0.40;
		const double coverThr = iouThreshold > 0.0 ? iouThreshold : 0.5;
		std::vector<MatchResult> kept;
		kept.reserve(std::min((int)results.size(), maxCount));
		for (const MatchResult& cand : results)
		{
			bool drop = false;
			for (const MatchResult& prev : kept)
			{
				const double dist = std::hypot(cand.Center.x - prev.Center.x, cand.Center.y - prev.Center.y);
				const double minSide = std::min(matchBoxSide(cand), matchBoxSide(prev));
				if (dist < centerFrac * minSide || smallerCoveredFraction(cand, prev) > coverThr)
				{
					drop = true;
					break;
				}
			}
			if (drop)
				continue;
			kept.push_back(cand);
			if ((int)kept.size() >= maxCount)
				break;
		}
		results.swap(kept);
	}
}
