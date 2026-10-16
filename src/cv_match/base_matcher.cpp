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
}
