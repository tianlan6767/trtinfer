#ifndef _BASEMATCHER_H 
#define _BASEMATCHER_H 

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <random>
#include <chrono>
#include "matcher.h"
#include "omp.h"
#include <vector>


namespace template_matching
{
	class BaseMatcher : public Matcher
	{
	public:
		BaseMatcher();

		~BaseMatcher();

		// 判断是否初始化成功
		bool isInited();

		virtual void drawResult(const cv::Mat& frame, std::vector<template_matching::MatchResult> matchResults);

		void setMetricsTime(const bool& enabled);

		bool getMetricsTime();

		// 多实例去重：同一中心的不同角度、相邻尺度上的错位锁，都算同一个目标。
		// 尺寸接近时，较小框被盖住的比例超过 iouThreshold 也去掉（<=0 时按 0.5，Halcon MaxOverlap 默认）。
		// 尺寸差一截且中心分开的，不因为大框的空白边包住小框而删掉。
		// dropWeakNearby：尺度明显不同、中心仍靠近、且分数低一截的，视为错尺度旁瓣。
		static void suppressMatchOverlap(std::vector<MatchResult>& results, double iouThreshold, int maxCount,
										 bool dropWeakNearby = false);

	protected:

		// 初始化匹配器
		bool initMatcher(const MatcherParam& param);

		// 初始化完成标志位
		bool initFinishedFlag_ = false;

		// 匹配参数
		MatcherParam matchParam_;

		// template
		cv::Mat templateImage_;

		// 计时
		std::chrono::steady_clock::time_point startTime_, endTime_;
		std::chrono::duration<double> timeUse_;

		// 评估时间
		bool metricsTime_ = false;

		std::shared_ptr<spdlog::logger> logger_;

		// 按 MatcherParam 采样尺度。默认只返回 {1}。
		static std::vector<double> sampleMatchScales(const MatcherParam& param);
		static bool scaleSearchEnabled(const MatcherParam& param);

	private:


	};
}

#endif
