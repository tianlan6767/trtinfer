#pragma once

#include <opencv2/core.hpp>

namespace sim
{

// 对齐 ROI 的相似度。PATTERN 匹配器已经是金字塔 + 旋转 + 尺度的 NCC，
// 这里是另一条更瘦的接口：两块同尺寸图打分，或只做平移的模板定位。
struct SimilarityScore {
	bool found = false;
	double ncc = 0;               // TM_CCORR_NORMED，相同图约 1
	double zncc = 0;              // TM_CCOEFF_NORMED，[-1, 1]
	double histCorr = 0;          // 灰度直方图相关，[-1, 1]
	double histBhattacharyya = 1; // 0 完全相同，1 完全不同
	double ssim = 0;              // 结构相似，约 [0, 1]
};

struct LocateResult {
	bool found = false;
	double score = 0;
	cv::Point2d topLeft;
	cv::Point2d center;
	int width = 0;
	int height = 0;
};

// 两张图必须同尺寸。彩色先转灰度。尺寸不合或空图时 found=false。
SimilarityScore compare(const cv::Mat& a, const cv::Mat& b);

// 只搜平移，不搜角度和尺度。score 是 ZNCC。
LocateResult locateZncc(const cv::Mat& image, const cv::Mat& templ, double threshold = 0.6);

// 只搜平移。score 是 NCC（TM_CCORR_NORMED）。
LocateResult locateNcc(const cv::Mat& image, const cv::Mat& templ, double threshold = 0.6);

}  // namespace sim
