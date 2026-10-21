#include "coarse_refine.h"

#include "matcher.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>

namespace template_matching
{
namespace
{

cv::Mat asGray(const cv::Mat& image)
{
	if (image.empty())
		return {};
	cv::Mat gray;
	if (image.channels() == 1)
		gray = image;
	else if (image.channels() == 3)
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	else if (image.channels() == 4)
		cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
	else
		return {};
	if (gray.depth() != CV_8U) {
		cv::Mat u8;
		gray.convertTo(u8, CV_8U);
		return u8;
	}
	return gray;
}

MatcherParam toMatcherParam(const CoarseRefineParam& param)
{
	MatcherParam out;
	out.matcherType = param.matcherType;
	out.maxCount = param.maxCount;
	out.scoreThreshold = param.scoreThreshold;
	out.iouThreshold = param.iouThreshold;
	out.angle = param.angle;
	out.minArea = param.minArea;
	out.meanBorder = param.meanBorder;
	out.stopLayer = param.stopLayer;
	out.edgeMinMag = param.edgeMinMag;
	out.maxEdgePoints = param.maxEdgePoints;
	out.usePolarity = param.usePolarity;
	out.greediness = param.greediness;
	out.minScale = param.minScale;
	out.maxScale = param.maxScale;
	out.scaleStep = param.scaleStep;
	out.numScales = param.numScales;
	return out;
}

void offsetHit(MatchResult& hit, double ox, double oy)
{
	hit.LeftTop.x += ox;
	hit.LeftTop.y += oy;
	hit.LeftBottom.x += ox;
	hit.LeftBottom.y += oy;
	hit.RightTop.x += ox;
	hit.RightTop.y += oy;
	hit.RightBottom.x += ox;
	hit.RightBottom.y += oy;
	hit.Center.x += ox;
	hit.Center.y += oy;
}

std::string coarseFailMessage(const feature::MatchResult& coarse, int need)
{
	if (coarse.method.find("unavailable") != std::string::npos)
		return coarse.method;
	std::ostringstream oss;
	oss << "coarse match failed (inliers=" << coarse.inliers << ", need=" << need << ")";
	return oss.str();
}

// 与验证过的脚本一致：floor(min)-margin .. ceil(max)+margin，再保证至少能放下模板。
bool roiFromCorners(const cv::Mat& corners, const cv::Size& imageSize, int margin,
					int minWidth, int minHeight, cv::Rect& roi)
{
	if (corners.rows < 4 || corners.cols < 2 || corners.type() != CV_64F)
		return false;
	if (imageSize.width <= 0 || imageSize.height <= 0)
		return false;

	double minX = std::numeric_limits<double>::infinity();
	double minY = minX;
	double maxX = -minX;
	double maxY = -minX;
	for (int i = 0; i < corners.rows; ++i) {
		const double x = corners.at<double>(i, 0);
		const double y = corners.at<double>(i, 1);
		if (!std::isfinite(x) || !std::isfinite(y))
			return false;
		minX = std::min(minX, x);
		maxX = std::max(maxX, x);
		minY = std::min(minY, y);
		maxY = std::max(maxY, y);
	}
	if (minX < -1e8 || minY < -1e8 || maxX > 1e8 || maxY > 1e8)
		return false;

	if (margin < 0)
		margin = 0;
	int x0 = static_cast<int>(std::floor(minX)) - margin;
	int y0 = static_cast<int>(std::floor(minY)) - margin;
	int x1 = static_cast<int>(std::ceil(maxX)) + margin;
	int y1 = static_cast<int>(std::ceil(maxY)) + margin;
	if (x1 < x0)
		std::swap(x0, x1);
	if (y1 < y0)
		std::swap(y0, y1);

	const int minW = std::max(1, minWidth);
	const int minH = std::max(1, minHeight);
	if (x1 - x0 < minW) {
		const int cx = (x0 + x1) / 2;
		x0 = cx - minW / 2;
		x1 = x0 + minW;
	}
	if (y1 - y0 < minH) {
		const int cy = (y0 + y1) / 2;
		y0 = cy - minH / 2;
		y1 = y0 + minH;
	}

	const int W = imageSize.width;
	const int H = imageSize.height;
	if (x1 - x0 > W) {
		x0 = 0;
		x1 = W;
	} else {
		if (x0 < 0) {
			x1 -= x0;
			x0 = 0;
		}
		if (x1 > W) {
			x0 -= x1 - W;
			x1 = W;
		}
	}
	if (y1 - y0 > H) {
		y0 = 0;
		y1 = H;
	} else {
		if (y0 < 0) {
			y1 -= y0;
			y0 = 0;
		}
		if (y1 > H) {
			y0 -= y1 - H;
			y1 = H;
		}
	}
	x0 = std::max(0, x0);
	y0 = std::max(0, y0);
	x1 = std::min(W, x1);
	y1 = std::min(H, y1);
	if (x1 <= x0 || y1 <= y0)
		return false;
	roi = cv::Rect(x0, y0, x1 - x0, y1 - y0);
	return true;
}

}  // namespace

CoarseRefineResult matchCoarseToFine(const cv::Mat& scene, const cv::Mat& templ,
									 const cv::Mat& mask, const CoarseRefineParam& param)
{
	CoarseRefineResult out;
	if (scene.empty() || templ.empty()) {
		out.message = "empty image";
		return out;
	}

	const cv::Mat sceneGray = asGray(scene);
	const cv::Mat templGray = asGray(templ);
	if (sceneGray.empty() || templGray.empty()) {
		out.message = "unsupported image";
		return out;
	}

	cv::Mat mask8;
	if (!mask.empty()) {
		if (mask.channels() != 1 || mask.size() != templGray.size()) {
			out.message = "mask size mismatch";
			return out;
		}
		if (mask.depth() == CV_8U)
			mask8 = mask;
		else
			mask.convertTo(mask8, CV_8U);
	}

	out.coarse = feature::matchHomography(sceneGray, templGray, param.detector, param.ratio, param.ransacReproj);
	const int need = std::max(4, param.minInliers);
	if (out.coarse.inliers < need || out.coarse.homography.empty() || out.coarse.corners.rows < 4) {
		out.message = coarseFailMessage(out.coarse, need);
		return out;
	}

	cv::Rect rect;
	if (!roiFromCorners(out.coarse.corners, sceneGray.size(), param.margin, templGray.cols, templGray.rows, rect)) {
		out.message = "coarse corners invalid";
		return out;
	}
	out.roi.x = rect.x;
	out.roi.y = rect.y;
	out.roi.width = rect.width;
	out.roi.height = rect.height;
	if (rect.width < templGray.cols || rect.height < templGray.rows) {
		out.message = "roi smaller than template";
		return out;
	}

	// 非连续 ROI（numpy 切片或 Mat 子图）直接送进 Matcher 会得到 0 个命中。这里强制 clone。
	const cv::Mat roi = sceneGray(rect).clone();

	const MatcherParam matcherParam = toMatcherParam(param);
	std::unique_ptr<Matcher> matcher = GetMatcher(matcherParam);
	if (!matcher) {
		out.message = "matcher init failed";
		return out;
	}

	std::vector<MatchResult> local;
	try {
		const int learned = matcher->setTemplate(templGray, mask8);
		if (learned != 0) {
			out.message = "setTemplate failed (" + std::to_string(learned) + ")";
			return out;
		}
		const int n = matcher->match(roi, local);
		if (n < 0) {
			out.message = "match failed (" + std::to_string(n) + ")";
			return out;
		}
	} catch (const cv::Exception& ex) {
		out.message = std::string("match failed: ") + ex.what();
		return out;
	}

	out.hits.reserve(local.size());
	for (MatchResult& hit : local) {
		offsetHit(hit, static_cast<double>(rect.x), static_cast<double>(rect.y));
		out.hits.push_back(hit);
	}
	if (out.hits.empty()) {
		out.message = "no fine hit";
		return out;
	}
	out.found = true;
	return out;
}

}  // namespace template_matching
