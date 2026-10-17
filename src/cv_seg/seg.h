#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace seg
{

struct Contour {
	double area = 0;
	double perimeter = 0;
	cv::Point2d center;
	int left = 0;
	int top = 0;
	int width = 0;
	int height = 0;
	cv::Mat points;  // Nx2 CV_64F
};

struct Region {
	int label = 0;
	double area = 0;
	cv::Point2d center;
	int left = 0;
	int top = 0;
	int width = 0;
	int height = 0;
};

struct WatershedResult {
	bool found = false;
	cv::Mat labels;  // CV_32S，边界为 -1
	std::vector<Region> regions;
};

// L2 距离变换。非 0 为前景。输出 CV_32F，与输入同尺寸。
cv::Mat distanceTransform(const cv::Mat& binary);

// Canny。彩色先转灰度。
cv::Mat canny(const cv::Mat& image, double low = 50, double high = 150, int aperture = 3);

// 外轮廓。externalOnly=false 时取全部轮廓（含孔）。
std::vector<Contour> extractContours(const cv::Mat& binary, bool externalOnly = true);

// markers：CV_32S，0 是未知，>0 是种子。image 灰度或 BGR。
WatershedResult watershed(const cv::Mat& image, const cv::Mat& markers);

// 距离变换 + 峰值种子的分水岭，用来切开粘连 Blob。
// minDistance：峰值之间的最小像素距离。distRatio：峰值至少是最大距离的这个比例。
WatershedResult splitBlobs(const cv::Mat& binary, double minDistance = 8, double distRatio = 0.4);

}  // namespace seg
