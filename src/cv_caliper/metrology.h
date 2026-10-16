#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace metrology
{

struct LineFit {
	bool found = false;
	cv::Point2d p1;
	cv::Point2d p2;
	cv::Point2d center;
	cv::Point2d direction;
	double angle = 0.0;
	double rms = 0.0;
	double length = 0.0;
	int numPoints = 0;
};

struct SizeSummary {
	bool found = false;
	cv::Point2d center;
	double width = 0.0;
	double height = 0.0;
	double angle = 0.0;
	double diameter = 0.0;
};

double distancePoints(cv::Point2d a, cv::Point2d b);

// 点到无限直线（过 a、b）的距离。a≈b 时返回 NaN。
double distancePointToLine(cv::Point2d p, cv::Point2d a, cv::Point2d b);

// 两条无限直线的间距。近似平行时是法向距离；夹角大时取两端中点到另一直线距离的平均。
double distanceBetweenLines(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2);

// 两直线方向夹角，度，范围 [0, 90]。
double angleBetweenLines(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2);

// 无限直线求交。近平行返回 false。
bool intersectLines(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2, cv::Point2d& out);

// 点集 SVD 直线。方向使 direction.y >= 0，angle 为相对 +X 的度（y 向下，正方向朝 +Y）。
LineFit fitLine(const std::vector<cv::Point2d>& points);

// 四个角：沿宽边从左到右为 tl→tr，再 br、bl。angle 与宽边方向一致。
SizeSummary summarizeRectangle(cv::Point2d topLeft, cv::Point2d topRight,
							   cv::Point2d bottomRight, cv::Point2d bottomLeft);

SizeSummary summarizeCircle(cv::Point2d center, double radius);

}  // namespace metrology
