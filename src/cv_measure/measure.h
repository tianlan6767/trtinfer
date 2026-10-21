#pragma once

#include "cv_caliper/metrology.h"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace measure
{

enum class FitMethod {
	LeastSquares = 0,
	Ransac = 1
};

struct FitParam {
	FitMethod method = FitMethod::LeastSquares;
	// 内点距离阈值（像素）
	double threshold = 1.5;
	int iterations = 100;
	unsigned seed = 1;
};

struct CircleFit {
	bool found = false;
	cv::Point2d center;
	double radius = 0.0;
	double rms = 0.0;
	int numPoints = 0;
	int numInliers = 0;
	std::vector<cv::Point2d> inliers;
};

// a、b 是半轴，a >= b。angle 是长轴相对 +X 的度（y 向下）。
struct EllipseFit {
	bool found = false;
	cv::Point2d center;
	double a = 0.0;
	double b = 0.0;
	double angle = 0.0;
	double rms = 0.0;
	int numPoints = 0;
	int numInliers = 0;
	std::vector<cv::Point2d> inliers;
};

struct BoxSize {
	bool found = false;
	cv::Point2d center;
	double width = 0.0;
	double height = 0.0;
	double angle = 0.0;
	cv::Point2d c0, c1, c2, c3;
};

// 平行四边形。rectangle 表示四角都接近直角。angleError 是相对 90° 的最大偏差（度）。
struct QuadFit {
	bool found = false;
	bool rectangle = false;
	cv::Point2d center;
	double width = 0.0;
	double height = 0.0;
	double angle = 0.0;
	double angleError = 0.0;
	double rms = 0.0;
	cv::Point2d c0, c1, c2, c3;
};

// value 是测得偏差，tolerance 是允许带宽。margin = tolerance - value。
struct FormResult {
	bool found = false;
	bool pass = false;
	double value = 0.0;
	double tolerance = 0.0;
	double margin = 0.0;
	double rms = 0.0;
};

struct CircleSize {
	bool found = false;
	cv::Point2d center;
	double radius = 0.0;
	double diameter = 0.0;
};

// 旋转卡尺宽度。min/maxAngle 是投影方向（度，[0, 180)）。
struct WidthFunction {
	bool found = false;
	double minWidth = 0.0;
	double maxWidth = 0.0;
	double minAngle = 0.0;
	double maxAngle = 0.0;
	cv::Point2d minA, minB, maxA, maxB;
};

// 量规读数，单位是像素。value 为间隙，负值表示干涉或重叠。
struct GageResult {
	bool found = false;
	double value = 0.0;
	double centerDistance = 0.0;
	double angle = 0.0;
	cv::Point2d a, b;
};

// mmPerPx 是各向同性近似（X/Y 平均）。各向异性用 mmPerPxX / mmPerPxY。
struct PixelScale {
	bool found = false;
	double mmPerPx = 1.0;
	double mmPerPxX = 1.0;
	double mmPerPxY = 1.0;
	std::string source;
};

// origin 是图像像素。angleDeg 是夹具 +X 相对图像 +X 的角度，y 向下。
struct FixtureFrame {
	cv::Point2d origin;
	double angleDeg = 0.0;
};

struct DimValue {
	bool found = false;
	bool scaled = false;
	double pixels = 0.0;
	double physical = 0.0;
	cv::Point2d fromPx, toPx;
	cv::Point2d fromFrame, toFrame;
};

metrology::LineFit fitLine(const std::vector<cv::Point2d>& points, const FitParam& param = FitParam());
CircleFit fitCircle(const std::vector<cv::Point2d>& points, const FitParam& param = FitParam());
EllipseFit fitEllipse(const std::vector<cv::Point2d>& points, const FitParam& param = FitParam());

BoxSize axisAlignedBox(const std::vector<cv::Point2d>& points);
BoxSize minAreaBox(const std::vector<cv::Point2d>& points);
QuadFit fitParallelogram(const std::vector<cv::Point2d>& points);

// 亮目标或 0/255 掩膜。取最大外轮廓。
std::vector<cv::Point2d> contourPoints(const cv::Mat& image);
BoxSize minAreaBoxContour(const cv::Mat& image);
QuadFit fitParallelogramContour(const cv::Mat& image);

// 平行度：value 是夹角（度），0 为平行。垂直度：value 是相对 90° 的偏差。
FormResult checkParallelism(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2, double tolDeg);
FormResult checkPerpendicularity(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2, double tolDeg);
// 直线度：点到拟合直线的峰谷带宽度。圆度：拟合圆半径的峰谷差。
FormResult checkStraightness(const std::vector<cv::Point2d>& points, double tolPx);
FormResult checkRoundness(const std::vector<cv::Point2d>& points, double tolPx);

CircleSize inscribedCircle(const std::vector<cv::Point2d>& points);
CircleSize inscribedCircleMask(const cv::Mat& image);
CircleSize circumscribedCircle(const std::vector<cv::Point2d>& points);
WidthFunction widthFunction(const std::vector<cv::Point2d>& points, double stepDeg = 1.0);

// 孔边到直线的间隙：圆心到直线距离 − 半径。
GageResult holeToEdge(cv::Point2d center, double radius, cv::Point2d a, cv::Point2d b);
// 两孔边缘间隙：圆心距 − r1 − r2。centerDistance 是圆心距。
GageResult holeToHole(cv::Point2d c0, double r0, cv::Point2d c1, double r1);
// 两边间距。不平行（默认超过 15°）时 found=false。
GageResult edgeToEdge(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2, double angleTolDeg = 15.0);

PixelScale scaleMmPerPx(double mmPerPx);
PixelScale scaleMmPerPxXY(double mmPerPxX, double mmPerPxY);
// 正对平面：mm/px = Z / f。Z 与标定平移同一单位，通常是毫米。
PixelScale scaleFromCameraMatrix(const cv::Mat& cameraMatrix, double zMm);
// YAML 里有 mmPerPx 时直接用。否则读 cameraMatrix，并要求 zMm > 0。
PixelScale scaleFromYaml(const std::string& path, double zMm = 0.0);

cv::Point2d toPlane(cv::Point2d px, const PixelScale& scale);
cv::Point2d toFixture(cv::Point2d px, const PixelScale& scale, const FixtureFrame& frame);

DimValue dimensionPoints(cv::Point2d a, cv::Point2d b, const PixelScale& scale = PixelScale(),
						 const FixtureFrame& frame = FixtureFrame());
DimValue dimensionPointLine(cv::Point2d p, cv::Point2d a, cv::Point2d b,
							const PixelScale& scale = PixelScale(),
							const FixtureFrame& frame = FixtureFrame());
DimValue dimensionLines(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2,
						const PixelScale& scale = PixelScale(),
						const FixtureFrame& frame = FixtureFrame(),
						double angleTolDeg = 15.0);
DimValue scaleGage(const GageResult& gage, const PixelScale& scale = PixelScale(),
				   const FixtureFrame& frame = FixtureFrame());

}  // namespace measure
