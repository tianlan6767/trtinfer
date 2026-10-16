#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace blob
{

// 固定阈值或 Otsu。invert=true 时暗目标变成前景（255）。
enum class ThresholdType {
	Fixed = 0,
	Otsu = 1
};

enum class MorphOp {
	Erode = 0,
	Dilate = 1,
	Open = 2,
	Close = 3
};

struct ThresholdParam {
	ThresholdType type = ThresholdType::Fixed;
	// Fixed 时的阈值。Otsu 忽略该值，由直方图决定。
	double value = 128;
	double maxValue = 255;
	bool invert = false;
};

struct MorphParam {
	MorphOp op = MorphOp::Open;
	// 方形核边长，偶数会自动加 1
	int ksize = 3;
	int iterations = 1;
};

// 连通域过滤。长宽比 = max(宽,高) / min(宽,高)，正方形为 1。
// 圆度 = 4π·面积 / 周长²，理想圆为 1（外轮廓，不含孔洞）。
struct BlobFilter {
	double minArea = 1;
	double maxArea = 1e18;
	double minCircularity = 0;
	double maxCircularity = 10;
	double minAspect = 0;
	double maxAspect = 1e9;
	// <=0 表示不截断（内部仍有上限，避免异常大图撑爆结果）
	int maxCount = 500;
};

struct ContourStats {
	double area = 0;
	double perimeter = 0;
	double circularity = 0;
	// minAreaRect 长边 / 短边
	double aspect = 1;
	// OpenCV minAreaRect 角度（度）
	double angle = 0;
	cv::Point2d center;
	int left = 0;
	int top = 0;
	int width = 0;
	int height = 0;
};

struct Blob {
	int label = 0;
	// 像素个数（连通域统计，不含背景孔）
	double area = 0;
	double perimeter = 0;
	double circularity = 0;
	// 外接矩形长边 / 短边
	double aspect = 1;
	double angle = 0;
	cv::Point2d center;
	int left = 0;
	int top = 0;
	int width = 0;
	int height = 0;
};

// 轴对齐裁剪，结果是连续图。越界会裁到图像内；无交集或宽高<=0 返回空图。
cv::Mat cropRoi(const cv::Mat& image, int x, int y, int width, int height);

// ksize 偶数会加 1。ksize<=0 且 sigma<=0 时原样拷贝。sigma=0 时由核大小推算。
cv::Mat gaussianBlur(const cv::Mat& image, int ksize = 5, double sigma = 0);

// 仅 uint8。ksize<3 时原样拷贝；偶数加 1。
cv::Mat medianBlur(const cv::Mat& image, int ksize = 5);

// 多通道先转灰度。输出单通道 8U，前景为 maxValue。
cv::Mat threshold(const cv::Mat& image, const ThresholdParam& param = ThresholdParam());

// 方形结构元。支持灰度 / 彩色 uint8。
cv::Mat morphology(const cv::Mat& image, const MorphParam& param = MorphParam());

// 二值图外轮廓统计。非 0 为前景。坐标与输入图一致。
std::vector<ContourStats> contourStats(const cv::Mat& binary);

// 8 连通域。非 0 为前景。
// roiW<=0 或 roiH<=0 表示整图；否则只分析该 ROI，但 center/left/top 仍是原图坐标。
std::vector<Blob> findBlobs(const cv::Mat& binary,
							const BlobFilter& filter = BlobFilter(),
							int roiX = 0,
							int roiY = 0,
							int roiW = 0,
							int roiH = 0);

}  // namespace blob
