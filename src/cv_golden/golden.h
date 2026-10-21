#pragma once

#include "cv_blob/blob.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace golden
{

// 待测相对金样的极性。差分是「对齐后的待测 - 金样」。
enum class Polarity {
	Both = 0,      // |差分| >= 阈值
	Brighter = 1,  // 待测更亮
	Darker = 2     // 待测更暗
};

enum class AlignMode {
	None = 0,        // 已经对齐，两图必须同尺寸
	Translate = 1,   // 复用 locate_zncc / locate_ncc，只平移
	Affine = 2,      // ORB + estimateAffine2D（不够内点时退回相似变换）
	Homography = 3   // ORB 单应，复用 feature::matchHomography
};

// 比对在金样坐标系里做。transform 把金样上的点变到原始待测图。
struct GoldenParam {
	AlignMode align = AlignMode::None;
	// Translate 时 false 用 ZNCC，true 用 NCC。
	bool useNcc = false;
	double alignThreshold = 0.6;
	// 同尺寸平移：金样四边各裁掉这么多再定位，最大平移大约就是这个值。
	// 金样比待测小（整图当模板）时忽略。会自动缩小，保证模板至少约 8px。
	int alignMargin = 16;

	double featureRatio = 0.75;
	double ransacReproj = 3.0;
	int minInliers = 6;

	Polarity polarity = Polarity::Both;
	double diffThreshold = 30;
	// <=0 且 blurKsize<=1 时不做模糊。模糊只参与差分，不改 alignedImage。
	double blurSigma = 0;
	int blurKsize = 0;

	// 金样坐标系。宽或高 <=0 表示整图。
	int roiX = 0;
	int roiY = 0;
	int roiW = 0;
	int roiH = 0;

	bool enableMorph = true;
	blob::MorphOp morphOp = blob::MorphOp::Open;
	int morphKsize = 3;
	int morphIterations = 1;

	double minArea = 16;
	// <=0 时不按业务截断（内部仍有上限）。按面积从大到小保留。
	int maxCount = 100;

	bool keepDiff = true;
	bool keepMask = true;
	bool keepAligned = true;
};

struct GoldenDefect {
	int label = 0;
	double area = 0;
	double meanAbs = 0;
	// 对齐后待测 - 金样。正数表示这块更亮。
	double meanSigned = 0;
	cv::Point2d center;
	int left = 0;
	int top = 0;
	int width = 0;
	int height = 0;
};

struct GoldenResult {
	bool ok = false;
	// 对齐步骤成功（None 同尺寸也算）。ROI 为空时 aligned 仍可能为 true，但 ok 为 false。
	bool aligned = false;
	std::string message;
	// none / zncc / ncc / orb-affine / orb-affine-partial / orb-homography
	std::string alignMethod;
	// 平移：ZNCC 或 NCC。特征：内点 / 匹配数。
	double alignScore = 0;
	int alignInliers = 0;
	// 3x3 CV_64F，金样坐标 → 待测坐标。失败时可能为空。
	cv::Mat transform;

	// 下面几个分数在有效像素上统计（ROI ∩ 掩膜 ∩ 对齐覆盖），不是只在缺陷里。
	double meanAbs = 0;
	double meanSigned = 0;
	double maxAbs = 0;
	int defectCount = 0;
	double defectArea = 0;
	int validPixels = 0;
	double defectRatio = 0;

	std::vector<GoldenDefect> defects;

	cv::Mat diff;          // CV_8U，绝对差分，有效区外为 0
	cv::Mat signedDiff;    // CV_32F，对齐后待测 - 金样
	cv::Mat mask;          // CV_8U，形态学之后的缺陷，255 为缺陷
	cv::Mat alignedImage;  // CV_8U，待测对齐到金样坐标系（模糊之前）
};

// mask 与金样同尺寸，非 0 为参与比对。空 mask 表示整图都参与。
// 彩色输入先转灰度。不改输入图。
GoldenResult compare(const cv::Mat& test,
					 const cv::Mat& golden,
					 const GoldenParam& param = GoldenParam(),
					 const cv::Mat& mask = cv::Mat());

}  // namespace golden
