#pragma once

#include "cv_feature/feature.h"
#include "cv_match/template_matching.h"

#include <string>
#include <vector>

namespace template_matching
{

// 粗定位四角在全图上的轴对齐裁切，像素坐标，x/y 为左上角。
struct LIB_API SceneRoi
{
	int x = 0;
	int y = 0;
	int width = 0;
	int height = 0;
};

// 特征粗定位 + ROI 上 Matcher 精匹配。
// 精匹配字段与 MatcherParam 同名，默认与 MatcherParam 一致，但 maxCount 默认为 1。
// angle 默认 0：只搜 0°，相信粗定位朝向。SHAPE 在大图上半宽过窄会漏（±3° 漏、±5° 可以），
// 走 SHAPE 时把 angle 设为至少 5。
// stopLayer 原样传给 Matcher。不要把 2 当成加速：大图上会锁到大约差半个分辨率的错误位姿。
struct LIB_API CoarseRefineParam
{
	feature::Detector detector = feature::Detector::ORB;
	double ratio = 0.75;
	double ransacReproj = 3.0;
	// 内点少于此数不做精匹配。低于 4 时按 4。match_features 的 found 要 6 个内点，
	// 这里默认 4，所以 4～5 个内点仍会进入精匹配。
	int minInliers = 4;
	// 粗定位四角轴对齐框向外扩的像素。大图样本用 120。
	int margin = 120;

	MatcherType matcherType = MatcherType::PATTERN;
	int maxCount = 1;
	double scoreThreshold = 0.5;
	double iouThreshold = 0.0;
	double angle = 0;
	double minArea = 256;
	bool meanBorder = true;
	int stopLayer = 0;
	double edgeMinMag = 18.0;
	double edgeContrast = 0.0;
	int maxEdgePoints = 320;
	// 0=浮点形状分；16=顶层 16 档方向粗搜。含义同 MatcherParam.oriBins。
	int oriBins = 0;
	bool usePolarity = true;
	double greediness = 0.9;
	double minScale = 1.0;
	double maxScale = 1.0;
	double scaleStep = 0.05;
	int numScales = 0;
};

// hits 的点在输入 scene 坐标系，不是 ROI 局部坐标。
// mask 只参与精匹配，不参与特征粗定位。
struct LIB_API CoarseRefineResult
{
	bool found = false;
	std::string message;
	feature::MatchResult coarse;
	SceneRoi roi;
	std::vector<MatchResult> hits;
};

// 特征粗定位 → 四角外扩得到轴对齐 ROI → clone 成连续图 → Matcher 精匹配 → 坐标加回全图。
// scene / templ 可以是灰度或 BGR。mask 为空则不用掩膜；非空时须与模板同尺寸、单通道。
LIB_API CoarseRefineResult matchCoarseToFine(const cv::Mat& scene,
											  const cv::Mat& templ,
											  const cv::Mat& mask = cv::Mat(),
											  const CoarseRefineParam& param = CoarseRefineParam());

}  // namespace template_matching
