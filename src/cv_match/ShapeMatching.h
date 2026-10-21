#pragma once

#include "base_matcher.h"
#include <vector>

namespace template_matching
{

struct ShapeEdgePt
{
	float dx = 0.f;
	float dy = 0.f;
	float nx = 0.f;
	float ny = 0.f;
};

struct ShapeLayer
{
	int width = 0;
	int height = 0;
	std::vector<ShapeEdgePt> edges;
};

struct ShapeCandidate
{
	double cx = 0;
	double cy = 0;
	double angle = 0;
	double score = 0;
	double scale = 1.0;
	// 顶层贴边用浮点粗分得到的种子。后面的去重和精修窗口都按 oriBins=0 那条走，
	// 不和响应图的内部高峰混在一个名额里。
	bool fromBorder = false;
	// 内部被更高的响应图峰压住、又从环带里捞回来的种子。第一层位置窗口用 ±4，
	// 角度仍大约 ±12°。贴边种子不会设这个标记。
	bool narrowRefine = false;
};

class ShapeMatcher : public BaseMatcher
{
public:
	ShapeMatcher(const MatcherParam& param);
	~ShapeMatcher() override = default;

	int match(const cv::Mat& frame, std::vector<MatchResult>& matchResults) override;
	int setTemplate(const cv::Mat& templateImage, const cv::Mat& mask = cv::Mat()) override;

private:
	std::vector<ShapeLayer> layers_;
	bool learned_ = false;
	bool scaleSearch_ = false;
	double scaleMin_ = 1.0;
	double scaleMax_ = 1.0;
	double scaleDs_ = 0.0;
	std::vector<double> scales_{1.0};
	int templW_ = 0;
	int templH_ = 0;
};

}  // namespace template_matching
