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
	int templW_ = 0;
	int templH_ = 0;
};

}  // namespace template_matching
