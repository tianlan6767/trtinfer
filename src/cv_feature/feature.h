#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace feature
{

enum class Detector {
	ORB = 0,
	SIFT = 1
};

struct MatchResult {
	bool found = false;
	std::string method;
	int keypointsModel = 0;
	int keypointsScene = 0;
	int matches = 0;
	int inliers = 0;
	cv::Mat homography;    // 3x3 CV_64F，失败时为空
	cv::Mat modelPoints;   // Nx2，比例测试后的匹配
	cv::Mat scenePoints;   // Nx2
	std::vector<unsigned char> inlierMask;
	cv::Mat corners;       // 4x2，模板四角投到场景
};

bool siftAvailable();

// 检测 → 描述 → 比值测试 → RANSAC 单应。SIFT 不可用时 method 会说明原因。
MatchResult matchHomography(const cv::Mat& scene,
							const cv::Mat& model,
							Detector detector = Detector::ORB,
							double ratio = 0.75,
							double ransacReproj = 3.0);

}  // namespace feature
