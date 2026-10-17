#include "cv_feature/feature.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace feature
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

cv::Mat toNx2(const std::vector<cv::Point2f>& pts)
{
	cv::Mat m(static_cast<int>(pts.size()), 2, CV_64F);
	for (int i = 0; i < m.rows; ++i) {
		m.at<double>(i, 0) = pts[i].x;
		m.at<double>(i, 1) = pts[i].y;
	}
	return m;
}

}  // namespace

bool siftAvailable()
{
	try {
		// Algorithm::empty() 对未重写的检测器会返回 true，不能用来判断 SIFT 是否可用。
		cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
		return static_cast<bool>(sift);
	} catch (const cv::Exception&) {
		return false;
	}
}

MatchResult matchHomography(const cv::Mat& scene, const cv::Mat& model, Detector detector, double ratio, double ransacReproj)
{
	MatchResult out;
	cv::Mat sceneG = asGray(scene);
	cv::Mat modelG = asGray(model);
	if (sceneG.empty() || modelG.empty())
		return out;

	cv::Ptr<cv::Feature2D> algo;
	int norm = cv::NORM_HAMMING;
	if (detector == Detector::SIFT) {
		out.method = "SIFT";
		if (!siftAvailable()) {
			out.method = "SIFT-unavailable";
			return out;
		}
		algo = cv::SIFT::create(800);
		norm = cv::NORM_L2;
	} else {
		out.method = "ORB";
		algo = cv::ORB::create(800);
	}

	std::vector<cv::KeyPoint> kpM, kpS;
	cv::Mat descM, descS;
	algo->detectAndCompute(modelG, cv::Mat(), kpM, descM);
	algo->detectAndCompute(sceneG, cv::Mat(), kpS, descS);
	out.keypointsModel = static_cast<int>(kpM.size());
	out.keypointsScene = static_cast<int>(kpS.size());
	if (descM.empty() || descS.empty() || kpM.size() < 4 || kpS.size() < 4)
		return out;

	cv::BFMatcher matcher(norm, false);
	std::vector<std::vector<cv::DMatch>> knn;
	matcher.knnMatch(descM, descS, knn, 2);
	const double keep = std::clamp(ratio, 0.1, 0.99);
	std::vector<cv::Point2f> src, dst;
	src.reserve(knn.size());
	dst.reserve(knn.size());
	for (const auto& pair : knn) {
		if (pair.size() < 2)
			continue;
		if (pair[0].distance < keep * pair[1].distance) {
			src.push_back(kpM[pair[0].queryIdx].pt);
			dst.push_back(kpS[pair[0].trainIdx].pt);
		}
	}
	out.matches = static_cast<int>(src.size());
	out.modelPoints = toNx2(src);
	out.scenePoints = toNx2(dst);
	out.inlierMask.assign(src.size(), 0);
	if (src.size() < 4)
		return out;

	cv::Mat inlier;
	cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, std::max(0.5, ransacReproj), inlier);
	if (H.empty())
		return out;
	H.convertTo(out.homography, CV_64F);
	int nIn = 0;
	for (int i = 0; i < inlier.rows; ++i) {
		const unsigned char v = inlier.at<unsigned char>(i);
		out.inlierMask[i] = v;
		nIn += v ? 1 : 0;
	}
	out.inliers = nIn;
	std::vector<cv::Point2f> corners = {
		cv::Point2f(0.f, 0.f),
		cv::Point2f(static_cast<float>(modelG.cols - 1), 0.f),
		cv::Point2f(static_cast<float>(modelG.cols - 1), static_cast<float>(modelG.rows - 1)),
		cv::Point2f(0.f, static_cast<float>(modelG.rows - 1)),
	};
	std::vector<cv::Point2f> projected;
	cv::perspectiveTransform(corners, projected, out.homography);
	out.corners = toNx2(projected);
	out.found = nIn >= 6 && std::isfinite(out.homography.at<double>(0, 0));
	return out;
}

}  // namespace feature
