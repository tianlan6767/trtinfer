#include "cv_sim/similarity.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace sim
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

double ssimOf(const cv::Mat& a, const cv::Mat& b)
{
	cv::Mat f1, f2;
	a.convertTo(f1, CV_32F);
	b.convertTo(f2, CV_32F);
	const int k = 7;
	const double sigma = 1.5;
	cv::Mat mu1, mu2;
	cv::GaussianBlur(f1, mu1, cv::Size(k, k), sigma);
	cv::GaussianBlur(f2, mu2, cv::Size(k, k), sigma);
	cv::Mat mu1_2 = mu1.mul(mu1);
	cv::Mat mu2_2 = mu2.mul(mu2);
	cv::Mat mu12 = mu1.mul(mu2);
	cv::Mat s1, s2, s12;
	cv::GaussianBlur(f1.mul(f1), s1, cv::Size(k, k), sigma);
	cv::GaussianBlur(f2.mul(f2), s2, cv::Size(k, k), sigma);
	cv::GaussianBlur(f1.mul(f2), s12, cv::Size(k, k), sigma);
	s1 -= mu1_2;
	s2 -= mu2_2;
	s12 -= mu12;
	const double C1 = (0.01 * 255.0) * (0.01 * 255.0);
	const double C2 = (0.03 * 255.0) * (0.03 * 255.0);
	cv::Mat num = (2.0 * mu12 + C1).mul(2.0 * s12 + C2);
	cv::Mat den = (mu1_2 + mu2_2 + C1).mul(s1 + s2 + C2);
	cv::Mat map;
	cv::divide(num, den, map);
	return cv::mean(map)[0];
}

double histScore(const cv::Mat& a, const cv::Mat& b, int method)
{
	int histSize = 64;
	float range[] = {0.f, 256.f};
	const float* ranges[] = {range};
	cv::Mat ha, hb;
	cv::calcHist(&a, 1, 0, cv::Mat(), ha, 1, &histSize, ranges, true, false);
	cv::calcHist(&b, 1, 0, cv::Mat(), hb, 1, &histSize, ranges, true, false);
	cv::normalize(ha, ha, 1.0, 0.0, cv::NORM_L1);
	cv::normalize(hb, hb, 1.0, 0.0, cv::NORM_L1);
	return cv::compareHist(ha, hb, method);
}

LocateResult locate(const cv::Mat& image, const cv::Mat& templ, int method, double threshold)
{
	LocateResult out;
	cv::Mat scene = asGray(image);
	cv::Mat patch = asGray(templ);
	if (scene.empty() || patch.empty())
		return out;
	if (patch.rows > scene.rows || patch.cols > scene.cols)
		return out;
	cv::Mat score;
	cv::matchTemplate(scene, patch, score, method);
	double minV = 0, maxV = 0;
	cv::Point minP, maxP;
	cv::minMaxLoc(score, &minV, &maxV, &minP, &maxP);
	out.score = maxV;
	out.width = patch.cols;
	out.height = patch.rows;
	out.topLeft = cv::Point2d(maxP.x, maxP.y);
	out.center = cv::Point2d(maxP.x + (patch.cols - 1) / 2.0, maxP.y + (patch.rows - 1) / 2.0);
	out.found = std::isfinite(maxV) && maxV >= threshold;
	return out;
}

}  // namespace

SimilarityScore compare(const cv::Mat& a, const cv::Mat& b)
{
	SimilarityScore out;
	cv::Mat ga = asGray(a);
	cv::Mat gb = asGray(b);
	if (ga.empty() || gb.empty() || ga.size() != gb.size())
		return out;
	cv::Mat ncc, zncc;
	cv::matchTemplate(ga, gb, ncc, cv::TM_CCORR_NORMED);
	cv::matchTemplate(ga, gb, zncc, cv::TM_CCOEFF_NORMED);
	out.ncc = ncc.at<float>(0, 0);
	out.zncc = zncc.at<float>(0, 0);
	out.histCorr = histScore(ga, gb, cv::HISTCMP_CORREL);
	out.histBhattacharyya = histScore(ga, gb, cv::HISTCMP_BHATTACHARYYA);
	out.ssim = ssimOf(ga, gb);
	out.found = std::isfinite(out.ncc) && std::isfinite(out.zncc) && std::isfinite(out.ssim);
	return out;
}

LocateResult locateZncc(const cv::Mat& image, const cv::Mat& templ, double threshold)
{
	return locate(image, templ, cv::TM_CCOEFF_NORMED, threshold);
}

LocateResult locateNcc(const cv::Mat& image, const cv::Mat& templ, double threshold)
{
	return locate(image, templ, cv::TM_CCORR_NORMED, threshold);
}

}  // namespace sim
