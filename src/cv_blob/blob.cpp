#include "blob.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace blob
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr int kHardCap = 100000;

int oddKernel(int k, int minOdd)
{
	if (k < minOdd)
		k = minOdd;
	if ((k % 2) == 0)
		++k;
	return k;
}

cv::Mat asGray8(const cv::Mat& image)
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
	if (gray.depth() != CV_8U)
	{
		cv::Mat u8;
		gray.convertTo(u8, CV_8U);
		return u8;
	}
	if (!gray.isContinuous())
		return gray.clone();
	return gray;
}

cv::Mat asBinary(const cv::Mat& image)
{
	cv::Mat gray = asGray8(image);
	if (gray.empty())
		return {};
	cv::Mat bin;
	cv::compare(gray, 0, bin, cv::CMP_GT);
	return bin;
}

cv::Rect clampedRect(const cv::Size& size, int x, int y, int w, int h, bool zeroMeansFull)
{
	if (size.width <= 0 || size.height <= 0)
		return {};
	if (zeroMeansFull && (w <= 0 || h <= 0))
		return cv::Rect(0, 0, size.width, size.height);
	if (w <= 0 || h <= 0)
		return {};
	return cv::Rect(x, y, w, h) & cv::Rect(0, 0, size.width, size.height);
}

double circularityOf(double area, double perimeter)
{
	if (!(area > 0.0) || !(perimeter > 1e-6))
		return 0.0;
	return 4.0 * kPi * area / (perimeter * perimeter);
}

double aspectOf(double a, double b)
{
	const double lo = std::min(std::fabs(a), std::fabs(b));
	const double hi = std::max(std::fabs(a), std::fabs(b));
	if (lo < 1.0)
		return hi >= 1.0 ? hi : 1.0;
	return hi / lo;
}

ContourStats statsFromContour(const std::vector<cv::Point>& contour, cv::Point origin)
{
	ContourStats s;
	if (contour.size() < 3)
		return s;
	s.area = std::fabs(cv::contourArea(contour));
	s.perimeter = cv::arcLength(contour, true);
	s.circularity = circularityOf(s.area, s.perimeter);
	const cv::RotatedRect rr = cv::minAreaRect(contour);
	s.angle = rr.angle;
	s.aspect = aspectOf(rr.size.width, rr.size.height);
	const cv::Rect box = cv::boundingRect(contour);
	s.left = box.x + origin.x;
	s.top = box.y + origin.y;
	s.width = box.width;
	s.height = box.height;
	cv::Moments m = cv::moments(contour);
	if (std::fabs(m.m00) > 1e-6)
		s.center = cv::Point2d(m.m10 / m.m00 + origin.x, m.m01 / m.m00 + origin.y);
	else
		s.center = cv::Point2d(rr.center.x + origin.x, rr.center.y + origin.y);
	return s;
}

}  // namespace

cv::Mat cropRoi(const cv::Mat& image, int x, int y, int width, int height)
{
	const cv::Rect r = clampedRect(image.size(), x, y, width, height, false);
	if (r.empty())
		return {};
	return image(r).clone();
}

cv::Mat gaussianBlur(const cv::Mat& image, int ksize, double sigma)
{
	if (image.empty())
		return {};
	int k = ksize;
	if (k < 0)
		k = 0;
	if (k > 0 && (k % 2) == 0)
		++k;
	if (k == 0 && !(sigma > 0.0))
		return image.clone();
	cv::Mat out;
	cv::GaussianBlur(image, out, cv::Size(k, k), sigma, sigma, cv::BORDER_REPLICATE);
	return out;
}

cv::Mat medianBlur(const cv::Mat& image, int ksize)
{
	if (image.empty())
		return {};
	if (image.depth() != CV_8U)
		return image.clone();
	if (ksize < 3)
		return image.clone();
	const int k = oddKernel(ksize, 3);
	cv::Mat out;
	cv::medianBlur(image, out, k);
	return out;
}

cv::Mat threshold(const cv::Mat& image, const ThresholdParam& param)
{
	cv::Mat gray = asGray8(image);
	if (gray.empty())
		return {};
	const double maxV = std::max(0.0, std::min(255.0, param.maxValue));
	int mode = param.invert ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
	if (param.type == ThresholdType::Otsu)
		mode |= cv::THRESH_OTSU;
	cv::Mat out;
	cv::threshold(gray, out, param.value, maxV, mode);
	return out;
}

cv::Mat morphology(const cv::Mat& image, const MorphParam& param)
{
	if (image.empty())
		return {};
	const int k = oddKernel(param.ksize, 1);
	const int iterations = std::max(1, param.iterations);
	const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
	int op = cv::MORPH_OPEN;
	switch (param.op)
	{
	case MorphOp::Erode:
		op = cv::MORPH_ERODE;
		break;
	case MorphOp::Dilate:
		op = cv::MORPH_DILATE;
		break;
	case MorphOp::Open:
		op = cv::MORPH_OPEN;
		break;
	case MorphOp::Close:
		op = cv::MORPH_CLOSE;
		break;
	}
	cv::Mat out;
	cv::morphologyEx(image, out, op, kernel, cv::Point(-1, -1), iterations);
	return out;
}

std::vector<ContourStats> contourStats(const cv::Mat& binary)
{
	std::vector<ContourStats> stats;
	cv::Mat bin = asBinary(binary);
	if (bin.empty())
		return stats;
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	stats.resize(contours.size());
#pragma omp parallel for schedule(dynamic) if (contours.size() >= 4)
	for (int i = 0; i < (int)contours.size(); ++i)
		stats[i] = statsFromContour(contours[i], cv::Point(0, 0));
	std::sort(stats.begin(), stats.end(), [](const ContourStats& a, const ContourStats& b) {
		return a.area > b.area;
	});
	return stats;
}

std::vector<Blob> findBlobs(const cv::Mat& binary, const BlobFilter& filter, int roiX, int roiY, int roiW, int roiH)
{
	std::vector<Blob> blobs;
	if (binary.empty())
		return blobs;
	const cv::Rect roi = clampedRect(binary.size(), roiX, roiY, roiW, roiH, true);
	if (roi.empty())
		return blobs;
	cv::Mat view = binary(roi).clone();
	cv::Mat bin = asBinary(view);
	if (bin.empty())
		return blobs;

	cv::Mat labels, stats, centroids;
	const int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
	if (n <= 1)
		return blobs;

	const cv::Point origin(roi.x, roi.y);
	std::vector<Blob> found(n);
	std::vector<char> ok(n, 0);

#pragma omp parallel for schedule(dynamic)
	for (int i = 1; i < n; ++i)
	{
		const int area = stats.at<int>(i, cv::CC_STAT_AREA);
		const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
		const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
		const int left = stats.at<int>(i, cv::CC_STAT_LEFT);
		const int top = stats.at<int>(i, cv::CC_STAT_TOP);
		if (area < filter.minArea || area > filter.maxArea)
			continue;
		const double aspect = aspectOf(w, h);
		if (aspect < filter.minAspect || aspect > filter.maxAspect)
			continue;

		cv::Mat mask(h, w, CV_8U);
		for (int r = 0; r < h; ++r)
		{
			const int* src = labels.ptr<int>(top + r) + left;
			uchar* dst = mask.ptr<uchar>(r);
			for (int c = 0; c < w; ++c)
				dst[c] = src[c] == i ? 255 : 0;
		}
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		double peri = 0.0;
		double angle = 0.0;
		if (!contours.empty())
		{
			for (const auto& c : contours)
				peri += cv::arcLength(c, true);
			const auto& best = *std::max_element(contours.begin(), contours.end(),
				[](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
					return std::fabs(cv::contourArea(a)) < std::fabs(cv::contourArea(b));
				});
			if (best.size() >= 3)
				angle = cv::minAreaRect(best).angle;
		}
		const double circ = circularityOf(area, peri);
		if (circ < filter.minCircularity || circ > filter.maxCircularity)
			continue;

		Blob b;
		b.label = i;
		b.area = area;
		b.perimeter = peri;
		b.circularity = circ;
		b.aspect = aspect;
		b.angle = angle;
		b.left = left + origin.x;
		b.top = top + origin.y;
		b.width = w;
		b.height = h;
		b.center = cv::Point2d(centroids.at<double>(i, 0) + origin.x, centroids.at<double>(i, 1) + origin.y);
		found[i] = b;
		ok[i] = 1;
	}

	for (int i = 1; i < n; ++i)
	{
		if (ok[i])
			blobs.push_back(found[i]);
	}
	std::sort(blobs.begin(), blobs.end(), [](const Blob& a, const Blob& b) { return a.area > b.area; });
	int limit = filter.maxCount;
	if (limit <= 0)
		limit = kHardCap;
	limit = std::min(limit, kHardCap);
	if ((int)blobs.size() > limit)
		blobs.resize(limit);
	return blobs;
}

}  // namespace blob
