#include "cv_seg/seg.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace seg
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

cv::Mat asBinary(const cv::Mat& image)
{
	cv::Mat gray = asGray(image);
	if (gray.empty())
		return {};
	cv::Mat bin;
	cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY);
	return bin;
}

cv::Mat toColor(const cv::Mat& image)
{
	cv::Mat gray = asGray(image);
	if (gray.empty())
		return {};
	cv::Mat bgr;
	cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
	return bgr;
}

cv::Mat pointsOf(const std::vector<cv::Point>& pts, const cv::Point& origin)
{
	cv::Mat m(static_cast<int>(pts.size()), 2, CV_64F);
	for (int i = 0; i < m.rows; ++i) {
		m.at<double>(i, 0) = pts[i].x + origin.x;
		m.at<double>(i, 1) = pts[i].y + origin.y;
	}
	return m;
}

std::vector<Region> regionsFromLabels(const cv::Mat& labels)
{
	std::vector<Region> regions;
	if (labels.empty() || labels.type() != CV_32S)
		return regions;
	double minV = 0, maxV = 0;
	cv::minMaxLoc(labels, &minV, &maxV);
	const int maxLabel = static_cast<int>(maxV);
	if (maxLabel < 1)
		return regions;
	std::vector<double> sumX(maxLabel + 1, 0), sumY(maxLabel + 1, 0);
	std::vector<int> area(maxLabel + 1, 0);
	std::vector<int> left(maxLabel + 1, labels.cols);
	std::vector<int> top(maxLabel + 1, labels.rows);
	std::vector<int> right(maxLabel + 1, -1);
	std::vector<int> bottom(maxLabel + 1, -1);
	for (int y = 0; y < labels.rows; ++y) {
		const int* row = labels.ptr<int>(y);
		for (int x = 0; x < labels.cols; ++x) {
			const int id = row[x];
			if (id <= 1)
				continue;
			area[id] += 1;
			sumX[id] += x;
			sumY[id] += y;
			left[id] = std::min(left[id], x);
			top[id] = std::min(top[id], y);
			right[id] = std::max(right[id], x);
			bottom[id] = std::max(bottom[id], y);
		}
	}
	for (int id = 2; id <= maxLabel; ++id) {
		if (area[id] <= 0)
			continue;
		Region r;
		r.label = id;
		r.area = area[id];
		r.center = cv::Point2d(sumX[id] / area[id], sumY[id] / area[id]);
		r.left = left[id];
		r.top = top[id];
		r.width = right[id] - left[id] + 1;
		r.height = bottom[id] - top[id] + 1;
		regions.push_back(r);
	}
	std::sort(regions.begin(), regions.end(), [](const Region& a, const Region& b) {
		return a.area > b.area;
	});
	return regions;
}

}  // namespace

cv::Mat distanceTransform(const cv::Mat& binary)
{
	cv::Mat bin = asBinary(binary);
	if (bin.empty())
		return {};
	cv::Mat dist;
	cv::distanceTransform(bin, dist, cv::DIST_L2, 5);
	return dist;
}

cv::Mat canny(const cv::Mat& image, double low, double high, int aperture)
{
	cv::Mat gray = asGray(image);
	if (gray.empty())
		return {};
	if (aperture != 3 && aperture != 5 && aperture != 7)
		aperture = 3;
	cv::Mat edges;
	cv::Canny(gray, edges, low, high, aperture, true);
	return edges;
}

std::vector<Contour> extractContours(const cv::Mat& binary, bool externalOnly)
{
	std::vector<Contour> out;
	cv::Mat bin = asBinary(binary);
	if (bin.empty())
		return out;
	std::vector<std::vector<cv::Point>> contours;
	const int mode = externalOnly ? cv::RETR_EXTERNAL : cv::RETR_LIST;
	cv::findContours(bin, contours, mode, cv::CHAIN_APPROX_NONE);
	out.reserve(contours.size());
	for (const auto& c : contours) {
		if (c.size() < 3)
			continue;
		Contour item;
		item.area = std::fabs(cv::contourArea(c));
		item.perimeter = cv::arcLength(c, true);
		cv::Moments m = cv::moments(c);
		if (std::fabs(m.m00) > 1e-6)
			item.center = cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
		cv::Rect box = cv::boundingRect(c);
		item.left = box.x;
		item.top = box.y;
		item.width = box.width;
		item.height = box.height;
		item.points = pointsOf(c, cv::Point(0, 0));
		out.push_back(item);
	}
	std::sort(out.begin(), out.end(), [](const Contour& a, const Contour& b) {
		return a.area > b.area;
	});
	return out;
}

WatershedResult watershed(const cv::Mat& image, const cv::Mat& markers)
{
	WatershedResult out;
	cv::Mat color = toColor(image);
	if (color.empty() || markers.empty() || markers.type() != CV_32S)
		return out;
	if (markers.size() != color.size())
		return out;
	out.labels = markers.clone();
	cv::watershed(color, out.labels);
	out.regions = regionsFromLabels(out.labels);
	out.found = !out.regions.empty();
	return out;
}

WatershedResult splitBlobs(const cv::Mat& binary, double minDistance, double distRatio)
{
	WatershedResult out;
	cv::Mat bin = asBinary(binary);
	if (bin.empty())
		return out;
	cv::Mat dist = distanceTransform(bin);
	double minV = 0, maxV = 0;
	cv::minMaxLoc(dist, &minV, &maxV);
	if (maxV <= 1.0)
		return out;
	const double ratio = std::clamp(distRatio, 0.05, 0.95);
	cv::Mat peaks;
	cv::threshold(dist, peaks, ratio * maxV, 255, cv::THRESH_BINARY);
	peaks.convertTo(peaks, CV_8U);
	const int rad = std::max(1, static_cast<int>(std::round(minDistance)));
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(rad * 2 + 1, rad * 2 + 1));
	cv::Mat dilated;
	cv::dilate(dist, dilated, kernel);
	cv::Mat localMax;
	cv::compare(dist, dilated, localMax, cv::CMP_GE);
	cv::bitwise_and(localMax, peaks, peaks);

	cv::Mat cc;
	const int n = cv::connectedComponents(peaks, cc);
	if (n <= 1)
		return out;
	cv::Mat markers = cc + 1;
	cv::Mat unknown;
	cv::Mat sureBg;
	cv::dilate(bin, sureBg, kernel);
	cv::subtract(sureBg, peaks, unknown);
	markers.setTo(0, unknown);

	cv::Mat color = toColor(bin);
	out.labels = markers;
	cv::watershed(color, out.labels);
	out.regions = regionsFromLabels(out.labels);
	out.found = !out.regions.empty();
	return out;
}

}  // namespace seg
