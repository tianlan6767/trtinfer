#include "cv_caliper/metrology.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>

namespace metrology
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

cv::Point2d unitDir(cv::Point2d a, cv::Point2d b, bool* ok)
{
	cv::Point2d d(b.x - a.x, b.y - a.y);
	const double n = std::hypot(d.x, d.y);
	if (n < 1e-12) {
		if (ok) {
			*ok = false;
		}
		return {};
	}
	if (ok) {
		*ok = true;
	}
	d.x /= n;
	d.y /= n;
	return d;
}

}  // namespace

double distancePoints(cv::Point2d a, cv::Point2d b)
{
	return std::hypot(a.x - b.x, a.y - b.y);
}

double distancePointToLine(cv::Point2d p, cv::Point2d a, cv::Point2d b)
{
	bool ok = false;
	const cv::Point2d d = unitDir(a, b, &ok);
	if (!ok) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const cv::Point2d n(-d.y, d.x);
	return std::abs((p.x - a.x) * n.x + (p.y - a.y) * n.y);
}

double distanceBetweenLines(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2)
{
	const cv::Point2d midA(0.5 * (a1.x + a2.x), 0.5 * (a1.y + a2.y));
	const cv::Point2d midB(0.5 * (b1.x + b2.x), 0.5 * (b1.y + b2.y));
	const double d1 = distancePointToLine(midB, a1, a2);
	const double d2 = distancePointToLine(midA, b1, b2);
	if (!std::isfinite(d1) || !std::isfinite(d2)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	return 0.5 * (d1 + d2);
}

double angleBetweenLines(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2)
{
	bool okA = false, okB = false;
	const cv::Point2d da = unitDir(a1, a2, &okA);
	const cv::Point2d db = unitDir(b1, b2, &okB);
	if (!okA || !okB) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	double c = std::abs(da.x * db.x + da.y * db.y);
	c = std::clamp(c, 0.0, 1.0);
	return std::acos(c) * 180.0 / kPi;
}

bool intersectLines(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2, cv::Point2d& out)
{
	const cv::Point2d r(a2.x - a1.x, a2.y - a1.y);
	const cv::Point2d s(b2.x - b1.x, b2.y - b1.y);
	const double denom = r.x * s.y - r.y * s.x;
	if (std::abs(denom) < 1e-8) {
		return false;
	}
	const cv::Point2d qp(b1.x - a1.x, b1.y - a1.y);
	const double t = (qp.x * s.y - qp.y * s.x) / denom;
	out = cv::Point2d(a1.x + r.x * t, a1.y + r.y * t);
	return std::isfinite(out.x) && std::isfinite(out.y);
}

LineFit fitLine(const std::vector<cv::Point2d>& points)
{
	LineFit fit;
	fit.numPoints = static_cast<int>(points.size());
	if (points.size() < 2) {
		return fit;
	}
	cv::Point2d mean(0, 0);
	for (const auto& p : points) {
		mean += p;
	}
	mean.x /= static_cast<double>(points.size());
	mean.y /= static_cast<double>(points.size());

	cv::Mat A(static_cast<int>(points.size()), 2, CV_64F);
	for (int i = 0; i < A.rows; ++i) {
		A.at<double>(i, 0) = points[i].x - mean.x;
		A.at<double>(i, 1) = points[i].y - mean.y;
	}
	cv::Mat w, u, vt;
	cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A);
	cv::Point2d dir(vt.at<double>(0, 0), vt.at<double>(0, 1));
	const double nrm = std::hypot(dir.x, dir.y);
	if (nrm < 1e-12) {
		return fit;
	}
	dir.x /= nrm;
	dir.y /= nrm;
	if (dir.y < 0.0) {
		dir.x = -dir.x;
		dir.y = -dir.y;
	}
	const cv::Point2d normal(-dir.y, dir.x);
	double acc = 0.0;
	double tmin = 1e9, tmax = -1e9;
	for (const auto& p : points) {
		const double r = (p.x - mean.x) * normal.x + (p.y - mean.y) * normal.y;
		acc += r * r;
		const double t = (p.x - mean.x) * dir.x + (p.y - mean.y) * dir.y;
		tmin = std::min(tmin, t);
		tmax = std::max(tmax, t);
	}
	fit.found = true;
	fit.direction = dir;
	fit.center = mean;
	fit.p1 = cv::Point2d(mean.x + dir.x * tmin, mean.y + dir.y * tmin);
	fit.p2 = cv::Point2d(mean.x + dir.x * tmax, mean.y + dir.y * tmax);
	fit.length = tmax - tmin;
	fit.rms = std::sqrt(acc / static_cast<double>(points.size()));
	fit.angle = std::atan2(dir.y, dir.x) * 180.0 / kPi;
	return fit;
}

SizeSummary summarizeRectangle(cv::Point2d topLeft, cv::Point2d topRight,
							   cv::Point2d bottomRight, cv::Point2d bottomLeft)
{
	SizeSummary s;
	const double w1 = distancePoints(topLeft, topRight);
	const double w2 = distancePoints(bottomLeft, bottomRight);
	const double h1 = distancePoints(topLeft, bottomLeft);
	const double h2 = distancePoints(topRight, bottomRight);
	s.width = 0.5 * (w1 + w2);
	s.height = 0.5 * (h1 + h2);
	s.center = cv::Point2d(0.25 * (topLeft.x + topRight.x + bottomRight.x + bottomLeft.x),
						   0.25 * (topLeft.y + topRight.y + bottomRight.y + bottomLeft.y));
	s.angle = std::atan2(topRight.y - topLeft.y, topRight.x - topLeft.x) * 180.0 / kPi;
	s.found = std::isfinite(s.width) && std::isfinite(s.height) && s.width > 1.0 && s.height > 1.0;
	return s;
}

SizeSummary summarizeCircle(cv::Point2d center, double radius)
{
	SizeSummary s;
	s.center = center;
	s.diameter = 2.0 * radius;
	s.width = s.diameter;
	s.height = s.diameter;
	s.found = std::isfinite(radius) && radius > 0.0 && std::isfinite(center.x) && std::isfinite(center.y);
	return s;
}

}  // namespace metrology
