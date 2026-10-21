#include "cv_measure/measure.h"

#include "cv_calib/calib.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <random>

namespace measure
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

bool finitePoint(cv::Point2d p)
{
	return std::isfinite(p.x) && std::isfinite(p.y);
}

double degNorm180(double deg)
{
	while (deg <= -90.0) {
		deg += 180.0;
	}
	while (deg > 90.0) {
		deg -= 180.0;
	}
	return deg;
}

FormResult finishForm(bool found, double value, double tol, double rms)
{
	FormResult r;
	r.found = found && std::isfinite(value) && std::isfinite(tol);
	r.value = r.found ? value : 0.0;
	r.tolerance = tol;
	r.rms = rms;
	r.margin = r.found ? (tol - value) : 0.0;
	r.pass = r.found && r.margin >= -1e-9;
	return r;
}

int ransacNeed(int n, int model)
{
	if (n < model) {
		return n + 1;
	}
	if (n < model + 2) {
		return model;
	}
	return std::max(model, static_cast<int>(std::lround(0.3 * n)));
}

cv::Mat asGray(const cv::Mat& image)
{
	if (image.empty()) {
		return {};
	}
	cv::Mat gray;
	if (image.channels() == 1) {
		gray = image;
	} else if (image.channels() == 3) {
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	} else if (image.channels() == 4) {
		cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
	} else {
		return {};
	}
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
	if (gray.empty()) {
		return {};
	}
	cv::Mat mid;
	cv::inRange(gray, 1, 254, mid);
	if (cv::countNonZero(mid) == 0) {
		return gray;
	}
	cv::Mat bin;
	cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	return bin;
}

bool fitCircleAlgebraic(const std::vector<cv::Point2d>& pts, cv::Point2d& c, double& r, double& rms)
{
	if (pts.size() < 3) {
		return false;
	}
	cv::Mat A(static_cast<int>(pts.size()), 3, CV_64F);
	cv::Mat b(static_cast<int>(pts.size()), 1, CV_64F);
	for (int i = 0; i < A.rows; ++i) {
		const double x = pts[i].x;
		const double y = pts[i].y;
		A.at<double>(i, 0) = x;
		A.at<double>(i, 1) = y;
		A.at<double>(i, 2) = 1.0;
		b.at<double>(i, 0) = -(x * x + y * y);
	}
	cv::Mat sol;
	if (!cv::solve(A, b, sol, cv::DECOMP_SVD)) {
		return false;
	}
	c.x = -0.5 * sol.at<double>(0, 0);
	c.y = -0.5 * sol.at<double>(1, 0);
	const double f = sol.at<double>(2, 0);
	const double r2 = c.x * c.x + c.y * c.y - f;
	if (!(r2 > 1e-8) || !std::isfinite(r2)) {
		return false;
	}
	r = std::sqrt(r2);
	double acc = 0.0;
	for (const auto& p : pts) {
		const double e = std::hypot(p.x - c.x, p.y - c.y) - r;
		acc += e * e;
	}
	rms = std::sqrt(acc / static_cast<double>(pts.size()));
	return std::isfinite(r) && r > 0.0 && std::isfinite(rms);
}

bool refineCircle(const std::vector<cv::Point2d>& pts, cv::Point2d& c, double& r, double& rms)
{
	const cv::Point2d c0 = c;
	const double r0 = r;
	const double rms0 = rms;
	for (int it = 0; it < 6; ++it) {
		cv::Mat A = cv::Mat::zeros(3, 3, CV_64F);
		cv::Mat b = cv::Mat::zeros(3, 1, CV_64F);
		for (const auto& p : pts) {
			const double dx = p.x - c.x;
			const double dy = p.y - c.y;
			const double ri = std::hypot(dx, dy);
			if (ri < 1e-8) {
				continue;
			}
			const double J[3] = {-dx / ri, -dy / ri, -1.0};
			const double resid = ri - r;
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					A.at<double>(i, j) += J[i] * J[j];
				}
				b.at<double>(i, 0) += J[i] * resid;
			}
		}
		cv::Mat delta;
		if (!cv::solve(A, b, delta, cv::DECOMP_SVD)) {
			break;
		}
		c.x -= delta.at<double>(0, 0);
		c.y -= delta.at<double>(1, 0);
		r -= delta.at<double>(2, 0);
		if (!(r > 0.2) || !std::isfinite(r) || !finitePoint(c)) {
			c = c0;
			r = r0;
			rms = rms0;
			return false;
		}
	}
	double acc = 0.0;
	for (const auto& p : pts) {
		const double e = std::hypot(p.x - c.x, p.y - c.y) - r;
		acc += e * e;
	}
	rms = std::sqrt(acc / static_cast<double>(std::max<size_t>(1, pts.size())));
	if (!std::isfinite(rms) || rms > rms0 + 1e-3) {
		c = c0;
		r = r0;
		rms = rms0;
		return false;
	}
	return true;
}

double ellipseDistance(cv::Point2d p, const EllipseFit& e)
{
	if (!(e.a > 1e-6) || !(e.b > 1e-6)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const double rad = e.angle * kPi / 180.0;
	const double ca = std::cos(rad);
	const double sa = std::sin(rad);
	const double dx = p.x - e.center.x;
	const double dy = p.y - e.center.y;
	const double x = dx * ca + dy * sa;
	const double y = -dx * sa + dy * ca;
	const double a2 = e.a * e.a;
	const double b2 = e.b * e.b;
	const double f = (x * x) / a2 + (y * y) / b2 - 1.0;
	const double g = std::hypot(2.0 * x / a2, 2.0 * y / b2);
	if (g < 1e-12) {
		return std::min(e.a, e.b);
	}
	return std::abs(f) / g;
}

bool ellipseFromPoints(const std::vector<cv::Point2d>& pts, EllipseFit& out)
{
	if (pts.size() < 5) {
		return false;
	}
	std::vector<cv::Point2f> pf;
	pf.reserve(pts.size());
	for (const auto& p : pts) {
		pf.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
	}
	cv::RotatedRect rr;
	try {
		rr = cv::fitEllipse(pf);
	} catch (const cv::Exception&) {
		return false;
	}
	if (!std::isfinite(rr.size.width) || !std::isfinite(rr.size.height)) {
		return false;
	}
	double w = rr.size.width;
	double h = rr.size.height;
	double ang = rr.angle;
	if (h > w) {
		std::swap(w, h);
		ang += 90.0;
	}
	out.center = cv::Point2d(rr.center.x, rr.center.y);
	out.a = 0.5 * w;
	out.b = 0.5 * h;
	out.angle = degNorm180(ang);
	if (!(out.a > 0.5) || !(out.b > 0.2) || !finitePoint(out.center)) {
		return false;
	}
	double acc = 0.0;
	int n = 0;
	for (const auto& p : pts) {
		const double d = ellipseDistance(p, out);
		if (!std::isfinite(d)) {
			continue;
		}
		acc += d * d;
		++n;
	}
	if (n < 5) {
		return false;
	}
	out.rms = std::sqrt(acc / static_cast<double>(n));
	return std::isfinite(out.rms);
}

std::vector<int> sampleIdx(std::mt19937& rng, int n, int k)
{
	std::uniform_int_distribution<int> dist(0, n - 1);
	std::vector<int> idx;
	int guard = 0;
	while (static_cast<int>(idx.size()) < k && guard < 40) {
		++guard;
		const int v = dist(rng);
		if (std::find(idx.begin(), idx.end(), v) == idx.end()) {
			idx.push_back(v);
		}
	}
	return idx;
}

void assignBox(const std::vector<cv::Point2d>& corners, BoxSize& box)
{
	if (corners.size() != 4) {
		return;
	}
	cv::Point2d center(0, 0);
	for (const auto& p : corners) {
		center += p;
	}
	center.x *= 0.25;
	center.y *= 0.25;
	std::vector<cv::Point2d> ordered = corners;
	std::sort(ordered.begin(), ordered.end(), [&](cv::Point2d a, cv::Point2d b) {
		return std::atan2(a.y - center.y, a.x - center.x) < std::atan2(b.y - center.y, b.x - center.x);
	});
	int best = 0;
	double bestLen = -1.0;
	for (int i = 0; i < 4; ++i) {
		const double L = metrology::distancePoints(ordered[i], ordered[(i + 1) % 4]);
		if (L > bestLen) {
			bestLen = L;
			best = i;
		}
	}
	cv::Point2d cyc[4];
	for (int i = 0; i < 4; ++i) {
		cyc[i] = ordered[(best + i) % 4];
	}
	box.c0 = cyc[0];
	box.c1 = cyc[1];
	box.c2 = cyc[2];
	box.c3 = cyc[3];
	box.width = metrology::distancePoints(box.c0, box.c1);
	box.height = metrology::distancePoints(box.c1, box.c2);
	box.angle = degNorm180(std::atan2(box.c1.y - box.c0.y, box.c1.x - box.c0.x) * 180.0 / kPi);
	box.center = center;
	box.found = box.width > 1e-6 && box.height > 1e-6 && finitePoint(center);
}

double sideDistance(cv::Point2d p, cv::Point2d a, cv::Point2d b)
{
	return metrology::distancePointToLine(p, a, b);
}

PixelScale makeScale(double sx, double sy, const std::string& source)
{
	PixelScale s;
	if (!(sx > 0.0) || !(sy > 0.0) || !std::isfinite(sx) || !std::isfinite(sy)) {
		return s;
	}
	s.found = true;
	s.mmPerPxX = sx;
	s.mmPerPxY = sy;
	s.mmPerPx = 0.5 * (sx + sy);
	s.source = source;
	return s;
}

cv::Point2d planeOf(cv::Point2d px, const PixelScale& scale)
{
	if (!scale.found) {
		return px;
	}
	return cv::Point2d(px.x * scale.mmPerPxX, px.y * scale.mmPerPxY);
}

}  // namespace

metrology::LineFit fitLine(const std::vector<cv::Point2d>& points, const FitParam& param)
{
	if (param.method != FitMethod::Ransac || points.size() < 4) {
		return metrology::fitLine(points);
	}
	const int n = static_cast<int>(points.size());
	const int need = ransacNeed(n, 2);
	std::mt19937 rng(param.seed);
	std::vector<cv::Point2d> best;
	double bestRms = 1e300;
	const int iters = std::clamp(param.iterations, 1, 2000);
	for (int it = 0; it < iters; ++it) {
		const auto idx = sampleIdx(rng, n, 2);
		if (idx.size() < 2) {
			break;
		}
		std::vector<cv::Point2d> in;
		for (const auto& p : points) {
			if (metrology::distancePointToLine(p, points[idx[0]], points[idx[1]]) <= param.threshold) {
				in.push_back(p);
			}
		}
		if (static_cast<int>(in.size()) < need) {
			continue;
		}
		const auto trial = metrology::fitLine(in);
		if (!trial.found) {
			continue;
		}
		if (in.size() > best.size() || (in.size() == best.size() && trial.rms < bestRms)) {
			best = std::move(in);
			bestRms = trial.rms;
		}
	}
	metrology::LineFit fit;
	fit.numPoints = n;
	if (static_cast<int>(best.size()) < need) {
		return fit;
	}
	fit = metrology::fitLine(best);
	fit.numPoints = n;
	fit.numInliers = static_cast<int>(best.size());
	fit.inliers = best;
	return fit;
}

CircleFit fitCircle(const std::vector<cv::Point2d>& points, const FitParam& param)
{
	CircleFit out;
	out.numPoints = static_cast<int>(points.size());
	if (points.size() < 3) {
		return out;
	}
	auto pack = [&](const std::vector<cv::Point2d>& in, bool subset) {
		cv::Point2d c;
		double r = 0, rms = 0;
		if (!fitCircleAlgebraic(in, c, r, rms)) {
			return;
		}
		refineCircle(in, c, r, rms);
		out.found = true;
		out.center = c;
		out.radius = r;
		out.rms = rms;
		out.inliers = subset ? in : points;
		out.numInliers = static_cast<int>(out.inliers.size());
	};
	if (param.method != FitMethod::Ransac || points.size() < 6) {
		pack(points, false);
		return out;
	}
	const int n = static_cast<int>(points.size());
	const int need = ransacNeed(n, 3);
	std::mt19937 rng(param.seed);
	std::vector<cv::Point2d> best;
	double bestRms = 1e300;
	const int iters = std::clamp(param.iterations, 1, 2000);
	for (int it = 0; it < iters; ++it) {
		const auto idx = sampleIdx(rng, n, 3);
		if (idx.size() < 3) {
			break;
		}
		std::vector<cv::Point2d> sample = {points[idx[0]], points[idx[1]], points[idx[2]]};
		cv::Point2d c;
		double r = 0, rms = 0;
		if (!fitCircleAlgebraic(sample, c, r, rms)) {
			continue;
		}
		std::vector<cv::Point2d> in;
		for (const auto& p : points) {
			if (std::abs(std::hypot(p.x - c.x, p.y - c.y) - r) <= param.threshold) {
				in.push_back(p);
			}
		}
		if (static_cast<int>(in.size()) < need) {
			continue;
		}
		if (!fitCircleAlgebraic(in, c, r, rms)) {
			continue;
		}
		if (in.size() > best.size() || (in.size() == best.size() && rms < bestRms)) {
			best = std::move(in);
			bestRms = rms;
		}
	}
	if (static_cast<int>(best.size()) < need) {
		return out;
	}
	pack(best, true);
	out.numPoints = n;
	return out;
}

EllipseFit fitEllipse(const std::vector<cv::Point2d>& points, const FitParam& param)
{
	EllipseFit out;
	out.numPoints = static_cast<int>(points.size());
	if (points.size() < 5) {
		return out;
	}
	if (param.method != FitMethod::Ransac || points.size() < 8) {
		if (!ellipseFromPoints(points, out)) {
			return EllipseFit{};
		}
		out.numPoints = static_cast<int>(points.size());
		out.numInliers = out.numPoints;
		out.inliers = points;
		out.found = true;
		return out;
	}
	const int n = static_cast<int>(points.size());
	const int need = std::max(5, ransacNeed(n, 5));
	std::mt19937 rng(param.seed);
	std::vector<cv::Point2d> best;
	double bestRms = 1e300;
	const int iters = std::clamp(param.iterations, 1, 2000);
	for (int it = 0; it < iters; ++it) {
		const auto idx = sampleIdx(rng, n, 5);
		if (idx.size() < 5) {
			break;
		}
		std::vector<cv::Point2d> sample;
		for (int id : idx) {
			sample.push_back(points[id]);
		}
		EllipseFit trial;
		if (!ellipseFromPoints(sample, trial)) {
			continue;
		}
		std::vector<cv::Point2d> in;
		for (const auto& p : points) {
			const double d = ellipseDistance(p, trial);
			if (std::isfinite(d) && d <= param.threshold) {
				in.push_back(p);
			}
		}
		if (static_cast<int>(in.size()) < need || in.size() < 5) {
			continue;
		}
		if (!ellipseFromPoints(in, trial)) {
			continue;
		}
		if (in.size() > best.size() || (in.size() == best.size() && trial.rms < bestRms)) {
			best = std::move(in);
			bestRms = trial.rms;
		}
	}
	if (static_cast<int>(best.size()) < need) {
		return out;
	}
	if (!ellipseFromPoints(best, out)) {
		return EllipseFit{};
	}
	out.found = true;
	out.numPoints = n;
	out.numInliers = static_cast<int>(best.size());
	out.inliers = best;
	return out;
}

BoxSize axisAlignedBox(const std::vector<cv::Point2d>& points)
{
	BoxSize box;
	if (points.empty()) {
		return box;
	}
	double minx = points[0].x, maxx = points[0].x, miny = points[0].y, maxy = points[0].y;
	for (const auto& p : points) {
		minx = std::min(minx, p.x);
		maxx = std::max(maxx, p.x);
		miny = std::min(miny, p.y);
		maxy = std::max(maxy, p.y);
	}
	box.c0 = cv::Point2d(minx, miny);
	box.c1 = cv::Point2d(maxx, miny);
	box.c2 = cv::Point2d(maxx, maxy);
	box.c3 = cv::Point2d(minx, maxy);
	box.width = maxx - minx;
	box.height = maxy - miny;
	box.angle = 0.0;
	box.center = cv::Point2d(0.5 * (minx + maxx), 0.5 * (miny + maxy));
	box.found = std::isfinite(box.width) && std::isfinite(box.height) && box.width > 0.0 && box.height > 0.0;
	return box;
}

BoxSize minAreaBox(const std::vector<cv::Point2d>& points)
{
	BoxSize box;
	if (points.size() < 3) {
		return box;
	}
	std::vector<cv::Point2f> pf;
	pf.reserve(points.size());
	for (const auto& p : points) {
		pf.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
	}
	const cv::RotatedRect rr = cv::minAreaRect(pf);
	cv::Point2f raw[4];
	rr.points(raw);
	std::vector<cv::Point2d> corners;
	for (int i = 0; i < 4; ++i) {
		corners.emplace_back(raw[i].x, raw[i].y);
	}
	assignBox(corners, box);
	return box;
}

QuadFit fitParallelogram(const std::vector<cv::Point2d>& points)
{
	QuadFit q;
	if (points.size() < 4) {
		return q;
	}
	std::vector<cv::Point2f> pf;
	pf.reserve(points.size());
	for (const auto& p : points) {
		pf.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
	}
	std::vector<cv::Point2f> hull;
	cv::convexHull(pf, hull, false);
	const int n = static_cast<int>(hull.size());
	if (n < 4) {
		return q;
	}
	std::vector<double> turn(n, 0.0);
	for (int i = 0; i < n; ++i) {
		const cv::Point2f p0 = hull[(i - 1 + n) % n];
		const cv::Point2f p1 = hull[i];
		const cv::Point2f p2 = hull[(i + 1) % n];
		const cv::Point2d v1(p1.x - p0.x, p1.y - p0.y);
		const cv::Point2d v2(p2.x - p1.x, p2.y - p1.y);
		const double cross = v1.x * v2.y - v1.y * v2.x;
		const double dot = v1.x * v2.x + v1.y * v2.y;
		turn[i] = std::abs(std::atan2(cross, dot));
	}
	std::vector<int> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](int a, int b) { return turn[a] > turn[b]; });
	std::vector<int> corners(order.begin(), order.begin() + 4);
	std::sort(corners.begin(), corners.end());

	auto sideOf = [&](int i0, int i1) {
		std::vector<cv::Point2d> side;
		for (int i = i0, guard = 0; guard <= n; i = (i + 1) % n, ++guard) {
			side.emplace_back(hull[i].x, hull[i].y);
			if (i == i1) {
				break;
			}
		}
		return side;
	};

	cv::Point2d lp1[4], lp2[4];
	for (int k = 0; k < 4; ++k) {
		const auto side = sideOf(corners[k], corners[(k + 1) % 4]);
		const auto fit = metrology::fitLine(side);
		if (!fit.found) {
			return q;
		}
		lp1[k] = fit.p1;
		lp2[k] = fit.p2;
	}
	cv::Point2d cs[4];
	for (int k = 0; k < 4; ++k) {
		const int prev = (k + 3) % 4;
		if (!metrology::intersectLines(lp1[prev], lp2[prev], lp1[k], lp2[k], cs[k])) {
			return q;
		}
	}
	BoxSize box;
	assignBox({cs[0], cs[1], cs[2], cs[3]}, box);
	if (!box.found) {
		return q;
	}
	q.found = true;
	q.center = box.center;
	q.width = box.width;
	q.height = box.height;
	q.angle = box.angle;
	q.c0 = box.c0;
	q.c1 = box.c1;
	q.c2 = box.c2;
	q.c3 = box.c3;

	const cv::Point2d corner[4] = {q.c0, q.c1, q.c2, q.c3};
	double maxErr = 0.0;
	for (int k = 0; k < 4; ++k) {
		const cv::Point2d prev = corner[(k + 3) % 4];
		const cv::Point2d cur = corner[k];
		const cv::Point2d next = corner[(k + 1) % 4];
		const cv::Point2d v1(prev.x - cur.x, prev.y - cur.y);
		const cv::Point2d v2(next.x - cur.x, next.y - cur.y);
		const double cross = v1.x * v2.y - v1.y * v2.x;
		const double dot = v1.x * v2.x + v1.y * v2.y;
		const double interior = std::atan2(std::abs(cross), dot) * 180.0 / kPi;
		maxErr = std::max(maxErr, std::abs(interior - 90.0));
	}
	q.angleError = maxErr;
	const double par02 = metrology::angleBetweenLines(q.c0, q.c1, q.c2, q.c3);
	const double par13 = metrology::angleBetweenLines(q.c1, q.c2, q.c3, q.c0);
	q.rectangle = std::isfinite(par02) && std::isfinite(par13) && par02 <= 5.0 && par13 <= 5.0 && maxErr <= 5.0;

	double acc = 0.0;
	for (const auto& p : points) {
		const double d0 = sideDistance(p, q.c0, q.c1);
		const double d1 = sideDistance(p, q.c1, q.c2);
		const double d2 = sideDistance(p, q.c2, q.c3);
		const double d3 = sideDistance(p, q.c3, q.c0);
		double d = d0;
		d = std::min(d, d1);
		d = std::min(d, d2);
		d = std::min(d, d3);
		if (std::isfinite(d)) {
			acc += d * d;
		}
	}
	q.rms = std::sqrt(acc / static_cast<double>(points.size()));
	return q;
}

std::vector<cv::Point2d> contourPoints(const cv::Mat& image)
{
	std::vector<cv::Point2d> out;
	const cv::Mat bin = asBinary(image);
	if (bin.empty()) {
		return out;
	}
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	if (contours.empty()) {
		return out;
	}
	size_t best = 0;
	double bestArea = -1.0;
	for (size_t i = 0; i < contours.size(); ++i) {
		const double area = std::abs(cv::contourArea(contours[i]));
		if (area > bestArea) {
			bestArea = area;
			best = i;
		}
	}
	out.reserve(contours[best].size());
	for (const auto& p : contours[best]) {
		out.emplace_back(p.x, p.y);
	}
	return out;
}

BoxSize minAreaBoxContour(const cv::Mat& image)
{
	return minAreaBox(contourPoints(image));
}

QuadFit fitParallelogramContour(const cv::Mat& image)
{
	return fitParallelogram(contourPoints(image));
}

FormResult checkParallelism(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2, double tolDeg)
{
	const double ang = metrology::angleBetweenLines(a1, a2, b1, b2);
	return finishForm(std::isfinite(ang), ang, tolDeg, ang);
}

FormResult checkPerpendicularity(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2, double tolDeg)
{
	const double ang = metrology::angleBetweenLines(a1, a2, b1, b2);
	if (!std::isfinite(ang)) {
		return finishForm(false, 0, tolDeg, 0);
	}
	const double value = 90.0 - ang;
	return finishForm(true, value, tolDeg, value);
}

FormResult checkStraightness(const std::vector<cv::Point2d>& points, double tolPx)
{
	const auto fit = metrology::fitLine(points);
	if (!fit.found || points.size() < 3) {
		return finishForm(false, 0, tolPx, 0);
	}
	const cv::Point2d normal(-fit.direction.y, fit.direction.x);
	double lo = 1e300, hi = -1e300;
	for (const auto& p : points) {
		const double r = (p.x - fit.center.x) * normal.x + (p.y - fit.center.y) * normal.y;
		lo = std::min(lo, r);
		hi = std::max(hi, r);
	}
	return finishForm(true, hi - lo, tolPx, fit.rms);
}

FormResult checkRoundness(const std::vector<cv::Point2d>& points, double tolPx)
{
	if (points.size() < 6) {
		return finishForm(false, 0, tolPx, 0);
	}
	const CircleFit fit = fitCircle(points, FitParam());
	if (!fit.found) {
		return finishForm(false, 0, tolPx, 0);
	}
	double lo = 1e300, hi = -1e300;
	for (const auto& p : points) {
		const double ri = std::hypot(p.x - fit.center.x, p.y - fit.center.y);
		lo = std::min(lo, ri);
		hi = std::max(hi, ri);
	}
	return finishForm(true, hi - lo, tolPx, fit.rms);
}

CircleSize inscribedCircle(const std::vector<cv::Point2d>& points)
{
	CircleSize out;
	if (points.size() < 3) {
		return out;
	}
	std::vector<cv::Point2f> pf;
	for (const auto& p : points) {
		pf.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
	}
	std::vector<cv::Point2f> hull;
	cv::convexHull(pf, hull, false);
	if (hull.size() < 3) {
		return out;
	}
	float minx = hull[0].x, maxx = hull[0].x, miny = hull[0].y, maxy = hull[0].y;
	for (const auto& p : hull) {
		minx = std::min(minx, p.x);
		maxx = std::max(maxx, p.x);
		miny = std::min(miny, p.y);
		maxy = std::max(maxy, p.y);
	}
	const int x0 = static_cast<int>(std::floor(minx)) - 2;
	const int y0 = static_cast<int>(std::floor(miny)) - 2;
	const int x1 = static_cast<int>(std::ceil(maxx)) + 2;
	const int y1 = static_cast<int>(std::ceil(maxy)) + 2;
	const int w = x1 - x0 + 1;
	const int h = y1 - y0 + 1;
	if (w < 3 || h < 3 || w > 8000 || h > 8000) {
		return out;
	}
	cv::Mat mask(h, w, CV_8U, cv::Scalar(0));
	std::vector<cv::Point> poly;
	poly.reserve(hull.size());
	for (const auto& p : hull) {
		poly.emplace_back(cvRound(p.x - x0), cvRound(p.y - y0));
	}
	cv::fillConvexPoly(mask, poly, cv::Scalar(255));
	cv::Mat dist;
	cv::distanceTransform(mask, dist, cv::DIST_L2, cv::DIST_MASK_5);
	double maxVal = 0;
	cv::Point maxLoc;
	cv::minMaxLoc(dist, nullptr, &maxVal, nullptr, &maxLoc);
	if (!(maxVal >= 0.5)) {
		return out;
	}
	out.found = true;
	out.radius = maxVal;
	out.diameter = 2.0 * maxVal;
	out.center = cv::Point2d(maxLoc.x + x0, maxLoc.y + y0);
	return out;
}

CircleSize inscribedCircleMask(const cv::Mat& image)
{
	CircleSize out;
	const cv::Mat bin = asBinary(image);
	if (bin.empty()) {
		return out;
	}
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	if (contours.empty()) {
		return out;
	}
	size_t best = 0;
	double bestArea = -1.0;
	for (size_t i = 0; i < contours.size(); ++i) {
		const double area = std::abs(cv::contourArea(contours[i]));
		if (area > bestArea) {
			bestArea = area;
			best = i;
		}
	}
	if (contours[best].size() < 3 || bestArea < 4.0) {
		return out;
	}
	cv::Mat mask = cv::Mat::zeros(bin.size(), CV_8U);
	cv::drawContours(mask, contours, static_cast<int>(best), cv::Scalar(255), cv::FILLED);
	cv::Mat dist;
	cv::distanceTransform(mask, dist, cv::DIST_L2, cv::DIST_MASK_5);
	double maxVal = 0;
	cv::Point maxLoc;
	cv::minMaxLoc(dist, nullptr, &maxVal, nullptr, &maxLoc);
	if (!(maxVal >= 0.5)) {
		return out;
	}
	out.found = true;
	out.radius = maxVal;
	out.diameter = 2.0 * maxVal;
	out.center = cv::Point2d(maxLoc.x, maxLoc.y);
	return out;
}

CircleSize circumscribedCircle(const std::vector<cv::Point2d>& points)
{
	CircleSize out;
	if (points.size() < 2) {
		return out;
	}
	std::vector<cv::Point2f> pf;
	pf.reserve(points.size());
	for (const auto& p : points) {
		pf.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
	}
	cv::Point2f c;
	float r = 0.f;
	cv::minEnclosingCircle(pf, c, r);
	if (!(r > 0.f) || !std::isfinite(r)) {
		return out;
	}
	out.found = true;
	out.center = cv::Point2d(c.x, c.y);
	out.radius = r;
	out.diameter = 2.0 * r;
	return out;
}

WidthFunction widthFunction(const std::vector<cv::Point2d>& points, double stepDeg)
{
	WidthFunction out;
	if (points.size() < 2) {
		return out;
	}
	const double step = stepDeg > 0.05 ? stepDeg : 1.0;
	double minW = 1e300, maxW = -1.0;
	double minA = 0, maxA = 0;
	cv::Point2d minP, maxP, minD, maxD;
	for (double deg = 0.0; deg < 180.0 - 1e-9; deg += step) {
		const double rad = deg * kPi / 180.0;
		const cv::Point2d dir(std::cos(rad), std::sin(rad));
		double tmin = 1e300, tmax = -1e300;
		cv::Point2d pmin = points[0], pmax = points[0];
		for (const auto& p : points) {
			const double t = p.x * dir.x + p.y * dir.y;
			if (t < tmin) {
				tmin = t;
				pmin = p;
			}
			if (t > tmax) {
				tmax = t;
				pmax = p;
			}
		}
		const double w = tmax - tmin;
		if (w < minW) {
			minW = w;
			minA = deg;
			minP = pmin;
			minD = dir;
		}
		if (w > maxW) {
			maxW = w;
			maxA = deg;
			maxP = pmax;
			maxD = dir;
		}
	}
	if (!(minW < 1e299) || !(maxW > 0.0)) {
		return out;
	}
	out.found = true;
	out.minWidth = minW;
	out.maxWidth = maxW;
	out.minAngle = minA;
	out.maxAngle = maxA;
	out.minA = minP;
	out.minB = cv::Point2d(minP.x + minD.x * minW, minP.y + minD.y * minW);
	out.maxA = maxP;
	out.maxB = cv::Point2d(maxP.x - maxD.x * maxW, maxP.y - maxD.y * maxW);
	return out;
}

GageResult holeToEdge(cv::Point2d center, double radius, cv::Point2d a, cv::Point2d b)
{
	GageResult g;
	if (!(radius >= 0.0) || !std::isfinite(radius)) {
		return g;
	}
	const auto foot = metrology::footOfPerpendicular(center, a, b, false);
	const double d = metrology::distancePointToLine(center, a, b);
	if (!foot.found || !std::isfinite(d)) {
		return g;
	}
	g.found = true;
	g.centerDistance = d;
	g.value = d - radius;
	g.angle = std::atan2(b.y - a.y, b.x - a.x) * 180.0 / kPi;
	const cv::Point2d dir(foot.point.x - center.x, foot.point.y - center.y);
	const double n = std::hypot(dir.x, dir.y);
	if (n < 1e-12) {
		g.a = center;
		g.b = center;
		return g;
	}
	const cv::Point2d u(dir.x / n, dir.y / n);
	g.a = cv::Point2d(center.x + u.x * radius, center.y + u.y * radius);
	g.b = foot.point;
	return g;
}

GageResult holeToHole(cv::Point2d c0, double r0, cv::Point2d c1, double r1)
{
	GageResult g;
	if (!(r0 >= 0.0) || !(r1 >= 0.0)) {
		return g;
	}
	const double d = metrology::distancePoints(c0, c1);
	if (!std::isfinite(d)) {
		return g;
	}
	g.found = true;
	g.centerDistance = d;
	g.value = d - r0 - r1;
	if (d < 1e-12) {
		g.a = c0;
		g.b = c0;
		return g;
	}
	const cv::Point2d u((c1.x - c0.x) / d, (c1.y - c0.y) / d);
	g.a = cv::Point2d(c0.x + u.x * r0, c0.y + u.y * r0);
	g.b = cv::Point2d(c1.x - u.x * r1, c1.y - u.y * r1);
	return g;
}

GageResult edgeToEdge(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2, double angleTolDeg)
{
	GageResult g;
	const double ang = metrology::angleBetweenLines(a1, a2, b1, b2);
	const double dist = metrology::distanceParallelLines(a1, a2, b1, b2, angleTolDeg);
	if (!std::isfinite(ang) || !std::isfinite(dist)) {
		return g;
	}
	const cv::Point2d mid = metrology::midpoint(a1, a2);
	const auto foot = metrology::footOfPerpendicular(mid, b1, b2, false);
	if (!foot.found) {
		return g;
	}
	g.found = true;
	g.value = dist;
	g.centerDistance = dist;
	g.angle = ang;
	g.a = mid;
	g.b = foot.point;
	return g;
}

PixelScale scaleMmPerPx(double mmPerPx)
{
	return makeScale(mmPerPx, mmPerPx, "explicit");
}

PixelScale scaleMmPerPxXY(double mmPerPxX, double mmPerPxY)
{
	return makeScale(mmPerPxX, mmPerPxY, "explicit-xy");
}

PixelScale scaleFromCameraMatrix(const cv::Mat& cameraMatrix, double zMm)
{
	PixelScale s;
	if (cameraMatrix.empty() || !(zMm > 0.0)) {
		return s;
	}
	cv::Mat K;
	cameraMatrix.convertTo(K, CV_64F);
	if (K.rows < 3 || K.cols < 3) {
		return s;
	}
	const double fx = K.at<double>(0, 0);
	const double fy = K.at<double>(1, 1);
	s = makeScale(zMm / fx, zMm / fy, "cameraMatrix");
	return s;
}

PixelScale scaleFromYaml(const std::string& path, double zMm)
{
	cv::FileStorage fs(path, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		return {};
	}
	cv::FileNode nx = fs["mmPerPxX"];
	cv::FileNode ny = fs["mmPerPxY"];
	if (!nx.empty() && !ny.empty()) {
		const PixelScale s = makeScale(static_cast<double>(nx), static_cast<double>(ny), "yaml:mmPerPxXY");
		if (s.found) {
			return s;
		}
	}
	cv::FileNode n = fs["mmPerPx"];
	if (!n.empty()) {
		const PixelScale s = makeScale(static_cast<double>(n), static_cast<double>(n), "yaml:mmPerPx");
		if (s.found) {
			return s;
		}
	}
	fs.release();
	const calib::CalibResult calib = calib::loadCalibration(path);
	if (!calib.found) {
		return {};
	}
	PixelScale s = scaleFromCameraMatrix(calib.cameraMatrix, zMm);
	if (s.found) {
		s.source = "yaml:cameraMatrix";
	}
	return s;
}

cv::Point2d toPlane(cv::Point2d px, const PixelScale& scale)
{
	return planeOf(px, scale);
}

cv::Point2d toFixture(cv::Point2d px, const PixelScale& scale, const FixtureFrame& frame)
{
	const cv::Point2d p = planeOf(px, scale);
	const cv::Point2d o = planeOf(frame.origin, scale);
	const double ang = -frame.angleDeg * kPi / 180.0;
	const double c = std::cos(ang);
	const double s = std::sin(ang);
	const double dx = p.x - o.x;
	const double dy = p.y - o.y;
	return cv::Point2d(c * dx - s * dy, s * dx + c * dy);
}

DimValue dimensionPoints(cv::Point2d a, cv::Point2d b, const PixelScale& scale, const FixtureFrame& frame)
{
	DimValue d;
	d.pixels = metrology::distancePoints(a, b);
	d.found = std::isfinite(d.pixels);
	d.fromPx = a;
	d.toPx = b;
	d.scaled = scale.found;
	const cv::Point2d pa = planeOf(a, scale);
	const cv::Point2d pb = planeOf(b, scale);
	d.physical = d.scaled ? metrology::distancePoints(pa, pb) : d.pixels;
	d.fromFrame = toFixture(a, scale, frame);
	d.toFrame = toFixture(b, scale, frame);
	return d;
}

DimValue dimensionPointLine(cv::Point2d p, cv::Point2d a, cv::Point2d b, const PixelScale& scale,
							const FixtureFrame& frame)
{
	DimValue d;
	const auto foot = metrology::footOfPerpendicular(p, a, b, false);
	d.pixels = metrology::distancePointToLine(p, a, b);
	d.found = foot.found && std::isfinite(d.pixels);
	if (!d.found) {
		return d;
	}
	d.fromPx = p;
	d.toPx = foot.point;
	d.scaled = scale.found;
	if (d.scaled) {
		d.physical = metrology::distancePointToLine(planeOf(p, scale), planeOf(a, scale), planeOf(b, scale));
	} else {
		d.physical = d.pixels;
	}
	d.fromFrame = toFixture(p, scale, frame);
	d.toFrame = toFixture(foot.point, scale, frame);
	return d;
}

DimValue dimensionLines(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2, const PixelScale& scale,
						const FixtureFrame& frame, double angleTolDeg)
{
	DimValue d;
	const GageResult g = edgeToEdge(a1, a2, b1, b2, angleTolDeg);
	if (!g.found) {
		return d;
	}
	d.found = true;
	d.pixels = g.value;
	d.fromPx = g.a;
	d.toPx = g.b;
	d.scaled = scale.found;
	if (d.scaled) {
		const GageResult gm = edgeToEdge(planeOf(a1, scale), planeOf(a2, scale), planeOf(b1, scale), planeOf(b2, scale),
										 angleTolDeg);
		d.physical = gm.found ? gm.value : metrology::distancePoints(planeOf(g.a, scale), planeOf(g.b, scale));
	} else {
		d.physical = d.pixels;
	}
	d.fromFrame = toFixture(g.a, scale, frame);
	d.toFrame = toFixture(g.b, scale, frame);
	return d;
}

DimValue scaleGage(const GageResult& gage, const PixelScale& scale, const FixtureFrame& frame)
{
	DimValue d;
	d.found = gage.found && std::isfinite(gage.value);
	if (!d.found) {
		return d;
	}
	d.pixels = gage.value;
	d.fromPx = gage.a;
	d.toPx = gage.b;
	d.scaled = scale.found;
	const double dx = gage.b.x - gage.a.x;
	const double dy = gage.b.y - gage.a.y;
	const double pixLen = std::hypot(dx, dy);
	double phys = pixLen;
	if (scale.found) {
		phys = std::hypot(dx * scale.mmPerPxX, dy * scale.mmPerPxY);
	}
	if (pixLen > 1e-9) {
		d.physical = (gage.value < 0.0 ? -phys : phys);
	} else {
		d.physical = gage.value * (scale.found ? scale.mmPerPx : 1.0);
	}
	d.fromFrame = toFixture(gage.a, scale, frame);
	d.toFrame = toFixture(gage.b, scale, frame);
	return d;
}

}  // namespace measure
