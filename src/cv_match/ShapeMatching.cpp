#include "ShapeMatching.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <opencv2/imgproc.hpp>

namespace template_matching
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kD2R = kPi / 180.0;
constexpr double kR2D = 180.0 / kPi;

cv::Mat toGray(const cv::Mat& image)
{
	cv::Mat gray;
	if (image.empty())
		return gray;
	if (image.channels() == 1)
	{
		if (image.depth() == CV_8U)
			return image;
		image.convertTo(gray, CV_8U);
		return gray;
	}
	if (image.channels() == 3)
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	else if (image.channels() == 4)
		cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
	else
		image.convertTo(gray, CV_8U);
	return gray;
}

int getTopLayer(const cv::Mat& templ, double minArea)
{
	int top = 0;
	int w = templ.cols, h = templ.rows;
	double area = (double)w * h;
	const double minA = std::max(16.0, minArea);
	// 顶层过小则角度步长大、近圆/L 形容易锁错角；最短边至少约 24 px
	while (area > minA && std::min(w, h) >= 48)
	{
		area /= 4.0;
		w /= 2;
		h /= 2;
		++top;
	}
	return top;
}

float modelRadius(const std::vector<ShapeEdgePt>& edges)
{
	float maxR = 1.f;
	for (const auto& e : edges)
		maxR = std::max(maxR, std::hypot(e.dx, e.dy));
	return maxR;
}

double angleStepDeg(const std::vector<ShapeEdgePt>& edges)
{
	const float maxR = modelRadius(edges);
	return std::max(0.25, std::atan(1.0 / maxR) * kR2D);
}

void computeUnitGrad(const cv::Mat& gray, cv::Mat& nx, cv::Mat& ny, cv::Mat& mag)
{
	cv::Mat gx, gy;
	cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
	cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
	mag.create(gray.size(), CV_32F);
	nx.create(gray.size(), CV_32F);
	ny.create(gray.size(), CV_32F);
	for (int y = 0; y < gray.rows; ++y)
	{
		const float* px = gx.ptr<float>(y);
		const float* py = gy.ptr<float>(y);
		float* pnx = nx.ptr<float>(y);
		float* pny = ny.ptr<float>(y);
		float* pm = mag.ptr<float>(y);
		for (int x = 0; x < gray.cols; ++x)
		{
			const float m = std::hypot(px[x], py[x]);
			pm[x] = m;
			if (m > 1e-3f)
			{
				pnx[x] = px[x] / m;
				pny[x] = py[x] / m;
			}
			else
			{
				pnx[x] = 0.f;
				pny[x] = 0.f;
			}
		}
	}
}

inline float sample32(const cv::Mat& m, float x, float y)
{
	const int w = m.cols, h = m.rows;
	if (w < 2 || h < 2)
		return 0.f;
	if (x < 0.f || y < 0.f || x > w - 1.f || y > h - 1.f)
		return 0.f;
	const int x0 = (int)std::floor(x);
	const int y0 = (int)std::floor(y);
	const int x1 = std::min(x0 + 1, w - 1);
	const int y1 = std::min(y0 + 1, h - 1);
	const float dx = x - x0;
	const float dy = y - y0;
	const float v00 = m.at<float>(y0, x0);
	const float v10 = m.at<float>(y0, x1);
	const float v01 = m.at<float>(y1, x0);
	const float v11 = m.at<float>(y1, x1);
	return (1.f - dx) * (1.f - dy) * v00 + dx * (1.f - dy) * v10 +
		   (1.f - dx) * dy * v01 + dx * dy * v11;
}

bool extractEdges(const cv::Mat& gray, const cv::Mat& mask8, double minMag, int maxPts,
				  std::vector<ShapeEdgePt>& edges)
{
	edges.clear();
	if (gray.empty())
		return false;
	cv::Mat nx, ny, mag;
	computeUnitGrad(gray, nx, ny, mag);
	if (!mask8.empty())
	{
		for (int y = 0; y < mag.rows; ++y)
		{
			const uchar* pm = mask8.ptr<uchar>(y);
			float* pmag = mag.ptr<float>(y);
			for (int x = 0; x < mag.cols; ++x)
			{
				if (pm[x] == 0)
					pmag[x] = 0.f;
			}
		}
	}

	const float th = (float)minMag;
	const float cx = (gray.cols - 1) * 0.5f;
	const float cy = (gray.rows - 1) * 0.5f;
	std::vector<ShapeEdgePt> cand;
	cand.reserve(256);
	for (int y = 1; y < gray.rows - 1; ++y)
	{
		const float* pm = mag.ptr<float>(y);
		const float* pup = mag.ptr<float>(y - 1);
		const float* pdn = mag.ptr<float>(y + 1);
		const float* pnx = nx.ptr<float>(y);
		const float* pny = ny.ptr<float>(y);
		for (int x = 1; x < gray.cols - 1; ++x)
		{
			const float m = pm[x];
			if (m < th)
				continue;
			if (m < pm[x - 1] || m < pm[x + 1] || m < pup[x] || m < pdn[x])
				continue;
			ShapeEdgePt e;
			e.dx = (float)x - cx;
			e.dy = (float)y - cy;
			e.nx = pnx[x];
			e.ny = pny[x];
			cand.push_back(e);
		}
	}
	if ((int)cand.size() > maxPts && maxPts > 0)
	{
		std::vector<std::pair<float, int>> magIdx;
		magIdx.reserve(cand.size());
		for (int i = 0; i < (int)cand.size(); ++i)
		{
			const int ix = (int)std::lround(cand[i].dx + cx);
			const int iy = (int)std::lround(cand[i].dy + cy);
			float mm = 0.f;
			if (ix >= 0 && iy >= 0 && ix < mag.cols && iy < mag.rows)
				mm = mag.at<float>(iy, ix);
			magIdx.push_back({mm, i});
		}
		std::sort(magIdx.begin(), magIdx.end(),
				  [](const auto& a, const auto& b) { return a.first > b.first; });
		std::vector<ShapeEdgePt> picked;
		picked.reserve(maxPts);
		const float minDist2 = 2.5f * 2.5f;
		for (const auto& it : magIdx)
		{
			const ShapeEdgePt& e = cand[it.second];
			bool ok = true;
			for (const auto& p : picked)
			{
				const float ddx = e.dx - p.dx, ddy = e.dy - p.dy;
				if (ddx * ddx + ddy * ddy < minDist2)
				{
					ok = false;
					break;
				}
			}
			if (!ok)
				continue;
			picked.push_back(e);
			if ((int)picked.size() >= maxPts)
				break;
		}
		edges.swap(picked);
	}
	else
		edges.swap(cand);
	return edges.size() >= 12;
}

std::vector<ShapeEdgePt> rotateModel(const std::vector<ShapeEdgePt>& src, double angleDeg)
{
	// 与 OpenCV getRotationMatrix2D 一致：图像坐标 y 向下，正角为画面逆时针
	const double rad = angleDeg * kD2R;
	const float c = (float)std::cos(rad);
	const float s = (float)std::sin(rad);
	std::vector<ShapeEdgePt> out(src.size());
	for (size_t i = 0; i < src.size(); ++i)
	{
		out[i].dx = src[i].dx * c + src[i].dy * s;
		out[i].dy = -src[i].dx * s + src[i].dy * c;
		out[i].nx = src[i].nx * c + src[i].ny * s;
		out[i].ny = -src[i].nx * s + src[i].ny * c;
	}
	return out;
}

double scorePose(const std::vector<ShapeEdgePt>& model, const cv::Mat& nx, const cv::Mat& ny,
				 double cx, double cy, bool polarity, double greediness, double minScore)
{
	const int n = (int)model.size();
	if (n <= 0)
		return 0.0;
	double sum = 0.0;
	const double need = minScore * n * std::max(0.0, std::min(1.0, greediness));
	for (int i = 0; i < n; ++i)
	{
		const float px = (float)(cx + model[i].dx);
		const float py = (float)(cy + model[i].dy);
		const float inx = sample32(nx, px, py);
		const float iny = sample32(ny, px, py);
		double d = model[i].nx * inx + model[i].ny * iny;
		if (!polarity)
			d = std::fabs(d);
		sum += d;
		if (sum + (n - i - 1) < need)
			return sum / n;
	}
	return sum / n;
}

MatchResult packResult(double cx, double cy, double angleDeg, double score, int tw, int th)
{
	const double rad = angleDeg * kD2R;
	const double c = std::cos(rad), s = std::sin(rad);
	const double hw = (tw - 1) * 0.5, hh = (th - 1) * 0.5;
	auto map = [&](double lx, double ly) {
		return cv::Point2d(cx + lx * c + ly * s, cy - lx * s + ly * c);
	};
	MatchResult r;
	r.LeftTop = map(-hw, -hh);
	r.RightTop = map(hw, -hh);
	r.RightBottom = map(hw, hh);
	r.LeftBottom = map(-hw, hh);
	r.Center = cv::Point2d(cx, cy);
	r.Angle = angleDeg;
	r.Score = score;
	return r;
}

void nmsCandidates(std::vector<ShapeCandidate>& cands, double minDist, double angTol, int maxCount)
{
	std::sort(cands.begin(), cands.end(),
			  [](const ShapeCandidate& a, const ShapeCandidate& b) { return a.score > b.score; });
	std::vector<ShapeCandidate> kept;
	for (const auto& c : cands)
	{
		bool ok = true;
		for (const auto& k : kept)
		{
			const double d = std::hypot(c.cx - k.cx, c.cy - k.cy);
			double da = std::fabs(c.angle - k.angle);
			while (da > 180.0)
				da -= 360.0;
			da = std::fabs(da);
			if (d < minDist && da < angTol)
			{
				ok = false;
				break;
			}
		}
		if (!ok)
			continue;
		kept.push_back(c);
		if ((int)kept.size() >= maxCount)
			break;
	}
	cands.swap(kept);
}

}  // namespace

ShapeMatcher::ShapeMatcher(const MatcherParam& param)
{
	initFinishedFlag_ = initMatcher(param);
}

int ShapeMatcher::setTemplate(const cv::Mat& templateImage, const cv::Mat& mask)
{
	learned_ = false;
	layers_.clear();
	cv::Mat gray = toGray(templateImage);
	if (gray.empty())
		return -1;
	cv::Mat mask8;
	if (!mask.empty())
	{
		if (mask.size() != gray.size())
			return -3;
		if (mask.channels() == 1)
			mask.convertTo(mask8, CV_8U);
		else
			return -3;
		cv::threshold(mask8, mask8, 0, 255, cv::THRESH_BINARY);
	}

	templateImage_ = gray.clone();
	templW_ = gray.cols;
	templH_ = gray.rows;
	const int top = getTopLayer(gray, matchParam_.minArea);
	std::vector<cv::Mat> pyr;
	cv::buildPyramid(gray, pyr, top);
	std::vector<cv::Mat> maskPyr;
	if (!mask8.empty())
	{
		maskPyr.resize(pyr.size());
		maskPyr[0] = mask8;
		for (int i = 1; i < (int)pyr.size(); ++i)
			cv::resize(maskPyr[0], maskPyr[i], pyr[i].size(), 0, 0, cv::INTER_NEAREST);
	}

	layers_.resize(pyr.size());
	double mag = matchParam_.edgeMinMag;
	for (int i = 0; i < (int)pyr.size(); ++i)
	{
		layers_[i].width = pyr[i].cols;
		layers_[i].height = pyr[i].rows;
		const cv::Mat& msk = maskPyr.empty() ? cv::Mat() : maskPyr[i];
		double layerMag = mag / std::pow(1.6, i);  // 上层更糊，阈值略降
		layerMag = std::max(8.0, layerMag);
		if (!extractEdges(pyr[i], msk, layerMag, matchParam_.maxEdgePoints, layers_[i].edges))
		{
			if (i == 0)
				return -4;
			layers_[i].edges = layers_[i - 1].edges;
			for (auto& e : layers_[i].edges)
			{
				e.dx *= 0.5f;
				e.dy *= 0.5f;
			}
		}
	}
	learned_ = !layers_.empty() && !layers_[0].edges.empty();
	return learned_ ? 0 : -4;
}

int ShapeMatcher::match(const cv::Mat& frame, std::vector<MatchResult>& matchResults)
{
	matchResults.clear();
	if (!learned_ || layers_.empty())
		return -4;
	cv::Mat gray = toGray(frame);
	if (gray.empty())
		return -1;
	if (gray.cols < templW_ || gray.rows < templH_)
		return -3;

	const int top = (int)layers_.size() - 1;
	int stopLayer = matchParam_.stopLayer;
	if (stopLayer < 0)
		stopLayer = 0;
	if (stopLayer > top)
		stopLayer = top;

	std::vector<cv::Mat> pyr;
	cv::buildPyramid(gray, pyr, top);
	std::vector<cv::Mat> nx(pyr.size()), ny(pyr.size()), mag(pyr.size());
	for (int i = 0; i < (int)pyr.size(); ++i)
		computeUnitGrad(pyr[i], nx[i], ny[i], mag[i]);

	const ShapeLayer& topModel = layers_[top];
	double angStep = angleStepDeg(topModel.edges);
	std::vector<double> angles;
	if (matchParam_.angle < 1e-6)
		angles.push_back(0.0);
	else
	{
		for (double a = 0; a <= matchParam_.angle + 1e-6; a += angStep)
			angles.push_back(a);
		for (double a = -angStep; a >= -matchParam_.angle - 1e-6; a -= angStep)
			angles.push_back(a);
	}

	const int tw = topModel.width, th = topModel.height;
	const float maxR = modelRadius(topModel.edges);
	int margin = (int)std::ceil(maxR) + 1;
	margin = std::min(margin, std::max(1, std::min(pyr[top].rows, pyr[top].cols) / 4));
	const int stride = std::max(1, std::min(tw, th) / 8);
	const double coarseTh = matchParam_.scoreThreshold * 0.72;
	const bool polarity = matchParam_.usePolarity;
	const double greed = matchParam_.greediness;

	std::vector<ShapeCandidate> coarse;
#pragma omp parallel
	{
		std::vector<ShapeCandidate> local;
#pragma omp for schedule(dynamic)
		for (int ai = 0; ai < (int)angles.size(); ++ai)
		{
			const auto model = rotateModel(topModel.edges, angles[ai]);
			for (int y = margin; y < pyr[top].rows - margin; y += stride)
			{
				for (int x = margin; x < pyr[top].cols - margin; x += stride)
				{
					const double sc = scorePose(model, nx[top], ny[top], x, y, polarity, greed, coarseTh);
					if (sc < coarseTh)
						continue;
					ShapeCandidate c;
					c.cx = x;
					c.cy = y;
					c.angle = angles[ai];
					c.score = sc;
					local.push_back(c);
				}
			}
		}
#pragma omp critical
		{
			coarse.insert(coarse.end(), local.begin(), local.end());
		}
	}

	const double minDistTop = 0.45 * std::min(tw, th);
	nmsCandidates(coarse, minDistTop, std::max(angStep * 1.5, 3.0),
				  std::max(matchParam_.maxCount * 3, matchParam_.maxCount + 5));

	std::vector<ShapeCandidate> refined = coarse;
	for (int layer = top - 1; layer >= stopLayer; --layer)
	{
		std::vector<ShapeCandidate> next;
		const auto& mdl = layers_[layer];
		double step = angleStepDeg(mdl.edges);
		for (const auto& c0 : refined)
		{
			const double cx0 = c0.cx * 2.0;
			const double cy0 = c0.cy * 2.0;
			double bestS = -1;
			ShapeCandidate best = c0;
			best.cx = cx0;
			best.cy = cy0;
			const int rad = 3;
			std::vector<double> angs;
			if (matchParam_.angle < 1e-6)
				angs.push_back(0.0);
			else
			{
				for (int k = -3; k <= 3; ++k)
					angs.push_back(c0.angle + k * step);
			}
			for (double ang : angs)
			{
				const auto model = rotateModel(mdl.edges, ang);
				for (int dy = -rad; dy <= rad; ++dy)
				{
					for (int dx = -rad; dx <= rad; ++dx)
					{
						const double cx = cx0 + dx;
						const double cy = cy0 + dy;
						if (cx < 2 || cy < 2 || cx >= pyr[layer].cols - 2 || cy >= pyr[layer].rows - 2)
							continue;
						const double sc = scorePose(model, nx[layer], ny[layer], cx, cy, polarity, greed,
												   matchParam_.scoreThreshold * 0.6);
						if (sc > bestS)
						{
							bestS = sc;
							best.cx = cx;
							best.cy = cy;
							best.angle = ang;
							best.score = sc;
						}
					}
				}
			}
			if (best.score >= matchParam_.scoreThreshold * (layer == stopLayer ? 1.0 : 0.65))
				next.push_back(best);
		}
		const double md = 0.45 * std::min(layers_[layer].width, layers_[layer].height);
		nmsCandidates(next, md, 4.0, matchParam_.maxCount + 4);
		refined.swap(next);
	}

	nmsCandidates(refined, 0.5 * std::min(templW_, templH_), 4.0, matchParam_.maxCount);
	const int scale = 1 << stopLayer;
	for (const auto& c : refined)
	{
		if (c.score < matchParam_.scoreThreshold)
			continue;
		matchResults.push_back(packResult(c.cx * scale, c.cy * scale, c.angle, c.score, templW_, templH_));
		if ((int)matchResults.size() >= matchParam_.maxCount)
			break;
	}
	return (int)matchResults.size();
}

}  // namespace template_matching
