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

// 出界的模型点直接跳过，不记 0。可见点少于模型的这个比例时，仍用全部点数做分母，
// 避免只擦到画布一条边就被归一化成高分。大约 68% 还在画面里的残件高于这条线。
// 68% 还在画面里的残件要高于这条线。再低就会把贴边的纹理归一化成 0.55 左右的假命中。
constexpr double kMinInFrameFrac = 0.62;

bool inFrame32(const cv::Mat& m, float x, float y)
{
	const int w = m.cols, h = m.rows;
	if (w < 2 || h < 2)
		return false;
	return x >= 0.f && y >= 0.f && x <= (float)(w - 1) && y <= (float)(h - 1);
}

double meanInFrame(double sum, int used, int n)
{
	if (n <= 0)
		return 0.0;
	if (used > 0 && (double)used >= (double)n * kMinInFrameFrac)
		return sum / (double)used;
	return sum / (double)n;
}

// 剩下的点就算全在画面里且方向完全对齐，也到不了贪心门槛。
bool poseHopeless(double sum, int used, int n, int index, double abortBelow)
{
	const int rem = n - index - 1;
	const int fut = used + rem;
	const double best = (fut > 0 && (double)fut >= (double)n * kMinInFrameFrac)
							? (sum + rem) / (double)fut
							: (sum + (double)rem) / (double)n;
	return best < abortBelow;
}

// 顶层不再用模型半径当搜索边距。半径换算回原图大约是 80–120 px，贴边残件的中心进不了粗搜。
int borderMargin(int rows, int cols, int stride)
{
	int m = 2;
	if (m * 2 + stride >= rows || m * 2 + stride >= cols)
		m = std::max(1, std::min(rows, cols) / 10);
	const int cap = std::max(1, std::min(rows, cols) / 4);
	if (m > cap)
		m = cap;
	return std::max(1, m);
}

inline float sample32(const cv::Mat& m, float x, float y)
{
	const int w = m.cols, h = m.rows;
	if (w < 2 || h < 2)
		return 0.f;
	if (x < 0.f || y < 0.f || x > w - 1.f || y > h - 1.f)
		return 0.f;
	const int x0 = (int)x;
	const int y0 = (int)y;
	const int x1 = x0 + 1 < w ? x0 + 1 : x0;
	const int y1 = y0 + 1 < h ? y0 + 1 : y0;
	const float dx = x - (float)x0;
	const float dy = y - (float)y0;
	const float* r0 = m.ptr<float>(y0);
	const float* r1 = m.ptr<float>(y1);
	const float v00 = r0[x0];
	const float v10 = r0[x1];
	const float v01 = r1[x0];
	const float v11 = r1[x1];
	return (1.f - dx) * (1.f - dy) * v00 + dx * (1.f - dy) * v10 +
		   (1.f - dx) * dy * v01 + dx * dy * v11;
}

// 梯度方向收成 16 档（每档 22.5°）。粗搜把每档的邻域最大余弦铺成响应图，位姿循环只查表。
int oriBin16(float nx, float ny)
{
	if (nx * nx + ny * ny < 1e-8f)
		return -1;
	double ang = std::atan2((double)ny, (double)nx);
	if (ang < 0.0)
		ang += 2.0 * kPi;
	int b = (int)(ang * (16.0 / (2.0 * kPi)));
	if (b < 0)
		b = 0;
	if (b >= 16)
		b = 0;
	return b;
}

struct OriCos16
{
	float pol[16][16];
	float any[16][16];
	OriCos16()
	{
		for (int i = 0; i < 16; ++i)
		{
			for (int j = 0; j < 16; ++j)
			{
				int d = std::abs(i - j);
				if (d > 8)
					d = 16 - d;
				const float c = (float)std::cos(d * (kPi / 8.0));
				pol[i][j] = c > 0.f ? c : 0.f;
				any[i][j] = std::fabs(c);
			}
		}
	}
};

const OriCos16& oriCos16()
{
	static const OriCos16 table;
	return table;
}

void buildOri16(const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag, float floorMag, cv::Mat& ori)
{
	ori.create(nx.size(), CV_8U);
	for (int y = 0; y < nx.rows; ++y)
	{
		const float* pnx = nx.ptr<float>(y);
		const float* pny = ny.ptr<float>(y);
		const float* pm = mag.ptr<float>(y);
		uchar* po = ori.ptr<uchar>(y);
		for (int x = 0; x < nx.cols; ++x)
		{
			if (pm[x] < floorMag)
			{
				po[x] = 255;
				continue;
			}
			const int b = oriBin16(pnx[x], pny[x]);
			po[x] = b < 0 ? (uchar)255 : (uchar)b;
		}
	}
}

// 16 张响应图。R_b(x,y) 是 b 档与场景方向在 ±spread 里的最大余弦（极性由建图时选定）。
// 扩散只做一次，位姿循环只查表。spread=3 盖住步长 4 的中心误差（≤2px）。
// 顶层角度仍用 atan(1/R)，不再加倍。末层精修仍用浮点 scorePose。
// oriBins!=16 不建这些图。
constexpr int kMapSpread = 3;
constexpr int kMapStride = 4;

// 粗搜进精修的名额。不能比步长加倍之前更紧：maxCount=200 时旧名额是 800，
// 收到 400 会在纹理图上把贴边弱分挤掉。oriBins=16 再垫一个下限。
int coarseKeepCount(int maxCount, bool quant)
{
	const int loose = std::max(maxCount * 4, maxCount + 8);
	if (!quant)
		return loose;
	return std::max(loose, 128);
}

int refineKeepCount(int maxCount, bool quant)
{
	const int loose = std::max(maxCount * 3, maxCount + 6);
	if (!quant)
		return loose;
	return std::max(loose, 96);
}
struct OriResponse16
{
	int width = 0;
	int height = 0;
	int spread = 0;
	std::vector<float> data;
};

struct MapPoint
{
	float dx = 0.f;
	float dy = 0.f;
	int bin = -1;
};

void buildOriResponse16(const cv::Mat& nx, const cv::Mat& ny, const cv::Mat& mag, float floorMag,
						bool polarity, int spread, OriResponse16& out)
{
	cv::Mat ori;
	buildOri16(nx, ny, mag, floorMag, ori);
	const int w = ori.cols;
	const int h = ori.rows;
	out.width = w;
	out.height = h;
	out.spread = std::max(0, spread);
	out.data.assign((size_t)16 * (size_t)w * (size_t)h, 0.f);
	if (w <= 0 || h <= 0)
		return;
	const OriCos16& table = oriCos16();
	cv::Mat raw(h, w, CV_32F);
	cv::Mat kernel;
	if (out.spread > 0)
		kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * out.spread + 1, 2 * out.spread + 1));
	for (int b = 0; b < 16; ++b)
	{
		const float* cosRow = polarity ? table.pol[b] : table.any[b];
		for (int y = 0; y < h; ++y)
		{
			const uchar* po = ori.ptr<uchar>(y);
			float* dst = raw.ptr<float>(y);
			for (int x = 0; x < w; ++x)
			{
				const int sl = po[x];
				dst[x] = sl <= 15 ? cosRow[sl] : 0.f;
			}
		}
		float* plane = out.data.data() + (size_t)b * (size_t)w * (size_t)h;
		cv::Mat spreaded(h, w, CV_32F, plane);
		if (out.spread > 0)
			cv::dilate(raw, spreaded, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
		else
			raw.copyTo(spreaded);
	}
}

std::vector<MapPoint> makeMapPoints(const std::vector<ShapeEdgePt>& model)
{
	std::vector<MapPoint> pts(model.size());
	for (size_t i = 0; i < model.size(); ++i)
	{
		pts[i].dx = model[i].dx;
		pts[i].dy = model[i].dy;
		pts[i].bin = oriBin16(model[i].nx, model[i].ny);
	}
	return pts;
}

struct EdgeCand
{
	float dx = 0.f;
	float dy = 0.f;
	float nx = 0.f;
	float ny = 0.f;
	float mag = 0.f;
	int bin = 0;
};

// 按角度轮流取点，避免强边缘（孔、高光）占满 maxEdgePoints，也避免残缺的那一段排在打分最前面被贪心提前丢掉。
void spreadEdgePoints(std::vector<EdgeCand>& cand, int maxPts, std::vector<ShapeEdgePt>& edges)
{
	const int bins = 16;
	std::vector<std::vector<int>> byBin(bins);
	for (int i = 0; i < (int)cand.size(); ++i)
		byBin[cand[i].bin].push_back(i);
	for (auto& b : byBin)
	{
		std::sort(b.begin(), b.end(), [&](int a, int c) { return cand[a].mag > cand[c].mag; });
	}
	std::vector<int> cursor(bins, 0);
	std::vector<ShapeEdgePt> picked;
	picked.reserve(std::min(maxPts > 0 ? maxPts : (int)cand.size(), (int)cand.size()));
	const float minDist2 = 2.5f * 2.5f;
	const int limit = maxPts > 0 ? maxPts : (int)cand.size();
	bool progressed = true;
	while ((int)picked.size() < limit && progressed)
	{
		progressed = false;
		for (int b = 0; b < bins && (int)picked.size() < limit; ++b)
		{
			while (cursor[b] < (int)byBin[b].size())
			{
				const EdgeCand& e = cand[byBin[b][cursor[b]]];
				++cursor[b];
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
				ShapeEdgePt pt;
				pt.dx = e.dx;
				pt.dy = e.dy;
				pt.nx = e.nx;
				pt.ny = e.ny;
				picked.push_back(pt);
				progressed = true;
				break;
			}
		}
	}
	edges.swap(picked);
}

// 沿法向对梯度幅值做抛物线插值，边缘点最多挪半个像素。
// 不要和 oriBins=16 绑在一起：挪点会改模型，末层浮点分就和全程浮点不是同一套边缘。
void nudgeSubpixel(EdgeCand& e, const cv::Mat& mag, float cx, float cy)
{
	const float x = e.dx + cx;
	const float y = e.dy + cy;
	const float m0 = sample32(mag, x, y);
	const float mm = sample32(mag, x - e.nx, y - e.ny);
	const float mp = sample32(mag, x + e.nx, y + e.ny);
	const float denom = mm - 2.f * m0 + mp;
	if (std::fabs(denom) < 1e-3f)
		return;
	float delta = 0.5f * (mm - mp) / denom;
	if (delta > 0.5f)
		delta = 0.5f;
	else if (delta < -0.5f)
		delta = -0.5f;
	e.dx += delta * e.nx;
	e.dy += delta * e.ny;
}

bool extractEdges(const cv::Mat& gray, const cv::Mat& mask8, double minMag, double highMag, int maxPts,
				  std::vector<ShapeEdgePt>& edges, bool subpixel)
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

	const float low = (float)minMag;
	float high = (float)highMag;
	if (!(high > low))
		high = low;
	const float cx = (gray.cols - 1) * 0.5f;
	const float cy = (gray.rows - 1) * 0.5f;
	std::vector<EdgeCand> cand;
	cand.reserve(256);
	cv::Mat seed(gray.size(), CV_8U, cv::Scalar(0));
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
			if (m < low)
				continue;
			if (m < pm[x - 1] || m < pm[x + 1] || m < pup[x] || m < pdn[x])
				continue;
			EdgeCand e;
			e.dx = (float)x - cx;
			e.dy = (float)y - cy;
			e.nx = pnx[x];
			e.ny = pny[x];
			e.mag = m;
			double ang = std::atan2((double)e.dy, (double)e.dx);
			if (ang < 0.0)
				ang += 2.0 * kPi;
			e.bin = std::min(15, (int)(ang / (2.0 * kPi) * 16.0));
			cand.push_back(e);
			if (m >= high)
				seed.at<uchar>(y, x) = 255;
		}
	}
	// 双阈值：弱边缘只有连到强种子上才留下。单阈值时 high==low，种子就是全部候选。
	if (high > low + 0.5f && !cand.empty())
	{
		cv::Mat alive = seed.clone();
		for (int step = 0; step < 5; ++step)
		{
			cv::Mat grown;
			cv::dilate(alive, grown, cv::Mat());
			bool any = false;
			for (int y = 1; y < gray.rows - 1; ++y)
			{
				uchar* pa = alive.ptr<uchar>(y);
				const uchar* pg = grown.ptr<uchar>(y);
				const float* pm = mag.ptr<float>(y);
				for (int x = 1; x < gray.cols - 1; ++x)
				{
					if (pa[x] || !pg[x] || pm[x] < low)
						continue;
					pa[x] = 255;
					any = true;
				}
			}
			if (!any)
				break;
		}
		std::vector<EdgeCand> linked;
		linked.reserve(cand.size());
		for (const auto& e : cand)
		{
			const int x = (int)std::lround(e.dx + cx);
			const int y = (int)std::lround(e.dy + cy);
			if (x < 0 || y < 0 || x >= alive.cols || y >= alive.rows)
				continue;
			if (alive.at<uchar>(y, x))
				linked.push_back(e);
		}
		cand.swap(linked);
	}
	else if (!cand.empty())
	{
		// 单阈值时丢掉远弱于最强边缘的点，避免径向灰阶或纹理把轮廓分数拉低。
		float peak = 0.f;
		for (const auto& e : cand)
			peak = std::max(peak, e.mag);
		const float rel = std::max(low, 0.45f * peak);
		std::vector<EdgeCand> strong;
		strong.reserve(cand.size());
		for (const auto& e : cand)
			if (e.mag >= rel)
				strong.push_back(e);
		if (strong.size() >= 12)
			cand.swap(strong);
	}
	if (subpixel)
	{
		for (auto& e : cand)
			nudgeSubpixel(e, mag, cx, cy);
	}
	spreadEdgePoints(cand, maxPts, edges);
	return edges.size() >= 12;
}

std::vector<ShapeEdgePt> scaleEdges(const std::vector<ShapeEdgePt>& src, double scale)
{
	std::vector<ShapeEdgePt> out = src;
	const float s = (float)scale;
	for (auto& e : out)
	{
		e.dx *= s;
		e.dy *= s;
	}
	return out;
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
				 const cv::Mat& mag, float minMag, double cx, double cy, bool polarity, double greediness,
				 double minScore, bool spatialBalance)
{
	const int n = (int)model.size();
	if (n <= 0)
		return 0.0;
	double sum = 0.0;
	int quadPts[4] = {0, 0, 0, 0};
	int quadHit[4] = {0, 0, 0, 0};
	int used = 0;
	const double abortBelow = minScore * std::max(0.0, std::min(1.0, greediness));
	// 只压很弱的纹理。门槛封顶，不随 edgeMinMag 抬到 119 那种对比度，否则正常边缘和残缺件会被乘到 0.5 下面。
	const float magGate = std::min(32.f, std::max(8.f, 2.5f * minMag));
	const int w = nx.cols, h = nx.rows;
	if (w < 2 || h < 2)
		return 0.0;
	const float xMax = (float)(w - 1);
	const float yMax = (float)(h - 1);
	for (int i = 0; i < n; ++i)
	{
		const float px = (float)(cx + model[i].dx);
		const float py = (float)(cy + model[i].dy);
		if (px < 0.f || py < 0.f || px > xMax || py > yMax)
			continue;
		const int x0 = (int)px;
		const int y0 = (int)py;
		const int x1 = x0 + 1 < w ? x0 + 1 : x0;
		const int y1 = y0 + 1 < h ? y0 + 1 : y0;
		const float dx = px - (float)x0;
		const float dy = py - (float)y0;
		const float* nx0 = nx.ptr<float>(y0);
		const float* nx1 = nx.ptr<float>(y1);
		const float* ny0 = ny.ptr<float>(y0);
		const float* ny1 = ny.ptr<float>(y1);
		const float w00 = (1.f - dx) * (1.f - dy);
		const float w10 = dx * (1.f - dy);
		const float w01 = (1.f - dx) * dy;
		const float w11 = dx * dy;
		const float inx = w00 * nx0[x0] + w10 * nx0[x1] + w01 * nx1[x0] + w11 * nx1[x1];
		const float iny = w00 * ny0[x0] + w10 * ny0[x1] + w01 * ny1[x0] + w11 * ny1[x1];
		float sm = magGate;
		if (spatialBalance)
		{
			const float* m0 = mag.ptr<float>(y0);
			const float* m1 = mag.ptr<float>(y1);
			sm = w00 * m0[x0] + w10 * m0[x1] + w01 * m1[x0] + w11 * m1[x1];
		}
		float w = std::min(1.f, sm / magGate);
		double raw = model[i].nx * inx + model[i].ny * iny;
		if (!polarity)
			raw = std::fabs(raw);
		else if (raw < 0.0)
			raw = 0.0;  // 与 16 档余弦表、Halcon use_polarity 一致：反方向记 0，不把整段分数拉成负数
		const double d = raw * (double)w;
		sum += d;
		++used;
		if (spatialBalance)
		{
			const int q = (model[i].dx >= 0.f ? 1 : 0) + (model[i].dy >= 0.f ? 2 : 0);
			quadPts[q] += 1;
			if (d > 0.35)
				quadHit[q] += 1;
		}
		if (poseHopeless(sum, used, n, i, abortBelow))
			return sum / n;
	}
	double mean = meanInFrame(sum, used, n);
	if (!spatialBalance)
		return mean;
	// 闭合轮廓（圆点、圆环）四个象限都有点。弧状干扰只贴住一侧，平均分仍可能过阈值。
	// 只在至少三个象限有点时打折，L 形这类开口轮廓维持原来的平均分。
	int present = 0;
	int good = 0;
	for (int q = 0; q < 4; ++q)
	{
		if (quadPts[q] < 3)
			continue;
		++present;
		if (quadHit[q] * 2 >= quadPts[q])
			++good;
	}
	if (present >= 3 && good < present)
	{
		const double balance = (double)good / (double)present;
		mean *= (0.55 + 0.45 * balance);
	}
	return mean;
}

// 顶层粗搜：响应图已经含 ±spread 的最大值。点本身在画布外就跳过。
// 极性在建图时写进余弦。末层精修不走这里。
double scorePoseMap(const std::vector<MapPoint>& model, const OriResponse16& resp, double cx, double cy,
					double greediness, double minScore)
{
	const int n = (int)model.size();
	if (n <= 0 || resp.width <= 0 || resp.height <= 0 || resp.data.empty())
		return 0.0;
	const int w = resp.width;
	const int h = resp.height;
	const size_t planeStride = (size_t)w * (size_t)h;
	const float* base = resp.data.data();
	double sum = 0.0;
	int used = 0;
	const double abortBelow = minScore * std::max(0.0, std::min(1.0, greediness));
	for (int i = 0; i < n; ++i)
	{
		const int px = (int)std::lround(cx + (double)model[i].dx);
		const int py = (int)std::lround(cy + (double)model[i].dy);
		// 点本身在画布外就跳过，不用邻域往画面里捞一条边来凑分。邻域最大已在响应图里。
		if ((unsigned)px >= (unsigned)w || (unsigned)py >= (unsigned)h)
			continue;
		float best = 0.f;
		const int ml = model[i].bin;
		if ((unsigned)ml < 16u)
			best = base[(size_t)ml * planeStride + (size_t)py * (size_t)w + (size_t)px];
		sum += (double)best;
		++used;
		if (poseHopeless(sum, used, n, i, abortBelow))
			return sum / n;
	}
	return meanInFrame(sum, used, n);
}

MatchResult packResult(double cx, double cy, double angleDeg, double score, int tw, int th, double objScale)
{
	const double rad = angleDeg * kD2R;
	const double c = std::cos(rad), s = std::sin(rad);
	const double hw = (tw - 1) * 0.5 * objScale, hh = (th - 1) * 0.5 * objScale;
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
	r.Scale = objScale;
	return r;
}

void nmsCandidates(std::vector<ShapeCandidate>& cands, double minDist, int maxCount, bool blendTies = false)
{
	std::sort(cands.begin(), cands.end(),
			  [](const ShapeCandidate& a, const ShapeCandidate& b) { return a.score > b.score; });
	// 响应图把分数铺平之后，同一实例上很多格点都是 1。只留排序第一个会停在平台边缘，
	// 精修窗口够不着真中心。同分的邻居收成质心，角度用圆周平均。
	struct Acc
	{
		ShapeCandidate c;
		double seedX = 0;
		double seedY = 0;
		double seedAng = 0;
		double w = 1;
		double angC = 0;
		double angS = 0;
	};
	std::vector<Acc> kept;
	kept.reserve(std::min((int)cands.size(), std::max(maxCount, 1)));
	for (const auto& c : cands)
	{
		int match = -1;
		for (int i = 0; i < (int)kept.size(); ++i)
		{
			// 用种子位置，不用挪过的质心，避免两个挨着的实例被收成一个。
			if (std::hypot(c.cx - kept[i].seedX, c.cy - kept[i].seedY) < minDist)
			{
				match = i;
				break;
			}
		}
		if (match >= 0)
		{
			Acc& a = kept[match];
			// 响应图把一片格点都抬到同一分上，最高的那一格经常在平台边上。
			// 几乎同分的邻居一起算质心。角度只平均离种子约一步的，
			// 避免把另一侧的高分角度卷进圆周平均。
			if (blendTies && c.score >= a.c.score - 1e-4)
			{
				const double w = a.w;
				a.c.cx = (a.c.cx * w + c.cx) / (w + 1.0);
				a.c.cy = (a.c.cy * w + c.cy) / (w + 1.0);
				double dAng = std::fabs(c.angle - a.seedAng);
				if (dAng > 180.0)
					dAng = 360.0 - dAng;
				if (dAng <= 4.0)
				{
					a.angC += std::cos(c.angle * kD2R);
					a.angS += std::sin(c.angle * kD2R);
					a.c.angle = std::atan2(a.angS, a.angC) * kR2D;
				}
				a.w = w + 1.0;
			}
			continue;
		}
		if ((int)kept.size() >= maxCount)
			continue;
		Acc a;
		a.c = c;
		a.seedX = c.cx;
		a.seedY = c.cy;
		a.seedAng = c.angle;
		a.w = 1;
		a.angC = std::cos(c.angle * kD2R);
		a.angS = std::sin(c.angle * kD2R);
		kept.push_back(a);
	}
	cands.clear();
	cands.reserve(kept.size());
	for (const auto& a : kept)
		cands.push_back(a.c);
}

}  // namespace

ShapeMatcher::ShapeMatcher(const MatcherParam& param)
{
	initFinishedFlag_ = initMatcher(param);
	scales_ = sampleMatchScales(matchParam_);
	scaleSearch_ = scaleSearchEnabled(matchParam_);
	scaleMin_ = scales_.front();
	scaleMax_ = scales_.back();
	for (double s : scales_)
	{
		scaleMin_ = std::min(scaleMin_, s);
		scaleMax_ = std::max(scaleMax_, s);
	}
	scaleDs_ = 0.0;
	if (scales_.size() >= 2)
	{
		double gap = 1e9;
		for (size_t i = 1; i < scales_.size(); ++i)
			gap = std::min(gap, std::fabs(scales_[i] - scales_[i - 1]));
		scaleDs_ = 0.5 * gap;
	}
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
	double contrast = matchParam_.edgeContrast;
	for (int i = 0; i < (int)pyr.size(); ++i)
	{
		layers_[i].width = pyr[i].cols;
		layers_[i].height = pyr[i].rows;
		const cv::Mat& msk = maskPyr.empty() ? cv::Mat() : maskPyr[i];
		double layerMag = mag / std::pow(1.6, i);  // 上层更糊，阈值略降
		layerMag = std::max(8.0, layerMag);
		double layerHigh = contrast > mag ? contrast / std::pow(1.6, i) : layerMag;
		if (layerHigh < layerMag)
			layerHigh = layerMag;
		// 16 档只用于顶层粗搜。亚像素挪点会改每一层的模型点，末层仍拿这些点做浮点分，
		// 贴边残件上可以把大约 0.53 的分压到 0.36。默认不挪。
		const bool subpixel = false;
		if (!extractEdges(pyr[i], msk, layerMag, layerHigh, matchParam_.maxEdgePoints, layers_[i].edges, subpixel))
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
	const double fitScale = scaleSearch_ ? scaleMin_ : 1.0;
	const int needW = std::max(1, (int)std::floor((double)templW_ * fitScale));
	const int needH = std::max(1, (int)std::floor((double)templH_ * fitScale));
	if (gray.cols < needW || gray.rows < needH)
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
	const bool quantCoarse = matchParam_.oriBins == 16;
	const bool polarity = matchParam_.usePolarity;
	// ±kMapSpread 扩散进响应图，不放进位姿循环。步长 kMapStride 的中心误差不超过 spread。
	OriResponse16 respTop;
	if (quantCoarse)
		buildOriResponse16(nx[top], ny[top], mag[top], 6.f, polarity, kMapSpread, respTop);

	const ShapeLayer& topModel = layers_[top];
	double angStep = angleStepDeg(topModel.edges);
	if (scaleSearch_)
	{
		const float scaledR = std::max(1.f, modelRadius(topModel.edges) * (float)scaleMax_);
		angStep = std::max(0.25, std::atan(1.0 / scaledR) * kR2D);
	}
	// 顶层角度步长保持 atan(1/R)。±180° 再乘 2 时，弱残件离最近样本大约一整步，
	// 粗分会掉到阈值下，下一层接不回来。oriBins=0 本来就是这个步长。
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
	int stride = std::max(1, std::min(tw, th) / 8);
	if (stride > 4)
		stride = 4;
	// 旧的「步长≤2」是在补逐点 ±2。响应图把邻域最大值铺开之后，步长 4 的格点误差 ≤2，
	// 下一层搜索半径盖得住。oriBins=0 仍用上面的 1–4。
	if (quantCoarse)
		stride = kMapStride;
	const int margin = borderMargin(pyr[top].rows, pyr[top].cols, stride);
	const double coarseTh = matchParam_.scoreThreshold * 0.72;
	const double greed = matchParam_.greediness;

	std::vector<ShapeCandidate> coarse;
	if (!scaleSearch_)
	{
#pragma omp parallel
		{
			std::vector<ShapeCandidate> local;
#pragma omp for schedule(dynamic)
			for (int ai = 0; ai < (int)angles.size(); ++ai)
			{
				const auto model = rotateModel(topModel.edges, angles[ai]);
				const std::vector<MapPoint> mapPts = quantCoarse ? makeMapPoints(model) : std::vector<MapPoint>();
				for (int y = margin; y < pyr[top].rows - margin; y += stride)
				{
					for (int x = margin; x < pyr[top].cols - margin; x += stride)
					{
						const double sc = quantCoarse
							? scorePoseMap(mapPts, respTop, x, y, greed, coarseTh)
							: scorePose(model, nx[top], ny[top], mag[top], (float)matchParam_.edgeMinMag, x, y, polarity, greed, coarseTh, false);
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
	}
	else
	{
		int scaleStride = std::max(1, std::min(tw, th) / 12);
		if (scaleStride > 4)
			scaleStride = 4;
		if (quantCoarse)
			scaleStride = kMapStride;
		std::vector<std::pair<int, int>> jobs;
		jobs.reserve(angles.size() * scales_.size());
		for (int ai = 0; ai < (int)angles.size(); ++ai)
			for (int si = 0; si < (int)scales_.size(); ++si)
				jobs.emplace_back(ai, si);
#pragma omp parallel
		{
			std::vector<ShapeCandidate> local;
#pragma omp for schedule(dynamic)
			for (int ji = 0; ji < (int)jobs.size(); ++ji)
			{
				const int ai = jobs[ji].first;
				const double objScale = scales_[jobs[ji].second];
				const auto model = scaleEdges(rotateModel(topModel.edges, angles[ai]), objScale);
				const std::vector<MapPoint> mapPts = quantCoarse ? makeMapPoints(model) : std::vector<MapPoint>();
				const int mgn = borderMargin(pyr[top].rows, pyr[top].cols, scaleStride);
				for (int y = mgn; y < pyr[top].rows - mgn; y += scaleStride)
				{
					for (int x = mgn; x < pyr[top].cols - mgn; x += scaleStride)
					{
						const double sc = quantCoarse
							? scorePoseMap(mapPts, respTop, x, y, greed, coarseTh)
							: scorePose(model, nx[top], ny[top], mag[top], (float)matchParam_.edgeMinMag, x, y, polarity, greed, coarseTh, false);
						if (sc < coarseTh)
							continue;
						ShapeCandidate c;
						c.cx = x;
						c.cy = y;
						c.angle = angles[ai];
						c.score = sc;
						c.scale = objScale;
						local.push_back(c);
					}
				}
			}
#pragma omp critical
			{
				coarse.insert(coarse.end(), local.begin(), local.end());
			}
		}
	}

	const double topSpan = std::min((double)std::min(tw, th), 2.0 * (double)maxR);
	const double minDistTop = 0.45 * std::max(4.0, topSpan);
	nmsCandidates(coarse, minDistTop, coarseKeepCount(matchParam_.maxCount, quantCoarse), quantCoarse);

	std::vector<ShapeCandidate> refined = coarse;
	for (int layer = top - 1; layer >= stopLayer; --layer)
	{
		std::vector<ShapeCandidate> next;
		const auto& mdl = layers_[layer];
		double step = angleStepDeg(mdl.edges);
		// 质心之后顶层误差大约 2px，翻到下一层是 4px。贴边残件的响应图峰值还会再偏出去，第一层用 ±10。
		const int rad0 = (quantCoarse && layer == top - 1) ? 10 : 4;
#pragma omp parallel
		{
			std::vector<ShapeCandidate> local;
#pragma omp for schedule(dynamic)
			for (int ci = 0; ci < (int)refined.size(); ++ci)
			{
			const auto& c0 = refined[ci];
			const double cx0 = c0.cx * 2.0;
			const double cy0 = c0.cy * 2.0;
			double bestS = -1;
			ShapeCandidate best = c0;
			best.cx = cx0;
			best.cy = cy0;
			const int rad = rad0;
			std::vector<double> angs;
			if (matchParam_.angle < 1e-6)
				angs.push_back(0.0);
			else
			{
				// 响应图在只露出大约 2/3 的件上，粗分最高的角度可以偏到真角一侧约 10°。
				// 第一层用浮点分把这个偏差找回来；更下面的层仍是 ±3 步。
				const int kMax = (quantCoarse && layer == top - 1)
									 ? std::max(3, (int)std::ceil(12.0 / std::max(step, 1e-3)))
									 : 3;
				for (int k = -kMax; k <= kMax; ++k)
				{
					double a = c0.angle + k * step;
					if (a > matchParam_.angle)
						a = matchParam_.angle;
					else if (a < -matchParam_.angle)
						a = -matchParam_.angle;
					bool dup = false;
					for (double prev : angs)
					{
						if (std::fabs(prev - a) < 1e-6)
							dup = true;
					}
					if (!dup)
						angs.push_back(a);
				}
			}
			if (!scaleSearch_)
			{
				for (double ang : angs)
				{
					const auto model = rotateModel(mdl.edges, ang);
					for (int dy = -rad; dy <= rad; ++dy)
					{
						for (int dx = -rad; dx <= rad; ++dx)
						{
							const double cx = cx0 + dx;
							const double cy = cy0 + dy;
							if (cx < 0 || cy < 0 || cx > pyr[layer].cols - 1 || cy > pyr[layer].rows - 1)
								continue;
							const double sc = scorePose(model, nx[layer], ny[layer], mag[layer], (float)matchParam_.edgeMinMag,
													   cx, cy, polarity, greed,
													   matchParam_.scoreThreshold * 0.6, layer == stopLayer);
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
			}
			else
			{
				std::vector<double> tryScales;
				for (int ks = -1; ks <= 1; ++ks)
				{
					double s = c0.scale + ks * scaleDs_;
					if (s < scaleMin_)
						s = scaleMin_;
					if (s > scaleMax_)
						s = scaleMax_;
					bool dup = false;
					for (double prev : tryScales)
					{
						if (std::fabs(prev - s) < 1e-6)
							dup = true;
					}
					if (!dup)
						tryScales.push_back(s);
				}
				const int radS = rad0;
				for (double ang : angs)
				{
					const auto rotated = rotateModel(mdl.edges, ang);
					for (double objScale : tryScales)
					{
						const auto model = scaleEdges(rotated, objScale);
						for (int dy = -radS; dy <= radS; ++dy)
						{
							for (int dx = -radS; dx <= radS; ++dx)
							{
								const double cx = cx0 + dx;
								const double cy = cy0 + dy;
								if (cx < 0 || cy < 0 || cx > pyr[layer].cols - 1 || cy > pyr[layer].rows - 1)
									continue;
								const double sc = scorePose(model, nx[layer], ny[layer], mag[layer], (float)matchParam_.edgeMinMag,
														   cx, cy, polarity, greed,
														   matchParam_.scoreThreshold * 0.6, layer == stopLayer);
								if (sc > bestS)
								{
									bestS = sc;
									best.cx = cx;
									best.cy = cy;
									best.angle = ang;
									best.scale = objScale;
									best.score = sc;
								}
							}
						}
					}
				}
			}
			if (best.score >= matchParam_.scoreThreshold * (layer == stopLayer ? 1.0 : 0.65))
				local.push_back(best);
			}
#pragma omp critical
			{
				next.insert(next.end(), local.begin(), local.end());
			}
		}
		const double layerSpan = std::min((double)std::min(layers_[layer].width, layers_[layer].height),
										  2.0 * (double)modelRadius(mdl.edges));
		const double md = 0.45 * std::max(4.0, layerSpan);
		nmsCandidates(next, md, refineKeepCount(matchParam_.maxCount, quantCoarse));
		refined.swap(next);
	}

	const double fullSpan = std::min((double)std::min(layers_[stopLayer].width, layers_[stopLayer].height),
									 2.0 * (double)modelRadius(layers_[stopLayer].edges));
	nmsCandidates(refined, 0.45 * std::max(4.0, fullSpan),
				  coarseKeepCount(matchParam_.maxCount, quantCoarse));
	const int pyrScale = 1 << stopLayer;
	std::vector<MatchResult> packed;
	packed.reserve(refined.size());
	for (const auto& c : refined)
	{
		if (c.score < matchParam_.scoreThreshold)
			continue;
		packed.push_back(packResult(c.cx * pyrScale, c.cy * pyrScale, c.angle, c.score, templW_, templH_, c.scale));
	}
	BaseMatcher::suppressMatchOverlap(packed, matchParam_.iouThreshold, matchParam_.maxCount, true);
	matchResults.swap(packed);
	return (int)matchResults.size();
}

}  // namespace template_matching
