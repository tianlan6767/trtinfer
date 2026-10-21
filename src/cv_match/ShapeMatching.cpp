#include "ShapeMatching.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
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

// packedXY 不为空时，顺手把单位法向收成一对。打分时每个模型点只碰一块内存。
// 幅值仍单独放着：最细一层的象限权重要用它，不能改计算顺序。
void computeUnitGrad(const cv::Mat& gray, cv::Mat& nx, cv::Mat& ny, cv::Mat& mag, cv::Mat* packedXY = nullptr)
{
	cv::Mat gx, gy;
	cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
	cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
	mag.create(gray.size(), CV_32F);
	nx.create(gray.size(), CV_32F);
	ny.create(gray.size(), CV_32F);
	if (packedXY)
		packedXY->create(gray.size(), CV_32FC2);
	for (int y = 0; y < gray.rows; ++y)
	{
		const float* px = gx.ptr<float>(y);
		const float* py = gy.ptr<float>(y);
		float* pnx = nx.ptr<float>(y);
		float* pny = ny.ptr<float>(y);
		float* pm = mag.ptr<float>(y);
		cv::Vec2f* pack = packedXY ? packedXY->ptr<cv::Vec2f>(y) : nullptr;
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
			if (pack)
				pack[x] = cv::Vec2f(pnx[x], pny[x]);
		}
	}
}

// 出界的模型点直接跳过，不记 0。可见点少于模型的这个比例时，仍用全部点数做分母，
// 避免只擦到画布一条边就被归一化成高分。大约 68% 还在画面里的残件高于这条线。
// 再低就会把贴边的纹理归一化成 0.55 左右的假命中。
// 233 上 T11 的可见比例大约 0.62–0.66，末层分大约 0.53，刚好贴着这条线的上侧。
// 往上提会改用全部点数做分母，0.53 会掉到 0.5 下面。不要往上调。
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

// 调用方传入的是「至少要够到的分数」。贪心再乘一次。已经有更好的位姿时，
// 把门槛抬到那个分数：算不赢就停。停下来的平均只会更低，不会换掉真正的最高分。
double abortMinScore(double floorMin, double bestSoFar, double greed)
{
	double need = floorMin;
	if (bestSoFar > need)
		need = bestSoFar;
	const double g = std::max(0.0, std::min(1.0, greed));
	if (g <= 1e-12)
		return need;
	return need / g;
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

// 响应图在贴边会把旁边的强边收成高分，这条带改回浮点 scorePose。
// 233 探针：T11 顶层 (6.6, 39) 的最近格点 (6, 38) 已经落在带宽 16 里，浮点粗分大约 0.55–0.61。
// 带宽不是漏检原因，不要为了这一件再加宽。
constexpr int kCoarseBorderBand = 16;

bool onCoarseBorder(int x, int y, int rows, int cols, int margin, int band)
{
	if (x < margin + band || y < margin + band)
		return true;
	if (x >= cols - margin - band || y >= rows - margin - band)
		return true;
	return false;
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
// 收到 400 会在纹理图上把贴边弱分挤掉。oriBins=16 再垫一个下限。贴边和内部都用这个数。
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
	// 16 张图叠在一起时，每个模型点要跳到另一块远处的内存。收成每个像素 16 档之后，
	// 一次粗搜只碰模板盖住的那一块。数值与按档分面相同。
	if (w > 0 && h > 0)
	{
		std::vector<float> planar;
		planar.swap(out.data);
		out.data.resize(planar.size());
		const size_t plane = (size_t)w * (size_t)h;
		float* dst = out.data.data();
		const float* src = planar.data();
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				float* pix = dst + ((size_t)y * (size_t)w + (size_t)x) * 16u;
				const size_t pixi = (size_t)y * (size_t)w + (size_t)x;
				for (int b = 0; b < 16; ++b)
					pix[b] = src[(size_t)b * plane + pixi];
			}
		}
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

// 不成对时，nx 和 ny 隔着整幅图。这里每个点只读一行里相邻的两个浮点，计算顺序与分开的两张图相同。
// 不加权幅值：权是 1，乘上去还是原来的点积。提前停的条件和返回的偏低分也不变。
double scorePosePacked(const ShapeEdgePt* pts, int n, const cv::Mat& packed, double cx, double cy,
					   bool polarity, double abortBelow)
{
	if (n <= 0)
		return 0.0;
	const int w = packed.cols, h = packed.rows;
	if (w < 2 || h < 2 || packed.type() != CV_32FC2 || (packed.step % sizeof(float)) != 0)
		return 0.0;
	const float xMax = (float)(w - 1);
	const float yMax = (float)(h - 1);
	const float* base = packed.ptr<float>(0);
	const size_t stride = packed.step / sizeof(float);
	double sum = 0.0;
	int used = 0;
	for (int i = 0; i < n; ++i)
	{
		const float px = (float)(cx + pts[i].dx);
		const float py = (float)(cy + pts[i].dy);
		if (px < 0.f || py < 0.f || px > xMax || py > yMax)
			continue;
		const int x0 = (int)px;
		const int y0 = (int)py;
		const int x1 = x0 + 1 < w ? x0 + 1 : x0;
		const int y1 = y0 + 1 < h ? y0 + 1 : y0;
		const float dx = px - (float)x0;
		const float dy = py - (float)y0;
		const float w00 = (1.f - dx) * (1.f - dy);
		const float w10 = dx * (1.f - dy);
		const float w01 = (1.f - dx) * dy;
		const float w11 = dx * dy;
		const float* r0 = base + (size_t)y0 * stride;
		const float* r1 = base + (size_t)y1 * stride;
		const int i0 = x0 * 2;
		const int i1 = x1 * 2;
		const float inx = w00 * r0[i0] + w10 * r0[i1] + w01 * r1[i0] + w11 * r1[i1];
		const float iny = w00 * r0[i0 + 1] + w10 * r0[i1 + 1] + w01 * r1[i0 + 1] + w11 * r1[i1 + 1];
		// 点积在 float 里加完再送进 double，和分开的 nx、ny 那条路径同一套舍入。
		double raw = (double)(pts[i].nx * inx + pts[i].ny * iny);
		if (!polarity)
			raw = std::fabs(raw);
		else if (raw < 0.0)
			raw = 0.0;
		sum += raw;
		++used;
		if (poseHopeless(sum, used, n, i, abortBelow))
			return sum / n;
	}
	return meanInFrame(sum, used, n);
}

// 同一角度的一批整数中心一起打分。每个中心仍按模型点顺序累加，过线的分数与逐个
// scorePosePacked 相同。到不了阈值的中心这里给出完整分，调用方照旧丢掉。
void scorePosePackedCenters(const ShapeEdgePt* pts, int n, const cv::Mat& packed, const int* xs, const int* ys,
							int npos, bool polarity, double* scores)
{
	if (npos <= 0)
		return;
	if (n <= 0 || packed.cols < 2 || packed.rows < 2 || packed.type() != CV_32FC2 ||
		(packed.step % sizeof(float)) != 0)
	{
		for (int p = 0; p < npos; ++p)
			scores[p] = 0.0;
		return;
	}
	const int w = packed.cols, h = packed.rows;
	const float xMax = (float)(w - 1);
	const float yMax = (float)(h - 1);
	const float* base = packed.ptr<float>(0);
	const size_t stride = packed.step / sizeof(float);
	std::vector<double> sum((size_t)npos, 0.0);
	std::vector<int> used((size_t)npos, 0);
	for (int i = 0; i < n; ++i)
	{
		const double pdx = (double)pts[i].dx;
		const double pdy = (double)pts[i].dy;
		const float pnx = pts[i].nx;
		const float pny = pts[i].ny;
		int p = 0;
		while (p < npos)
		{
			const int y = ys[p];
			int q = p + 1;
			while (q < npos && ys[q] == y)
				++q;
			const float py = (float)((double)y + pdy);
			if (py < 0.f || py > yMax)
			{
				p = q;
				continue;
			}
			const int y0 = (int)py;
			const int y1 = y0 + 1 < h ? y0 + 1 : y0;
			const float dy = py - (float)y0;
			const float wy0 = 1.f - dy;
			const float wy1 = dy;
			const float* r0 = base + (size_t)y0 * stride;
			const float* r1 = base + (size_t)y1 * stride;
			for (int k = p; k < q; ++k)
			{
				const float px = (float)((double)xs[k] + pdx);
				if (px < 0.f || px > xMax)
					continue;
				const int x0 = (int)px;
				const int x1 = x0 + 1 < w ? x0 + 1 : x0;
				const float dx = px - (float)x0;
				const float w00 = (1.f - dx) * wy0;
				const float w10 = dx * wy0;
				const float w01 = (1.f - dx) * wy1;
				const float w11 = dx * wy1;
				const int i0 = x0 * 2;
				const int i1 = x1 * 2;
				const float inx = w00 * r0[i0] + w10 * r0[i1] + w01 * r1[i0] + w11 * r1[i1];
				const float iny = w00 * r0[i0 + 1] + w10 * r0[i1 + 1] + w01 * r1[i0 + 1] + w11 * r1[i1 + 1];
				double raw = (double)(pnx * inx + pny * iny);
				if (!polarity)
					raw = std::fabs(raw);
				else if (raw < 0.0)
					raw = 0.0;
				sum[(size_t)k] += raw;
				used[(size_t)k] += 1;
			}
			p = q;
		}
	}
	for (int p = 0; p < npos; ++p)
		scores[p] = meanInFrame(sum[(size_t)p], used[(size_t)p], n);
}

double scorePose(const std::vector<ShapeEdgePt>& model, const cv::Mat& nx, const cv::Mat& ny,
				 const cv::Mat& mag, float minMag, double cx, double cy, bool polarity, double greediness,
				 double minScore, bool spatialBalance, int pointStep = 1, const cv::Mat* packedXY = nullptr)
{
	const int nAll = (int)model.size();
	// 隔点会把按方向轮流排好的模型拆成一半方向，角度会排反。只用前一半点，每个方向都还在。
	int n = nAll;
	if (pointStep > 1)
		n = std::max(8, nAll / 2);
	if (n > nAll)
		n = nAll;
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
	const bool usePacked = packedXY != nullptr && packedXY->type() == CV_32FC2 &&
						   packedXY->rows == h && packedXY->cols == w &&
						   (packedXY->step % sizeof(float)) == 0;
	// 不看幅值时权恒为 1。走成对法向，点积和提前停与下面这条循环相同。
	if (usePacked && !spatialBalance)
		return scorePosePacked(model.data(), n, *packedXY, cx, cy, polarity, abortBelow);
	const float* packBase = usePacked ? packedXY->ptr<float>(0) : nullptr;
	const size_t packStride = usePacked ? packedXY->step / sizeof(float) : 0;
	const ShapeEdgePt* pts = model.data();
	for (int i = 0; i < n; ++i)
	{
		const float px = (float)(cx + pts[i].dx);
		const float py = (float)(cy + pts[i].dy);
		if (px < 0.f || py < 0.f || px > xMax || py > yMax)
			continue;
		const int x0 = (int)px;
		const int y0 = (int)py;
		const int x1 = x0 + 1 < w ? x0 + 1 : x0;
		const int y1 = y0 + 1 < h ? y0 + 1 : y0;
		const float dx = px - (float)x0;
		const float dy = py - (float)y0;
		const float w00 = (1.f - dx) * (1.f - dy);
		const float w10 = dx * (1.f - dy);
		const float w01 = (1.f - dx) * dy;
		const float w11 = dx * dy;
		float inx = 0.f;
		float iny = 0.f;
		float sm = magGate;
		if (usePacked)
		{
			const float* r0 = packBase + (size_t)y0 * packStride;
			const float* r1 = packBase + (size_t)y1 * packStride;
			const int i0 = x0 * 2;
			const int i1 = x1 * 2;
			inx = w00 * r0[i0] + w10 * r0[i1] + w01 * r1[i0] + w11 * r1[i1];
			iny = w00 * r0[i0 + 1] + w10 * r0[i1 + 1] + w01 * r1[i0 + 1] + w11 * r1[i1 + 1];
		}
		else
		{
			const float* nx0 = nx.ptr<float>(y0);
			const float* nx1 = nx.ptr<float>(y1);
			const float* ny0 = ny.ptr<float>(y0);
			const float* ny1 = ny.ptr<float>(y1);
			inx = w00 * nx0[x0] + w10 * nx0[x1] + w01 * nx1[x0] + w11 * nx1[x1];
			iny = w00 * ny0[x0] + w10 * ny0[x1] + w01 * ny1[x0] + w11 * ny1[x1];
		}
		if (spatialBalance)
		{
			const float* m0 = mag.ptr<float>(y0);
			const float* m1 = mag.ptr<float>(y1);
			sm = w00 * m0[x0] + w10 * m0[x1] + w01 * m1[x0] + w11 * m1[x1];
		}
		float wt = std::min(1.f, sm / magGate);
		double raw = pts[i].nx * inx + pts[i].ny * iny;
		if (!polarity)
			raw = std::fabs(raw);
		else if (raw < 0.0)
			raw = 0.0;  // 与 16 档余弦表、Halcon use_polarity 一致：反方向记 0，不把整段分数拉成负数
		const double d = raw * (double)wt;
		sum += d;
		++used;
		if (spatialBalance)
		{
			const int q = (pts[i].dx >= 0.f ? 1 : 0) + (pts[i].dy >= 0.f ? 2 : 0);
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
	const float* base = resp.data.data();
	double sum = 0.0;
	int used = 0;
	const double abortBelow = minScore * std::max(0.0, std::min(1.0, greediness));
	for (int i = 0; i < n; ++i)
	{
		// 先按浮点位置判断出界。lround 会把 [-0.5,0) 收成 0，把裁掉的轮廓当成画面里的点。
		const double fx = cx + (double)model[i].dx;
		const double fy = cy + (double)model[i].dy;
		if (fx < 0.0 || fy < 0.0 || fx > (double)(w - 1) || fy > (double)(h - 1))
			continue;
		// 负数已经在上面丢掉。非负时 (int)(x+0.5) 与 lround 同为远离 0 的一侧，但不会进 libm。
		const int px = (int)(fx + 0.5);
		const int py = (int)(fy + 0.5);
		if ((unsigned)px >= (unsigned)w || (unsigned)py >= (unsigned)h)
			continue;
		float best = 0.f;
		const int ml = model[i].bin;
		if ((unsigned)ml < 16u)
			best = base[(((size_t)py * (size_t)w + (size_t)px) << 4) + (size_t)ml];
		sum += (double)best;
		++used;
		// 过线的位置占大多数时，每个点都做无望判断不划算。隔几个点查一次，结论不变。
		if ((i & 3) == 3 && poseHopeless(sum, used, n, i, abortBelow))
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

// 分数相同就按位置、角度、尺度排死。std::sort 不稳定，OpenMP 合并顺序一变，
// 平台上谁当种子就跟着变，去重个数会在 9 和 10 之间跳。
bool candidatePrecedes(const ShapeCandidate& a, const ShapeCandidate& b)
{
	if (a.score != b.score)
		return a.score > b.score;
	if (a.cy != b.cy)
		return a.cy < b.cy;
	if (a.cx != b.cx)
		return a.cx < b.cx;
	if (a.angle != b.angle)
		return a.angle < b.angle;
	if (a.scale != b.scale)
		return a.scale < b.scale;
	if (a.fromBorder != b.fromBorder)
		return !a.fromBorder && b.fromBorder;
	return false;
}

void nmsCandidates(std::vector<ShapeCandidate>& cands, double minDist, int maxCount, bool blendTies = false,
				   double blendRadius = 1e9)
{
	std::sort(cands.begin(), cands.end(), candidatePrecedes);
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
	// 格子边长等于去重半径。严格小于 minDist 的种子一定落在周围 3×3 里。
	// 名额放到几千之后，逐个和全部已留种子比会把纹理图的去重拖回秒级。
	const double cell = std::max(minDist, 1e-6);
	struct CellKey
	{
		int x;
		int y;
		bool operator==(const CellKey& o) const { return x == o.x && y == o.y; }
	};
	struct CellHash
	{
		size_t operator()(const CellKey& k) const
		{
			return (std::hash<int>{}(k.x) * 1315423911u) ^ std::hash<int>{}(k.y);
		}
	};
	std::unordered_map<CellKey, std::vector<int>, CellHash> grid;
	for (const auto& c : cands)
	{
		int match = -1;
		const int ix = (int)std::floor(c.cx / cell);
		const int iy = (int)std::floor(c.cy / cell);
		for (int oy = -1; oy <= 1; ++oy)
		{
			for (int ox = -1; ox <= 1; ++ox)
			{
				const auto it = grid.find(CellKey{ix + ox, iy + oy});
				if (it == grid.end())
					continue;
				for (int id : it->second)
				{
					// 用种子位置，不用挪过的质心，避免两个挨着的实例被收成一个。
					// 两点都小于半径时留更早的那个，和按名次线性扫一致。
					if (std::hypot(c.cx - kept[id].seedX, c.cy - kept[id].seedY) < minDist &&
						(match < 0 || id < match))
						match = id;
				}
			}
		}
		if (match >= 0)
		{
			Acc& a = kept[match];
			// 响应图把一片格点都抬到同一分上，最高的那一格经常在平台边上。
			// 几乎同分的邻居一起算质心。角度只平均离种子约一步的，
			// 避免把另一侧的高分角度卷进圆周平均。
			// 只把平台内部的同分邻居收成质心。更远的高分格点仍按去重丢掉，
			// 但不拉质心，避免两个挨着的实例被收到中间、两边都对不上。
			if (blendTies && c.score >= a.c.score - 1e-4 &&
				std::hypot(c.cx - a.seedX, c.cy - a.seedY) <= blendRadius)
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
		grid[CellKey{ix, iy}].push_back((int)kept.size());
		kept.push_back(a);
	}
	cands.clear();
	cands.reserve(kept.size());
	for (const auto& a : kept)
		cands.push_back(a.c);
}

// 内部响应图上，弱真峰常常落在更高假峰的整幅去重半径里（大约 8～16 个顶层像素）。
// 233 上缺的 (318,180) 和 (656,134) 就是这种：响应图假峰更高，真峰被删，假峰到末层又不到 0.5。
// 扩散把一片格点抬成几乎同分，按响应图分数挑环带会挑中假峰尾巴。这里只列出环带上的候选，
// 调用方再用浮点 scorePose 决定留哪一个。贴边名单不走这里。
// 同分平台只在大约 4px 内收质心。整幅半径都收，会把两个挨着的峰收到中间，两边都对不上。
constexpr double kInteriorRescueMin = 8.0;
constexpr double kInteriorBlendRadius = 4.0;
constexpr int kRingSectors = 8;
constexpr int kMaxInteriorRescue = 24;

struct RingPick
{
	ShapeCandidate c;
	int seed = -1;
};

void nmsInteriorRescue(std::vector<ShapeCandidate>& cands, double minDist, int maxCount,
					   std::vector<RingPick>& picks)
{
	// 同一像素上只有角度不同，距离是 0，去重也只会留最高的那个。
	// 先收成每个像素一个再排序，避免纹理图上几十万条候选把排序拖到几十毫秒。
	// 尺度不是 1 时不收，避免把同一中心的另一个尺度删掉。
	{
		bool unitScale = true;
		int maxX = 1;
		int maxY = 1;
		for (const auto& c : cands)
		{
			if (std::fabs(c.scale - 1.0) > 1e-6)
				unitScale = false;
			maxX = std::max(maxX, (int)std::lround(c.cx) + 1);
			maxY = std::max(maxY, (int)std::lround(c.cy) + 1);
		}
		if (unitScale && maxX > 1 && maxY > 1 && (long long)maxX * maxY <= 2000000ll)
		{
			std::vector<int> bestPix((size_t)maxX * (size_t)maxY, -1);
			std::vector<ShapeCandidate> extra;
			for (int i = 0; i < (int)cands.size(); ++i)
			{
				const int x = (int)std::lround(cands[i].cx);
				const int y = (int)std::lround(cands[i].cy);
				if (x < 0 || y < 0 || x >= maxX || y >= maxY)
				{
					extra.push_back(cands[i]);
					continue;
				}
				int& slot = bestPix[(size_t)y * (size_t)maxX + (size_t)x];
				if (slot < 0 || candidatePrecedes(cands[i], cands[slot]))
					slot = i;
			}
			std::vector<ShapeCandidate> uniq;
			uniq.reserve(extra.size() + 1024);
			uniq.insert(uniq.end(), extra.begin(), extra.end());
			for (int slot : bestPix)
			{
				if (slot >= 0)
					uniq.push_back(cands[slot]);
			}
			cands.swap(uniq);
		}
	}
	std::sort(cands.begin(), cands.end(), candidatePrecedes);
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
	const double cell = std::max(minDist, 1e-6);
	struct CellKey
	{
		int x;
		int y;
		bool operator==(const CellKey& o) const { return x == o.x && y == o.y; }
	};
	struct CellHash
	{
		size_t operator()(const CellKey& k) const
		{
			return (std::hash<int>{}(k.x) * 1315423911u) ^ std::hash<int>{}(k.y);
		}
	};
	std::unordered_map<CellKey, std::vector<int>, CellHash> grid;
	for (const auto& c : cands)
	{
		int match = -1;
		const int ix = (int)std::floor(c.cx / cell);
		const int iy = (int)std::floor(c.cy / cell);
		for (int oy = -1; oy <= 1; ++oy)
		{
			for (int ox = -1; ox <= 1; ++ox)
			{
				const auto it = grid.find(CellKey{ix + ox, iy + oy});
				if (it == grid.end())
					continue;
				for (int id : it->second)
				{
					if (std::hypot(c.cx - kept[id].seedX, c.cy - kept[id].seedY) < minDist &&
						(match < 0 || id < match))
						match = id;
				}
			}
		}
		if (match >= 0)
		{
			Acc& a = kept[match];
			const double tieDist = std::hypot(c.cx - a.seedX, c.cy - a.seedY);
			if (c.score >= a.c.score - 1e-4 && tieDist <= kInteriorBlendRadius)
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
		grid[CellKey{ix, iy}].push_back((int)kept.size());
		kept.push_back(a);
	}

	// 每个像素只留排序后的第一名。cands 已按 candidatePrecedes 排过，先出现的就是该像素最高分。
	// 每个种子的环带再切成 8 个方向、两段距离。每格留下响应图最高的和更远的那个。
	// 4 个方向时 (318,180) 落在假峰尾巴同一格里，后面无论怎么打分都看不到它。
	struct PixKey
	{
		int x;
		int y;
		bool operator==(const PixKey& o) const { return x == o.x && y == o.y; }
	};
	struct PixHash
	{
		size_t operator()(const PixKey& k) const
		{
			return (std::hash<int>{}(k.x) * 1315423911u) ^ std::hash<int>{}(k.y);
		}
	};
	std::unordered_map<PixKey, int, PixHash> bestAt;
	bestAt.reserve(std::min(cands.size(), (size_t)4096));
	for (int i = 0; i < (int)cands.size(); ++i)
	{
		if (cands[i].score < 0.42)
			break;
		const PixKey pk{(int)std::lround(cands[i].cx), (int)std::lround(cands[i].cy)};
		if (!bestAt.count(pk))
			bestAt.emplace(pk, i);
	}
	struct Slot
	{
		int bestIdx = -1;
		int farIdx = -1;
		double farDist = 0;
	};
	const int bins = kRingSectors * 2;
	std::vector<Slot> slots((size_t)kept.size() * (size_t)bins);
	for (const auto& kv : bestAt)
	{
		const int idx = kv.second;
		const ShapeCandidate& c = cands[idx];
		int match = -1;
		double matchDist = 0;
		const int ix = (int)std::floor(c.cx / cell);
		const int iy = (int)std::floor(c.cy / cell);
		for (int oy = -1; oy <= 1; ++oy)
		{
			for (int ox = -1; ox <= 1; ++ox)
			{
				const auto it = grid.find(CellKey{ix + ox, iy + oy});
				if (it == grid.end())
					continue;
				for (int id : it->second)
				{
					const double dist = std::hypot(c.cx - kept[id].seedX, c.cy - kept[id].seedY);
					if (dist >= kInteriorRescueMin && dist < minDist && (match < 0 || id < match))
					{
						match = id;
						matchDist = dist;
					}
				}
			}
		}
		if (match < 0)
			continue;
		double ang = std::atan2(c.cy - kept[match].seedY, c.cx - kept[match].seedX);
		int sector = (int)std::floor((ang + kPi) / (2.0 * kPi) * kRingSectors);
		if (sector < 0)
			sector = 0;
		if (sector >= kRingSectors)
			sector = kRingSectors - 1;
		const int band = matchDist < 12.0 ? 0 : 1;
		Slot& s = slots[(size_t)match * (size_t)bins + (size_t)(band * kRingSectors + sector)];
		if (s.bestIdx < 0 || candidatePrecedes(c, cands[s.bestIdx]))
			s.bestIdx = idx;
		if (s.farIdx < 0 || matchDist > s.farDist + 1e-9 ||
			(std::fabs(matchDist - s.farDist) <= 1e-9 && candidatePrecedes(c, cands[s.farIdx])))
		{
			s.farIdx = idx;
			s.farDist = matchDist;
		}
	}
	picks.clear();
	picks.reserve(kept.size() * 4);
	for (int id = 0; id < (int)kept.size(); ++id)
	{
		for (int b = 0; b < bins; ++b)
		{
			const Slot& s = slots[(size_t)id * (size_t)bins + (size_t)b];
			if (s.bestIdx >= 0)
			{
				RingPick p;
				p.c = cands[s.bestIdx];
				p.c.fromBorder = false;
				p.c.narrowRefine = true;
				p.seed = id;
				picks.push_back(p);
			}
			if (s.farIdx >= 0 && s.farIdx != s.bestIdx)
			{
				RingPick p;
				p.c = cands[s.farIdx];
				p.c.fromBorder = false;
				p.c.narrowRefine = true;
				p.seed = id;
				picks.push_back(p);
			}
		}
	}
	cands.clear();
	cands.reserve(kept.size());
	for (const auto& a : kept)
		cands.push_back(a.c);
}

// 贴边种子和内部种子分开去重。内部响应图峰可以在下一层浮点分掉到阈值下，
// 但不能在那之前把贴边残件从名单里抹掉。
void nmsKeepOrigins(std::vector<ShapeCandidate>& cands, double minDist, int maxCount, bool blendInterior,
					double blendRadius = 1e9, bool preserveNarrow = false)
{
	std::vector<ShapeCandidate> border;
	std::vector<ShapeCandidate> interior;
	border.reserve(cands.size() / 8 + 1);
	interior.reserve(cands.size());
	for (const auto& c : cands)
	{
		if (c.fromBorder)
			border.push_back(c);
		else
			interior.push_back(c);
	}
	// 中间层的浮点分还没落到末层。假峰这时仍可能高于真峰，整幅半径会把刚捞回的内部种子再删掉。
	// 只把 narrowRefine 留到末层，末层再用浮点分做整幅去重。贴边名单不走这条。
	std::vector<ShapeCandidate> pinned;
	if (preserveNarrow)
	{
		for (const auto& c : interior)
		{
			if (c.narrowRefine)
				pinned.push_back(c);
		}
	}
	nmsCandidates(interior, minDist, maxCount, blendInterior, blendRadius);
	if (preserveNarrow)
	{
		for (const auto& r : pinned)
		{
			bool present = false;
			for (const auto& k : interior)
			{
				if (k.cx == r.cx && k.cy == r.cy && k.angle == r.angle && k.scale == r.scale)
				{
					present = true;
					break;
				}
			}
			if (!present)
				interior.push_back(r);
		}
	}
	nmsCandidates(border, minDist, maxCount, false);
	cands.clear();
	cands.reserve(interior.size() + border.size());
	cands.insert(cands.end(), interior.begin(), interior.end());
	cands.insert(cands.end(), border.begin(), border.end());
}

double boxMinSide(const MatchResult& r)
{
	const double w = std::hypot(r.RightTop.x - r.LeftTop.x, r.RightTop.y - r.LeftTop.y);
	const double h = std::hypot(r.LeftBottom.x - r.LeftTop.x, r.LeftBottom.y - r.LeftTop.y);
	return std::max(1.0, std::min(w, h));
}

// 内部峰和贴边峰重叠时留下贴边。dropWeakNearby 会把 0.53 的 T11 让给旁边更高的内部假峰。
bool interiorConflictsBorder(const MatchResult& interior, const MatchResult& border)
{
	const double minSide = std::min(boxMinSide(interior), boxMinSide(border));
	const double dist = std::hypot(interior.Center.x - border.Center.x, interior.Center.y - border.Center.y);
	if (dist < 0.40 * minSide)
		return true;
	if (dist < 0.65 * minSide && border.Score < 0.78 && border.Score < interior.Score * 0.74)
		return true;
	return false;
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
	std::vector<cv::Mat> nx(pyr.size()), ny(pyr.size()), mag(pyr.size()), packed(pyr.size());
	for (int i = 0; i < (int)pyr.size(); ++i)
		computeUnitGrad(pyr[i], nx[i], ny[i], mag[i], &packed[i]);
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
		// 贴边格子与角度无关。同一角度把这些中心合在一起打，浮点分仍按模型点顺序累加。
		std::vector<int> borderXs, borderYs, innerXs, innerYs;
		if (quantCoarse)
		{
			for (int y = margin; y < pyr[top].rows - margin; y += stride)
			{
				for (int x = margin; x < pyr[top].cols - margin; x += stride)
				{
					if (onCoarseBorder(x, y, pyr[top].rows, pyr[top].cols, margin, kCoarseBorderBand))
					{
						borderXs.push_back(x);
						borderYs.push_back(y);
					}
					else
					{
						innerXs.push_back(x);
						innerYs.push_back(y);
					}
				}
			}
		}
		// 每个角度写自己的桶，再按角度下标拼起来。critical 按线程完成顺序拼接，同分平台的种子会变。
		std::vector<std::vector<ShapeCandidate>> buckets(angles.size());
#pragma omp parallel for schedule(dynamic)
		for (int ai = 0; ai < (int)angles.size(); ++ai)
		{
			std::vector<ShapeCandidate> local;
			const auto model = rotateModel(topModel.edges, angles[ai]);
			// 内部每隔一个角度查响应图。角度误差最多大约一步，第一层 ±12° 接得回来。
			// 贴边每个角度都用浮点分，T11 的 ±3 步窗口接不住被拉大的角度步长。
			// 响应图在裁掉的轮廓旁边会把任意强边收成高分，不能拿它当贴边的入选分数。
			const bool interiorAngle = (ai & 1) == 0;
			if (!quantCoarse)
			{
				for (int y = margin; y < pyr[top].rows - margin; y += stride)
				{
					for (int x = margin; x < pyr[top].cols - margin; x += stride)
					{
						const double sc = scorePose(model, nx[top], ny[top], mag[top], (float)matchParam_.edgeMinMag,
													 x, y, polarity, greed, abortMinScore(coarseTh, -1.0, greed),
													 false, 1, &packed[top]);
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
			else
			{
				if (interiorAngle)
				{
					const std::vector<MapPoint> mapPts = makeMapPoints(model);
					for (int p = 0; p < (int)innerXs.size(); ++p)
					{
						const double sc = scorePoseMap(mapPts, respTop, innerXs[p], innerYs[p], greed,
														abortMinScore(coarseTh, -1.0, greed));
						if (sc < coarseTh)
							continue;
						ShapeCandidate c;
						c.cx = innerXs[p];
						c.cy = innerYs[p];
						c.angle = angles[ai];
						c.score = sc;
						local.push_back(c);
					}
				}
				if (!borderXs.empty())
				{
					std::vector<double> bsc(borderXs.size());
					scorePosePackedCenters(model.data(), (int)model.size(), packed[top], borderXs.data(),
										   borderYs.data(), (int)borderXs.size(), polarity, bsc.data());
					for (int p = 0; p < (int)borderXs.size(); ++p)
					{
						if (bsc[(size_t)p] < coarseTh)
							continue;
						ShapeCandidate c;
						c.cx = borderXs[p];
						c.cy = borderYs[p];
						c.angle = angles[ai];
						c.score = bsc[(size_t)p];
						c.fromBorder = true;
						local.push_back(c);
					}
				}
			}
			buckets[ai].swap(local);
		}
		for (auto& b : buckets)
			coarse.insert(coarse.end(), b.begin(), b.end());
	}
	else
	{
		int scaleStride = std::max(1, std::min(tw, th) / 12);
		if (scaleStride > 4)
			scaleStride = 4;
		if (quantCoarse)
			scaleStride = kMapStride;
		const int mgn = borderMargin(pyr[top].rows, pyr[top].cols, scaleStride);
		std::vector<int> borderXs, borderYs, innerXs, innerYs;
		if (quantCoarse)
		{
			for (int y = mgn; y < pyr[top].rows - mgn; y += scaleStride)
			{
				for (int x = mgn; x < pyr[top].cols - mgn; x += scaleStride)
				{
					if (onCoarseBorder(x, y, pyr[top].rows, pyr[top].cols, mgn, kCoarseBorderBand))
					{
						borderXs.push_back(x);
						borderYs.push_back(y);
					}
					else
					{
						innerXs.push_back(x);
						innerYs.push_back(y);
					}
				}
			}
		}
		std::vector<std::pair<int, int>> jobs;
		jobs.reserve(angles.size() * scales_.size());
		for (int ai = 0; ai < (int)angles.size(); ++ai)
			for (int si = 0; si < (int)scales_.size(); ++si)
				jobs.emplace_back(ai, si);
		std::vector<std::vector<ShapeCandidate>> buckets(jobs.size());
#pragma omp parallel for schedule(dynamic)
		for (int ji = 0; ji < (int)jobs.size(); ++ji)
		{
			std::vector<ShapeCandidate> local;
			const int ai = jobs[ji].first;
			const double objScale = scales_[jobs[ji].second];
			const auto model = scaleEdges(rotateModel(topModel.edges, angles[ai]), objScale);
			const bool interiorAngle = (ai & 1) == 0;
			if (!quantCoarse)
			{
				for (int y = mgn; y < pyr[top].rows - mgn; y += scaleStride)
				{
					for (int x = mgn; x < pyr[top].cols - mgn; x += scaleStride)
					{
						const double sc = scorePose(model, nx[top], ny[top], mag[top], (float)matchParam_.edgeMinMag,
													 x, y, polarity, greed, abortMinScore(coarseTh, -1.0, greed),
													 false, 1, &packed[top]);
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
			else
			{
				if (interiorAngle)
				{
					const std::vector<MapPoint> mapPts = makeMapPoints(model);
					for (int p = 0; p < (int)innerXs.size(); ++p)
					{
						const double sc = scorePoseMap(mapPts, respTop, innerXs[p], innerYs[p], greed,
														abortMinScore(coarseTh, -1.0, greed));
						if (sc < coarseTh)
							continue;
						ShapeCandidate c;
						c.cx = innerXs[p];
						c.cy = innerYs[p];
						c.angle = angles[ai];
						c.score = sc;
						c.scale = objScale;
						local.push_back(c);
					}
				}
				if (!borderXs.empty())
				{
					std::vector<double> bsc(borderXs.size());
					scorePosePackedCenters(model.data(), (int)model.size(), packed[top], borderXs.data(),
										   borderYs.data(), (int)borderXs.size(), polarity, bsc.data());
					for (int p = 0; p < (int)borderXs.size(); ++p)
					{
						if (bsc[(size_t)p] < coarseTh)
							continue;
						ShapeCandidate c;
						c.cx = borderXs[p];
						c.cy = borderYs[p];
						c.angle = angles[ai];
						c.score = bsc[(size_t)p];
						c.scale = objScale;
						c.fromBorder = true;
						local.push_back(c);
					}
				}
			}
			buckets[ji].swap(local);
		}
		for (auto& b : buckets)
			coarse.insert(coarse.end(), b.begin(), b.end());
	}

	const double topSpan = std::min((double)std::min(tw, th), 2.0 * (double)maxR);
	const double minDistTop = 0.45 * std::max(4.0, topSpan);
	if (quantCoarse)
	{
		// 贴边去重与 743fd21 相同：整幅半径、不收质心、不捞环带。
		// 内部在同样的整幅半径里多留环带上的第二个峰，数量封顶，避免把第一层 ±10 px 撑满。
		std::vector<ShapeCandidate> border;
		std::vector<ShapeCandidate> interior;
		border.reserve(coarse.size() / 8 + 1);
		interior.reserve(coarse.size());
		for (const auto& c : coarse)
		{
			if (c.fromBorder)
				border.push_back(c);
			else
				interior.push_back(c);
		}
		const int keep = coarseKeepCount(matchParam_.maxCount, true);
		std::vector<RingPick> ringPicks;
		nmsInteriorRescue(interior, minDistTop, keep, ringPicks);
		if (!ringPicks.empty())
		{
			// 隔点模型会把 (318,180) 排到假峰尾巴后面，完整浮点分只打了错误的那一个。
			// 环带上每个候选都用完整模型打分，再按种子留最高的。贴边不在这份名单里。
			std::unordered_map<long long, int> angleSlot;
			std::vector<std::vector<ShapeEdgePt>> fullModels;
			std::vector<int> pickModel(ringPicks.size(), -1);
			for (int i = 0; i < (int)ringPicks.size(); ++i)
			{
				const long long key = (long long)std::llround(ringPicks[i].c.angle * 1000.0);
				auto it = angleSlot.find(key);
				if (it == angleSlot.end())
				{
					const int id = (int)fullModels.size();
					fullModels.push_back(rotateModel(topModel.edges, ringPicks[i].c.angle));
					it = angleSlot.emplace(key, id).first;
				}
				pickModel[i] = it->second;
			}
			std::vector<double> floatScore(ringPicks.size(), 0.0);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < (int)ringPicks.size(); ++i)
			{
				floatScore[i] = scorePose(fullModels[pickModel[i]], nx[top], ny[top], mag[top],
										  (float)matchParam_.edgeMinMag, ringPicks[i].c.cx, ringPicks[i].c.cy,
										  polarity, greed, abortMinScore(coarseTh, -1.0, greed), false, 1,
										  &packed[top]);
			}
			int maxSeed = 0;
			for (const auto& p : ringPicks)
				maxSeed = std::max(maxSeed, p.seed);
			std::vector<int> best(maxSeed + 1, -1);
			for (int i = 0; i < (int)ringPicks.size(); ++i)
			{
				if (floatScore[i] < coarseTh)
					continue;
				const int s = ringPicks[i].seed;
				if (best[s] < 0 || floatScore[i] > floatScore[best[s]] ||
					(floatScore[i] == floatScore[best[s]] && candidatePrecedes(ringPicks[i].c, ringPicks[best[s]].c)))
					best[s] = i;
			}
			std::vector<ShapeCandidate> rings;
			rings.reserve(best.size());
			for (int b : best)
			{
				if (b < 0)
					continue;
				ShapeCandidate ring = ringPicks[b].c;
				ring.score = floatScore[b];
				ring.fromBorder = false;
				ring.narrowRefine = true;
				rings.push_back(ring);
			}
			std::sort(rings.begin(), rings.end(), candidatePrecedes);
			if ((int)rings.size() > kMaxInteriorRescue)
				rings.resize(kMaxInteriorRescue);
			interior.insert(interior.end(), rings.begin(), rings.end());
		}
		nmsCandidates(border, minDistTop, keep, false);
		coarse.clear();
		coarse.reserve(interior.size() + border.size());
		coarse.insert(coarse.end(), interior.begin(), interior.end());
		coarse.insert(coarse.end(), border.begin(), border.end());
	}
	else
		nmsCandidates(coarse, minDistTop, coarseKeepCount(matchParam_.maxCount, false), false);

	std::vector<ShapeCandidate> refined = coarse;
	for (int layer = top - 1; layer >= stopLayer; --layer)
	{
		std::vector<ShapeCandidate> next;
		const auto& mdl = layers_[layer];
		double step = angleStepDeg(mdl.edges);
		// 贴边第一层必须是 ±4 px、±3 步。233 上 ±10 px 会把 T11 拖离 (25, 159)。
		// 内部响应图种子第一层仍是 ±10 px、大约 ±12°，和 8e6638d 一样。
		std::vector<ShapeCandidate> slots(refined.size());
		std::vector<char> slotOk(refined.size(), 0);
#pragma omp parallel for schedule(dynamic, 2)
		for (int ci = 0; ci < (int)refined.size(); ++ci)
		{
			const auto& c0 = refined[ci];
			const double cx0 = c0.cx * 2.0;
			const double cy0 = c0.cy * 2.0;
			double bestS = -1;
			ShapeCandidate best = c0;
			best.cx = cx0;
			best.cy = cy0;
			// 贴边恒为 ±4。普通内部种子先在 ±4 里找；这里已经过 0.5 就不再往外扩，
			// 避免 233 上 (318,180)、(656,134) 这类真峰被 ±10 拖到旁边的假峰上。
			// 内圈不够 0.5 时再扩到 ±10，接响应图平台边上的种子。环带捞回的种子只用 ±4。
			// 贴边第一层保持逐像素、逐角度。内部第一层用步长 2 的网格再在胜者周围补 ±1，
			// 角度也隔一个再补相邻一步。T11 不走这里。
			const bool interiorFirst = quantCoarse && layer == top - 1 && !c0.fromBorder;
			const bool widenPos = interiorFirst && !c0.narrowRefine;
			std::vector<double> angs;
			if (matchParam_.angle < 1e-6)
				angs.push_back(0.0);
			else
			{
				// 响应图在只露出大约 2/3 的件上，粗分最高的角度可以偏到真角一侧约 10°。
				// 第一层用浮点分把这个偏差找回来；贴边浮点种子和更下面的层仍是 ±3 步。
				const int kMax = (quantCoarse && layer == top - 1 && !c0.fromBorder)
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
			// 种子角度、中心位置先算。后面的位姿只要不可能超过当前最高分就停。
			// 内部第一层仍跳过原来的奇数角度下标。贴边不跳，窗口大小也不变。
			std::vector<int> angOrder;
			angOrder.reserve(angs.size());
			int seedI = 0;
			double seedDist = 1e9;
			for (int ia = 0; ia < (int)angs.size(); ++ia)
			{
				const double d = std::fabs(angs[ia] - c0.angle);
				if (d < seedDist)
				{
					seedDist = d;
					seedI = ia;
				}
			}
			auto angleKept = [&](int ia) {
				return !(interiorFirst && (int)angs.size() > 8 && (ia & 1) &&
						 std::fabs(angs[ia] - c0.angle) > 1e-6);
			};
			if (angleKept(seedI))
				angOrder.push_back(seedI);
			for (int ia = 0; ia < (int)angs.size(); ++ia)
			{
				if (ia != seedI && angleKept(ia))
					angOrder.push_back(ia);
			}
			// 只加快内部精修。贴边仍逐点打分，弱残件（T11 大约 0.53）用前一半点会排丢。
			// 模型点按方向轮流排，取前一半时每个方向都还在。隔开的几个高分再用完整模型重打。
			const bool fastInterior = quantCoarse && !c0.fromBorder && (int)mdl.edges.size() >= 16;
			const int searchStep = fastInterior ? 2 : 1;
			struct SubPeak
			{
				double score = -1;
				double cx = 0;
				double cy = 0;
				double ang = 0;
				double scale = 1;
			};
			SubPeak peaks[8];
			int nPeaks = 0;
			auto consider = [&](const std::vector<ShapeEdgePt>& model, double ang, double cx, double cy,
								double objScale) {
				if (cx < 0 || cy < 0 || cx > pyr[layer].cols - 1 || cy > pyr[layer].rows - 1)
					return;
				// 内部只为了保住隔开的几个高分。3px 内要超过那一个峰；名额满了要超过最低的那个。
				// 用全局最低去挡旁边的弱峰会把 0.4 的真位置丢掉。门槛乘过贪心之后，提前返回的偏低分
				// 严格低于要超过的峰，算得过的位姿仍会把点打完。贴边仍对着当前最高分停。
				int slot = -1;
				double rival = -1.0;
				if (fastInterior)
				{
					for (int k = 0; k < nPeaks; ++k)
					{
						const double ddx = peaks[k].cx - cx;
						const double ddy = peaks[k].cy - cy;
						if (ddx * ddx + ddy * ddy <= 9.0)
						{
							slot = k;
							break;
						}
					}
					if (slot >= 0)
						rival = peaks[slot].score;
					else if (nPeaks >= 8)
					{
						rival = peaks[0].score;
						for (int k = 1; k < nPeaks; ++k)
						{
							if (peaks[k].score < rival)
								rival = peaks[k].score;
						}
					}
				}
				double bestForAbort = fastInterior ? -1.0 : bestS;
				if (rival >= 0.0)
				{
					const double g = std::max(1e-12, std::min(1.0, greed));
					if (rival >= (matchParam_.scoreThreshold * 0.6) / g)
						bestForAbort = rival * g;
				}
				const double sc = scorePose(model, nx[layer], ny[layer], mag[layer], (float)matchParam_.edgeMinMag,
											cx, cy, polarity, greed,
											abortMinScore(matchParam_.scoreThreshold * 0.6, bestForAbort, greed),
											fastInterior ? false : layer == stopLayer, searchStep, &packed[layer]);
				if (!fastInterior)
				{
					if (sc > bestS)
					{
						bestS = sc;
						best.cx = cx;
						best.cy = cy;
						best.angle = ang;
						best.score = sc;
						if (scaleSearch_)
							best.scale = objScale;
					}
					return;
				}
				if (sc < 0.0)
					return;
				if (slot >= 0)
				{
					if (sc > peaks[slot].score)
					{
						peaks[slot].score = sc;
						peaks[slot].cx = cx;
						peaks[slot].cy = cy;
						peaks[slot].ang = ang;
						peaks[slot].scale = objScale;
					}
					return;
				}
				if (nPeaks < 8)
				{
					peaks[nPeaks].score = sc;
					peaks[nPeaks].cx = cx;
					peaks[nPeaks].cy = cy;
					peaks[nPeaks].ang = ang;
					peaks[nPeaks].scale = objScale;
					++nPeaks;
					return;
				}
				int worst = 0;
				for (int k = 1; k < nPeaks; ++k)
				{
					if (peaks[k].score < peaks[worst].score)
						worst = k;
				}
				if (sc > peaks[worst].score)
				{
					peaks[worst].score = sc;
					peaks[worst].cx = cx;
					peaks[worst].cy = cy;
					peaks[worst].ang = ang;
					peaks[worst].scale = objScale;
				}
			};
			// 末层中心的完整分低于 0.18 时，±4 窗口到不了 0.5。弱贴边残件的中心仍高于这条线。
			// 只用于 ori16、阈值 0.5。更低的阈值，或全程浮点，中心偏低的种子仍可能在邻格过线。
			bool poorCenter = false;
			if (layer == stopLayer && quantCoarse && matchParam_.scoreThreshold >= 0.5)
			{
				const auto probeRot = rotateModel(mdl.edges, c0.angle);
				const auto probeModel = scaleSearch_ ? scaleEdges(probeRot, c0.scale) : probeRot;
				const double probe = scorePose(probeModel, nx[layer], ny[layer], mag[layer],
											   (float)matchParam_.edgeMinMag, cx0, cy0, polarity, greed,
											   abortMinScore(0.18, -1.0, greed), true, 1, &packed[layer]);
				if (probe < 0.18)
				{
					poorCenter = true;
					bestS = probe;
					best.score = probe;
					best.cx = cx0;
					best.cy = cy0;
					best.angle = c0.angle;
					if (scaleSearch_)
						best.scale = c0.scale;
				}
			}
			for (int pass = 0; pass < (poorCenter ? 0 : (widenPos ? 2 : 1)); ++pass)
			{
			if (pass == 1 && bestS >= matchParam_.scoreThreshold)
				break;
			// ±4 且角度已经铺开之后仍低于粗阈值，±10 里不会突然变成合格命中。
			// 贴边不进这圈。环带种子 narrowRefine 也不会把位置扩到 ±10。
			if (pass == 1 && bestS < matchParam_.scoreThreshold * 0.6)
				break;
			const int rad = (pass == 1) ? 10 : 4;
			const int posStep = interiorFirst ? 2 : 1;
			if (!scaleSearch_)
			{
				for (int ia : angOrder)
				{
					const double ang = angs[ia];
					// ±4 已经选出角度。外圈只在这个角度附近补位置，避免每个外点再扫一整段角。
					if (pass == 1 && std::fabs(ang - best.angle) > step * 2.0 + 1e-6 &&
						std::fabs(ang - c0.angle) > 1e-6)
						continue;
					const auto model = rotateModel(mdl.edges, ang);
					// 第二圈只补 ±4 外面。里面上一圈已经用浮点分留下最高的。
					if (pass == 0)
						consider(model, ang, cx0, cy0, c0.scale);
					for (int dy = -rad; dy <= rad; dy += posStep)
					{
						for (int dx = -rad; dx <= rad; dx += posStep)
						{
							if (dx == 0 && dy == 0)
								continue;
							if (pass == 1 && std::abs(dx) <= 4 && std::abs(dy) <= 4)
								continue;
							consider(model, ang, cx0 + dx, cy0 + dy, c0.scale);
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
				const int radS = rad;
				for (int ia : angOrder)
				{
					const double ang = angs[ia];
					if (pass == 1 && std::fabs(ang - best.angle) > step * 2.0 + 1e-6 &&
						std::fabs(ang - c0.angle) > 1e-6)
						continue;
					const auto rotated = rotateModel(mdl.edges, ang);
					for (double objScale : tryScales)
					{
						const auto model = scaleEdges(rotated, objScale);
						if (pass == 0)
							consider(model, ang, cx0, cy0, objScale);
						for (int dy = -radS; dy <= radS; dy += posStep)
						{
							for (int dx = -radS; dx <= radS; dx += posStep)
							{
								if (dx == 0 && dy == 0)
									continue;
								if (pass == 1 && std::abs(dx) <= 4 && std::abs(dy) <= 4)
									continue;
								consider(model, ang, cx0 + dx, cy0 + dy, objScale);
							}
						}
					}
				}
			}
			for (int k = 0; fastInterior && k < nPeaks; ++k)
			{
				const double tryAng[2] = {peaks[k].ang, c0.angle};
				const int nTry = std::fabs(peaks[k].ang - c0.angle) < 1e-6 ? 1 : 2;
				for (int t = 0; t < nTry; ++t)
				{
					const auto rotated = rotateModel(mdl.edges, tryAng[t]);
					const auto model = scaleSearch_ ? scaleEdges(rotated, peaks[k].scale) : rotated;
					const double full = scorePose(model, nx[layer], ny[layer], mag[layer],
												  (float)matchParam_.edgeMinMag, peaks[k].cx, peaks[k].cy, polarity,
												  greed, abortMinScore(matchParam_.scoreThreshold * 0.6, bestS, greed),
												  layer == stopLayer, 1, &packed[layer]);
					if (full > bestS)
					{
						bestS = full;
						best.cx = peaks[k].cx;
						best.cy = peaks[k].cy;
						best.angle = tryAng[t];
						best.score = full;
						if (scaleSearch_)
							best.scale = peaks[k].scale;
					}
				}
			}
			nPeaks = 0;
			}
			if (interiorFirst && bestS > 0.0)
			{
				const double a0 = best.angle;
				const double cxb = best.cx;
				const double cyb = best.cy;
				for (int ka = -1; ka <= 1; ++ka)
				{
					double ang = a0 + ka * step;
					if (ang > matchParam_.angle)
						ang = matchParam_.angle;
					else if (ang < -matchParam_.angle)
						ang = -matchParam_.angle;
					const auto model = rotateModel(mdl.edges, ang);
					for (int dy = -1; dy <= 1; ++dy)
					{
						for (int dx = -1; dx <= 1; ++dx)
						{
							if (ka == 0 && dx == 0 && dy == 0)
								continue;
							const double cx = cxb + dx;
							const double cy = cyb + dy;
							if (cx < 0 || cy < 0 || cx > pyr[layer].cols - 1 || cy > pyr[layer].rows - 1)
								continue;
							const double sc = scorePose(model, nx[layer], ny[layer], mag[layer],
													   (float)matchParam_.edgeMinMag, cx, cy, polarity, greed,
													   abortMinScore(matchParam_.scoreThreshold * 0.6, bestS, greed),
													   layer == stopLayer, 1, &packed[layer]);
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
			if (best.score >= matchParam_.scoreThreshold * (layer == stopLayer ? 1.0 : 0.65))
			{
				slots[ci] = best;
				slotOk[ci] = 1;
			}
		}
		next.reserve(refined.size());
		for (int ci = 0; ci < (int)refined.size(); ++ci)
		{
			if (slotOk[ci])
				next.push_back(slots[ci]);
		}
		const double layerSpan = std::min((double)std::min(layers_[layer].width, layers_[layer].height),
										  2.0 * (double)modelRadius(mdl.edges));
		const double md = 0.45 * std::max(4.0, layerSpan);
		if (quantCoarse)
			nmsKeepOrigins(next, md, refineKeepCount(matchParam_.maxCount, true), false, 1e9,
						   layer != stopLayer);
		else
			nmsCandidates(next, md, refineKeepCount(matchParam_.maxCount, false));
		refined.swap(next);
	}

	const double fullSpan = std::min((double)std::min(layers_[stopLayer].width, layers_[stopLayer].height),
									 2.0 * (double)modelRadius(layers_[stopLayer].edges));
	const double fullDist = 0.45 * std::max(4.0, fullSpan);
	if (quantCoarse)
		nmsKeepOrigins(refined, fullDist, coarseKeepCount(matchParam_.maxCount, true), false);
	else
		nmsCandidates(refined, fullDist, coarseKeepCount(matchParam_.maxCount, false));
	const int pyrScale = 1 << stopLayer;
	if (!quantCoarse)
	{
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

	// 贴边和内部分开做最后的重叠抑制。内部假峰分数更高时，联合名单会把 0.53 的贴边残件删掉。
	std::vector<MatchResult> borderHits;
	std::vector<MatchResult> interiorHits;
	for (const auto& c : refined)
	{
		if (c.score < matchParam_.scoreThreshold)
			continue;
		MatchResult r = packResult(c.cx * pyrScale, c.cy * pyrScale, c.angle, c.score, templW_, templH_, c.scale);
		if (c.fromBorder)
			borderHits.push_back(r);
		else
			interiorHits.push_back(r);
	}
	const int cap = std::max(matchParam_.maxCount, 1);
	BaseMatcher::suppressMatchOverlap(borderHits, matchParam_.iouThreshold, cap, true);
	BaseMatcher::suppressMatchOverlap(interiorHits, matchParam_.iouThreshold, cap, true);
	std::vector<MatchResult> keptInterior;
	keptInterior.reserve(interiorHits.size());
	for (const auto& in : interiorHits)
	{
		bool drop = false;
		for (const auto& b : borderHits)
		{
			if (interiorConflictsBorder(in, b))
			{
				drop = true;
				break;
			}
		}
		if (!drop)
			keptInterior.push_back(in);
	}
	const int room = std::max(0, cap - (int)borderHits.size());
	if ((int)keptInterior.size() > room)
	{
		std::sort(keptInterior.begin(), keptInterior.end(), [](const MatchResult& a, const MatchResult& b) {
			if (a.Score != b.Score)
				return a.Score > b.Score;
			if (a.Center.y != b.Center.y)
				return a.Center.y < b.Center.y;
			if (a.Center.x != b.Center.x)
				return a.Center.x < b.Center.x;
			if (a.Angle != b.Angle)
				return a.Angle < b.Angle;
			return a.Scale < b.Scale;
		});
		keptInterior.resize((size_t)room);
	}
	matchResults.clear();
	matchResults.reserve(borderHits.size() + keptInterior.size());
	matchResults.insert(matchResults.end(), borderHits.begin(), borderHits.end());
	matchResults.insert(matchResults.end(), keptInterior.begin(), keptInterior.end());
	return (int)matchResults.size();
}

}  // namespace template_matching
