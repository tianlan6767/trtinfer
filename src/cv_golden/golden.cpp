#include "cv_golden/golden.h"

#include "cv_feature/feature.h"
#include "cv_sim/similarity.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>

namespace golden
{
namespace
{

struct Aligned {
	bool ok = false;
	std::string message;
	std::string method;
	double score = 0;
	int inliers = 0;
	bool interpolated = false;
	cv::Mat image;
	cv::Mat valid;
	cv::Mat transform;
};

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
	if (gray.depth() != CV_8U) {
		cv::Mat u8;
		gray.convertTo(u8, CV_8U);
		return u8;
	}
	return gray.clone();
}

// 非 0 表示参与比对。空 mask 表示全图。
cv::Mat asIncludeMask(const cv::Mat& mask, const cv::Size& size, std::string& err)
{
	err.clear();
	if (mask.empty())
		return cv::Mat(size, CV_8U, cv::Scalar(255));
	cv::Mat gray;
	if (mask.channels() == 1)
		gray = mask;
	else if (mask.channels() == 3)
		cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
	else if (mask.channels() == 4)
		cv::cvtColor(mask, gray, cv::COLOR_BGRA2GRAY);
	else {
		err = "bad mask";
		return {};
	}
	if (gray.size() != size) {
		err = "mask size mismatch";
		return {};
	}
	cv::Mat bin;
	cv::compare(gray, 0, bin, cv::CMP_GT);
	return bin;
}

cv::Mat eye33()
{
	return cv::Mat::eye(3, 3, CV_64F);
}

cv::Mat translationH(double dx, double dy)
{
	cv::Mat H = eye33();
	H.at<double>(0, 2) = dx;
	H.at<double>(1, 2) = dy;
	return H;
}

cv::Mat toHomography(const cv::Mat& A)
{
	if (A.empty())
		return {};
	cv::Mat Ad;
	A.convertTo(Ad, CV_64F);
	if (Ad.rows == 3 && Ad.cols == 3)
		return Ad;
	cv::Mat H = eye33();
	if (Ad.rows == 2 && Ad.cols == 3)
		Ad.copyTo(H(cv::Rect(0, 0, 3, 2)));
	return H;
}

bool finiteHomography(const cv::Mat& H)
{
	if (H.rows != 3 || H.cols != 3)
		return false;
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			if (!std::isfinite(H.at<double>(r, c)))
				return false;
		}
	}
	return true;
}

// 轻微姿态：线性部分的尺度别太离谱，否则多半是配错了。
bool linearReasonable(const cv::Mat& H)
{
	if (!finiteHomography(H))
		return false;
	const double n1 = std::hypot(H.at<double>(0, 0), H.at<double>(1, 0));
	const double n2 = std::hypot(H.at<double>(0, 1), H.at<double>(1, 1));
	return n1 >= 0.35 && n1 <= 3.0 && n2 >= 0.35 && n2 <= 3.0;
}

bool centerMapsInside(const cv::Mat& H, const cv::Size& goldenSize, const cv::Size& testSize)
{
	std::vector<cv::Point2f> src{
		cv::Point2f(0.5f * static_cast<float>(goldenSize.width - 1),
					0.5f * static_cast<float>(goldenSize.height - 1))};
	std::vector<cv::Point2f> dst;
	cv::perspectiveTransform(src, dst, H);
	if (dst.empty() || !std::isfinite(dst[0].x) || !std::isfinite(dst[0].y))
		return false;
	const float mx = 0.5f * static_cast<float>(testSize.width);
	const float my = 0.5f * static_cast<float>(testSize.height);
	return dst[0].x >= -mx && dst[0].x < testSize.width + mx && dst[0].y >= -my && dst[0].y < testSize.height + my;
}

int countInliers(const cv::Mat& mask)
{
	if (mask.empty())
		return 0;
	return cv::countNonZero(mask);
}

// 金样 (x, y) 取自待测 (x+dx, y+dy)。重叠区精确拷贝，不插值。
void blitShift(const cv::Mat& src, cv::Mat& dst, cv::Mat& valid, int dx, int dy)
{
	dst = cv::Mat::zeros(dst.size(), CV_8U);
	valid = cv::Mat::zeros(dst.size(), CV_8U);
	const int x0 = std::max(0, -dx);
	const int y0 = std::max(0, -dy);
	const int x1 = std::min(dst.cols, src.cols - dx);
	const int y1 = std::min(dst.rows, src.rows - dy);
	if (x1 <= x0 || y1 <= y0)
		return;
	const cv::Rect g(x0, y0, x1 - x0, y1 - y0);
	const cv::Rect t(x0 + dx, y0 + dy, g.width, g.height);
	src(t).copyTo(dst(g));
	valid(g).setTo(255);
}

void warpToGolden(const cv::Mat& test, const cv::Size& goldenSize, const cv::Mat& H, cv::Mat& image, cv::Mat& valid)
{
	// H 把金样坐标变到待测坐标。WARP_INVERSE_MAP 时 dst(p) = src(H*p)；
	// 不设这个标志的话 OpenCV 会先把 H 求逆。
	const int flags = cv::INTER_LINEAR | cv::WARP_INVERSE_MAP;
	cv::warpPerspective(test, image, H, goldenSize, flags, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::Mat ones(test.size(), CV_8U, cv::Scalar(255));
	cv::warpPerspective(ones, valid, H, goldenSize, cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
						cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::compare(valid, 0, valid, cv::CMP_GT);
}

Aligned failAlign(const std::string& message, double score = 0, const std::string& method = {})
{
	Aligned out;
	out.message = message;
	out.score = score;
	out.method = method;
	return out;
}

Aligned alignNone(const cv::Mat& test, const cv::Mat& golden)
{
	if (test.size() != golden.size())
		return failAlign("size mismatch");
	Aligned out;
	out.ok = true;
	out.method = "none";
	out.score = 1;
	out.image = test;
	out.valid = cv::Mat(golden.size(), CV_8U, cv::Scalar(255));
	out.transform = eye33();
	return out;
}

Aligned alignTranslate(const cv::Mat& test, const cv::Mat& golden, const GoldenParam& param)
{
	const bool smaller = golden.cols < test.cols || golden.rows < test.rows;
	const bool fits = golden.cols <= test.cols && golden.rows <= test.rows;
	if (!fits)
		return failAlign("golden larger than test");

	const std::string method = param.useNcc ? "ncc" : "zncc";
	auto locate = [&](const cv::Mat& scene, const cv::Mat& templ) {
		if (param.useNcc)
			return sim::locateNcc(scene, templ, param.alignThreshold);
		return sim::locateZncc(scene, templ, param.alignThreshold);
	};

	// 金样能塞进待测、且至少有一个方向还有搜索空间：整张金样当模板。
	if (smaller) {
		const sim::LocateResult hit = locate(test, golden);
		if (!hit.found)
			return failAlign("locate failed", hit.score, method);
		const int x = cvRound(hit.topLeft.x);
		const int y = cvRound(hit.topLeft.y);
		if (x < 0 || y < 0 || x + golden.cols > test.cols || y + golden.rows > test.rows)
			return failAlign("locate failed", hit.score, method);
		Aligned out;
		out.ok = true;
		out.method = method;
		out.score = hit.score;
		out.image = test(cv::Rect(x, y, golden.cols, golden.rows)).clone();
		out.valid = cv::Mat(golden.size(), CV_8U, cv::Scalar(255));
		out.transform = translationH(x, y);
		return out;
	}

	// 同尺寸：内缩一块再定位，得到整像素平移。matchTemplate 在同尺寸时只有一个位置。
	int margin = std::max(0, param.alignMargin);
	while (margin > 0 && (golden.cols - 2 * margin < 8 || golden.rows - 2 * margin < 8))
		--margin;
	if (margin <= 0) {
		// 图太小，内缩后没有搜索空间，退回逐像素对齐。
		Aligned out = alignNone(test, golden);
		out.method = method;
		return out;
	}

	const cv::Rect inner(margin, margin, golden.cols - 2 * margin, golden.rows - 2 * margin);
	const sim::LocateResult hit = locate(test, golden(inner));
	if (!hit.found)
		return failAlign("locate failed", hit.score, method);
	const int dx = cvRound(hit.topLeft.x - margin);
	const int dy = cvRound(hit.topLeft.y - margin);
	Aligned out;
	out.ok = true;
	out.method = method;
	out.score = hit.score;
	out.image = cv::Mat(golden.size(), CV_8U);
	blitShift(test, out.image, out.valid, dx, dy);
	out.transform = translationH(dx, dy);
	if (cv::countNonZero(out.valid) <= 0)
		return failAlign("locate failed", hit.score, method);
	return out;
}

std::vector<cv::Point2f> pointsOf(const cv::Mat& nx2)
{
	std::vector<cv::Point2f> pts;
	if (nx2.empty() || nx2.cols < 2 || nx2.type() != CV_64F)
		return pts;
	pts.reserve(nx2.rows);
	for (int i = 0; i < nx2.rows; ++i)
		pts.emplace_back(static_cast<float>(nx2.at<double>(i, 0)), static_cast<float>(nx2.at<double>(i, 1)));
	return pts;
}

Aligned alignFeature(const cv::Mat& test, const cv::Mat& golden, const GoldenParam& param)
{
	const feature::MatchResult mr = feature::matchHomography(
		test, golden, feature::Detector::ORB, param.featureRatio, param.ransacReproj);
	const std::vector<cv::Point2f> src = pointsOf(mr.modelPoints);
	const std::vector<cv::Point2f> dst = pointsOf(mr.scenePoints);
	if (src.size() < 4 || dst.size() != src.size())
		return failAlign("not enough features", 0, param.align == AlignMode::Affine ? "orb-affine" : "orb-homography");

	cv::Mat H;
	std::string method;
	int inliers = 0;
	double score = 0;
	if (param.align == AlignMode::Homography) {
		if (!mr.found || mr.homography.empty() || mr.inliers < std::max(4, param.minInliers))
			return failAlign("homography failed", mr.matches > 0 ? static_cast<double>(mr.inliers) / mr.matches : 0, "orb-homography");
		mr.homography.convertTo(H, CV_64F);
		const double h22 = H.at<double>(2, 2);
		if (!(std::abs(h22) > 1e-12))
			return failAlign("homography failed", 0, "orb-homography");
		H /= h22;
		method = "orb-homography";
		inliers = mr.inliers;
		score = mr.matches > 0 ? static_cast<double>(mr.inliers) / mr.matches : 0;
	} else {
		cv::Mat inliersA;
		cv::Mat A = cv::estimateAffine2D(src, dst, inliersA, cv::RANSAC, std::max(0.5, param.ransacReproj));
		int nIn = countInliers(inliersA);
		method = "orb-affine";
		if (A.empty() || nIn < std::max(3, param.minInliers)) {
			cv::Mat inliersP;
			cv::Mat P = cv::estimateAffinePartial2D(src, dst, inliersP, cv::RANSAC, std::max(0.5, param.ransacReproj));
			const int nP = countInliers(inliersP);
			if (!P.empty() && nP >= std::max(3, param.minInliers) && nP >= nIn) {
				A = P;
				nIn = nP;
				method = "orb-affine-partial";
			}
		}
		if (A.empty() || nIn < std::max(3, param.minInliers))
			return failAlign("affine failed", src.empty() ? 0 : static_cast<double>(nIn) / static_cast<double>(src.size()), method);
		H = toHomography(A);
		inliers = nIn;
		score = static_cast<double>(nIn) / static_cast<double>(src.size());
	}

	if (!linearReasonable(H) || !centerMapsInside(H, golden.size(), test.size()))
		return failAlign("transform rejected", score, method);

	Aligned out;
	out.ok = true;
	out.method = method;
	out.score = score;
	out.inliers = inliers;
	out.interpolated = true;
	out.transform = H;
	warpToGolden(test, golden.size(), H, out.image, out.valid);
	if (cv::countNonZero(out.valid) <= 0)
		return failAlign("transform rejected", score, method);
	return out;
}

Aligned align(const cv::Mat& test, const cv::Mat& golden, const GoldenParam& param)
{
	switch (param.align) {
	case AlignMode::None:
		return alignNone(test, golden);
	case AlignMode::Translate:
		return alignTranslate(test, golden, param);
	case AlignMode::Affine:
	case AlignMode::Homography:
		return alignFeature(test, golden, param);
	}
	return failAlign("bad align mode");
}

int blurRadius(const GoldenParam& param, bool wantBlur)
{
	if (!wantBlur)
		return 0;
	int k = param.blurKsize;
	if (k > 0 && (k % 2) == 0)
		++k;
	if (k > 1)
		return k / 2;
	if (param.blurSigma > 0)
		return std::max(1, static_cast<int>(std::ceil(param.blurSigma * 2.0)));
	return 0;
}

void fillDefectMeans(const cv::Mat& labels,
					 const cv::Mat& centroids,
					 const cv::Mat& absF,
					 const cv::Mat& signedF,
					 const std::vector<blob::Blob>& blobs,
					 std::vector<GoldenDefect>& defects)
{
	const int nlab = labels.empty() ? 0 : static_cast<int>(centroids.rows);
	std::vector<double> sumA(std::max(nlab, 1), 0.0);
	std::vector<double> sumS(std::max(nlab, 1), 0.0);
	std::vector<int> cnt(std::max(nlab, 1), 0);
	if (nlab > 1) {
		for (int y = 0; y < labels.rows; ++y) {
			const int* lab = labels.ptr<int>(y);
			const float* ab = absF.ptr<float>(y);
			const float* sg = signedF.ptr<float>(y);
			for (int x = 0; x < labels.cols; ++x) {
				const int L = lab[x];
				if (L <= 0 || L >= nlab)
					continue;
				sumA[L] += ab[x];
				sumS[L] += sg[x];
				cnt[L] += 1;
			}
		}
	}

	defects.reserve(blobs.size());
	for (const blob::Blob& b : blobs) {
		int lab = b.label;
		if (lab < 1 || lab >= nlab || cnt[lab] <= 0) {
			lab = -1;
			double best = 1e18;
			for (int i = 1; i < nlab; ++i) {
				const double dx = centroids.at<double>(i, 0) - b.center.x;
				const double dy = centroids.at<double>(i, 1) - b.center.y;
				const double d2 = dx * dx + dy * dy;
				if (d2 < best) {
					best = d2;
					lab = i;
				}
			}
		}
		GoldenDefect d;
		d.label = b.label;
		d.area = b.area;
		d.center = b.center;
		d.left = b.left;
		d.top = b.top;
		d.width = b.width;
		d.height = b.height;
		if (lab >= 1 && lab < nlab && cnt[lab] > 0) {
			d.meanAbs = sumA[lab] / cnt[lab];
			d.meanSigned = sumS[lab] / cnt[lab];
		}
		defects.push_back(d);
	}
}

}  // namespace

GoldenResult compare(const cv::Mat& test, const cv::Mat& goldenImage, const GoldenParam& param, const cv::Mat& mask)
{
	GoldenResult out;
	const cv::Mat testG = asGray8(test);
	const cv::Mat goldG = asGray8(goldenImage);
	if (testG.empty() || goldG.empty()) {
		out.message = "empty image";
		return out;
	}

	std::string maskErr;
	const cv::Mat userMask = asIncludeMask(mask, goldG.size(), maskErr);
	if (!maskErr.empty()) {
		out.message = maskErr;
		return out;
	}

	Aligned pack = align(testG, goldG, param);
	out.aligned = pack.ok;
	out.alignMethod = pack.method;
	out.alignScore = pack.score;
	out.alignInliers = pack.inliers;
	out.transform = pack.transform;
	out.message = pack.message;
	if (!pack.ok)
		return out;

	cv::Rect roi(0, 0, goldG.cols, goldG.rows);
	if (param.roiW > 0 && param.roiH > 0)
		roi = cv::Rect(param.roiX, param.roiY, param.roiW, param.roiH) & roi;
	if (roi.width <= 0 || roi.height <= 0) {
		out.message = "empty compare region";
		return out;
	}
	cv::Mat roiMask(goldG.size(), CV_8U, cv::Scalar(0));
	roiMask(roi).setTo(255);
	cv::bitwise_and(pack.valid, roiMask, pack.valid);
	cv::bitwise_and(pack.valid, userMask, pack.valid);

	const bool wantBlur = param.blurSigma > 0.0 || param.blurKsize > 1;
	int radius = blurRadius(param, wantBlur);
	if (pack.interpolated)
		radius = std::max(radius, 1);
	if (radius > 0) {
		const int k = radius * 2 + 1;
		const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
		cv::erode(pack.valid, pack.valid, kernel);
	}
	out.validPixels = cv::countNonZero(pack.valid);
	if (out.validPixels <= 0) {
		out.message = "empty compare region";
		return out;
	}

	cv::Mat goldBlur = goldG;
	cv::Mat testBlur = pack.image;
	if (wantBlur) {
		goldBlur = blob::gaussianBlur(goldG, param.blurKsize, param.blurSigma);
		testBlur = blob::gaussianBlur(pack.image, param.blurKsize, param.blurSigma);
	}

	cv::Mat goldF, testF;
	goldBlur.convertTo(goldF, CV_32F);
	testBlur.convertTo(testF, CV_32F);
	cv::Mat signedF = testF - goldF;
	cv::Mat absF;
	cv::absdiff(testF, goldF, absF);

	out.meanAbs = cv::mean(absF, pack.valid)[0];
	out.meanSigned = cv::mean(signedF, pack.valid)[0];
	cv::minMaxLoc(absF, nullptr, &out.maxAbs, nullptr, nullptr, pack.valid);

	const double th = std::max(0.0, param.diffThreshold);
	cv::Mat binary;
	if (param.polarity == Polarity::Brighter)
		cv::compare(signedF, th, binary, cv::CMP_GE);
	else if (param.polarity == Polarity::Darker)
		cv::compare(signedF, -th, binary, cv::CMP_LE);
	else
		cv::compare(absF, th, binary, cv::CMP_GE);
	cv::bitwise_and(binary, pack.valid, binary);

	if (param.enableMorph && param.morphKsize > 0 && param.morphIterations > 0) {
		blob::MorphParam morph;
		morph.op = param.morphOp;
		morph.ksize = param.morphKsize;
		morph.iterations = param.morphIterations;
		binary = blob::morphology(binary, morph);
		cv::bitwise_and(binary, pack.valid, binary);
	}

	blob::BlobFilter filter;
	filter.minArea = std::max(0.0, param.minArea);
	filter.maxCount = param.maxCount;
	std::vector<blob::Blob> blobs = blob::findBlobs(binary, filter);

	cv::Mat labels, stats, centroids;
	cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
	(void)stats;
	fillDefectMeans(labels, centroids, absF, signedF, blobs, out.defects);

	out.defectCount = static_cast<int>(out.defects.size());
	out.defectArea = 0;
	for (const GoldenDefect& d : out.defects)
		out.defectArea += d.area;
	out.defectRatio = out.defectArea / static_cast<double>(out.validPixels);
	out.ok = true;
	out.message.clear();

	if (param.keepDiff) {
		cv::Mat inv;
		cv::bitwise_not(pack.valid, inv);
		absF.setTo(0, inv);
		signedF.setTo(0, inv);
		absF.convertTo(out.diff, CV_8U);
		out.signedDiff = signedF;
	}
	if (param.keepMask)
		out.mask = binary;
	if (param.keepAligned)
		out.alignedImage = pack.image;
	return out;
}

}  // namespace golden
