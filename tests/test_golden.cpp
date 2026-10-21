// 金样比对。不依赖 TensorRT。编译见 Makefile 的 test-golden。
#include "cv_golden/golden.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <string>

namespace
{

int g_fails = 0;

void expect(bool cond, const std::string& msg)
{
	if (cond) {
		std::cout << "OK  " << msg << std::endl;
		return;
	}
	std::cerr << "FAIL " << msg << std::endl;
	++g_fails;
}

cv::Mat makePlate()
{
	const int h = 240;
	const int w = 320;
	cv::Mat img(h, w, CV_8U, cv::Scalar(160));
	cv::rectangle(img, cv::Rect(12, 12, w - 24, h - 24), cv::Scalar(30), 3);
	cv::circle(img, cv::Point(78, 86), 24, cv::Scalar(80), cv::FILLED);
	cv::rectangle(img, cv::Rect(150, 48, 90, 72), cv::Scalar(210), cv::FILLED);
	cv::line(img, cv::Point(36, 190), cv::Point(280, 210), cv::Scalar(45), 5);
	cv::rectangle(img, cv::Rect(40, 150, 14, 10), cv::Scalar(20), cv::FILLED);
	cv::rectangle(img, cv::Rect(260, 40, 16, 12), cv::Scalar(240), cv::FILLED);
	cv::rectangle(img, cv::Rect(40, 40, 12, 10), cv::Scalar(15), cv::FILLED);
	cv::rectangle(img, cv::Rect(180, 180, 18, 14), cv::Scalar(245), cv::FILLED);
	return img;
}

void paintDefects(cv::Mat& img)
{
	cv::rectangle(img, cv::Rect(112, 44, 32, 18), cv::Scalar(255), cv::FILLED);
	cv::circle(img, cv::Point(214, 168), 11, cv::Scalar(12), cv::FILLED);
	img.at<uchar>(32, 32) = 255;
	img.at<uchar>(32, 33) = 255;
	img.at<uchar>(33, 32) = 255;
	img.at<uchar>(33, 33) = 255;
}

const golden::GoldenDefect* nearest(const std::vector<golden::GoldenDefect>& defects, double x, double y)
{
	const golden::GoldenDefect* best = nullptr;
	double bestD = 1e18;
	for (const auto& d : defects) {
		const double dist = std::hypot(d.center.x - x, d.center.y - y);
		if (dist < bestD) {
			bestD = dist;
			best = &d;
		}
	}
	return best;
}

golden::GoldenParam baseParam()
{
	golden::GoldenParam p;
	p.diffThreshold = 30;
	p.minArea = 40;
	p.enableMorph = true;
	p.morphKsize = 3;
	p.morphIterations = 1;
	return p;
}

void testCleanAndPolarity()
{
	const cv::Mat plate = makePlate();
	golden::GoldenResult same = golden::compare(plate, plate, baseParam());
	expect(same.ok && same.defectCount == 0, "identical has no defects");
	expect(same.meanAbs == 0 && same.maxAbs == 0, "identical meanAbs is 0");
	expect(same.alignMethod == "none" && same.aligned, "none alignment");

	cv::Mat bgr;
	cv::cvtColor(plate, bgr, cv::COLOR_GRAY2BGR);
	golden::GoldenResult color = golden::compare(bgr, bgr, baseParam());
	expect(color.ok && color.defectCount == 0 && color.meanAbs == 0, "bgr identical");

	cv::Mat test = plate.clone();
	paintDefects(test);
	golden::GoldenParam both = baseParam();
	golden::GoldenResult all = golden::compare(test, plate, both);
	expect(all.ok && all.defectCount == 2, "both polarities -> 2 defects, got " + std::to_string(all.defectCount));
	expect(all.meanAbs > 1.0 && all.maxAbs > 80.0, "scores move when defects exist");
	expect(all.validPixels == plate.rows * plate.cols, "full frame is valid");
	const golden::GoldenDefect* bright = nearest(all.defects, 127.5, 52.5);
	const golden::GoldenDefect* dark = nearest(all.defects, 214, 168);
	expect(bright && std::hypot(bright->center.x - 127.5, bright->center.y - 52.5) < 3.0, "bright blob at rect");
	expect(dark && std::hypot(dark->center.x - 214, dark->center.y - 168) < 3.0, "dark blob at circle");
	expect(bright && bright->meanSigned > 40.0, "bright meanSigned");
	expect(dark && dark->meanSigned < -40.0, "dark meanSigned");
	expect(!all.diff.empty() && !all.mask.empty() && all.mask.size() == plate.size(), "diff and mask returned");
	expect(cv::countNonZero(all.mask) > 0, "mask has foreground");

	golden::GoldenParam onlyBright = baseParam();
	onlyBright.polarity = golden::Polarity::Brighter;
	golden::GoldenResult br = golden::compare(test, plate, onlyBright);
	expect(br.ok && br.defectCount == 1, "brighter keeps one");
	expect(br.defectCount == 1 && std::hypot(br.defects[0].center.x - 127.5, br.defects[0].center.y - 52.5) < 3.0,
		   "brighter is the rect");

	golden::GoldenParam onlyDark = baseParam();
	onlyDark.polarity = golden::Polarity::Darker;
	golden::GoldenResult dk = golden::compare(test, plate, onlyDark);
	expect(dk.ok && dk.defectCount == 1, "darker keeps one");
	expect(dk.defectCount == 1 && std::hypot(dk.defects[0].center.x - 214, dk.defects[0].center.y - 168) < 3.0,
		   "darker is the circle");

	golden::GoldenParam noImg = baseParam();
	noImg.keepDiff = false;
	noImg.keepMask = false;
	noImg.keepAligned = false;
	golden::GoldenResult hidden = golden::compare(test, plate, noImg);
	expect(hidden.ok && hidden.defectCount == 2 && hidden.diff.empty() && hidden.mask.empty() && hidden.alignedImage.empty(),
		   "images are optional");
}

void testRoiAndMask()
{
	const cv::Mat plate = makePlate();
	cv::Mat test = plate.clone();
	paintDefects(test);

	golden::GoldenParam roi = baseParam();
	roi.roiX = 0;
	roi.roiY = 0;
	roi.roiW = 170;
	roi.roiH = plate.rows;
	golden::GoldenResult left = golden::compare(test, plate, roi);
	expect(left.ok && left.defectCount == 1, "roi drops the dark circle");
	expect(left.defectCount == 1 && left.defects[0].center.x < 160, "roi defect stays on the left");

	cv::Mat mask(plate.size(), CV_8U, cv::Scalar(255));
	mask(cv::Rect(108, 40, 40, 26)).setTo(0);
	golden::GoldenResult masked = golden::compare(test, plate, baseParam(), mask);
	expect(masked.ok && masked.defectCount == 1, "mask drops the bright rect");
	expect(masked.defectCount == 1 && std::hypot(masked.defects[0].center.x - 214, masked.defects[0].center.y - 168) < 3.0,
		   "mask keeps the circle");

	cv::Mat bad(plate.rows, plate.cols - 4, CV_8U, cv::Scalar(255));
	golden::GoldenResult mismatch = golden::compare(test, plate, baseParam(), bad);
	expect(!mismatch.ok && mismatch.message == "mask size mismatch", "mask size is checked");
}

cv::Mat shiftImage(const cv::Mat& img, int dx, int dy, uchar fill)
{
	cv::Mat out(img.size(), img.type(), cv::Scalar(fill));
	const int tw = img.cols - std::abs(dx);
	const int th = img.rows - std::abs(dy);
	if (tw <= 0 || th <= 0)
		return out;
	const int srcX = std::max(0, -dx);
	const int srcY = std::max(0, -dy);
	const int dstX = std::max(0, dx);
	const int dstY = std::max(0, dy);
	img(cv::Rect(srcX, srcY, tw, th)).copyTo(out(cv::Rect(dstX, dstY, tw, th)));
	return out;
}

void testTranslate()
{
	const cv::Mat plate = makePlate();
	const int ox = 15;
	const int oy = 9;
	cv::Mat canvas(plate.rows + 40, plate.cols + 36, CV_8U, cv::Scalar(35));
	plate.copyTo(canvas(cv::Rect(ox, oy, plate.cols, plate.rows)));
	cv::rectangle(canvas, cv::Rect(ox + 112, oy + 44, 32, 18), cv::Scalar(255), cv::FILLED);

	golden::GoldenParam p = baseParam();
	p.align = golden::AlignMode::Translate;
	p.alignThreshold = 0.6;
	golden::GoldenResult hit = golden::compare(canvas, plate, p);
	expect(hit.ok && hit.alignMethod == "zncc", "zncc locate " + hit.message);
	expect(hit.alignScore > 0.85, "zncc score");
	expect(hit.transform.rows == 3 && std::abs(hit.transform.at<double>(0, 2) - ox) < 1.0, "translate x");
	expect(std::abs(hit.transform.at<double>(1, 2) - oy) < 1.0, "translate y");
	expect(hit.defectCount == 1, "one defect after locate, got " + std::to_string(hit.defectCount));
	expect(hit.defectCount == 1 && std::hypot(hit.defects[0].center.x - 127.5, hit.defects[0].center.y - 52.5) < 3.0,
		   "defect is in golden frame");

	p.useNcc = true;
	golden::GoldenResult ncc = golden::compare(canvas, plate, p);
	expect(ncc.ok && ncc.alignMethod == "ncc" && std::abs(ncc.transform.at<double>(0, 2) - ox) < 1.0, "ncc locate");

	const int dx = 5;
	const int dy = -3;
	cv::Mat shifted = shiftImage(plate, dx, dy, 0);
	cv::rectangle(shifted, cv::Rect(112 + dx, 44 + dy, 32, 18), cv::Scalar(255), cv::FILLED);
	golden::GoldenParam shift = baseParam();
	shift.align = golden::AlignMode::Translate;
	shift.alignMargin = 16;
	golden::GoldenResult moved = golden::compare(shifted, plate, shift);
	expect(moved.ok, "same-size shift " + moved.message);
	expect(std::abs(moved.transform.at<double>(0, 2) - dx) < 1.0, "shift dx");
	expect(std::abs(moved.transform.at<double>(1, 2) - dy) < 1.0, "shift dy");
	expect(moved.defectCount == 1, "shift keeps the rect, got " + std::to_string(moved.defectCount));
	expect(moved.defectCount == 1 && std::hypot(moved.defects[0].center.x - 127.5, moved.defects[0].center.y - 52.5) < 3.0,
		   "shift defect back in golden frame");

	golden::GoldenParam raw = baseParam();
	golden::GoldenResult unaligned = golden::compare(shifted, plate, raw);
	expect(unaligned.ok && unaligned.meanAbs > moved.meanAbs, "unaligned diff is larger");
}

cv::Mat makePosePlate()
{
	// 四周高对比纹理给 ORB，中间一块平场放缺陷，避免重采样边缘被当成缺陷。
	const int h = 260;
	const int w = 340;
	cv::Mat img(h, w, CV_8U, cv::Scalar(210));
	cv::RNG rng(5);
	for (int i = 0; i < 90; ++i) {
		const int ww = rng.uniform(8, 26);
		const int hh = rng.uniform(6, 20);
		const int x = rng.uniform(6, std::max(7, w - ww - 6));
		const int y = rng.uniform(6, std::max(7, h - hh - 6));
		img(cv::Rect(x, y, ww, hh)).setTo(rng.uniform(0, 45));
	}
	cv::rectangle(img, cv::Rect(4, 4, w - 8, h - 8), cv::Scalar(15), 2);
	cv::circle(img, cv::Point(28, 28), 10, cv::Scalar(250), cv::FILLED);
	cv::rectangle(img, cv::Rect(w - 44, 16, 22, 18), cv::Scalar(250), cv::FILLED);
	cv::circle(img, cv::Point(26, h - 26), 12, cv::Scalar(20), cv::FILLED);
	cv::rectangle(img, cv::Rect(w - 50, h - 40, 28, 14), cv::Scalar(30), cv::FILLED);
	cv::rectangle(img, cv::Rect(w / 2 - 80, h / 2 - 70, 160, 140), cv::Scalar(180), cv::FILLED);
	return img;
}

golden::GoldenParam poseParam(const cv::Mat& plate)
{
	golden::GoldenParam p = baseParam();
	p.diffThreshold = 40;
	p.minArea = 40;
	p.blurSigma = 0.8;
	p.featureRatio = 0.8;
	p.ransacReproj = 4.0;
	p.minInliers = 6;
	p.roiX = plate.cols / 2 - 40;
	p.roiY = plate.rows / 2 - 36;
	p.roiW = 80;
	p.roiH = 72;
	return p;
}

void expectDominant(const golden::GoldenResult& r, double x, double y, const std::string& tag)
{
	expect(r.ok, tag + " ok " + r.message + " method=" + r.alignMethod);
	expect(r.alignInliers >= 6, tag + " inliers " + std::to_string(r.alignInliers));
	expect(!r.defects.empty(), tag + " has a defect, n=" + std::to_string(r.defectCount));
	if (r.defects.empty())
		return;
	const golden::GoldenDefect* best = &r.defects[0];
	for (const auto& d : r.defects) {
		if (d.area > best->area)
			best = &d;
	}
	const double dist = std::hypot(best->center.x - x, best->center.y - y);
	expect(dist < 10.0 && best->area > 150.0 && best->meanSigned < -20.0,
		   tag + " dominant at (" + std::to_string(best->center.x) + "," + std::to_string(best->center.y)
			   + ") area=" + std::to_string(best->area) + " dist=" + std::to_string(dist)
			   + " n=" + std::to_string(r.defectCount));
}

void testPose()
{
	const cv::Mat plate = makePosePlate();
	const double cx = plate.cols / 2.0;
	const double cy = plate.rows / 2.0;
	// 不要放在旋转中心上，否则正变换和逆变换分不出来。
	const double gx = cx - 18.0;
	const double gy = cy + 12.0;

	cv::Mat rot;
	const cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)), 3.0, 1.0);
	cv::warpAffine(plate, rot, M, plate.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
	const double tx = M.at<double>(0, 0) * gx + M.at<double>(0, 1) * gy + M.at<double>(0, 2);
	const double ty = M.at<double>(1, 0) * gx + M.at<double>(1, 1) * gy + M.at<double>(1, 2);
	cv::circle(rot, cv::Point(cvRound(tx), cvRound(ty)), 14, cv::Scalar(0), cv::FILLED);

	golden::GoldenParam affine = poseParam(plate);
	affine.align = golden::AlignMode::Affine;
	expectDominant(golden::compare(rot, plate, affine), gx, gy, "affine");

	std::vector<cv::Point2f> src = {
		cv::Point2f(0, 0),
		cv::Point2f(static_cast<float>(plate.cols - 1), 0),
		cv::Point2f(static_cast<float>(plate.cols - 1), static_cast<float>(plate.rows - 1)),
		cv::Point2f(0, static_cast<float>(plate.rows - 1))};
	std::vector<cv::Point2f> dst = {
		cv::Point2f(18, 10),
		cv::Point2f(static_cast<float>(plate.cols + 8), 16),
		cv::Point2f(static_cast<float>(plate.cols - 4), static_cast<float>(plate.rows + 12)),
		cv::Point2f(8, static_cast<float>(plate.rows - 6))};
	const cv::Mat H = cv::getPerspectiveTransform(src, dst);
	cv::Mat scene;
	cv::warpPerspective(plate, scene, H, cv::Size(plate.cols + 30, plate.rows + 30), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
	std::vector<cv::Point2f> gpt{cv::Point2f(static_cast<float>(gx), static_cast<float>(gy))};
	std::vector<cv::Point2f> tpt;
	cv::perspectiveTransform(gpt, tpt, H);
	cv::circle(scene, cv::Point(cvRound(tpt[0].x), cvRound(tpt[0].y)), 14, cv::Scalar(0), cv::FILLED);

	golden::GoldenParam homo = poseParam(plate);
	homo.align = golden::AlignMode::Homography;
	expectDominant(golden::compare(scene, plate, homo), gx, gy, "homography");
}

void testFailures()
{
	const cv::Mat plate = makePlate();
	golden::GoldenResult empty = golden::compare(cv::Mat(), plate, {});
	expect(!empty.ok && empty.message == "empty image", "empty image");

	golden::GoldenParam none;
	golden::GoldenResult size = golden::compare(plate.colRange(0, 40), plate, none);
	expect(!size.ok && size.message == "size mismatch", "size mismatch");

	cv::Mat blank(plate.size(), CV_8U, cv::Scalar(20));
	golden::GoldenParam locate;
	locate.align = golden::AlignMode::Translate;
	golden::GoldenResult missed = golden::compare(blank, plate, locate);
	expect(!missed.ok && !missed.aligned && missed.defectCount == 0, "locate failure does not invent defects");
}

}  // namespace

int main()
{
	testCleanAndPolarity();
	testRoiAndMask();
	testTranslate();
	testPose();
	testFailures();
	if (g_fails) {
		std::cerr << g_fails << " failed" << std::endl;
		return 1;
	}
	std::cout << "all golden checks passed" << std::endl;
	return 0;
}
