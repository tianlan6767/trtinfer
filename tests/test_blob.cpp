// 预处理 + Blob。不依赖 TensorRT。编译见 Makefile 的 test-blob。
#include "cv_blob/blob.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <string>

namespace
{

int g_fails = 0;

void expect(bool cond, const std::string& msg)
{
	if (cond)
	{
		std::cout << "OK  " << msg << std::endl;
		return;
	}
	std::cerr << "FAIL " << msg << std::endl;
	++g_fails;
}

const blob::Blob* roundest(const std::vector<blob::Blob>& blobs)
{
	const blob::Blob* best = nullptr;
	for (const auto& b : blobs)
	{
		if (!best || b.circularity > best->circularity)
			best = &b;
	}
	return best;
}

void testCropAndBlur()
{
	cv::Mat img(40, 30, CV_8U, cv::Scalar(0));
	img.at<uchar>(12, 8) = 200;
	cv::Mat roi = blob::cropRoi(img, 5, 10, 10, 8);
	expect(roi.rows == 8 && roi.cols == 10, "crop size");
	expect(!roi.empty() && roi.at<uchar>(2, 3) == 200, "crop pixel");
	cv::Mat hung = blob::cropRoi(img, -4, -2, 10, 6);
	expect(hung.rows == 4 && hung.cols == 6, "crop clamps to image");
	expect(blob::cropRoi(img, 0, 0, 0, 10).empty(), "crop rejects non-positive size");

	cv::Mat flat(16, 16, CV_8U, cv::Scalar(80));
	cv::Mat g = blob::gaussianBlur(flat, 5, 1.0);
	expect(!g.empty() && g.at<uchar>(8, 8) == 80, "gaussian keeps flat field");

	cv::Mat salt(9, 9, CV_8U, cv::Scalar(10));
	salt.at<uchar>(4, 4) = 255;
	cv::Mat med = blob::medianBlur(salt, 3);
	expect(med.at<uchar>(4, 4) == 10, "median removes salt pixel");
}

void testThresholdMorph()
{
	cv::Mat img(20, 20, CV_8U, cv::Scalar(40));
	img.at<uchar>(3, 3) = 180;
	blob::ThresholdParam fixed;
	fixed.type = blob::ThresholdType::Fixed;
	fixed.value = 100;
	cv::Mat bin = blob::threshold(img, fixed);
	expect(bin.at<uchar>(0, 0) == 0 && bin.at<uchar>(3, 3) == 255, "fixed threshold");

	cv::Mat bimodal(30, 80, CV_8U, cv::Scalar(15));
	bimodal.colRange(40, 80).setTo(220);
	blob::ThresholdParam otsu;
	otsu.type = blob::ThresholdType::Otsu;
	cv::Mat split = blob::threshold(bimodal, otsu);
	const int fg = cv::countNonZero(split);
	expect(std::abs(fg - 30 * 40) < 50, "otsu splits bimodal image");

	cv::Mat block(20, 20, CV_8U, cv::Scalar(0));
	cv::rectangle(block, cv::Rect(6, 6, 7, 7), cv::Scalar(255), cv::FILLED);
	blob::MorphParam erode;
	erode.op = blob::MorphOp::Erode;
	erode.ksize = 3;
	expect(cv::countNonZero(blob::morphology(block, erode)) == 25, "erode 7x7 -> 5x5");

	blob::MorphParam dilate;
	dilate.op = blob::MorphOp::Dilate;
	dilate.ksize = 3;
	cv::Mat small = cv::Mat::zeros(20, 20, CV_8U);
	cv::rectangle(small, cv::Rect(8, 8, 5, 5), cv::Scalar(255), cv::FILLED);
	expect(cv::countNonZero(blob::morphology(small, dilate)) == 49, "dilate 5x5 -> 7x7");

	cv::Mat gap = cv::Mat::zeros(40, 50, CV_8U);
	cv::rectangle(gap, cv::Rect(8, 10, 12, 12), cv::Scalar(255), cv::FILLED);
	cv::rectangle(gap, cv::Rect(21, 10, 12, 12), cv::Scalar(255), cv::FILLED);
	blob::MorphParam close;
	close.op = blob::MorphOp::Close;
	close.ksize = 3;
	cv::Mat closed = blob::morphology(gap, close);
	blob::BlobFilter filter;
	filter.minArea = 10;
	auto blobs = blob::findBlobs(closed, filter);
	expect(blobs.size() == 1, "close bridges a 1px gap");

	cv::Mat speck = cv::Mat::zeros(30, 30, CV_8U);
	cv::rectangle(speck, cv::Rect(8, 8, 12, 12), cv::Scalar(255), cv::FILLED);
	speck.at<uchar>(2, 2) = 255;
	blob::MorphParam open;
	open.op = blob::MorphOp::Open;
	open.ksize = 3;
	cv::Mat opened = blob::morphology(speck, open);
	expect(cv::countNonZero(opened) == 12 * 12, "open removes a single speck");
}

void testBlobsAndContours()
{
	cv::Mat img = cv::Mat::zeros(180, 220, CV_8U);
	cv::circle(img, cv::Point(55, 60), 22, cv::Scalar(255), cv::FILLED);
	cv::rectangle(img, cv::Rect(130, 30, 14, 70), cv::Scalar(255), cv::FILLED);
	img.at<uchar>(150, 100) = 255;
	img.at<uchar>(150, 101) = 255;
	img.at<uchar>(151, 100) = 255;

	cv::Mat blurred = blob::gaussianBlur(img, 3, 0.6);
	blob::ThresholdParam th;
	th.value = 80;
	cv::Mat bin = blob::threshold(blurred, th);
	blob::MorphParam open;
	open.op = blob::MorphOp::Open;
	open.ksize = 3;
	bin = blob::morphology(bin, open);

	blob::BlobFilter filter;
	filter.minArea = 30;
	auto blobs = blob::findBlobs(bin, filter);
	expect(blobs.size() == 2, "pipeline keeps circle and bar");
	const blob::Blob* circ = roundest(blobs);
	expect(circ != nullptr && circ->circularity > 0.7, "circle circularity");
	expect(circ && std::hypot(circ->center.x - 55, circ->center.y - 60) < 3.0, "circle center");
	expect(circ && circ->aspect < 1.35, "circle aspect");

	filter.minCircularity = 0.7;
	auto rounds = blob::findBlobs(bin, filter);
	expect(rounds.size() == 1, "circularity filter drops the bar");

	blob::BlobFilter bars;
	bars.minArea = 30;
	bars.minAspect = 3.0;
	auto onlyBar = blob::findBlobs(bin, bars);
	expect(onlyBar.size() == 1 && onlyBar[0].aspect > 3.0, "aspect filter keeps the bar");

	blob::BlobFilter roiFilter;
	roiFilter.minArea = 30;
	auto roiBlobs = blob::findBlobs(bin, roiFilter, 100, 0, 120, 160);
	expect(roiBlobs.size() == 1 && roiBlobs[0].center.x > 100, "roi limits blobs, coords stay global");

	cv::Mat square = cv::Mat::zeros(80, 90, CV_8U);
	cv::rectangle(square, cv::Rect(10, 15, 40, 40), cv::Scalar(255), cv::FILLED);
	auto stats = blob::contourStats(square);
	expect(stats.size() == 1, "one external contour");
	if (!stats.empty())
	{
		std::cout << "    contour area=" << stats[0].area << " peri=" << stats[0].perimeter
				  << " circ=" << stats[0].circularity << " aspect=" << stats[0].aspect << std::endl;
		expect(stats[0].area > 1400 && stats[0].area <= 1600, "square contour area");
		auto pixels = blob::findBlobs(square);
		expect(pixels.size() == 1 && pixels[0].area == 1600, "blob area counts pixels");
		expect(stats[0].circularity > 0.6 && stats[0].circularity < 0.95, "square circularity");
		expect(stats[0].width == 40 && stats[0].height == 40, "square bbox");
		expect(std::fabs(stats[0].center.x - 29.5) < 2.0 && std::fabs(stats[0].center.y - 34.5) < 2.0,
			   "square center");
	}

	bars.maxCount = 1;
	auto capped = blob::findBlobs(bin, bars);
	expect((int)capped.size() <= 1, "maxCount truncates");
}

}  // namespace

int main()
{
	testCropAndBlur();
	testThresholdMorph();
	testBlobsAndContours();
	if (g_fails)
	{
		std::cerr << g_fails << " check(s) failed" << std::endl;
		return 1;
	}
	std::cout << "all blob checks passed" << std::endl;
	return 0;
}
