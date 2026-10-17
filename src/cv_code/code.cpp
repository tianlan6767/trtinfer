#include "cv_code/code.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#if __has_include(<opencv2/barcode.hpp>)
#include <opencv2/barcode.hpp>
#define CVTER_HAS_BARCODE 1
#endif

#if defined(CVTER_WITH_TESSERACT)
#include <tesseract/baseapi.h>
#define CVTER_HAS_TESSERACT 1
#endif

#if __has_include(<zbar.h>)
#include <zbar.h>
#define CVTER_HAS_ZBAR 1
#endif

#include <algorithm>

namespace code
{
namespace
{

cv::Mat asBgr(const cv::Mat& image)
{
	if (image.empty())
		return {};
	if (image.channels() == 3 && image.depth() == CV_8U)
		return image;
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
		gray = u8;
	}
	cv::Mat bgr;
	cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
	return bgr;
}

cv::Mat polygonFrom(const cv::Mat& pts, int offset, int count)
{
	cv::Mat out(count, 2, CV_64F);
	for (int i = 0; i < count; ++i) {
		const int k = offset + i;
		double x = 0, y = 0;
		if (pts.channels() == 2 && pts.depth() == CV_32F)
			{ cv::Point2f p = pts.at<cv::Point2f>(k); x = p.x; y = p.y; }
		else if (pts.depth() == CV_32F && pts.cols >= 2)
			{ x = pts.at<float>(k, 0); y = pts.at<float>(k, 1); }
		else if (pts.depth() == CV_64F && pts.cols >= 2)
			{ x = pts.at<double>(k, 0); y = pts.at<double>(k, 1); }
		else if (pts.depth() == CV_32F)
			{ x = pts.at<float>(k, 0); y = pts.at<float>(k, 1); }
		out.at<double>(i, 0) = x;
		out.at<double>(i, 1) = y;
	}
	return out;
}

std::string barcodeName(int type)
{
#ifdef CVTER_HAS_BARCODE
	switch (static_cast<cv::barcode::BarcodeType>(type)) {
	case cv::barcode::EAN_8: return "EAN_8";
	case cv::barcode::EAN_13: return "EAN_13";
	case cv::barcode::UPC_A: return "UPC_A";
	case cv::barcode::UPC_E: return "UPC_E";
	case cv::barcode::UPC_EAN_EXTENSION: return "UPC_EAN_EXTENSION";
	default: return "NONE";
	}
#else
	(void)type;
	return "NONE";
#endif
}

}  // namespace

bool barcodeAvailable()
{
#if defined(CVTER_HAS_BARCODE) || defined(CVTER_HAS_ZBAR)
	return true;
#else
	return false;
#endif
}

bool ocrAvailable()
{
#ifdef CVTER_HAS_TESSERACT
	tesseract::TessBaseAPI api;
	const char* paths[] = {nullptr, "/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tessdata"};
	for (const char* path : paths) {
		if (api.Init(path, "eng") == 0) {
			api.End();
			return true;
		}
	}
	return false;
#else
	return false;
#endif
}

std::vector<DecodeHit> decodeQr(const cv::Mat& image)
{
	std::vector<DecodeHit> hits;
	cv::Mat bgr = asBgr(image);
	if (bgr.empty())
		return hits;
	try {
		cv::QRCodeDetector det;
		std::vector<std::string> texts;
		cv::Mat points;
		std::vector<cv::Mat> straight;
		const bool multi = det.detectAndDecodeMulti(bgr, texts, points, straight);
		if (multi && !texts.empty() && points.total() >= 4) {
			const int n = static_cast<int>(texts.size());
			const int stride = static_cast<int>(points.total()) / std::max(1, n);
			for (int i = 0; i < n; ++i) {
				if (texts[i].empty())
					continue;
				DecodeHit hit;
				hit.found = true;
				hit.text = texts[i];
				hit.format = "QR";
				const int take = std::min(4, stride);
				hit.polygon = polygonFrom(points.reshape(points.channels(), static_cast<int>(points.total())), i * stride, take);
				hits.push_back(hit);
			}
			return hits;
		}
		cv::Mat one;
		const std::string text = det.detectAndDecode(bgr, one);
		if (!text.empty()) {
			DecodeHit hit;
			hit.found = true;
			hit.text = text;
			hit.format = "QR";
			if (one.total() >= 4)
				hit.polygon = polygonFrom(one.reshape(one.channels(), static_cast<int>(one.total())), 0, 4);
			hits.push_back(hit);
		}
	} catch (const cv::Exception&) {
	}
	return hits;
}

std::vector<DecodeHit> decodeBarcodeZbar(const cv::Mat& bgr)
{
	std::vector<DecodeHit> hits;
#ifdef CVTER_HAS_ZBAR
	cv::Mat gray;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
	if (!gray.isContinuous())
		gray = gray.clone();
	zbar::ImageScanner scanner;
	scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);
	zbar::Image image(gray.cols, gray.rows, "Y800", gray.data, static_cast<unsigned long>(gray.total()));
	if (scanner.scan(image) <= 0)
		return hits;
	for (auto symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
		if (symbol->get_data().empty())
			continue;
		DecodeHit hit;
		hit.found = true;
		hit.text = symbol->get_data();
		hit.format = symbol->get_type_name();
		const int n = symbol->get_location_size();
		if (n > 0) {
			hit.polygon = cv::Mat(n, 2, CV_64F);
			for (int i = 0; i < n; ++i) {
				hit.polygon.at<double>(i, 0) = symbol->get_location_x(i);
				hit.polygon.at<double>(i, 1) = symbol->get_location_y(i);
			}
		}
		hits.push_back(hit);
	}
#else
	(void)bgr;
#endif
	return hits;
}

std::vector<DecodeHit> decodeBarcode(const cv::Mat& image)
{
	std::vector<DecodeHit> hits;
	cv::Mat bgr = asBgr(image);
	if (bgr.empty())
		return hits;
#ifdef CVTER_HAS_BARCODE
	try {
		cv::barcode::BarcodeDetector det;
		std::vector<std::string> info;
		std::vector<cv::barcode::BarcodeType> types;
		cv::Mat points;
		if (det.detectAndDecode(bgr, info, types, points)) {
			const int n = static_cast<int>(info.size());
			int stride = 4;
			if (n > 0 && points.total() >= static_cast<size_t>(n))
				stride = std::max(1, static_cast<int>(points.total()) / n);
			cv::Mat flat = points.empty() ? cv::Mat() : points.reshape(points.channels(), static_cast<int>(points.total()));
			for (int i = 0; i < n; ++i) {
				if (info[i].empty())
					continue;
				DecodeHit hit;
				hit.found = true;
				hit.text = info[i];
				hit.format = i < static_cast<int>(types.size()) ? barcodeName(static_cast<int>(types[i])) : "BARCODE";
				if (!flat.empty() && (i + 1) * std::min(stride, 4) <= flat.rows)
					hit.polygon = polygonFrom(flat, i * stride, std::min(4, stride));
				hits.push_back(hit);
			}
		}
	} catch (const cv::Exception&) {
	}
#endif
	if (!hits.empty())
		return hits;
	return decodeBarcodeZbar(bgr);
}

std::vector<DecodeHit> decodeCodes(const cv::Mat& image)
{
	std::vector<DecodeHit> hits = decodeQr(image);
	std::vector<DecodeHit> bars = decodeBarcode(image);
	hits.insert(hits.end(), bars.begin(), bars.end());
	return hits;
}

OcrResult recognizeText(const cv::Mat& image, const std::string& lang)
{
	OcrResult out;
	out.backend = "none";
#ifndef CVTER_HAS_TESSERACT
	(void)image;
	(void)lang;
	return out;
#else
	cv::Mat bgr = asBgr(image);
	if (bgr.empty())
		return out;
	cv::Mat gray;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
	tesseract::TessBaseAPI api;
	const std::string useLang = lang.empty() ? "eng" : lang;
	const char* paths[] = {nullptr, "/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tessdata"};
	bool ready = false;
	for (const char* path : paths) {
		if (api.Init(path, useLang.c_str()) == 0) {
			ready = true;
			break;
		}
	}
	if (!ready)
		return out;
	out.backend = "tesseract";
	api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
	api.SetImage(gray.data, gray.cols, gray.rows, 1, static_cast<int>(gray.step));
	char* text = api.GetUTF8Text();
	if (text) {
		out.text = text;
		delete[] text;
	}
	api.End();
	while (!out.text.empty() && (out.text.back() == '\n' || out.text.back() == '\r' || out.text.back() == ' '))
		out.text.pop_back();
	out.found = !out.text.empty();
	return out;
#endif
}

}  // namespace code
