#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace code
{

struct DecodeHit {
	bool found = false;
	std::string text;
	std::string format;
	cv::Mat polygon;  // Nx2 CV_64F
};

struct OcrResult {
	bool found = false;
	std::string text;
	std::string backend;
};

bool barcodeAvailable();
bool ocrAvailable();

std::vector<DecodeHit> decodeQr(const cv::Mat& image);
std::vector<DecodeHit> decodeBarcode(const cv::Mat& image);
std::vector<DecodeHit> decodeCodes(const cv::Mat& image);

// lang 例如 "eng"。没编进 Tesseract 时 backend="none"。
OcrResult recognizeText(const cv::Mat& image, const std::string& lang = "eng");

}  // namespace code
