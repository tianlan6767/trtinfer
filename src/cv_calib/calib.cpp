#include "cv_calib/calib.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#if __has_include(<opencv2/aruco.hpp>) && __has_include(<opencv2/aruco/charuco.hpp>)
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#define CVTER_HAS_ARUCO 1
#endif

#include <algorithm>
#include <cmath>
#include <string>

namespace calib
{
namespace
{

cv::Mat asGray(const cv::Mat& image)
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
	return gray;
}

cv::Mat toNx2(const std::vector<cv::Point2f>& pts)
{
	cv::Mat m(static_cast<int>(pts.size()), 2, CV_64F);
	for (int i = 0; i < m.rows; ++i) {
		m.at<double>(i, 0) = pts[i].x;
		m.at<double>(i, 1) = pts[i].y;
	}
	return m;
}

cv::Mat toNx3(const std::vector<cv::Point3f>& pts)
{
	cv::Mat m(static_cast<int>(pts.size()), 3, CV_64F);
	for (int i = 0; i < m.rows; ++i) {
		m.at<double>(i, 0) = pts[i].x;
		m.at<double>(i, 1) = pts[i].y;
		m.at<double>(i, 2) = pts[i].z;
	}
	return m;
}

std::vector<cv::Point2f> fromNx2(const cv::Mat& src)
{
	std::vector<cv::Point2f> pts;
	if (src.empty())
		return pts;
	cv::Mat m;
	src.convertTo(m, CV_64F);
	if (m.channels() == 2 && m.cols == 1)
		m = m.reshape(1, static_cast<int>(m.total()));
	if (m.cols < 2)
		return pts;
	pts.reserve(m.rows);
	for (int i = 0; i < m.rows; ++i)
		pts.emplace_back(static_cast<float>(m.at<double>(i, 0)), static_cast<float>(m.at<double>(i, 1)));
	return pts;
}

std::vector<cv::Point3f> fromNx3(const cv::Mat& src)
{
	std::vector<cv::Point3f> pts;
	if (src.empty())
		return pts;
	cv::Mat m;
	src.convertTo(m, CV_64F);
	if (m.cols < 3)
		return pts;
	pts.reserve(m.rows);
	for (int i = 0; i < m.rows; ++i) {
		pts.emplace_back(static_cast<float>(m.at<double>(i, 0)),
						 static_cast<float>(m.at<double>(i, 1)),
						 static_cast<float>(m.at<double>(i, 2)));
	}
	return pts;
}

cv::TermCriteria criteriaOf(const BoardConfig& cfg)
{
	const int n = std::max(1, cfg.subpixIters);
	const double eps = cfg.subpixEps > 0.0 ? cfg.subpixEps : 0.001;
	return cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, n, eps);
}

int halfWin(int win)
{
	if (win < 2)
		win = 2;
	return win;
}

bool refineSubpix(const cv::Mat& gray, std::vector<cv::Point2f>& pts, int win, const cv::TermCriteria& crit)
{
	if (pts.empty() || gray.empty())
		return false;
	for (int w = halfWin(win); w >= 2; --w) {
		bool inside = true;
		for (const auto& p : pts) {
			if (p.x < w + 1 || p.y < w + 1 || p.x >= gray.cols - w - 1 || p.y >= gray.rows - w - 1) {
				inside = false;
				break;
			}
		}
		if (!inside)
			continue;
		try {
			cv::cornerSubPix(gray, pts, cv::Size(w, w), cv::Size(-1, -1), crit);
			return true;
		} catch (const cv::Exception&) {
		}
	}
	return false;
}

double markerLenOf(const BoardConfig& cfg)
{
	double m = cfg.markerLength;
	if (m <= 0.0)
		m = cfg.squareLength * 0.6;
	if (cfg.type == BoardType::Charuco && m >= cfg.squareLength)
		m = cfg.squareLength * 0.6;
	return m;
}

#ifdef CVTER_HAS_ARUCO
int dictId(const BoardConfig& cfg)
{
	if (cfg.dictionary >= 0)
		return cfg.dictionary;
	if (cfg.type == BoardType::AprilTag)
		return cv::aruco::DICT_APRILTAG_36h11;
	return cv::aruco::DICT_4X4_50;
}

cv::Ptr<cv::aruco::Dictionary> dictionaryOf(const BoardConfig& cfg)
{
	return cv::aruco::getPredefinedDictionary(dictId(cfg));
}

cv::Ptr<cv::aruco::DetectorParameters> detectorParams(const BoardConfig& cfg, bool april)
{
	cv::Ptr<cv::aruco::DetectorParameters> p = cv::aruco::DetectorParameters::create();
	p->cornerRefinementMethod = april ? cv::aruco::CORNER_REFINE_APRILTAG : cv::aruco::CORNER_REFINE_SUBPIX;
	p->cornerRefinementWinSize = std::min(7, std::max(3, cfg.subpixWin));
	p->cornerRefinementMaxIterations = std::max(1, cfg.subpixIters);
	p->cornerRefinementMinAccuracy = cfg.subpixEps > 0.0 ? cfg.subpixEps : 0.001;
	p->minMarkerPerimeterRate = 0.02;
	return p;
}

struct MarkerPack {
	std::vector<std::vector<cv::Point2f>> corners;
	std::vector<int> ids;
};

MarkerPack detectMarkersSubpix(const cv::Mat& gray, const BoardConfig& cfg, bool april)
{
	MarkerPack pack;
	cv::Ptr<cv::aruco::Dictionary> dict = dictionaryOf(cfg);
	cv::aruco::detectMarkers(gray, dict, pack.corners, pack.ids, detectorParams(cfg, april));
	const cv::TermCriteria crit = criteriaOf(cfg);
	const int win = std::min(cfg.subpixWin, 5);
	for (auto& c : pack.corners)
		refineSubpix(gray, c, win, crit);
	return pack;
}
#endif

BoardDetection finish(const std::string& board, const std::string& refine, bool subpixel,
					  const std::vector<cv::Point2f>& imagePts, const std::vector<cv::Point3f>& objectPts,
					  const std::vector<int>& ids)
{
	BoardDetection det;
	det.board = board;
	det.refine = refine;
	det.subpixel = subpixel && !imagePts.empty();
	det.count = static_cast<int>(imagePts.size());
	det.found = det.count >= 4;
	det.imagePoints = toNx2(imagePts);
	det.objectPoints = toNx3(objectPts);
	det.ids = ids;
	if (!det.found)
		det.subpixel = false;
	return det;
}

BoardDetection detectChessboard(const cv::Mat& gray, const BoardConfig& cfg)
{
	const cv::Size pattern(cfg.squaresX, cfg.squaresY);
	std::vector<cv::Point2f> corners;
	bool usedSb = false;
	if (cfg.useSb) {
		const int flags = cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY;
		usedSb = cv::findChessboardCornersSB(gray, pattern, corners, flags);
	}
	if (!usedSb || corners.empty()) {
		usedSb = false;
		corners.clear();
		const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
		if (!cv::findChessboardCorners(gray, pattern, corners, flags))
			corners.clear();
	}
	if (corners.empty())
		return finish("Chessboard", "not-found", false, {}, {}, {});
	const bool refined = refineSubpix(gray, corners, cfg.subpixWin, criteriaOf(cfg));
	std::string how = usedSb ? "findChessboardCornersSB" : "findChessboardCorners";
	how += refined ? "+cornerSubPix" : "+subpix";
	std::vector<cv::Point3f> obj;
	obj.reserve(corners.size());
	const float s = static_cast<float>(cfg.squareLength);
	for (int i = 0; i < pattern.height; ++i)
		for (int j = 0; j < pattern.width; ++j)
			obj.emplace_back(j * s, i * s, 0.f);
	if (obj.size() != corners.size())
		return finish("Chessboard", "size-mismatch", false, {}, {}, {});
	return finish("Chessboard", how, true, corners, obj, {});
}

BoardDetection detectCircles(const cv::Mat& gray, const BoardConfig& cfg, bool asymmetric)
{
	const cv::Size pattern(cfg.squaresX, cfg.squaresY);
	std::vector<cv::Point2f> centers;
	const int flags = asymmetric ? cv::CALIB_CB_ASYMMETRIC_GRID : cv::CALIB_CB_SYMMETRIC_GRID;
	const std::string name = asymmetric ? "CirclesAsymmetric" : "CirclesSymmetric";
	if (!cv::findCirclesGrid(gray, pattern, centers, flags | cv::CALIB_CB_CLUSTERING)) {
		centers.clear();
		if (!cv::findCirclesGrid(gray, pattern, centers, flags))
			return finish(name, "not-found", false, {}, {}, {});
	}
	std::vector<cv::Point3f> obj;
	const float s = static_cast<float>(cfg.squareLength);
	for (int i = 0; i < pattern.height; ++i) {
		for (int j = 0; j < pattern.width; ++j) {
			if (asymmetric)
				obj.emplace_back((2.f * j + (i % 2)) * s, i * s, 0.f);
			else
				obj.emplace_back(j * s, i * s, 0.f);
		}
	}
	if (obj.size() != centers.size())
		return finish(name, "size-mismatch", false, {}, {}, {});
	// findCirclesGrid 的斑点中心已经是亚像素，不再套 cornerSubPix。
	return finish(name, "findCirclesGrid+blobSubpix", true, centers, obj, {});
}

#ifdef CVTER_HAS_ARUCO
BoardDetection detectCharuco(const cv::Mat& gray, const BoardConfig& cfg)
{
	const double mlen = markerLenOf(cfg);
	cv::Ptr<cv::aruco::Dictionary> dict = dictionaryOf(cfg);
	cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
		cfg.squaresX, cfg.squaresY, static_cast<float>(cfg.squareLength), static_cast<float>(mlen), dict);
	MarkerPack markers = detectMarkersSubpix(gray, cfg, false);
	if (markers.ids.empty())
		return finish("Charuco", "not-found", false, {}, {}, {});
	std::vector<cv::Point2f> corners;
	std::vector<int> ids;
	cv::aruco::interpolateCornersCharuco(markers.corners, markers.ids, gray, board, corners, ids);
	if (corners.empty())
		return finish("Charuco", "interpolate-failed", false, {}, {}, {});
	refineSubpix(gray, corners, cfg.subpixWin, criteriaOf(cfg));
	std::vector<cv::Point3f> obj;
	obj.reserve(ids.size());
	for (int id : ids) {
		if (id < 0 || id >= static_cast<int>(board->chessboardCorners.size()))
			return finish("Charuco", "bad-id", false, {}, {}, {});
		obj.push_back(board->chessboardCorners[id]);
	}
	return finish("Charuco", "detectMarkers+cornerSubPix+interpolateCornersCharuco+cornerSubPix", true, corners, obj, ids);
}

BoardDetection detectMarkerBoard(const cv::Mat& gray, const BoardConfig& cfg, bool april)
{
	const std::string name = april ? "AprilTag" : "ArucoGrid";
	const double mlen = cfg.markerLength > 0 ? cfg.markerLength : cfg.squareLength;
	const double sep = cfg.markerSeparation > 0 ? cfg.markerSeparation : mlen * 0.25;
	cv::Ptr<cv::aruco::Dictionary> dict = dictionaryOf(cfg);
	cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(
		cfg.squaresX, cfg.squaresY, static_cast<float>(mlen), static_cast<float>(sep), dict, cfg.firstMarker);
	MarkerPack markers = detectMarkersSubpix(gray, cfg, april);
	if (markers.ids.empty())
		return finish(name, "not-found", false, {}, {}, {});
	std::vector<cv::Point2f> img;
	std::vector<cv::Point3f> obj;
	std::vector<int> ids;
	for (size_t i = 0; i < markers.ids.size(); ++i) {
		const auto it = std::find(board->ids.begin(), board->ids.end(), markers.ids[i]);
		if (it == board->ids.end())
			continue;
		const int k = static_cast<int>(std::distance(board->ids.begin(), it));
		if (k < 0 || k >= static_cast<int>(board->objPoints.size()) || markers.corners[i].size() < 4)
			continue;
		for (int c = 0; c < 4; ++c) {
			img.push_back(markers.corners[i][c]);
			obj.push_back(board->objPoints[k][c]);
		}
		ids.push_back(markers.ids[i]);
	}
	const char* how = april ? "apriltag+CORNER_REFINE_APRILTAG+cornerSubPix"
							: "aruco+CORNER_REFINE_SUBPIX+cornerSubPix";
	return finish(name, how, true, img, obj, ids);
}

BoardDetection detectDiamond(const cv::Mat& gray, const BoardConfig& cfg)
{
	const double mlen = markerLenOf(cfg);
	const float rate = static_cast<float>(cfg.squareLength / std::max(1e-9, mlen));
	MarkerPack markers = detectMarkersSubpix(gray, cfg, false);
	if (markers.ids.empty())
		return finish("CharucoDiamond", "not-found", false, {}, {}, {});
	std::vector<std::vector<cv::Point2f>> diamondCorners;
	std::vector<cv::Vec4i> diamondIds;
	cv::aruco::detectCharucoDiamond(gray, markers.corners, markers.ids, rate, diamondCorners, diamondIds,
									cv::noArray(), cv::noArray(), dictionaryOf(cfg));
	std::vector<cv::Point2f> img;
	std::vector<cv::Point3f> obj;
	std::vector<int> ids;
	const float side = static_cast<float>(cfg.squareLength);
	for (size_t i = 0; i < diamondCorners.size(); ++i) {
		if (diamondCorners[i].size() < 4)
			continue;
		refineSubpix(gray, diamondCorners[i], cfg.subpixWin, criteriaOf(cfg));
		const float ox = static_cast<float>(i) * side * 3.f;
		const cv::Point3f local[4] = {
			{ox, 0.f, 0.f},
			{ox + side, 0.f, 0.f},
			{ox + side, side, 0.f},
			{ox, side, 0.f},
		};
		for (int c = 0; c < 4; ++c) {
			img.push_back(diamondCorners[i][c]);
			obj.push_back(local[c]);
		}
		if (i < diamondIds.size())
			ids.push_back(diamondIds[i][0]);
	}
	return finish("CharucoDiamond", "detectCharucoDiamond+cornerSubPix", true, img, obj, ids);
}
#endif

cv::Mat as64(const cv::Mat& m, int rows, int cols)
{
	if (m.empty())
		return {};
	cv::Mat d;
	m.convertTo(d, CV_64F);
	if (d.total() == static_cast<size_t>(rows * cols))
		return d.reshape(1, rows);
	return {};
}

}  // namespace

bool arucoAvailable()
{
#ifdef CVTER_HAS_ARUCO
	return true;
#else
	return false;
#endif
}

std::string boardTypeName(BoardType type)
{
	switch (type) {
	case BoardType::Chessboard: return "Chessboard";
	case BoardType::Charuco: return "Charuco";
	case BoardType::CirclesSymmetric: return "CirclesSymmetric";
	case BoardType::CirclesAsymmetric: return "CirclesAsymmetric";
	case BoardType::ArucoGrid: return "ArucoGrid";
	case BoardType::CharucoDiamond: return "CharucoDiamond";
	case BoardType::AprilTag: return "AprilTag";
	}
	return "Unknown";
}

cv::Mat drawBoard(const BoardConfig& cfg, int pixelsPerUnit, int margin)
{
	const int cell = std::max(8, pixelsPerUnit);
	const int border = std::max(0, margin);
	if (cfg.squaresX < 2 || cfg.squaresY < 2)
		return {};

	if (cfg.type == BoardType::Chessboard) {
		const int sx = cfg.squaresX + 1;
		const int sy = cfg.squaresY + 1;
		cv::Mat img(sy * cell + 2 * border, sx * cell + 2 * border, CV_8U, cv::Scalar(255));
		for (int y = 0; y < sy; ++y) {
			for (int x = 0; x < sx; ++x) {
				if (((x + y) & 1) == 0) {
					cv::rectangle(img,
								  cv::Rect(border + x * cell, border + y * cell, cell, cell),
								  cv::Scalar(0), cv::FILLED);
				}
			}
		}
		return img;
	}

	if (cfg.type == BoardType::CirclesSymmetric || cfg.type == BoardType::CirclesAsymmetric) {
		const bool asym = cfg.type == BoardType::CirclesAsymmetric;
		const int cols = cfg.squaresX;
		const int rows = cfg.squaresY;
		const int rad = std::max(4, cell / 4);
		const int spanX = asym ? (2 * (cols - 1) + 1) * cell : (cols - 1) * cell;
		const int spanY = (rows - 1) * cell;
		const int W = spanX + 2 * rad + 2 * border;
		const int H = spanY + 2 * rad + 2 * border;
		cv::Mat img(H, W, CV_8U, cv::Scalar(255));
		const int ox = border + rad;
		const int oy = border + rad;
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				const int x = asym ? ox + (2 * j + (i % 2)) * cell : ox + j * cell;
				const int y = oy + i * cell;
				cv::circle(img, cv::Point(x, y), rad, cv::Scalar(0), cv::FILLED, cv::LINE_AA);
			}
		}
		return img;
	}

#ifndef CVTER_HAS_ARUCO
	return {};
#else
	try {
		cv::Ptr<cv::aruco::Dictionary> dict = dictionaryOf(cfg);
		if (cfg.type == BoardType::Charuco) {
			const double mlen = markerLenOf(cfg);
			cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
				cfg.squaresX, cfg.squaresY, static_cast<float>(cfg.squareLength), static_cast<float>(mlen), dict);
			cv::Mat img;
			board->draw(cv::Size(cfg.squaresX * cell + 2 * border, cfg.squaresY * cell + 2 * border), img, border, 1);
			return img;
		}
		if (cfg.type == BoardType::ArucoGrid || cfg.type == BoardType::AprilTag) {
			const double mlen = cfg.markerLength > 0 ? cfg.markerLength : 1.0;
			const double sep = cfg.markerSeparation > 0 ? cfg.markerSeparation : 0.25;
			cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(
				cfg.squaresX, cfg.squaresY, static_cast<float>(mlen), static_cast<float>(sep), dict, cfg.firstMarker);
			const int gap = std::max(2, static_cast<int>(std::round(cell * sep / std::max(1e-6, mlen))));
			const int W = cfg.squaresX * cell + (cfg.squaresX - 1) * gap + 2 * border;
			const int H = cfg.squaresY * cell + (cfg.squaresY - 1) * gap + 2 * border;
			cv::Mat img;
			board->draw(cv::Size(W, H), img, border, 1);
			return img;
		}
		if (cfg.type == BoardType::CharucoDiamond) {
			const double mlen = markerLenOf(cfg);
			const int mk = std::max(8, static_cast<int>(std::round(cell * mlen / std::max(1e-6, cfg.squareLength))));
			cv::Vec4i ids(cfg.firstMarker, cfg.firstMarker + 1, cfg.firstMarker + 2, cfg.firstMarker + 3);
			cv::Mat img;
			cv::aruco::drawCharucoDiamond(dict, ids, cell, mk, img, border, 1);
			return img;
		}
	} catch (const cv::Exception&) {
		return {};
	}
	return {};
#endif
}

BoardDetection detectBoard(const cv::Mat& image, const BoardConfig& cfg)
{
	cv::Mat gray = asGray(image);
	if (gray.empty() || cfg.squaresX < 2 || cfg.squaresY < 2)
		return finish(boardTypeName(cfg.type), "bad-input", false, {}, {}, {});
	try {
		switch (cfg.type) {
		case BoardType::Chessboard:
			return detectChessboard(gray, cfg);
		case BoardType::CirclesSymmetric:
			return detectCircles(gray, cfg, false);
		case BoardType::CirclesAsymmetric:
			return detectCircles(gray, cfg, true);
		case BoardType::Charuco:
		case BoardType::ArucoGrid:
		case BoardType::CharucoDiamond:
		case BoardType::AprilTag:
#ifndef CVTER_HAS_ARUCO
			return finish(boardTypeName(cfg.type), "aruco-unavailable", false, {}, {}, {});
#else
			if (cfg.type == BoardType::Charuco)
				return detectCharuco(gray, cfg);
			if (cfg.type == BoardType::CharucoDiamond)
				return detectDiamond(gray, cfg);
			return detectMarkerBoard(gray, cfg, cfg.type == BoardType::AprilTag);
#endif
		}
	} catch (const cv::Exception& e) {
		return finish(boardTypeName(cfg.type), e.what(), false, {}, {}, {});
	}
	return finish(boardTypeName(cfg.type), "not-found", false, {}, {}, {});
}

CalibResult calibrateDetections(const std::vector<BoardDetection>& detections, int width, int height)
{
	CalibResult out;
	out.width = width;
	out.height = height;
	std::vector<std::vector<cv::Point3f>> objs;
	std::vector<std::vector<cv::Point2f>> imgs;
	for (const auto& d : detections) {
		if (!d.found)
			continue;
		auto op = fromNx3(d.objectPoints);
		auto ip = fromNx2(d.imagePoints);
		if (op.size() < 4 || op.size() != ip.size())
			continue;
		objs.push_back(std::move(op));
		imgs.push_back(std::move(ip));
	}
	if (objs.size() < 3 || width < 8 || height < 8)
		return out;
	try {
		cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
		const int flags = cv::CALIB_FIX_K3 | cv::CALIB_ZERO_TANGENT_DIST;
		const double rms = cv::calibrateCamera(objs, imgs, cv::Size(width, height), K, D, out.rvecs, out.tvecs, flags);
		if (!std::isfinite(rms) || K.rows != 3 || K.cols != 3)
			return out;
		out.rms = rms;
		out.cameraMatrix = K;
		out.distCoeffs = D;
		out.found = true;
	} catch (const cv::Exception&) {
		out.found = false;
	}
	return out;
}

CalibResult calibrateImages(const std::vector<cv::Mat>& images, const BoardConfig& cfg)
{
	std::vector<BoardDetection> dets;
	int w = 0, h = 0;
	for (const auto& im : images) {
		if (im.empty())
			continue;
		if (w == 0) {
			w = im.cols;
			h = im.rows;
		}
		dets.push_back(detectBoard(im, cfg));
	}
	return calibrateDetections(dets, w, h);
}

cv::Mat undistortImage(const cv::Mat& image, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
{
	if (image.empty() || cameraMatrix.empty() || distCoeffs.empty())
		return {};
	cv::Mat K = as64(cameraMatrix, 3, 3);
	cv::Mat D;
	distCoeffs.convertTo(D, CV_64F);
	D = D.reshape(1, 1);
	if (K.empty() || D.empty())
		return {};
	cv::Mat out;
	cv::undistort(image, out, K, D);
	return out;
}

bool saveCalibration(const std::string& path, const CalibResult& result)
{
	cv::FileStorage fs(path, cv::FileStorage::WRITE);
	if (!fs.isOpened())
		return false;
	fs << "found" << (result.found ? 1 : 0);
	fs << "rms" << result.rms;
	fs << "imageWidth" << result.width;
	fs << "imageHeight" << result.height;
	fs << "cameraMatrix" << result.cameraMatrix;
	fs << "distCoeffs" << result.distCoeffs;
	fs << "nviews" << static_cast<int>(result.rvecs.size());
	for (size_t i = 0; i < result.rvecs.size(); ++i) {
		fs << ("rvec" + std::to_string(i)) << result.rvecs[i];
		if (i < result.tvecs.size())
			fs << ("tvec" + std::to_string(i)) << result.tvecs[i];
	}
	return true;
}

CalibResult loadCalibration(const std::string& path)
{
	CalibResult out;
	cv::FileStorage fs(path, cv::FileStorage::READ);
	if (!fs.isOpened())
		return out;
	int found = 0;
	fs["found"] >> found;
	fs["rms"] >> out.rms;
	fs["imageWidth"] >> out.width;
	fs["imageHeight"] >> out.height;
	fs["cameraMatrix"] >> out.cameraMatrix;
	fs["distCoeffs"] >> out.distCoeffs;
	int n = 0;
	fs["nviews"] >> n;
	for (int i = 0; i < n; ++i) {
		cv::Mat r, t;
		fs["rvec" + std::to_string(i)] >> r;
		fs["tvec" + std::to_string(i)] >> t;
		if (!r.empty())
			out.rvecs.push_back(r);
		if (!t.empty())
			out.tvecs.push_back(t);
	}
	out.found = found != 0 && out.cameraMatrix.rows == 3 && out.cameraMatrix.cols == 3;
	return out;
}

HandEyeResult calibrateHandEye(const std::vector<cv::Mat>& R_gripper2base,
							   const std::vector<cv::Mat>& t_gripper2base,
							   const std::vector<cv::Mat>& R_target2cam,
							   const std::vector<cv::Mat>& t_target2cam,
							   HandEyeMethod method)
{
	HandEyeResult out;
	const size_t n = R_gripper2base.size();
	if (n < 3 || t_gripper2base.size() != n || R_target2cam.size() != n || t_target2cam.size() != n)
		return out;
	try {
		std::vector<cv::Mat> Rg, tg, Rt, tt;
		for (size_t i = 0; i < n; ++i) {
			cv::Mat R = as64(R_gripper2base[i], 3, 3);
			cv::Mat t = as64(t_gripper2base[i], 3, 1);
			cv::Mat Rc = as64(R_target2cam[i], 3, 3);
			cv::Mat tc = as64(t_target2cam[i], 3, 1);
			if (R.empty() || t.empty() || Rc.empty() || tc.empty())
				return out;
			Rg.push_back(R);
			tg.push_back(t);
			Rt.push_back(Rc);
			tt.push_back(tc);
		}
		cv::calibrateHandEye(Rg, tg, Rt, tt, out.R, out.t, static_cast<cv::HandEyeCalibrationMethod>(method));
		out.found = out.R.rows == 3 && out.R.cols == 3 && out.t.total() == 3;
	} catch (const cv::Exception&) {
		out.found = false;
	}
	return out;
}

}  // namespace calib
