#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace calib
{

// OpenCV 标准工业标定板。没有单独的「不对称棋盘」检测器：
// 部分遮挡 / 透视用棋盘 SB + 亚像素；棋盘和标记的混合板走 ChArUco。
enum class BoardType {
	Chessboard = 0,
	Charuco = 1,
	CirclesSymmetric = 2,
	CirclesAsymmetric = 3,
	ArucoGrid = 4,
	CharucoDiamond = 5,
	AprilTag = 6
};

enum class HandEyeMethod {
	Tsai = 0,
	Park = 1,
	Horaud = 2,
	Andreff = 3,
	Daniilidis = 4
};

// 亚像素终止条件。subpixWin 是 cornerSubPix 的半窗（OpenCV winSize）。
// 默认 5 → 实际搜索窗 11×11；迭代 40 次、epsilon 0.001。
// 棋盘还会优先用 findChessboardCornersSB（本身就是亚像素），再 cornerSubPix。
// 圆点阵用 findCirclesGrid 的斑点亚像素中心，不再套 cornerSubPix。
// ChArUco / ArUco / AprilTag：检测参数里的 CORNER_REFINE_SUBPIX（AprilTag 用
// CORNER_REFINE_APRILTAG），然后对角点再做 cornerSubPix。
// ChArUco 棋盘角用 interpolateCornersCharuco，再 cornerSubPix。
struct BoardConfig {
	BoardType type = BoardType::Chessboard;
	// Chessboard：内角点数。Circles*：圆点行列数。
	// Charuco：棋盘格格子数。ArucoGrid / AprilTag：标记行列数。
	// CharucoDiamond：只画/检一块菱形，这两个数不参与布局。
	int squaresX = 9;
	int squaresY = 6;
	double squareLength = 1.0;
	double markerLength = 0.7;
	double markerSeparation = 0.25;
	// <0 时按板型选默认字典：AprilTag → 36h11，其余 → 4X4_50。
	int dictionary = -1;
	int firstMarker = 0;
	bool useSb = true;
	int subpixWin = 5;
	int subpixIters = 40;
	double subpixEps = 0.001;
};

struct BoardDetection {
	bool found = false;
	bool subpixel = false;
	std::string board;
	std::string refine;
	int count = 0;
	cv::Mat imagePoints;   // Nx2 CV_64F
	cv::Mat objectPoints;  // Nx3 CV_64F
	std::vector<int> ids;
};

struct CalibResult {
	bool found = false;
	double rms = 0;
	int width = 0;
	int height = 0;
	cv::Mat cameraMatrix;  // 3x3 CV_64F
	cv::Mat distCoeffs;    // 1xN CV_64F
	std::vector<cv::Mat> rvecs;
	std::vector<cv::Mat> tvecs;
};

struct HandEyeResult {
	bool found = false;
	cv::Mat R;  // 3x3，相机到夹爪
	cv::Mat t;  // 3x1
};

bool arucoAvailable();

std::string boardTypeName(BoardType type);

// pixelsPerUnit：每个方格/圆间距的像素。margin：白边。
cv::Mat drawBoard(const BoardConfig& cfg, int pixelsPerUnit = 40, int margin = 30);

BoardDetection detectBoard(const cv::Mat& image, const BoardConfig& cfg = BoardConfig());

CalibResult calibrateDetections(const std::vector<BoardDetection>& detections, int width, int height);

CalibResult calibrateImages(const std::vector<cv::Mat>& images, const BoardConfig& cfg = BoardConfig());

cv::Mat undistortImage(const cv::Mat& image, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);

bool saveCalibration(const std::string& path, const CalibResult& result);
CalibResult loadCalibration(const std::string& path);

// 输入每帧 gripper→base 与 target→camera。输出 cam→gripper。
HandEyeResult calibrateHandEye(const std::vector<cv::Mat>& R_gripper2base,
							   const std::vector<cv::Mat>& t_gripper2base,
							   const std::vector<cv::Mat>& R_target2cam,
							   const std::vector<cv::Mat>& t_target2cam,
							   HandEyeMethod method = HandEyeMethod::Tsai);

}  // namespace calib
