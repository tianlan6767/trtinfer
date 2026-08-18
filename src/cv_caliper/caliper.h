#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace caliper
{

enum class Polarity {
    DarkToLight = 0,
    LightToDark = 1,
    Both = 2
};

enum class EdgeSelect {
    First = 0,
    Last = 1,
    Strongest = 2
};

struct CaliperParam {
    Polarity polarity = Polarity::LightToDark;
    EdgeSelect select = EdgeSelect::Strongest;
    // 每个采样位置沿卡尺长度方向平均的像素数（Halcon 切片）
    int projection = 5;
    // 卡尺条数，0=按 stride 自动
    int numCalipers = 0;
    // 沿工具方向的间距（像素）
    int stride = 2;
    // 一维梯度幅值阈值（约等于灰度落差）
    double contrast = 18.0;
    // 剖面高斯平滑
    double sigma = 0.8;
    // 拟合时丢掉的最差比例
    double outlierRatio = 0.3;
    int minPoints = 8;
    // 圆：true=从外向内搜
    bool radialInward = false;
};

struct LineResult {
    bool found = false;
    cv::Point2d p1;
    cv::Point2d p2;
    cv::Point2d center;
    double angle = 0.0;
    double rms = 0.0;
    double length = 0.0;
    int numPoints = 0;
    int numInliers = 0;
    std::vector<cv::Point2d> points;
    std::vector<cv::Point2d> inliers;
};

struct CircleResult {
    bool found = false;
    cv::Point2d center;
    double radius = 0.0;
    double rms = 0.0;
    int numPoints = 0;
    int numInliers = 0;
    std::vector<cv::Point2d> points;
    std::vector<cv::Point2d> inliers;
};

// 旋转矩形卡尺。phi 为长度轴相对 +X 的弧度；搜索方向为长度轴顺时针 90°。
// phi=π/2 时沿 +Y 布卡尺、沿 +X 搜索，适合竖边。
LineResult findLineOriented(const cv::Mat& image,
                            cv::Point2d center,
                            double phi,
                            double length1,
                            double length2,
                            const CaliperParam& param = CaliperParam());

// 轴对齐矩形。searchHorizontal=true：沿 X 搜索竖边。
LineResult findLineRect(const cv::Mat& image,
                        double x0, double y0, double x1, double y1,
                        bool searchHorizontal = true,
                        const CaliperParam& param = CaliperParam());

// 环形卡尺拟合圆。radius 为预期半径，search 为径向半宽。
CircleResult findCircle(const cv::Mat& image,
                        cv::Point2d center,
                        double radius,
                        double search,
                        double startDeg = 0.0,
                        double endDeg = 360.0,
                        const CaliperParam& param = CaliperParam());

}  // namespace caliper
