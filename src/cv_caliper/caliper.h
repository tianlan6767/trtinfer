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
    Strongest = 2,
    // 一条剖面上的全部边缘，按搜索方向编号。单边接口只返回第 edgeIndex 条（默认 0）。
    All = 3,
    // 沿搜索方向第 edgeIndex 条过阈值的局部峰，从 0 开始。
    Nth = 4
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
    // Nth / All 的单边接口：沿搜索方向的边缘序号，从 0 开始。
    int edgeIndex = 0;
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

// 四边卡尺拟合矩形。phi 与 findLineOriented 相同：弧度，0 沿 +X，正方向朝 +Y。
// 每条边的搜索方向指向矩形内部。亮目标、暗背景用 DarkToLight。
struct RectResult {
    bool found = false;
    cv::Point2d center;
    double width = 0.0;
    double height = 0.0;
    double angle = 0.0;
    double rms = 0.0;
    cv::Point2d topLeft;
    cv::Point2d topRight;
    cv::Point2d bottomRight;
    cv::Point2d bottomLeft;
    LineResult top;
    LineResult right;
    LineResult bottom;
    LineResult left;
};

RectResult findRectangle(const cv::Mat& image,
                         cv::Point2d center,
                         double width,
                         double height,
                         double phi,
                         double search,
                         const CaliperParam& param = CaliperParam());

// 轴对齐预期框。search 是每条边的搜索半宽。
RectResult findRectangleBox(const cv::Mat& image,
                            double x0, double y0, double x1, double y1,
                            double search,
                            const CaliperParam& param = CaliperParam());

struct IndexedLine {
    int index = 0;
    LineResult line;
};

struct IndexedCircle {
    int index = 0;
    CircleResult circle;
};

struct MultiLineResult {
    bool found = false;
    std::vector<IndexedLine> edges;
};

struct MultiCircleResult {
    bool found = false;
    std::vector<IndexedCircle> edges;
};

// 圆弧段。startDeg/endDeg 是测得的图像角（度，y 向下）；spanDeg 为张角。
struct ArcResult {
    bool found = false;
    cv::Point2d center;
    double radius = 0.0;
    double rms = 0.0;
    double startDeg = 0.0;
    double endDeg = 0.0;
    double spanDeg = 0.0;
    double arcLength = 0.0;
    double chord = 0.0;
    cv::Point2d chordStart;
    cv::Point2d chordEnd;
    int numPoints = 0;
    int numInliers = 0;
    std::vector<cv::Point2d> points;
    std::vector<cv::Point2d> inliers;
};

// 沿同一组卡尺取两条边，测壁厚或槽宽。edge0 是序号 edgeIndex，edge1 是下一条。
// radial 为 true 时读 circle0 / circle1，width = |r1 - r0|。
struct DualWidthResult {
    bool found = false;
    bool radial = false;
    double width = 0.0;
    double rms = 0.0;
    double angle = 0.0;
    int numSamples = 0;
    LineResult edge0;
    LineResult edge1;
    CircleResult circle0;
    CircleResult circle1;
    std::vector<double> samples;
};

// 开口两侧对向卡尺。phi 与 findLineOriented 相同，沿边缘方向。
struct GapResult {
    bool found = false;
    bool parallel = false;
    double width = 0.0;
    double angle = 0.0;
    double rms = 0.0;
    LineResult first;
    LineResult second;
    cv::Point2d p1;
    cv::Point2d p2;
};

// 沿搜索方向的每一条边各拟合一条，序号 0 是最先碰到的。select 不参与这里。
MultiLineResult findEdgesOriented(const cv::Mat& image,
                                  cv::Point2d center,
                                  double phi,
                                  double length1,
                                  double length2,
                                  const CaliperParam& param = CaliperParam());

MultiLineResult findEdgesRect(const cv::Mat& image,
                              double x0, double y0, double x1, double y1,
                              bool searchHorizontal = true,
                              const CaliperParam& param = CaliperParam());

// 径向每一条边各拟合一个圆。select 不参与，序号 0 是搜索方向上先碰到的。
MultiCircleResult findCircleEdges(const cv::Mat& image,
                                  cv::Point2d center,
                                  double radius,
                                  double search,
                                  double startDeg = 0.0,
                                  double endDeg = 360.0,
                                  const CaliperParam& param = CaliperParam());

// 与 findCircle 相同的弧段采样，并给出张角、弧长、弦长。
ArcResult findArc(const cv::Mat& image,
                  cv::Point2d center,
                  double radius,
                  double search,
                  double startDeg = 0.0,
                  double endDeg = 360.0,
                  const CaliperParam& param = CaliperParam());

// 搜索窗要盖住两条边。极性相反时用 Both。width 是两条拟合边的间距。
DualWidthResult measureDualWidth(const cv::Mat& image,
                                 cv::Point2d center,
                                 double phi,
                                 double length1,
                                 double length2,
                                 const CaliperParam& param = CaliperParam());

// 环形壁厚。radius 取在两条边之间，search 要盖住内边和外边。
DualWidthResult measureRadialWidth(const cv::Mat& image,
                                   cv::Point2d center,
                                   double radius,
                                   double search,
                                   double startDeg = 0.0,
                                   double endDeg = 360.0,
                                   const CaliperParam& param = CaliperParam());

// halfGap 是开口半宽，卡尺中心落在预期的两条边上，向开口内侧搜索。
GapResult measureGap(const cv::Mat& image,
                     cv::Point2d center,
                     double phi,
                     double length1,
                     double halfGap,
                     double search,
                     const CaliperParam& param = CaliperParam());

GapResult gapBetween(const LineResult& a, const LineResult& b);

}  // namespace caliper
