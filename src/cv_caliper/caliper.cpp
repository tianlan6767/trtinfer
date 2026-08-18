#include "cv_caliper/caliper.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace caliper
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

cv::Mat toGray(const cv::Mat& image)
{
    cv::Mat gray;
    if (image.empty()) {
        return gray;
    }
    if (image.channels() == 1) {
        if (image.depth() == CV_8U) {
            return image;
        }
        image.convertTo(gray, CV_8U);
        return gray;
    }
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        return gray;
    }
    image.convertTo(gray, CV_8U);
    return gray;
}

inline float sampleBilinear(const cv::Mat& gray, double x, double y)
{
    const int w = gray.cols;
    const int h = gray.rows;
    if (w <= 1 || h <= 1) {
        return 0.f;
    }
    if (x < 0.0 || y < 0.0 || x > w - 1.0 || y > h - 1.0) {
        const int ix = std::clamp(static_cast<int>(std::lround(x)), 0, w - 1);
        const int iy = std::clamp(static_cast<int>(std::lround(y)), 0, h - 1);
        return static_cast<float>(gray.at<uchar>(iy, ix));
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float dx = static_cast<float>(x - x0);
    const float dy = static_cast<float>(y - y0);
    const float v00 = static_cast<float>(gray.at<uchar>(y0, x0));
    const float v10 = static_cast<float>(gray.at<uchar>(y0, x1));
    const float v01 = static_cast<float>(gray.at<uchar>(y1, x0));
    const float v11 = static_cast<float>(gray.at<uchar>(y1, x1));
    return (1.f - dx) * (1.f - dy) * v00 + dx * (1.f - dy) * v10 +
           (1.f - dx) * dy * v01 + dx * dy * v11;
}

std::vector<float> gaussKernel(double sigma)
{
    const double s = std::max(0.3, sigma);
    const int r = std::max(1, static_cast<int>(std::ceil(3.0 * s)));
    std::vector<float> k(2 * r + 1);
    float sum = 0.f;
    for (int i = -r; i <= r; ++i) {
        const float v = static_cast<float>(std::exp(-(i * i) / (2.0 * s * s)));
        k[i + r] = v;
        sum += v;
    }
    for (float& v : k) {
        v /= sum;
    }
    return k;
}

std::vector<float> conv1d(const std::vector<float>& src, const std::vector<float>& k)
{
    const int n = static_cast<int>(src.size());
    const int r = static_cast<int>(k.size() / 2);
    std::vector<float> dst(n, 0.f);
    for (int i = 0; i < n; ++i) {
        float acc = 0.f;
        for (int j = -r; j <= r; ++j) {
            int t = i + j;
            if (t < 0) t = 0;
            if (t >= n) t = n - 1;
            acc += src[t] * k[j + r];
        }
        dst[i] = acc;
    }
    return dst;
}

struct ProfileEdge {
    bool found = false;
    double offset = 0.0;
    double contrast = 0.0;
};

ProfileEdge findProfileEdge(const std::vector<float>& profile, const CaliperParam& param)
{
    ProfileEdge out;
    const int n = static_cast<int>(profile.size());
    if (n < 5) {
        return out;
    }
    std::vector<float> deriv(n, 0.f);
    for (int i = 1; i < n - 1; ++i) {
        deriv[i] = 0.5f * (profile[i + 1] - profile[i - 1]);
    }

    auto scoreAt = [&](int i) -> float {
        const float g = deriv[i];
        if (param.polarity == Polarity::LightToDark) {
            return -g;
        }
        if (param.polarity == Polarity::DarkToLight) {
            return g;
        }
        return std::abs(g);
    };

    const float th = static_cast<float>(param.contrast);
    int best = -1;
    float bestScore = th;

    if (param.select == EdgeSelect::First) {
        for (int i = 1; i < n - 1; ++i) {
            if (scoreAt(i) >= th) {
                best = i;
                bestScore = scoreAt(i);
                break;
            }
        }
    } else if (param.select == EdgeSelect::Last) {
        for (int i = n - 2; i >= 1; --i) {
            if (scoreAt(i) >= th) {
                best = i;
                bestScore = scoreAt(i);
                break;
            }
        }
    } else {
        for (int i = 1; i < n - 1; ++i) {
            const float s = scoreAt(i);
            if (s > bestScore) {
                bestScore = s;
                best = i;
            }
        }
    }
    if (best < 1) {
        return out;
    }

    const float a = deriv[best - 1];
    const float b = deriv[best];
    const float c = deriv[best + 1];
    double sub = 0.0;
    const float denom = a - 2.f * b + c;
    if (std::abs(denom) > 1e-6f) {
        sub = 0.5 * static_cast<double>(a - c) / static_cast<double>(denom);
        sub = std::clamp(sub, -0.75, 0.75);
    }

    out.found = true;
    out.offset = static_cast<double>(best) + sub;
    out.contrast = static_cast<double>(bestScore);
    return out;
}

int autoCount(double length, const CaliperParam& param, int minCount, int maxCount)
{
    if (param.numCalipers > 0) {
        return std::clamp(param.numCalipers, minCount, maxCount);
    }
    const int stride = std::max(1, param.stride);
    const int n = static_cast<int>(std::floor(2.0 * length / stride)) + 1;
    return std::clamp(n, minCount, maxCount);
}

bool fitLineSvd(const std::vector<cv::Point2d>& pts, cv::Point2d& mean, cv::Point2d& dir, double& rms)
{
    if (pts.size() < 2) {
        return false;
    }
    mean = cv::Point2d(0, 0);
    for (const auto& p : pts) {
        mean += p;
    }
    mean.x /= static_cast<double>(pts.size());
    mean.y /= static_cast<double>(pts.size());

    cv::Mat A(static_cast<int>(pts.size()), 2, CV_64F);
    for (int i = 0; i < A.rows; ++i) {
        A.at<double>(i, 0) = pts[i].x - mean.x;
        A.at<double>(i, 1) = pts[i].y - mean.y;
    }
    cv::Mat w, u, vt;
    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A);
    dir = cv::Point2d(vt.at<double>(0, 0), vt.at<double>(0, 1));
    const double nrm = std::hypot(dir.x, dir.y);
    if (nrm < 1e-12) {
        return false;
    }
    dir.x /= nrm;
    dir.y /= nrm;
    const cv::Point2d normal(-dir.y, dir.x);
    double acc = 0.0;
    for (const auto& p : pts) {
        const double r = (p.x - mean.x) * normal.x + (p.y - mean.y) * normal.y;
        acc += r * r;
    }
    rms = std::sqrt(acc / static_cast<double>(pts.size()));
    return std::isfinite(rms);
}

std::vector<cv::Point2d> rejectLineOutliers(const std::vector<cv::Point2d>& pts,
                                            const cv::Point2d& mean,
                                            const cv::Point2d& dir,
                                            double ratio)
{
    if (pts.size() < 3) {
        return pts;
    }
    const cv::Point2d normal(-dir.y, dir.x);
    std::vector<double> resid;
    resid.reserve(pts.size());
    for (const auto& p : pts) {
        resid.push_back(std::abs((p.x - mean.x) * normal.x + (p.y - mean.y) * normal.y));
    }
    std::vector<double> sorted = resid;
    std::sort(sorted.begin(), sorted.end());
    const double q = sorted[static_cast<int>(sorted.size() * 0.7)];
    const double th = std::max(1.2, 2.5 * q);
    std::vector<cv::Point2d> keep;
    keep.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        if (resid[i] <= th) {
            keep.push_back(pts[i]);
        }
    }
    if (keep.size() < pts.size() && keep.size() >= 3) {
        return keep;
    }
    // 再按比例砍掉最差的
    std::vector<size_t> idx(pts.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return resid[a] < resid[b]; });
    const int keepN = std::max(3, static_cast<int>(std::ceil(pts.size() * (1.0 - std::clamp(ratio, 0.05, 0.6)))));
    keep.clear();
    for (int i = 0; i < keepN; ++i) {
        keep.push_back(pts[idx[i]]);
    }
    return keep;
}

bool fitCircleAlgebraic(const std::vector<cv::Point2d>& pts, cv::Point2d& c, double& r, double& rms)
{
    if (pts.size() < 3) {
        return false;
    }
    cv::Mat A(static_cast<int>(pts.size()), 3, CV_64F);
    cv::Mat b(static_cast<int>(pts.size()), 1, CV_64F);
    for (int i = 0; i < A.rows; ++i) {
        const double x = pts[i].x;
        const double y = pts[i].y;
        A.at<double>(i, 0) = x;
        A.at<double>(i, 1) = y;
        A.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) = -(x * x + y * y);
    }
    cv::Mat sol;
    if (!cv::solve(A, b, sol, cv::DECOMP_SVD)) {
        return false;
    }
    c.x = -0.5 * sol.at<double>(0, 0);
    c.y = -0.5 * sol.at<double>(1, 0);
    const double f = sol.at<double>(2, 0);
    const double r2 = c.x * c.x + c.y * c.y - f;
    if (!(r2 > 1.0) || !std::isfinite(r2)) {
        return false;
    }
    r = std::sqrt(r2);
    if (!std::isfinite(r) || r < 1.0) {
        return false;
    }
    double acc = 0.0;
    for (const auto& p : pts) {
        const double e = std::hypot(p.x - c.x, p.y - c.y) - r;
        acc += e * e;
    }
    rms = std::sqrt(acc / static_cast<double>(pts.size()));
    return std::isfinite(rms);
}

std::vector<cv::Point2d> rejectCircleOutliers(const std::vector<cv::Point2d>& pts,
                                              const cv::Point2d& c,
                                              double r,
                                              double ratio)
{
    if (pts.size() < 4) {
        return pts;
    }
    std::vector<double> resid;
    resid.reserve(pts.size());
    for (const auto& p : pts) {
        resid.push_back(std::abs(std::hypot(p.x - c.x, p.y - c.y) - r));
    }
    std::vector<double> sorted = resid;
    std::sort(sorted.begin(), sorted.end());
    const double q = sorted[static_cast<int>(sorted.size() * 0.7)];
    const double th = std::max(1.2, 2.5 * q);
    std::vector<cv::Point2d> keep;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (resid[i] <= th) {
            keep.push_back(pts[i]);
        }
    }
    if (keep.size() >= 4) {
        return keep;
    }
    std::vector<size_t> idx(pts.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return resid[a] < resid[b]; });
    const int keepN = std::max(4, static_cast<int>(std::ceil(pts.size() * (1.0 - std::clamp(ratio, 0.05, 0.6)))));
    keep.clear();
    for (int i = 0; i < keepN; ++i) {
        keep.push_back(pts[idx[i]]);
    }
    return keep;
}

LineResult packLine(const std::vector<cv::Point2d>& allPts,
                    const std::vector<cv::Point2d>& inliers,
                    const cv::Point2d& mean,
                    const cv::Point2d& dir,
                    double rms)
{
    LineResult r;
    r.points = allPts;
    r.inliers = inliers;
    r.numPoints = static_cast<int>(allPts.size());
    r.numInliers = static_cast<int>(inliers.size());
    double tmin = 1e9, tmax = -1e9;
    for (const auto& p : inliers) {
        const double t = (p.x - mean.x) * dir.x + (p.y - mean.y) * dir.y;
        tmin = std::min(tmin, t);
        tmax = std::max(tmax, t);
    }
    cv::Point2d d = dir;
    if (d.y < 0) {
        d.x = -d.x;
        d.y = -d.y;
        const double tmp = tmin;
        tmin = -tmax;
        tmax = -tmp;
    }
    r.p1 = cv::Point2d(mean.x + d.x * tmin, mean.y + d.y * tmin);
    r.p2 = cv::Point2d(mean.x + d.x * tmax, mean.y + d.y * tmax);
    r.center = cv::Point2d(0.5 * (r.p1.x + r.p2.x), 0.5 * (r.p1.y + r.p2.y));
    r.length = tmax - tmin;
    r.angle = std::atan2(d.y, d.x) * 180.0 / kPi;
    r.rms = rms;
    r.found = r.length >= 8.0 && r.numInliers >= 3 && std::isfinite(r.rms);
    return r;
}

std::vector<float> sampleSearchProfile(const cv::Mat& gray,
                                       cv::Point2d origin,
                                       cv::Point2d searchDir,
                                       cv::Point2d lengthDir,
                                       double searchHalf,
                                       int projection)
{
    const int n = std::max(5, static_cast<int>(std::lround(2.0 * searchHalf)) + 1);
    const double step = (n <= 1) ? 1.0 : (2.0 * searchHalf) / static_cast<double>(n - 1);
    const int proj = std::max(1, projection);
    const int half = proj / 2;
    std::vector<float> profile(n, 0.f);
    for (int i = 0; i < n; ++i) {
        const double s = -searchHalf + step * i;
        float acc = 0.f;
        int cnt = 0;
        for (int k = -half; k <= half; ++k) {
            const double x = origin.x + searchDir.x * s + lengthDir.x * k;
            const double y = origin.y + searchDir.y * s + lengthDir.y * k;
            acc += sampleBilinear(gray, x, y);
            ++cnt;
        }
        profile[i] = acc / static_cast<float>(std::max(1, cnt));
    }
    return profile;
}

double profileOffsetToSearch(double index, double searchHalf, int n)
{
    if (n <= 1) {
        return 0.0;
    }
    const double step = (2.0 * searchHalf) / static_cast<double>(n - 1);
    return -searchHalf + step * index;
}

}  // namespace

LineResult findLineOriented(const cv::Mat& image,
                            cv::Point2d center,
                            double phi,
                            double length1,
                            double length2,
                            const CaliperParam& param)
{
    LineResult empty;
    const cv::Mat gray = toGray(image);
    if (gray.empty() || length1 < 4.0 || length2 < 3.0) {
        return empty;
    }

    const cv::Point2d lengthDir(std::cos(phi), std::sin(phi));
    const cv::Point2d searchDir(std::sin(phi), -std::cos(phi));
    const int nCal = autoCount(length1, param, 8, 400);
    const auto kernel = gaussKernel(param.sigma);

    std::vector<cv::Point2d> pts;
    pts.reserve(nCal);
    for (int i = 0; i < nCal; ++i) {
        const double t = (nCal == 1) ? 0.0 : (-length1 + (2.0 * length1) * i / (nCal - 1));
        const cv::Point2d origin(center.x + lengthDir.x * t, center.y + lengthDir.y * t);
        auto profile = sampleSearchProfile(gray, origin, searchDir, lengthDir, length2, param.projection);
        profile = conv1d(profile, kernel);
        const ProfileEdge edge = findProfileEdge(profile, param);
        if (!edge.found) {
            continue;
        }
        const double s = profileOffsetToSearch(edge.offset, length2, static_cast<int>(profile.size()));
        pts.emplace_back(origin.x + searchDir.x * s, origin.y + searchDir.y * s);
    }
    if (static_cast<int>(pts.size()) < param.minPoints) {
        empty.points = pts;
        empty.numPoints = static_cast<int>(pts.size());
        return empty;
    }

    cv::Point2d mean, dir;
    double rms = 0.0;
    if (!fitLineSvd(pts, mean, dir, rms)) {
        empty.points = pts;
        empty.numPoints = static_cast<int>(pts.size());
        return empty;
    }
    auto inliers = rejectLineOutliers(pts, mean, dir, param.outlierRatio);
    if (static_cast<int>(inliers.size()) < std::max(3, param.minPoints / 2)) {
        inliers = pts;
    }
    if (!fitLineSvd(inliers, mean, dir, rms)) {
        empty.points = pts;
        empty.numPoints = static_cast<int>(pts.size());
        return empty;
    }
    return packLine(pts, inliers, mean, dir, rms);
}

LineResult findLineRect(const cv::Mat& image,
                        double x0, double y0, double x1, double y1,
                        bool searchHorizontal,
                        const CaliperParam& param)
{
    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);
    const cv::Point2d center(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    if (searchHorizontal) {
        const double length1 = 0.5 * (y1 - y0);
        const double length2 = 0.5 * (x1 - x0);
        return findLineOriented(image, center, kPi / 2.0, length1, length2, param);
    }
    const double length1 = 0.5 * (x1 - x0);
    const double length2 = 0.5 * (y1 - y0);
    return findLineOriented(image, center, 0.0, length1, length2, param);
}

CircleResult findCircle(const cv::Mat& image,
                        cv::Point2d center,
                        double radius,
                        double search,
                        double startDeg,
                        double endDeg,
                        const CaliperParam& param)
{
    CircleResult empty;
    const cv::Mat gray = toGray(image);
    if (gray.empty() || radius < 4.0 || search < 2.0) {
        return empty;
    }

    double a0 = startDeg;
    double a1 = endDeg;
    if (a1 < a0) {
        a1 += 360.0;
    }
    const double span = std::max(1.0, a1 - a0);
    const double arcLen = radius * span * kPi / 180.0;
    int nCal = param.numCalipers;
    if (nCal <= 0) {
        nCal = static_cast<int>(std::floor(arcLen / std::max(1, param.stride))) + 1;
    }
    nCal = std::clamp(nCal, 12, 720);

    const auto kernel = gaussKernel(param.sigma);

    std::vector<cv::Point2d> pts;
    pts.reserve(nCal);
    for (int i = 0; i < nCal; ++i) {
        const double deg = a0 + span * i / std::max(1, nCal - 1);
        const double rad = deg * kPi / 180.0;
        const cv::Point2d radial(std::cos(rad), std::sin(rad));
        const cv::Point2d tangent(-radial.y, radial.x);
        // 极性沿 searchDir：InnerToOuter 向外，radialInward 向内
        const cv::Point2d searchDir = param.radialInward ? cv::Point2d(-radial.x, -radial.y) : radial;
        const cv::Point2d origin(center.x + radial.x * radius, center.y + radial.y * radius);
        auto profile = sampleSearchProfile(gray, origin, searchDir, tangent, search, param.projection);
        profile = conv1d(profile, kernel);
        const ProfileEdge edge = findProfileEdge(profile, param);
        if (!edge.found) {
            continue;
        }
        const double s = profileOffsetToSearch(edge.offset, search, static_cast<int>(profile.size()));
        pts.emplace_back(origin.x + searchDir.x * s, origin.y + searchDir.y * s);
    }

    empty.points = pts;
    empty.numPoints = static_cast<int>(pts.size());
    if (static_cast<int>(pts.size()) < param.minPoints) {
        return empty;
    }

    cv::Point2d c;
    double r = 0.0, rms = 0.0;
    if (!fitCircleAlgebraic(pts, c, r, rms)) {
        return empty;
    }
    auto inliers = rejectCircleOutliers(pts, c, r, param.outlierRatio);
    if (static_cast<int>(inliers.size()) < std::max(4, param.minPoints / 2)) {
        inliers = pts;
    }
    if (!fitCircleAlgebraic(inliers, c, r, rms)) {
        return empty;
    }
    // 相对预期半径过飞则失败
    if (std::abs(r - radius) > std::max(8.0, 2.0 * search + 0.25 * radius)) {
        empty.inliers = inliers;
        empty.numInliers = static_cast<int>(inliers.size());
        return empty;
    }

    CircleResult out;
    out.found = std::isfinite(r) && std::isfinite(rms) && r > 1.0;
    out.center = c;
    out.radius = r;
    out.rms = rms;
    out.points = pts;
    out.inliers = inliers;
    out.numPoints = static_cast<int>(pts.size());
    out.numInliers = static_cast<int>(inliers.size());
    return out;
}

}  // namespace caliper
