#include "cv_caliper/caliper.h"
#include "cv_caliper/metrology.h"

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

ProfileEdge refineProfilePeak(const std::vector<float>& deriv, int best, float score)
{
    ProfileEdge out;
    if (best < 1 || best + 1 >= static_cast<int>(deriv.size())) {
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
    out.contrast = static_cast<double>(score);
    return out;
}

std::vector<ProfileEdge> findProfilePeaks(const std::vector<float>& profile, const CaliperParam& param)
{
    std::vector<ProfileEdge> peaks;
    const int n = static_cast<int>(profile.size());
    if (n < 5) {
        return peaks;
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
    for (int i = 1; i < n - 1; ++i) {
        const float s = scoreAt(i);
        if (s < th) {
            continue;
        }
        if (s < scoreAt(i - 1) || s < scoreAt(i + 1)) {
            continue;
        }
        ProfileEdge edge = refineProfilePeak(deriv, i, s);
        if (!edge.found) {
            continue;
        }
        if (!peaks.empty() && edge.offset - peaks.back().offset < 2.5) {
            if (edge.contrast > peaks.back().contrast) {
                peaks.back() = edge;
            }
        } else {
            peaks.push_back(edge);
        }
    }
    return peaks;
}

ProfileEdge findProfileEdge(const std::vector<float>& profile, const CaliperParam& param)
{
    ProfileEdge out;
    const int n = static_cast<int>(profile.size());
    if (n < 5) {
        return out;
    }
    if (param.select == EdgeSelect::Nth || param.select == EdgeSelect::All) {
        const std::vector<ProfileEdge> peaks = findProfilePeaks(profile, param);
        const int idx = (param.select == EdgeSelect::All) ? 0 : param.edgeIndex;
        if (idx < 0 || idx >= static_cast<int>(peaks.size())) {
            return out;
        }
        return peaks[idx];
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

LineResult fitPointsToLine(const std::vector<cv::Point2d>& pts, const CaliperParam& param)
{
    LineResult empty;
    empty.points = pts;
    empty.numPoints = static_cast<int>(pts.size());
    if (static_cast<int>(pts.size()) < std::max(3, param.minPoints)) {
        return empty;
    }
    cv::Point2d mean, dir;
    double rms = 0.0;
    if (!fitLineSvd(pts, mean, dir, rms)) {
        return empty;
    }
    auto inliers = rejectLineOutliers(pts, mean, dir, param.outlierRatio);
    if (static_cast<int>(inliers.size()) < std::max(3, param.minPoints / 2)) {
        inliers = pts;
    }
    if (!fitLineSvd(inliers, mean, dir, rms)) {
        return empty;
    }
    return packLine(pts, inliers, mean, dir, rms);
}

CircleResult fitPointsToCircle(const std::vector<cv::Point2d>& pts,
                               cv::Point2d nominalCenter,
                               double search,
                               const CaliperParam& param)
{
    CircleResult empty;
    empty.points = pts;
    empty.numPoints = static_cast<int>(pts.size());
    const int need = std::max(4, param.minPoints);
    if (static_cast<int>(pts.size()) < need) {
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
    if (std::hypot(c.x - nominalCenter.x, c.y - nominalCenter.y) > std::max(10.0, 1.5 * search)) {
        empty.inliers = inliers;
        empty.numInliers = static_cast<int>(inliers.size());
        return empty;
    }
    if (!(rms < 4.0) || !(r > 1.0)) {
        empty.inliers = inliers;
        empty.numInliers = static_cast<int>(inliers.size());
        return empty;
    }
    CircleResult out;
    out.found = std::isfinite(r) && std::isfinite(rms);
    out.center = c;
    out.radius = r;
    out.rms = rms;
    out.points = pts;
    out.inliers = inliers;
    out.numPoints = static_cast<int>(pts.size());
    out.numInliers = static_cast<int>(inliers.size());
    return out;
}

std::vector<std::vector<cv::Point2d>> collectLinearPeaks(const cv::Mat& gray,
                                                        cv::Point2d center,
                                                        double phi,
                                                        double length1,
                                                        double length2,
                                                        const CaliperParam& param)
{
    std::vector<std::vector<cv::Point2d>> spokes;
    if (gray.empty() || length1 < 4.0 || length2 < 3.0) {
        return spokes;
    }
    const cv::Point2d lengthDir(std::cos(phi), std::sin(phi));
    const cv::Point2d searchDir(std::sin(phi), -std::cos(phi));
    const int nCal = autoCount(length1, param, 8, 400);
    const auto kernel = gaussKernel(param.sigma);
    spokes.reserve(nCal);
    for (int i = 0; i < nCal; ++i) {
        const double t = (nCal == 1) ? 0.0 : (-length1 + (2.0 * length1) * i / (nCal - 1));
        const cv::Point2d origin(center.x + lengthDir.x * t, center.y + lengthDir.y * t);
        auto profile = sampleSearchProfile(gray, origin, searchDir, lengthDir, length2, param.projection);
        profile = conv1d(profile, kernel);
        const std::vector<ProfileEdge> peaks = findProfilePeaks(profile, param);
        std::vector<cv::Point2d> pts;
        pts.reserve(peaks.size());
        for (const auto& edge : peaks) {
            const double s = profileOffsetToSearch(edge.offset, length2, static_cast<int>(profile.size()));
            pts.emplace_back(origin.x + searchDir.x * s, origin.y + searchDir.y * s);
        }
        spokes.push_back(std::move(pts));
    }
    return spokes;
}

std::vector<std::vector<cv::Point2d>> collectRadialPeaks(const cv::Mat& gray,
                                                        cv::Point2d center,
                                                        double radius,
                                                        double search,
                                                        double startDeg,
                                                        double endDeg,
                                                        const CaliperParam& param)
{
    std::vector<std::vector<cv::Point2d>> spokes;
    if (gray.empty() || radius < 4.0 || search < 2.0) {
        return spokes;
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
    nCal = std::clamp(nCal, span >= 350.0 ? 12 : 6, 720);
    const auto kernel = gaussKernel(param.sigma);
    spokes.reserve(nCal);
    for (int i = 0; i < nCal; ++i) {
        const double deg = a0 + span * i / std::max(1, nCal - 1);
        const double rad = deg * kPi / 180.0;
        const cv::Point2d radial(std::cos(rad), std::sin(rad));
        const cv::Point2d tangent(-radial.y, radial.x);
        const cv::Point2d searchDir = param.radialInward ? cv::Point2d(-radial.x, -radial.y) : radial;
        const cv::Point2d origin(center.x + radial.x * radius, center.y + radial.y * radius);
        auto profile = sampleSearchProfile(gray, origin, searchDir, tangent, search, param.projection);
        profile = conv1d(profile, kernel);
        const std::vector<ProfileEdge> peaks = findProfilePeaks(profile, param);
        std::vector<cv::Point2d> pts;
        pts.reserve(peaks.size());
        for (const auto& edge : peaks) {
            const double s = profileOffsetToSearch(edge.offset, search, static_cast<int>(profile.size()));
            pts.emplace_back(origin.x + searchDir.x * s, origin.y + searchDir.y * s);
        }
        spokes.push_back(std::move(pts));
    }
    return spokes;
}

double unwrapAngle(double angle, double ref)
{
    while (angle - ref > kPi) {
        angle -= 2.0 * kPi;
    }
    while (ref - angle > kPi) {
        angle += 2.0 * kPi;
    }
    return angle;
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

namespace
{

double angDiffDeg(double a, double b)
{
    double d = std::fmod(a - b, 180.0);
    if (d > 90.0) {
        d -= 180.0;
    }
    if (d < -90.0) {
        d += 180.0;
    }
    return std::abs(d);
}

double sideHalfSpan(double sideLen, double search)
{
    const double half = 0.5 * sideLen;
    const double inset = std::min(half * 0.35, std::max(4.0, 0.5 * search));
    double span = half - inset;
    if (span < 4.0) {
        span = std::max(4.0, half * 0.7);
    }
    return span;
}

LineResult measureSide(const cv::Mat& image,
                       cv::Point2d mid,
                       cv::Point2d along,
                       cv::Point2d inward,
                       double halfLen,
                       double search,
                       const CaliperParam& param)
{
    cv::Point2d searchDir(along.y, -along.x);
    const double s = std::hypot(searchDir.x, searchDir.y);
    if (s > 1e-12) {
        searchDir.x /= s;
        searchDir.y /= s;
    }
    if (searchDir.x * inward.x + searchDir.y * inward.y < 0.0) {
        along.x = -along.x;
        along.y = -along.y;
    }
    const double phi = std::atan2(along.y, along.x);
    return findLineOriented(image, mid, phi, halfLen, search, param);
}

}  // namespace

RectResult findRectangle(const cv::Mat& image,
                         cv::Point2d center,
                         double width,
                         double height,
                         double phi,
                         double search,
                         const CaliperParam& param)
{
    RectResult out;
    if (image.empty() || width < 12.0 || height < 12.0 || search < 3.0) {
        return out;
    }
    const cv::Point2d ux(std::cos(phi), std::sin(phi));
    const cv::Point2d uy(-std::sin(phi), std::cos(phi));
    const double hw = 0.5 * width;
    const double hh = 0.5 * height;

    out.right = measureSide(image, center + ux * hw, uy, -ux, sideHalfSpan(height, search), search, param);
    out.left = measureSide(image, center - ux * hw, uy, ux, sideHalfSpan(height, search), search, param);
    out.bottom = measureSide(image, center + uy * hh, ux, -uy, sideHalfSpan(width, search), search, param);
    out.top = measureSide(image, center - uy * hh, ux, uy, sideHalfSpan(width, search), search, param);

    if (!out.top.found || !out.right.found || !out.bottom.found || !out.left.found) {
        return out;
    }
    cv::Point2d tl, tr, br, bl;
    if (!metrology::intersectLines(out.top.p1, out.top.p2, out.left.p1, out.left.p2, tl) ||
        !metrology::intersectLines(out.top.p1, out.top.p2, out.right.p1, out.right.p2, tr) ||
        !metrology::intersectLines(out.bottom.p1, out.bottom.p2, out.right.p1, out.right.p2, br) ||
        !metrology::intersectLines(out.bottom.p1, out.bottom.p2, out.left.p1, out.left.p2, bl)) {
        return out;
    }
    const metrology::SizeSummary summary = metrology::summarizeRectangle(tl, tr, br, bl);
    out.topLeft = tl;
    out.topRight = tr;
    out.bottomRight = br;
    out.bottomLeft = bl;
    out.center = summary.center;
    out.width = summary.width;
    out.height = summary.height;
    out.angle = summary.angle;
    out.rms = 0.25 * (out.top.rms + out.right.rms + out.bottom.rms + out.left.rms);

    const double wTol = std::max(8.0, 2.0 * search + 0.25 * width);
    const double hTol = std::max(8.0, 2.0 * search + 0.25 * height);
    const double expectDeg = phi * 180.0 / kPi;
    out.found = summary.found &&
                std::abs(out.width - width) <= wTol &&
                std::abs(out.height - height) <= hTol &&
                angDiffDeg(out.angle, expectDeg) <= 20.0 &&
                std::isfinite(out.rms);
    return out;
}

RectResult findRectangleBox(const cv::Mat& image,
                            double x0, double y0, double x1, double y1,
                            double search,
                            const CaliperParam& param)
{
    if (x1 < x0) {
        std::swap(x0, x1);
    }
    if (y1 < y0) {
        std::swap(y0, y1);
    }
    const cv::Point2d center(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    return findRectangle(image, center, x1 - x0, y1 - y0, 0.0, search, param);
}

namespace
{

std::vector<std::vector<cv::Point2d>> groupSpokes(const std::vector<std::vector<cv::Point2d>>& spokes)
{
    std::vector<std::vector<cv::Point2d>> groups;
    for (const auto& spoke : spokes) {
        if (static_cast<int>(groups.size()) < static_cast<int>(spoke.size())) {
            groups.resize(spoke.size());
        }
        for (int k = 0; k < static_cast<int>(spoke.size()); ++k) {
            groups[k].push_back(spoke[k]);
        }
    }
    return groups;
}

std::vector<double> pairWidths(const std::vector<std::vector<cv::Point2d>>& spokes, int i0)
{
    std::vector<double> samples;
    if (i0 < 0) {
        return samples;
    }
    for (const auto& spoke : spokes) {
        const int i1 = i0 + 1;
        if (i1 >= static_cast<int>(spoke.size())) {
            continue;
        }
        samples.push_back(std::hypot(spoke[i1].x - spoke[i0].x, spoke[i1].y - spoke[i0].y));
    }
    return samples;
}

double sampleRms(const std::vector<double>& samples, double width)
{
    if (samples.empty() || !std::isfinite(width)) {
        return 0.0;
    }
    double acc = 0.0;
    for (double s : samples) {
        const double d = s - width;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(samples.size()));
}

}  // namespace

MultiLineResult findEdgesOriented(const cv::Mat& image,
                                  cv::Point2d center,
                                  double phi,
                                  double length1,
                                  double length2,
                                  const CaliperParam& param)
{
    MultiLineResult out;
    const cv::Mat gray = toGray(image);
    const auto groups = groupSpokes(collectLinearPeaks(gray, center, phi, length1, length2, param));
    for (int k = 0; k < static_cast<int>(groups.size()); ++k) {
        IndexedLine item;
        item.index = k;
        item.line = fitPointsToLine(groups[k], param);
        if (item.line.numPoints <= 0) {
            continue;
        }
        out.edges.push_back(item);
        if (item.line.found) {
            out.found = true;
        }
    }
    return out;
}

MultiLineResult findEdgesRect(const cv::Mat& image,
                              double x0, double y0, double x1, double y1,
                              bool searchHorizontal,
                              const CaliperParam& param)
{
    if (x1 < x0) {
        std::swap(x0, x1);
    }
    if (y1 < y0) {
        std::swap(y0, y1);
    }
    const cv::Point2d center(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    if (searchHorizontal) {
        return findEdgesOriented(image, center, kPi / 2.0, 0.5 * (y1 - y0), 0.5 * (x1 - x0), param);
    }
    return findEdgesOriented(image, center, 0.0, 0.5 * (x1 - x0), 0.5 * (y1 - y0), param);
}

MultiCircleResult findCircleEdges(const cv::Mat& image,
                                  cv::Point2d center,
                                  double radius,
                                  double search,
                                  double startDeg,
                                  double endDeg,
                                  const CaliperParam& param)
{
    MultiCircleResult out;
    const cv::Mat gray = toGray(image);
    const auto groups = groupSpokes(collectRadialPeaks(gray, center, radius, search, startDeg, endDeg, param));
    for (int k = 0; k < static_cast<int>(groups.size()); ++k) {
        IndexedCircle item;
        item.index = k;
        item.circle = fitPointsToCircle(groups[k], center, search, param);
        if (item.circle.numPoints <= 0) {
            continue;
        }
        out.edges.push_back(item);
        if (item.circle.found) {
            out.found = true;
        }
    }
    return out;
}

ArcResult findArc(const cv::Mat& image,
                  cv::Point2d center,
                  double radius,
                  double search,
                  double startDeg,
                  double endDeg,
                  const CaliperParam& param)
{
    ArcResult arc;
    const CircleResult hit = findCircle(image, center, radius, search, startDeg, endDeg, param);
    arc.center = hit.center;
    arc.radius = hit.radius;
    arc.rms = hit.rms;
    arc.points = hit.points;
    arc.inliers = hit.inliers;
    arc.numPoints = hit.numPoints;
    arc.numInliers = hit.numInliers;
    arc.found = hit.found;
    if (!hit.found || hit.inliers.size() < 3) {
        arc.found = false;
        return arc;
    }
    double a0 = startDeg;
    double a1 = endDeg;
    if (a1 < a0) {
        a1 += 360.0;
    }
    const double requested = a1 - a0;
    if (requested >= 350.0) {
        arc.spanDeg = 360.0;
        arc.arcLength = 2.0 * kPi * hit.radius;
        arc.chord = 0.0;
        arc.startDeg = startDeg;
        arc.endDeg = startDeg + 360.0;
        arc.chordStart = hit.center;
        arc.chordEnd = hit.center;
        return arc;
    }
    const double mid = 0.5 * (a0 + a1) * kPi / 180.0;
    double lo = 1e300;
    double hi = -1e300;
    cv::Point2d startPt = hit.inliers.front();
    cv::Point2d endPt = hit.inliers.front();
    for (const auto& p : hit.inliers) {
        const double u = unwrapAngle(std::atan2(p.y - hit.center.y, p.x - hit.center.x), mid);
        if (u < lo) {
            lo = u;
            startPt = p;
        }
        if (u > hi) {
            hi = u;
            endPt = p;
        }
    }
    arc.spanDeg = (hi - lo) * 180.0 / kPi;
    arc.startDeg = lo * 180.0 / kPi;
    arc.endDeg = arc.startDeg + arc.spanDeg;
    arc.arcLength = hit.radius * std::max(0.0, hi - lo);
    arc.chord = std::hypot(endPt.x - startPt.x, endPt.y - startPt.y);
    arc.chordStart = startPt;
    arc.chordEnd = endPt;
    if (!(arc.spanDeg > 1.0) || !std::isfinite(arc.arcLength)) {
        arc.found = false;
    }
    return arc;
}

DualWidthResult measureDualWidth(const cv::Mat& image,
                                 cv::Point2d center,
                                 double phi,
                                 double length1,
                                 double length2,
                                 const CaliperParam& param)
{
    DualWidthResult out;
    CaliperParam all = param;
    all.select = EdgeSelect::All;
    const MultiLineResult edges = findEdgesOriented(image, center, phi, length1, length2, all);
    for (const auto& edge : edges.edges) {
        if (edge.index == param.edgeIndex) {
            out.edge0 = edge.line;
        } else if (edge.index == param.edgeIndex + 1) {
            out.edge1 = edge.line;
        }
    }
    const auto samples = pairWidths(collectLinearPeaks(toGray(image), center, phi, length1, length2, param),
                                    param.edgeIndex);
    out.samples = samples;
    out.numSamples = static_cast<int>(samples.size());
    if (!out.edge0.found || !out.edge1.found) {
        return out;
    }
    const double parallel = metrology::distanceParallelLines(out.edge0.p1, out.edge0.p2, out.edge1.p1, out.edge1.p2, 12.0);
    const double fallback = metrology::distanceBetweenLines(out.edge0.p1, out.edge0.p2, out.edge1.p1, out.edge1.p2);
    out.angle = metrology::angleBetweenLines(out.edge0.p1, out.edge0.p2, out.edge1.p1, out.edge1.p2);
    out.width = std::isfinite(parallel) ? parallel : fallback;
    out.rms = samples.empty() ? 0.5 * (out.edge0.rms + out.edge1.rms) : sampleRms(samples, out.width);
    const int need = std::max(4, param.minPoints / 2);
    out.found = std::isfinite(out.width) && out.width > 0.5 && out.numSamples >= need;
    return out;
}

DualWidthResult measureRadialWidth(const cv::Mat& image,
                                   cv::Point2d center,
                                   double radius,
                                   double search,
                                   double startDeg,
                                   double endDeg,
                                   const CaliperParam& param)
{
    DualWidthResult out;
    out.radial = true;
    CaliperParam all = param;
    all.select = EdgeSelect::All;
    const MultiCircleResult edges = findCircleEdges(image, center, radius, search, startDeg, endDeg, all);
    for (const auto& edge : edges.edges) {
        if (edge.index == param.edgeIndex) {
            out.circle0 = edge.circle;
        } else if (edge.index == param.edgeIndex + 1) {
            out.circle1 = edge.circle;
        }
    }
    const auto samples = pairWidths(
        collectRadialPeaks(toGray(image), center, radius, search, startDeg, endDeg, param), param.edgeIndex);
    out.samples = samples;
    out.numSamples = static_cast<int>(samples.size());
    if (!out.circle0.found || !out.circle1.found) {
        return out;
    }
    out.width = std::abs(out.circle1.radius - out.circle0.radius);
    out.rms = samples.empty() ? 0.5 * (out.circle0.rms + out.circle1.rms) : sampleRms(samples, out.width);
    out.angle = 0.0;
    const int need = std::max(4, param.minPoints / 2);
    out.found = std::isfinite(out.width) && out.width > 0.5 && out.numSamples >= need;
    return out;
}

GapResult gapBetween(const LineResult& a, const LineResult& b)
{
    GapResult g;
    g.first = a;
    g.second = b;
    if (!a.found || !b.found) {
        return g;
    }
    g.angle = metrology::angleBetweenLines(a.p1, a.p2, b.p1, b.p2);
    g.parallel = std::isfinite(g.angle) && g.angle <= 12.0;
    double width = metrology::distanceParallelLines(a.p1, a.p2, b.p1, b.p2, 12.0);
    if (!std::isfinite(width)) {
        width = metrology::distanceBetweenLines(a.p1, a.p2, b.p1, b.p2);
    }
    const auto foot = metrology::footOfPerpendicular(a.center, b.p1, b.p2, false);
    g.p1 = a.center;
    g.p2 = foot.found ? foot.point : b.center;
    g.width = width;
    g.rms = 0.5 * (a.rms + b.rms);
    g.found = std::isfinite(width) && width > 0.5;
    return g;
}

GapResult measureGap(const cv::Mat& image,
                     cv::Point2d center,
                     double phi,
                     double length1,
                     double halfGap,
                     double search,
                     const CaliperParam& param)
{
    GapResult empty;
    if (image.empty() || length1 < 4.0 || halfGap < 1.0 || search < 2.0) {
        return empty;
    }
    const cv::Point2d across(std::sin(phi), -std::cos(phi));
    const cv::Point2d leftMid(center.x - across.x * halfGap, center.y - across.y * halfGap);
    const cv::Point2d rightMid(center.x + across.x * halfGap, center.y + across.y * halfGap);
    const LineResult first = findLineOriented(image, leftMid, phi, length1, search, param);
    const LineResult second = findLineOriented(image, rightMid, phi + kPi, length1, search, param);
    return gapBetween(first, second);
}

}  // namespace caliper
