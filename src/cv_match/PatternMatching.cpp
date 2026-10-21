#include "PatternMatching.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <cmath>
#include <algorithm>
#include <omp.h>


namespace template_matching {

	namespace {
		bool isFinitePoint(const Point2d& pt)
		{
			return std::isfinite(pt.x) && std::isfinite(pt.y);
		}

		Mat asGray(const Mat& src)
		{
			if (src.empty() || src.channels() == 1)
				return src;
			Mat gray;
			if (src.channels() == 3)
				cvtColor(src, gray, COLOR_BGR2GRAY);
			else if (src.channels() == 4)
				cvtColor(src, gray, COLOR_BGRA2GRAY);
			else
				extractChannel(src, gray, 0);
			return gray;
		}

		double nccOf(const Mat& a, const Mat& b)
		{
			if (a.empty() || b.empty() || a.size() != b.size())
				return 0.0;
			Scalar ma, sa, mb, sb;
			meanStdDev(a, ma, sa);
			meanStdDev(b, mb, sb);
			if (sa[0] < 1e-6 || sb[0] < 1e-6)
				return 0.0;
			Mat af, bf;
			a.convertTo(af, CV_32F);
			b.convertTo(bf, CV_32F);
			const double num = sum((af - (float)ma[0]).mul(bf - (float)mb[0]))[0];
			return num / (sa[0] * sb[0] * (double)a.total());
		}

		double nccMasked(const Mat& a, const Mat& b, const Mat& mask)
		{
			if (a.empty() || b.empty() || a.size() != b.size() || mask.size() != a.size())
				return -1.0;
			double sa = 0.0, sb = 0.0;
			int n = 0;
			for (int y = 0; y < a.rows; ++y)
			{
				const uchar* pa = a.ptr<uchar>(y);
				const uchar* pb = b.ptr<uchar>(y);
				const uchar* pm = mask.ptr<uchar>(y);
				for (int x = 0; x < a.cols; ++x)
				{
					if (!pm[x])
						continue;
					sa += pa[x];
					sb += pb[x];
					++n;
				}
			}
			if (n < 24)
				return -1.0;
			const double ma = sa / n, mb = sb / n;
			double va = 0.0, vb = 0.0, cov = 0.0;
			for (int y = 0; y < a.rows; ++y)
			{
				const uchar* pa = a.ptr<uchar>(y);
				const uchar* pb = b.ptr<uchar>(y);
				const uchar* pm = mask.ptr<uchar>(y);
				for (int x = 0; x < a.cols; ++x)
				{
					if (!pm[x])
						continue;
					const double da = pa[x] - ma;
					const double db = pb[x] - mb;
					va += da * da;
					vb += db * db;
					cov += da * db;
				}
			}
			if (va < 16.0 * n || vb < 1.0)
				return -1.0;
			return cov / std::sqrt(va * vb);
		}

		struct AlignedQuads
		{
			bool valid = false;
			int present = 0;
			int bad = 0;
			double worst = 1.0;
			double second = 1.0;
			double best = 0.0;
			double infoNcc = -1.0;
			int infoQuads = 0;
		};

		AlignedQuads alignQuads(const Mat& sceneIn, const Mat& templIn, const Mat& maskIn, const MatchResult& hit)
		{
			AlignedQuads out;
			const double w = std::hypot(hit.RightTop.x - hit.LeftTop.x, hit.RightTop.y - hit.LeftTop.y);
			const double h = std::hypot(hit.LeftBottom.x - hit.LeftTop.x, hit.LeftBottom.y - hit.LeftTop.y);
			if (w < 12.0 || h < 12.0)
				return out;
			int W = std::max(12, std::min(180, (int)std::lround(w)));
			int H = std::max(12, std::min(180, (int)std::lround(h)));
			Point2f src[3] = {
				Point2f((float)hit.LeftTop.x, (float)hit.LeftTop.y),
				Point2f((float)hit.RightTop.x, (float)hit.RightTop.y),
				Point2f((float)hit.LeftBottom.x, (float)hit.LeftBottom.y)};
			Point2f dst[3] = {{0.f, 0.f}, {(float)(W - 1), 0.f}, {0.f, (float)(H - 1)}};
			Mat affine = getAffineTransform(src, dst);
			if (affine.empty())
				return out;
			Mat scene = asGray(sceneIn);
			Mat templSrc = asGray(templIn);
			if (scene.empty() || templSrc.empty())
				return out;
			Mat aligned;
			warpAffine(scene, aligned, affine, Size(W, H), INTER_LINEAR, BORDER_REPLICATE);
			if (aligned.type() != CV_8U)
				aligned.convertTo(aligned, CV_8U);
			Mat templ;
			const int interp = (W * (double)H < templSrc.cols * (double)templSrc.rows) ? INTER_AREA : INTER_LINEAR;
			resize(templSrc, templ, Size(W, H), 0, 0, interp);
			if (templ.type() != CV_8U)
				templ.convertTo(templ, CV_8U);
			Mat userMask;
			if (!maskIn.empty())
			{
				Mat maskGray = maskIn.channels() == 1 ? maskIn : asGray(maskIn);
				resize(maskGray, userMask, Size(W, H), 0, 0, INTER_NEAREST);
			}

			double qncc[4];
			int qn = 0;
			for (int qy = 0; qy < 2; ++qy)
			{
				for (int qx = 0; qx < 2; ++qx)
				{
					const int x0 = qx * W / 2;
					const int y0 = qy * H / 2;
					const int x1 = (qx + 1) * W / 2;
					const int y1 = (qy + 1) * H / 2;
					if (x1 <= x0 || y1 <= y0)
						continue;
					Rect roi(x0, y0, x1 - x0, y1 - y0);
					if (!userMask.empty() && mean(userMask(roi))[0] < 80.0)
						continue;
					Scalar ignored, templStd, sceneStd;
					meanStdDev(templ(roi), ignored, templStd);
					if (templStd[0] < 5.0)
						continue;
					meanStdDev(aligned(roi), ignored, sceneStd);
					const double q = nccOf(templ(roi), aligned(roi));
					qncc[qn++] = q;
					++out.present;
					if (sceneStd[0] < 3.0 || q < 0.35)
						++out.bad;
				}
			}
			if (qn > 0)
			{
				std::sort(qncc, qncc + qn);
				out.worst = qncc[0];
				out.second = qn >= 2 ? qncc[1] : qncc[0];
				out.best = qncc[qn - 1];
			}

			// 只在没人给 mask 时，用模板边框估计背景，拿非背景像素再打一次分。
			// 大空白边对不上时，全局 NCC 会掉到阈值下，物体本身仍然对齐。
			if (userMask.empty())
			{
				const int border = std::max(2, std::min(W, H) / 12);
				double bgSum = 0.0;
				int bgN = 0;
				for (int y = 0; y < H; ++y)
				{
					const uchar* row = templ.ptr<uchar>(y);
					for (int x = 0; x < W; ++x)
					{
						if (x < border || y < border || x >= W - border || y >= H - border)
						{
							bgSum += row[x];
							++bgN;
						}
					}
				}
				const double bg = bgN > 0 ? bgSum / bgN : 0.0;
				Mat info(H, W, CV_8U, Scalar(0));
				for (int y = 0; y < H; ++y)
				{
					const uchar* row = templ.ptr<uchar>(y);
					uchar* mi = info.ptr<uchar>(y);
					for (int x = 0; x < W; ++x)
						if (std::abs((int)row[x] - (int)std::lround(bg)) > 22)
							mi[x] = 255;
				}
				const int infoCount = countNonZero(info);
				if (infoCount >= 80 && infoCount >= (int)(0.04 * W * H))
				{
					out.infoNcc = nccMasked(templ, aligned, info);
					for (int qy = 0; qy < 2; ++qy)
					{
						for (int qx = 0; qx < 2; ++qx)
						{
							Rect roi(qx * W / 2, qy * H / 2, (qx == 0 ? W / 2 : W - W / 2), (qy == 0 ? H / 2 : H - H / 2));
							if (roi.width < 2 || roi.height < 2)
								continue;
							if (countNonZero(info(roi)) < 12)
								continue;
							const double q = nccMasked(templ(roi), aligned(roi), info(roi));
							if (q >= 0.55)
								++out.infoQuads;
						}
					}
				}
			}
			out.valid = true;
			return out;
		}

		// 缺一口的弧：只有一个象限偏弱（大约 0.25–0.70），其余象限仍然高。
		// 整圈四个象限接近。只挡死一个象限时，那个象限接近 0，不在这个区间里。
		// 旁边另一颗点落进空白边时，往往有两个象限一起变差，次差也低，这里不删。
		bool partialFromQuads(const AlignedQuads& q)
		{
			if (!q.valid)
				return false;
			if (q.present >= 3 && q.bad * 2 >= q.present)
				return true;
			if (q.present >= 3 && q.worst >= 0.22 && q.worst < 0.72 && q.second >= 0.70 && (q.best - q.worst) > 0.28)
				return true;
			return false;
		}

		void refinePatternHits(vector<MatchResult>& hits, const Mat& scene, const Mat& templ, const Mat& mask,
							   double scoreThreshold)
		{
			if (hits.empty() || scene.empty() || templ.empty())
				return;
			const double rescueFloor = std::max(0.30, scoreThreshold * 0.82);
			vector<MatchResult> kept;
			kept.reserve(hits.size());
			for (MatchResult hit : hits)
			{
				if (hit.Score < rescueFloor)
					continue;
				AlignedQuads q = alignQuads(scene, templ, mask, hit);
				if (partialFromQuads(q))
					continue;
				if (hit.Score + 1e-9 >= scoreThreshold)
				{
					kept.push_back(hit);
					continue;
				}
				// 全局分刚低于阈值，但非背景像素在至少三个象限都对齐：这是弱的完整实例，不是残弧。
				if (q.valid && q.infoQuads >= 3 && q.infoNcc >= scoreThreshold)
				{
					hit.Score = q.infoNcc;
					kept.push_back(hit);
				}
			}
			hits.swap(kept);
		}
	}

	int GetTopLayer(Mat* matTempl, int iMinDstLength)
	{
		int iTopLayer = 0;
		int iMinReduceArea = iMinDstLength * iMinDstLength;
		int iArea = matTempl->cols * matTempl->rows;
		while (iArea > iMinReduceArea)
		{
			iArea /= 4;
			iTopLayer++;
		}
		return iTopLayer;
	}

	void LearnPattern(s_TemplData& m_TemplData, Mat& m_matDst, double& m_iMinReduceArea, const Mat& mask)
	{
		m_TemplData.clear();

		int iTopLayer = GetTopLayer(&m_matDst, (int)sqrt((double)m_iMinReduceArea));
		buildPyramid(m_matDst, m_TemplData.vecPyramid, iTopLayer);
		s_TemplData* templData = &m_TemplData;
		int iSize = templData->vecPyramid.size();
		templData->resize(iSize);

		Mat maskFull;
		templData->bHasMask = false;
		if (!mask.empty())
		{
			if (mask.size() != m_matDst.size() || mask.channels() != 1)
			{
				templData->clear();
				templData->bIsPatternLearned = false;
				return;
			}
			maskFull = mask.clone();
			if (maskFull.depth() != CV_8U)
				maskFull.convertTo(maskFull, CV_8U);
			threshold(maskFull, maskFull, 0, 255, THRESH_BINARY);
			templData->bHasMask = countNonZero(maskFull) > 0;
		}

		Scalar fullMean = mean(m_matDst, templData->bHasMask ? maskFull : Mat());
		templData->iBorderColor = fullMean.val[0] < 128 ? 255 : 0;

		for (int i = 0; i < iSize; i++)
		{
			Mat layerMask;
			if (templData->bHasMask)
			{
				resize(maskFull, layerMask, templData->vecPyramid[i].size(), 0, 0, INTER_NEAREST);
				threshold(layerMask, layerMask, 0, 255, THRESH_BINARY);
				templData->vecMask[i] = layerMask;
			}

			double invArea = 1. / ((double)templData->vecPyramid[i].rows * templData->vecPyramid[i].cols);
			Scalar templMean, templSdv;
			double templNorm = 0, templSum2 = 0;

			if (templData->bHasMask)
			{
				int n = countNonZero(layerMask);
				templData->vecMaskCount[i] = (double)std::max(n, 1);
				templMean = mean(templData->vecPyramid[i], layerMask);
				Mat f32, tm;
				templData->vecPyramid[i].convertTo(f32, CV_32F);
				subtract(f32, Scalar(templMean[0]), tm);
				tm.setTo(0, layerMask == 0);
				templData->vecTemplMasked[i] = tm;
				templNorm = (double)norm(tm, NORM_L2);
				if (templNorm < DBL_EPSILON)
					templData->vecResultEqual1[i] = true;
			}
			else
			{
				meanStdDev(templData->vecPyramid[i], templMean, templSdv);
				templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

				if (templNorm < DBL_EPSILON)
				{
					templData->vecResultEqual1[i] = true;
				}
				templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];


				templSum2 /= invArea;
				templNorm = std::sqrt(templNorm);
				templNorm /= std::sqrt(invArea); // care of accuracy here
			}

			templData->vecInvArea[i] = invArea;
			templData->vecTemplMean[i] = templMean;
			templData->vecTemplNorm[i] = templNorm;
		}
		templData->bIsPatternLearned = true;
	}

	Point2f ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle)
	{
		double dWidth = ptOrg.x * 2;
		double dHeight = ptOrg.y * 2;
		double dY1 = dHeight - ptInput.y, dY2 = dHeight - ptOrg.y;

		double dX = (ptInput.x - ptOrg.x) * cos(dAngle) - (dY1 - ptOrg.y) * sin(dAngle) + ptOrg.x;
		double dY = (ptInput.x - ptOrg.x) * sin(dAngle) + (dY1 - ptOrg.y) * cos(dAngle) + dY2;

		dY = -dY + dHeight;
		return Point2f((float)dX, (float)dY);
	}

	Size GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle)
	{
		double dRAngle_radian = dRAngle * D2R;
		Point ptLT(0, 0), ptLB(0, sizeSrc.height - 1), ptRB(sizeSrc.width - 1, sizeSrc.height - 1), ptRT(sizeSrc.width - 1, 0);
		Point2f ptCenter((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
		Point2f ptLT_R = ptRotatePt2f(Point2f(ptLT), ptCenter, dRAngle_radian);
		Point2f ptLB_R = ptRotatePt2f(Point2f(ptLB), ptCenter, dRAngle_radian);
		Point2f ptRB_R = ptRotatePt2f(Point2f(ptRB), ptCenter, dRAngle_radian);
		Point2f ptRT_R = ptRotatePt2f(Point2f(ptRT), ptCenter, dRAngle_radian);

		float fTopY = max(max(ptLT_R.y, ptLB_R.y), max(ptRB_R.y, ptRT_R.y));
		float fBottomY = min(min(ptLT_R.y, ptLB_R.y), min(ptRB_R.y, ptRT_R.y));
		float fRightX = max(max(ptLT_R.x, ptLB_R.x), max(ptRB_R.x, ptRT_R.x));
		float fLeftX = min(min(ptLT_R.x, ptLB_R.x), min(ptRB_R.x, ptRT_R.x));

		if (dRAngle > 360)
			dRAngle -= 360;
		else if (dRAngle < 0)
			dRAngle += 360;

		if (fabs(fabs(dRAngle) - 90) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 270) < VISION_TOLERANCE)
		{
			return Size(sizeSrc.height, sizeSrc.width);
		}
		else if (fabs(dRAngle) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 180) < VISION_TOLERANCE)
		{
			return sizeSrc;
		}

		double dAngle = dRAngle;

		if (dAngle > 0 && dAngle < 90)
		{
			;
		}
		else if (dAngle > 90 && dAngle < 180)
		{
			dAngle -= 90;
		}
		else if (dAngle > 180 && dAngle < 270)
		{
			dAngle -= 180;
		}
		else if (dAngle > 270 && dAngle < 360)
		{
			dAngle -= 270;
		}
		else//Debug
		{
			cout << "Unkown" << endl;;
		}

		float fH1 = sizeDst.width * sin(dAngle * D2R) * cos(dAngle * D2R);
		float fH2 = sizeDst.height * sin(dAngle * D2R) * cos(dAngle * D2R);

		int iHalfHeight = (int)ceil(fTopY - ptCenter.y - fH1);
		int iHalfWidth = (int)ceil(fRightX - ptCenter.x - fH2);

		Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

		bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height)
			|| (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height
				|| sizeDst.area() > sizeRet.area());
		if (bWrongSize)
			sizeRet = Size(int(fRightX - fLeftX + 0.5), int(fTopY - fBottomY + 0.5));

		return sizeRet;
	}

#ifdef __aarch64__
/*
	inline int32_t vaddvq_s32(int32x4_t v) {
		int32x2_t tmp = vpadd_s32(vget_low_s32(v), vget_high_s32(v));
		return vget_lane_s32(vpadd_s32(tmp, tmp), 0);
	}

	inline int32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength) {
		const int iBlockSize = 16, Block = iLength / iBlockSize;
		int32x4_t SumV = vdupq_n_s32(0);
		uint8x16_t Zero = vdupq_n_u8(0);

		for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize) {
			uint8x16_t SrcK = vld1q_u8(pCharKernel + Y);
			uint8x16_t SrcC = vld1q_u8(pCharConv + Y);
			int16x8_t SrcK_L = vmovl_u8(vget_low_u8(SrcK));
			int16x8_t SrcK_H = vmovl_u8(vget_high_u8(SrcK));
			int16x8_t SrcC_L = vmovl_u8(vget_low_u8(SrcC));
			int16x8_t SrcC_H = vmovl_u8(vget_high_u8(SrcC));
			int32x4_t SumT = vaddq_s32(vmull_s16(vget_low_s16(SrcK_L), vget_low_s16(SrcC_L)), vmull_s16(vget_low_s16(SrcK_H), vget_low_s16(SrcC_H)));
			SumV = vaddq_s32(SumV, SumT);
		}

		int32_t Sum = vaddvq_s32(SumV);

		for (int Y = Block * iBlockSize; Y < iLength; Y++) {
			Sum += pCharKernel[Y] * pCharConv[Y];
		}

		return Sum;
	}
*/
	inline uint32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength) {
		uint32_t Sum = 0;

		for (int Y = 0; Y < iLength; Y++) {
			Sum += pCharKernel[Y] * pCharConv[Y];
		}

		return Sum;
	}
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_X64)
	//From ImageShop
	// 4個有符號的32位的數據相加的和。
	inline int _mm_hsum_epi32(__m128i V)      // V3 V2 V1 V0
	{
		// 實測這個速度要快些，_mm_extract_epi32最慢。
		__m128i T = _mm_add_epi32(V, _mm_srli_si128(V, 8));  // V3+V1   V2+V0  V1  V0  
		T = _mm_add_epi32(T, _mm_srli_si128(T, 4));    // V3+V1+V2+V0  V2+V0+V1 V1+V0 V0 
		return _mm_cvtsi128_si32(T);       // 提取低位 
	}
	// 基於SSE的字節數據的乘法。
	// <param name="Kernel">需要卷積的核矩陣。 </param>
	// <param name="Conv">卷積矩陣。 </param>
	// <param name="Length">矩陣所有元素的長度。 </param>
	inline int IM_Conv_SIMD(unsigned char* pCharKernel, unsigned char* pCharConv, int iLength)
	{
		const int iBlockSize = 16, Block = iLength / iBlockSize;
		__m128i SumV = _mm_setzero_si128();
		__m128i Zero = _mm_setzero_si128();
		for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
		{
			__m128i SrcK = _mm_loadu_si128((__m128i*)(pCharKernel + Y));
			__m128i SrcC = _mm_loadu_si128((__m128i*)(pCharConv + Y));
			__m128i SrcK_L = _mm_unpacklo_epi8(SrcK, Zero);
			__m128i SrcK_H = _mm_unpackhi_epi8(SrcK, Zero);
			__m128i SrcC_L = _mm_unpacklo_epi8(SrcC, Zero);
			__m128i SrcC_H = _mm_unpackhi_epi8(SrcC, Zero);
			__m128i SumT = _mm_add_epi32(_mm_madd_epi16(SrcK_L, SrcC_L), _mm_madd_epi16(SrcK_H, SrcC_H));
			SumV = _mm_add_epi32(SumV, SumT);
		}
		int Sum = _mm_hsum_epi32(SumV);
		for (int Y = Block * iBlockSize; Y < iLength; Y++)
		{
			Sum += pCharKernel[Y] * pCharConv[Y];
		}
		return Sum;
	}
#else
	// Other architecture
#endif


	void CCOEFF_Denominator(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
	{
		if (pTemplData->vecResultEqual1[iLayer])
		{
			matResult = Scalar::all(1);
			return;
		}
		double* q0 = 0, * q1 = 0, * q2 = 0, * q3 = 0;

		Mat sum, sqsum;
		integral(matSrc, sum, sqsum, CV_64F);

		q0 = (double*)sqsum.data;
		q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
		q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
		q3 = q2 + pTemplData->vecPyramid[iLayer].cols;

		double* p0 = (double*)sum.data;
		double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
		double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows * sum.step);
		double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;

		int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
		int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

		//
		double dTemplMean0 = pTemplData->vecTemplMean[iLayer][0];
		double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
		double dInvArea = pTemplData->vecInvArea[iLayer];
		//

		int i, j;
		for (i = 0; i < matResult.rows; i++)
		{
			float* rrow = matResult.ptr<float>(i);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			for (j = 0; j < matResult.cols; j += 1, idx += 1, idx2 += 1)
			{
				double num = rrow[j], t;
				double wndMean2 = 0, wndSum2 = 0;

				t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
				wndMean2 += t * t;
				num -= t * dTemplMean0;
				wndMean2 *= dInvArea;


				t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
				wndSum2 += t;


				//t = std::sqrt (MAX (wndSum2 - wndMean2, 0)) * dTemplNorm;

				double diff2 = MAX(wndSum2 - wndMean2, 0);
				if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * wndSum2))
					t = 0; // avoid rounding errors
				else
					t = std::sqrt(diff2) * dTemplNorm;

				if (fabs(num) < t)
					num /= t;
				else if (fabs(num) < t * 1.125)
					num = num > 0 ? 1 : -1;
				else
					num = 0;

				rrow[j] = (float)num;
			}
		}
	}

	void MatchTemplateMasked(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
	{
		if (pTemplData->vecResultEqual1[iLayer])
		{
			matResult.create(matSrc.rows - pTemplData->vecPyramid[iLayer].rows + 1,
				matSrc.cols - pTemplData->vecPyramid[iLayer].cols + 1, CV_32FC1);
			matResult.setTo(1);
			return;
		}
		Mat src32, mask32, src2, num, sumI, sumI2;
		matSrc.convertTo(src32, CV_32F);
		pTemplData->vecMask[iLayer].convertTo(mask32, CV_32F, 1.0 / 255.0);
		multiply(src32, src32, src2);
		matchTemplate(src32, pTemplData->vecTemplMasked[iLayer], num, TM_CCORR);
		matchTemplate(src32, mask32, sumI, TM_CCORR);
		matchTemplate(src2, mask32, sumI2, TM_CCORR);

		const double n = std::max(1.0, pTemplData->vecMaskCount[iLayer]);
		const double tnorm = pTemplData->vecTemplNorm[iLayer];
		matResult.create(num.size(), CV_32FC1);
		for (int y = 0; y < num.rows; ++y)
		{
			const float* pn = num.ptr<float>(y);
			const float* pi = sumI.ptr<float>(y);
			const float* p2 = sumI2.ptr<float>(y);
			float* po = matResult.ptr<float>(y);
			for (int x = 0; x < num.cols; ++x)
			{
				const double meanI = pi[x] / n;
				double varI = p2[x] - n * meanI * meanI;
				if (varI < 0)
					varI = 0;
				double den = std::sqrt(varI) * tnorm;
				double v = pn[x];
				if (den <= 1e-12)
					po[x] = 0.f;
				else if (std::fabs(v) < den)
					po[x] = (float)(v / den);
				else if (std::fabs(v) < den * 1.125)
					po[x] = v > 0 ? 1.f : -1.f;
				else
					po[x] = 0.f;
			}
		}
	}

	void MatchTemplate(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, bool bUseSIMD)
	{
		if (pTemplData->bHasMask)
		{
			MatchTemplateMasked(matSrc, pTemplData, matResult, iLayer);
			return;
		}
		// 手写 SIMD 按 cols 跨行，忽略 step，默认不用；OpenCV 内部已有 SIMD/FFT
		if (bUseSIMD)
		{
			matResult.create(matSrc.rows - pTemplData->vecPyramid[iLayer].rows + 1,
				matSrc.cols - pTemplData->vecPyramid[iLayer].cols + 1, CV_32FC1);
			matResult.setTo(0);
			cv::Mat& matTemplate = pTemplData->vecPyramid[iLayer];
			const size_t srcStep = matSrc.step1();

			int  t_r_end = matTemplate.rows, t_r = 0;
			for (int r = 0; r < matResult.rows; r++)
			{
				float* r_matResult = matResult.ptr<float>(r);
				uchar* r_source = matSrc.ptr<uchar>(r);
				uchar* r_template, * r_sub_source;
				for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source)
				{
					r_template = matTemplate.ptr<uchar>();
					r_sub_source = r_source;
					for (t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += srcStep, r_template += matTemplate.cols)
					{
						*r_matResult = *r_matResult + IM_Conv_SIMD(r_template, r_sub_source, matTemplate.cols);
					}
				}
			}
		}
		else
			matchTemplate(matSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCORR);

		CCOEFF_Denominator(matSrc, pTemplData, matResult, iLayer);
	}

	Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap)
	{
		//比對到的區域完全不重疊 : +-一個樣板寬高
		//int iStartX = ptMaxLoc.x - iTemplateW;
		//int iStartY = ptMaxLoc.y - iTemplateH;
		//int iEndX = ptMaxLoc.x + iTemplateW;

		//int iEndY = ptMaxLoc.y + iTemplateH;
		////塗黑
		//rectangle (matResult, Rect (iStartX, iStartY, 2 * iTemplateW * (1-dMaxOverlap * 2), 2 * iTemplateH * (1-dMaxOverlap * 2)), Scalar (dMinValue), CV_FILLED);
		////得到下一個最大值
		//Point ptNewMaxLoc;
		//minMaxLoc (matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
		//return ptNewMaxLoc;

		//比對到的區域需考慮重疊比例
		int iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
		int iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);
		//塗黑
		rectangle(matResult, Rect(iStartX, iStartY, 2 * sizeTemplate.width * (1 - dMaxOverlap), 2 * sizeTemplate.height * (1 - dMaxOverlap)), Scalar(-1), CV_FILLED);
		//得到下一個最大值
		Point ptNewMaxLoc;
		minMaxLoc(matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
		return ptNewMaxLoc;
	}

	Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax)
	{
		//比對到的區域需考慮重疊比例
		int iStartX = int(ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
		int iStartY = int(ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
		Rect rectIgnore(iStartX, iStartY, int(2 * sizeTemplate.width * (1 - dMaxOverlap))
			, int(2 * sizeTemplate.height * (1 - dMaxOverlap)));
		//塗黑
		rectangle(matResult, rectIgnore, Scalar(-1), CV_FILLED);
		blockMax.UpdateMax(rectIgnore);
		Point ptReturn;
		blockMax.GetMaxValueLoc(dMaxValue, ptReturn);
		return ptReturn;
	}

	bool compareScoreBig2Small(const s_MatchParameter& lhs, const s_MatchParameter& rhs) { return  lhs.dMatchScore > rhs.dMatchScore; }
	bool comparePtWithAngle(const pair<Point2f, double> lhs, const pair<Point2f, double> rhs) { return lhs.second < rhs.second; }

	void GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI, Scalar borderFill)
	{
		double dAngle_radian = dAngle * D2R;
		Point2f ptC((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
		Point2f ptLT_rotate = ptRotatePt2f(ptLT, ptC, dAngle_radian);
		Size sizePadding(size.width + 6, size.height + 6);


		Mat rMat = getRotationMatrix2D(ptC, dAngle, 1);
		rMat.at<double>(0, 2) -= ptLT_rotate.x - 3;
		rMat.at<double>(1, 2) -= ptLT_rotate.y - 3;
		warpAffine(matSrc, matROI, rMat, sizePadding, INTER_LINEAR, BORDER_CONSTANT, borderFill);
	}

	bool SubPixEsimation(vector<s_MatchParameter>* vec, double* dNewX, double* dNewY, double* dNewAngle, double dAngleStep, int iMaxScoreIndex)
	{
		//Az=S, (A.T)Az=(A.T)s, z = ((A.T)A).inv (A.T)s
		if (vec == nullptr || vec->empty())
			return false;
		if (iMaxScoreIndex < 1 || iMaxScoreIndex + 1 >= (int)vec->size())
			return false;

		Mat matA(27, 10, CV_64F);
		Mat matZ(10, 1, CV_64F);
		Mat matS(27, 1, CV_64F);

		double dX_maxScore = (*vec)[iMaxScoreIndex].pt.x;
		double dY_maxScore = (*vec)[iMaxScoreIndex].pt.y;
		double dTheata_maxScore = (*vec)[iMaxScoreIndex].dMatchAngle;
		if (!std::isfinite(dX_maxScore) || !std::isfinite(dY_maxScore) || !std::isfinite(dTheata_maxScore))
			return false;
		int iRow = 0;
		/*for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
			{
				for (int theta = 0; theta <= 2; theta++)
				{*/
		for (int theta = 0; theta <= 2; theta++)
		{
			for (int y = -1; y <= 1; y++)
			{
				for (int x = -1; x <= 1; x++)
				{
					//xx yy tt xy xt yt x y t 1
					//0  1  2  3  4  5  6 7 8 9
					double dX = dX_maxScore + x;
					double dY = dY_maxScore + y;
					//double dT = (*vec)[theta].dMatchAngle + (theta - 1) * dAngleStep;
					double dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * D2R;
					matA.at<double>(iRow, 0) = dX * dX;
					matA.at<double>(iRow, 1) = dY * dY;
					matA.at<double>(iRow, 2) = dT * dT;
					matA.at<double>(iRow, 3) = dX * dY;
					matA.at<double>(iRow, 4) = dX * dT;
					matA.at<double>(iRow, 5) = dY * dT;
					matA.at<double>(iRow, 6) = dX;
					matA.at<double>(iRow, 7) = dY;
					matA.at<double>(iRow, 8) = dT;
					matA.at<double>(iRow, 9) = 1.0;
					matS.at<double>(iRow, 0) = (*vec)[iMaxScoreIndex + (theta - 1)].vecResult[x + 1][y + 1];
					iRow++;
#ifdef _DEBUG
					/*string str = format ("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f", dValueA[0], dValueA[1], dValueA[2], dValueA[3], dValueA[4], dValueA[5], dValueA[6], dValueA[7], dValueA[8], dValueA[9]);
					fileA <<  str << endl;
					str = format ("%.6f", dValueS[iRow]);
					fileS << str << endl;*/
#endif
				}
			}
		}
		//求解Z矩陣，得到k0~k9
		//[ x* ] = [ 2k0 k3 k4 ]-1 [ -k6 ]
		//| y* | = | k3 2k1 k5 |   | -k7 |
		//[ t* ] = [ k4 k5 2k2 ]   [ -k8 ]

		Mat matAtA = matA.t() * matA;
		Mat matAtS = matA.t() * matS;
		if (!solve(matAtA, matAtS, matZ, DECOMP_SVD))
			return false;

		Mat matZ_t;
		transpose(matZ, matZ_t);
		double* dZ = matZ_t.ptr<double>(0);
		for (int i = 0; i < 10; ++i)
		{
			if (!std::isfinite(dZ[i]))
				return false;
		}

		Mat matK1 = (Mat_<double>(3, 3) <<
			(2 * dZ[0]), dZ[3], dZ[4],
			dZ[3], (2 * dZ[1]), dZ[5],
			dZ[4], dZ[5], (2 * dZ[2]));
		Mat matK2 = (Mat_<double>(3, 1) << -dZ[6], -dZ[7], -dZ[8]);
		Mat matDelta;
		if (!solve(matK1, matK2, matDelta, DECOMP_SVD))
			return false;

		*dNewX = matDelta.at<double>(0, 0);
		*dNewY = matDelta.at<double>(1, 0);
		*dNewAngle = matDelta.at<double>(2, 0) * R2D;
		return std::isfinite(*dNewX) && std::isfinite(*dNewY) && std::isfinite(*dNewAngle);
	}

	void FilterWithScore(vector<s_MatchParameter>* vec, double dScore)
	{
		std::sort(vec->begin(), vec->end(), compareScoreBig2Small);
		int iSize = vec->size(), iIndexDelete = iSize + 1;
		for (int i = 0; i < iSize; i++)
		{
			if ((*vec)[i].dMatchScore < dScore)
			{
				iIndexDelete = i;
				break;
			}
		}
		if (iIndexDelete == iSize + 1)//沒有任何元素小於dScore
			return;
		vec->erase(vec->begin() + iIndexDelete, vec->end());
		return;
	}

	void SortPtWithCenter(vector<Point2f>& vecSort)
	{
		int iSize = (int)vecSort.size();
		Point2f ptCenter;
		for (int i = 0; i < iSize; i++)
			ptCenter += vecSort[i];
		ptCenter /= iSize;

		Point2f vecX(1, 0);

		vector<pair<Point2f, double>> vecPtAngle(iSize);
		for (int i = 0; i < iSize; i++)
		{
			vecPtAngle[i].first = vecSort[i];//pt
			Point2f vec1(vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
			float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
			float fDot = vec1.x;

			if (vec1.y < 0)//若點在中心的上方
			{
				vecPtAngle[i].second = acos(fDot / fNormVec1) * R2D;
			}
			else if (vec1.y > 0)//下方
			{
				vecPtAngle[i].second = 360 - acos(fDot / fNormVec1) * R2D;
			}
			else//點與中心在相同Y
			{
				if (vec1.x - ptCenter.x > 0)
					vecPtAngle[i].second = 0;
				else
					vecPtAngle[i].second = 180;
			}

		}
		std::sort(vecPtAngle.begin(), vecPtAngle.end(), comparePtWithAngle);
		for (int i = 0; i < iSize; i++)
			vecSort[i] = vecPtAngle[i].first;
	}

	cv::RotatedRect RotatedRect2(const Point2f& _point1, const Point2f& _point2, const Point2f& _point3)
	{
		Point2f _center = 0.5f * (_point1 + _point3);
		Vec2f vecs[2];
		vecs[0] = Vec2f(_point1 - _point2);
		vecs[1] = Vec2f(_point2 - _point3);
		double x = std::max(norm(_point1), std::max(norm(_point2), norm(_point3)));
		double a = std::min(norm(vecs[0]), norm(vecs[1]));
		// check that given sides are perpendicular
		//CV_Assert(std::fabs(vecs[0].ddot(vecs[1])) * a <= FLT_EPSILON * 9 * x * (norm(vecs[0]) * norm(vecs[1])));

		// wd_i stores which vector (0,1) or (1,2) will make the width
		// One of them will definitely have slope within -1 to 1
		int wd_i = 0;
		if (std::fabs(vecs[1][1]) < std::fabs(vecs[1][0])) wd_i = 1;
		int ht_i = (wd_i + 1) % 2;

		float _angle = std::atan(vecs[wd_i][1] / vecs[wd_i][0]) * 180.0f / (float)CV_PI;
		float _width = (float)norm(vecs[wd_i]);
		float _height = (float)norm(vecs[ht_i]);

		return cv::RotatedRect(_center, Size2f(_width, _height), _angle);
	}

	void FilterWithRotatedRect(vector<s_MatchParameter>* vec, int iMethod, double dMaxOverLap)
	{
		int iMatchSize = (int)vec->size();
		RotatedRect rect1, rect2;
		for (int i = 0; i < iMatchSize - 1; i++)
		{
			if (vec->at(i).bDelete)
				continue;
			for (int j = i + 1; j < iMatchSize; j++)
			{
				if (vec->at(j).bDelete)
					continue;
				rect1 = vec->at(i).rectR;
				rect2 = vec->at(j).rectR;
				const double side1 = std::max(1.0, (double)std::min(rect1.size.width, rect1.size.height));
				const double side2 = std::max(1.0, (double)std::min(rect2.size.width, rect2.size.height));
				const double minSide = std::min(side1, side2);
				const double maxSide = std::max(side1, side2);
				const double centerDist = norm(rect1.center - rect2.center);
				// 大模板的空白边把旁边较小的实例包进去时，两个都留。尺寸接近的重叠仍按原来的比例删。
				if (centerDist >= 0.45 * minSide && maxSide > minSide * 1.10)
					continue;
				vector<Point2f> vecInterSec;
				int iInterSecType = rotatedRectangleIntersection(rect1, rect2, vecInterSec);
				if (iInterSecType == INTERSECT_NONE)//無交集
					continue;
				else if (iInterSecType == INTERSECT_FULL) //一個矩形包覆另一個
				{
					int iDeleteIndex;
					if (iMethod == CV_TM_SQDIFF)
						iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
					else
						iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
					vec->at(iDeleteIndex).bDelete = true;
				}
				else//交點 > 0
				{
					if (vecInterSec.size() < 3)//一個或兩個交點
						continue;
					else
					{
						int iDeleteIndex;
						//求面積與交疊比例
						//SortPtWithCenter(vecInterSec);
						vector<cv::Point2f> order_pts;
						cv::convexHull(cv::Mat(vecInterSec), order_pts, true);
						double dArea = contourArea(order_pts);
						double dRatio = dArea / rect1.size.area();
						//若大於最大交疊比例，選分數高的
						if (dRatio > dMaxOverLap)
						{
							if (iMethod == CV_TM_SQDIFF)
								iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
							else
								iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
							vec->at(iDeleteIndex).bDelete = true;
						}
					}
				}
			}
		}
		vector<s_MatchParameter>::iterator it;
		for (it = vec->begin(); it != vec->end();)
		{
			if ((*it).bDelete)
				it = vec->erase(it);
			else
				++it;
		}
	}

	PatternMatcher::PatternMatcher(const MatcherParam& param)
	{
		initFinishedFlag_ = false;

		initFinishedFlag_ = initMatcher(param);
		if (initFinishedFlag_ == false)
		{
			logger_->info("Failed to init matcher!");
			return;
		}

		logger_->info("成功!");

		initFinishedFlag_ = true;
	}

	PatternMatcher::~PatternMatcher()
	{

	}

	void SuppressOverlap(vector<MatchResult>& results, double maxOverlap, int maxCount)
	{
		// 尺度之间的合并：同一中心的不同角度、错位的相邻尺度都去掉。尺寸差开的邻实例保留。
		BaseMatcher::suppressMatchOverlap(results, maxOverlap, maxCount, true);
	}

	void FinishPatternHits(vector<MatchResult>& hits, const Mat& scene, const Mat& templ, const Mat& mask,
						   double scoreThreshold, double maxOverlap, int maxCount)
	{
		refinePatternHits(hits, scene, templ, mask, scoreThreshold);
		SuppressOverlap(hits, maxOverlap, maxCount);
	}

	int PatternMatcher::matchPrepared(const cv::Mat& image, const cv::Mat& templImage, s_TemplData& templData,
									  std::vector<MatchResult>& matchResults)
	{
		matchResults.clear();
		if (image.empty())
			return -1;
		if (templImage.empty() || templImage.channels() != 1)
		{
			logger_->error("image empty or channels != 1.");
			return -5;
		}
		if ((templImage.cols < image.cols && templImage.rows > image.rows) || (templImage.cols > image.cols && templImage.rows < image.rows))
			return -2;
		if (templImage.size().area() > image.size().area())
			return -3;
		if (!templData.bIsPatternLearned)
			return -4;

		double d1 = clock();
		//決定金字塔層數 總共為1 + iLayer層
		cv::Mat templView = templImage;
		int iTopLayer = GetTopLayer(&templView, (int)sqrt((double)matchParam_.minArea));
		//建立金字塔
		vector<Mat> vecMatSrcPyr;
		buildPyramid(image, vecMatSrcPyr, iTopLayer);

		s_TemplData* pTemplData = &templData;
		if (!pTemplData->vecPyramid.empty())
			iTopLayer = std::min(iTopLayer, (int)pTemplData->vecPyramid.size() - 1);

		//第一階段以最頂層找出大致角度與ROI
		double dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[iTopLayer].cols, pTemplData->vecPyramid[iTopLayer].rows)) * R2D;
		//double dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[0].cols, pTemplData->vecPyramid[0].rows)) * R2D;

		vector<double> vecAngles;

		if (matchParam_.angle < VISION_TOLERANCE)
			vecAngles.push_back(0.0);
		else
		{
			// 步长大于剩余角度时，旧条件 `angle + step` 会搜出范围（小模板顶层一步就能到 ±20°）。
			for (double dAngle = 0; dAngle < matchParam_.angle - 1e-9; dAngle += dAngleStep)
				vecAngles.push_back(dAngle);
			vecAngles.push_back(matchParam_.angle);
			for (double dAngle = -dAngleStep; dAngle > -matchParam_.angle + 1e-9; dAngle -= dAngleStep)
				vecAngles.push_back(dAngle);
			vecAngles.push_back(-matchParam_.angle);
		}


		int iTopSrcW = vecMatSrcPyr[iTopLayer].cols, iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
		Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

		int iSize = (int)vecAngles.size();
		//vector<s_MatchParameter> vecMatchParameter (iSize * (m_iMaxPos + MATCH_CANDIDATE_NUM));
		vector<s_MatchParameter> vecMatchParameter;
		//Caculate lowest score at every layer
		vector<double> vecLayerScore(iTopLayer + 1, matchParam_.scoreThreshold);
		for (int iLayer = 1; iLayer <= iTopLayer; iLayer++)
			vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;

		Size sizePat = pTemplData->vecPyramid[iTopLayer].size();
		bool bCalMaxByBlock = (vecMatSrcPyr[iTopLayer].size().area() / sizePat.area() > 500) && matchParam_.maxCount > 10;
		// 尺度搜索时每个尺度先多留一些峰，等跨尺度去重后再按 maxCount 截断。
		// 否则同一圆点的多个角度会占满名额，把更弱的另一个实例挤掉。
		const int peakBudget = scaleSearch_
			? std::min(64, matchParam_.maxCount + MATCH_CANDIDATE_NUM - 1 + std::max(matchParam_.maxCount, 6))
			: matchParam_.maxCount + MATCH_CANDIDATE_NUM - 1;
		const Scalar borderFill = matchParam_.meanBorder
			? Scalar(pTemplData->vecTemplMean[0][0])
			: Scalar(pTemplData->iBorderColor);

#pragma omp parallel
		{
			vector<s_MatchParameter> local;
#pragma omp for schedule(dynamic)
			for (int i = 0; i < iSize; i++)
			{
				Mat matRotatedSrc, matR = getRotationMatrix2D(ptCenter, vecAngles[i], 1);
				Mat matResult;
				Point ptMaxLoc;
				double dValue, dMaxVal;
				Size sizeBest = GetBestRotationSize(vecMatSrcPyr[iTopLayer].size(), pTemplData->vecPyramid[iTopLayer].size(), vecAngles[i]);

				float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
				float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
				matR.at<double>(0, 2) += fTranslationX;
				matR.at<double>(1, 2) += fTranslationY;
				warpAffine(vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest, INTER_LINEAR, BORDER_CONSTANT, borderFill);

				MatchTemplate(matRotatedSrc, pTemplData, matResult, iTopLayer, false);

				if (bCalMaxByBlock)
				{
					s_BlockMax blockMax(matResult, pTemplData->vecPyramid[iTopLayer].size());
					blockMax.GetMaxValueLoc(dMaxVal, ptMaxLoc);
					if (dMaxVal < vecLayerScore[iTopLayer])
						continue;
					local.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
					for (int j = 0; j < peakBudget; j++)
					{
						ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size(), dValue, matchParam_.iouThreshold, blockMax);
						if (dValue < vecLayerScore[iTopLayer])
							break;
						local.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
					}
				}
				else
				{
					minMaxLoc(matResult, 0, &dMaxVal, 0, &ptMaxLoc);
					if (dMaxVal < vecLayerScore[iTopLayer])
						continue;
					local.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
					for (int j = 0; j < peakBudget; j++)
					{
						ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size(), dValue, matchParam_.iouThreshold);
						if (dValue < vecLayerScore[iTopLayer])
							break;
						local.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
					}
				}
			}
#pragma omp critical
			{
				vecMatchParameter.insert(vecMatchParameter.end(), local.begin(), local.end());
			}
		}
		std::sort(vecMatchParameter.begin(), vecMatchParameter.end(), compareScoreBig2Small);


		int iMatchSize = (int)vecMatchParameter.size();
		int iDstW = pTemplData->vecPyramid[iTopLayer].cols, iDstH = pTemplData->vecPyramid[iTopLayer].rows;

		//顯示第一層結果
		if (m_bDebugMode)
		{
			int iDebugScale = 2;

			Mat matShow, matResize;
			resize(vecMatSrcPyr[iTopLayer], matResize, vecMatSrcPyr[iTopLayer].size() * iDebugScale);
			cvtColor(matResize, matShow, CV_GRAY2BGR);
			string str = format("Toplayer, Candidate:%d", iMatchSize);
			vector<Point2f> vec;
			for (int i = 0; i < iMatchSize; i++)
			{
				Point2f ptLT, ptRT, ptRB, ptLB;
				double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
				ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);
				ptRT = Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
				ptLB = Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
				ptRB = Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
				line(matShow, ptLT * iDebugScale, ptLB * iDebugScale, Scalar(0, 255, 0));
				line(matShow, ptLB * iDebugScale, ptRB * iDebugScale, Scalar(0, 255, 0));
				line(matShow, ptRB * iDebugScale, ptRT * iDebugScale, Scalar(0, 255, 0));
				line(matShow, ptRT * iDebugScale, ptLT * iDebugScale, Scalar(0, 255, 0));
				circle(matShow, ptLT * iDebugScale, 1, Scalar(0, 0, 255));
				vec.push_back(ptLT * iDebugScale);
				vec.push_back(ptRT * iDebugScale);
				vec.push_back(ptLB * iDebugScale);
				vec.push_back(ptRB * iDebugScale);

				string strText = format("%d", i);
				cv::putText(matShow, strText, ptLT * iDebugScale, FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
			}
			cvNamedWindow(str.c_str(), 0x10000000);
			Rect rectShow = boundingRect(vec);
			cv::imshow(str, matShow);// (rectShow));
			cv::waitKey();
			//moveWindow (str, 0, 0);
		}
		//顯示第一層結果

		//第一階段結束
		bool bSubPixelEstimation = m_bSubPixel;
		int iStopLayer = matchParam_.stopLayer;
		if (iStopLayer < 0)
			iStopLayer = 0;
		if (iStopLayer > iTopLayer)
			iStopLayer = iTopLayer;
		//int iSearchSize = min (m_iMaxPos + MATCH_CANDIDATE_NUM, (int)vecMatchParameter.size ());//可能不需要搜尋到全部 太浪費時間
		vector<s_MatchParameter> vecAllResult;
		for (int i = 0; i < (int)vecMatchParameter.size(); i++)
			//for (int i = 0; i < iSearchSize; i++)
		{
			double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
			Point2f ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);

			double dAngleStep = atan(2.0 / max(iDstW, iDstH)) * R2D;//min改為max
			vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep;
			vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep;

			if (iTopLayer <= iStopLayer)
			{
				vecMatchParameter[i].pt = Point2d(ptLT * ((iTopLayer == 0) ? 1 : 2));
				vecAllResult.push_back(vecMatchParameter[i]);
			}
			else
			{
				for (int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--)
				{
					//搜尋角度
					dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[iLayer].cols, pTemplData->vecPyramid[iLayer].rows)) * R2D;//min改為max
					vector<double> vecAngles;
					double dMatchedAngle = vecMatchParameter[i].dMatchAngle;
					if (matchParam_.angle < VISION_TOLERANCE)
						vecAngles.push_back(0.0);
					else
					{
						for (int k = -2; k <= 2; k++)
						{
							double a = dMatchedAngle + dAngleStep * k;
							if (a > matchParam_.angle)
								a = matchParam_.angle;
							else if (a < -matchParam_.angle)
								a = -matchParam_.angle;
							bool dup = false;
							for (double prev : vecAngles)
							{
								if (std::fabs(prev - a) < 1e-6)
									dup = true;
							}
							if (!dup)
								vecAngles.push_back(a);
						}
					}
					Point2f ptSrcCenter((vecMatSrcPyr[iLayer].cols - 1) / 2.0f, (vecMatSrcPyr[iLayer].rows - 1) / 2.0f);
					iSize = (int)vecAngles.size();
					vector<s_MatchParameter> vecNewMatchParameter(iSize);
					int iMaxScoreIndex = 0;
					double dBigValue = -1;
					for (int j = 0; j < iSize; j++)
					{
						Mat matResult, matRotatedSrc;
						double dMaxValue = 0;
						Point ptMaxLoc;
						GetRotatedROI(vecMatSrcPyr[iLayer], pTemplData->vecPyramid[iLayer].size(), ptLT * 2, vecAngles[j], matRotatedSrc, borderFill);

						MatchTemplate(matRotatedSrc, pTemplData, matResult, iLayer, false);
						//matchTemplate (matRotatedSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCOEFF_NORMED);
						minMaxLoc(matResult, 0, &dMaxValue, 0, &ptMaxLoc);
						vecNewMatchParameter[j] = s_MatchParameter(ptMaxLoc, dMaxValue, vecAngles[j]);

						if (vecNewMatchParameter[j].dMatchScore > dBigValue)
						{
							iMaxScoreIndex = j;
							dBigValue = vecNewMatchParameter[j].dMatchScore;
						}
						//次像素估計
						if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 || ptMaxLoc.x == matResult.cols - 1 || ptMaxLoc.y == matResult.rows - 1)
							vecNewMatchParameter[j].bPosOnBorder = true;
						if (!vecNewMatchParameter[j].bPosOnBorder)
						{
							for (int y = -1; y <= 1; y++)
								for (int x = -1; x <= 1; x++)
									vecNewMatchParameter[j].vecResult[x + 1][y + 1] = matResult.at<float>(ptMaxLoc + Point(x, y));
						}
						//次像素估計
					}
					// 最细一层放到略低于阈值，好让「空白边对不上、物体本身对齐」的弱实例活到后面的补救。
					const double layerGate = (iLayer == iStopLayer)
						? std::max(0.30, matchParam_.scoreThreshold * 0.82)
						: vecLayerScore[iLayer];
					if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < layerGate)
						break;
					//次像素估計
					if (bSubPixelEstimation
						&& iLayer == 0
						&& (!vecNewMatchParameter[iMaxScoreIndex].bPosOnBorder)
						&& iMaxScoreIndex >= 1
						&& iMaxScoreIndex + 1 < (int)vecNewMatchParameter.size())
					{
						double dNewX = 0, dNewY = 0, dNewAngle = 0;
						if (SubPixEsimation(&vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle, dAngleStep, iMaxScoreIndex))
						{
							// 圆对称目标上角度曲面是平的，二次拟合会把角度甩出搜索范围。
							// 位置只收 1px 以内的修正，角度必须贴近离散峰并且落在 ±angle 里。
							const double ox = vecNewMatchParameter[iMaxScoreIndex].pt.x;
							const double oy = vecNewMatchParameter[iMaxScoreIndex].pt.y;
							const double oa = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;
							if (std::fabs(dNewX - ox) <= 1.0 && std::fabs(dNewY - oy) <= 1.0)
								vecNewMatchParameter[iMaxScoreIndex].pt = Point2d(dNewX, dNewY);
							const double angLim = std::max(1.0, dAngleStep * 1.5);
							if (std::fabs(dNewAngle - oa) <= angLim
								&& (matchParam_.angle < VISION_TOLERANCE || std::fabs(dNewAngle) <= matchParam_.angle + 1e-3))
								vecNewMatchParameter[iMaxScoreIndex].dMatchAngle = dNewAngle;
						}
					}
					//次像素估計

					double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;

					//讓坐標系回到旋轉時(GetRotatedROI)的(0, 0)
					Point2f ptPaddingLT = ptRotatePt2f(ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - Point2f(3, 3);
					Point2f pt(vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x, vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
					//再旋轉
					pt = ptRotatePt2f(pt, ptSrcCenter, -dNewMatchAngle * D2R);

					if (iLayer == iStopLayer)
					{
						vecNewMatchParameter[iMaxScoreIndex].pt = pt * (iStopLayer == 0 ? 1 : 2);
						vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
					}
					else
					{
						//更新MatchAngle ptLT
						vecMatchParameter[i].dMatchAngle = dNewMatchAngle;
						vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep / 2;
						vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep / 2;
						ptLT = pt;
					}
				}

			}
		}
		FilterWithScore(&vecAllResult, std::max(0.30, matchParam_.scoreThreshold * 0.82));

		//最後濾掉重疊
		iDstW = pTemplData->vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
		iDstH = pTemplData->vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

		for (int i = 0; i < (int)vecAllResult.size(); i++)
		{
			Point2f ptLT, ptRT, ptRB, ptLB;
			double dRAngle = -vecAllResult[i].dMatchAngle * D2R;
			ptLT = vecAllResult[i].pt;
			ptRT = Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
			ptLB = Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
			ptRB = Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
			//紀錄旋轉矩形
			vecAllResult[i].rectR = RotatedRect2(ptLT, ptRT, ptRB);
		}
		FilterWithRotatedRect(&vecAllResult, CV_TM_CCOEFF_NORMED, matchParam_.iouThreshold);
		//最後濾掉重疊

		//根據分數排序
		std::sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);

		iMatchSize = (int)vecAllResult.size();
		if (vecAllResult.size() == 0)
			return false;
		int iW = pTemplData->vecPyramid[0].cols, iH = pTemplData->vecPyramid[0].rows;

		for (int i = 0; i < iMatchSize; i++)
		{
			MatchResult sstm;
			double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

			sstm.LeftTop = vecAllResult[i].pt;

			sstm.RightTop = Point2d(sstm.LeftTop.x + iW * cos(dRAngle), sstm.LeftTop.y - iW * sin(dRAngle));
			sstm.LeftBottom = Point2d(sstm.LeftTop.x + iH * sin(dRAngle), sstm.LeftTop.y + iH * cos(dRAngle));
			sstm.RightBottom = Point2d(sstm.RightTop.x + iH * sin(dRAngle), sstm.RightTop.y + iH * cos(dRAngle));
			sstm.Center = Point2d((sstm.LeftTop.x + sstm.RightTop.x + sstm.RightBottom.x + sstm.LeftBottom.x) / 4, (sstm.LeftTop.y + sstm.RightTop.y + sstm.RightBottom.y + sstm.LeftBottom.y) / 4);
			sstm.Angle = -vecAllResult[i].dMatchAngle;
			sstm.Score = vecAllResult[i].dMatchScore;

			if (sstm.Angle < -180)
				sstm.Angle += 360;
			if (sstm.Angle > 180)
				sstm.Angle -= 360;
			if (!isFinitePoint(sstm.LeftTop)
				|| !isFinitePoint(sstm.RightTop)
				|| !isFinitePoint(sstm.RightBottom)
				|| !isFinitePoint(sstm.LeftBottom)
				|| !isFinitePoint(sstm.Center)
				|| !std::isfinite(sstm.Angle)
				|| !std::isfinite(sstm.Score))
				continue;
			matchResults.push_back(sstm);

			// 尺度搜索先按 outLimit 放行，match() 里跨尺度去重后再截到 maxCount。
			const int outLimit = scaleSearch_
				? std::min(64, std::max(matchParam_.maxCount, 1) * 3 + 6)
				: std::max(matchParam_.maxCount, 1);
			if ((int)matchResults.size() >= outLimit)
				break;

			//std::cout << "x: " << sstm.Center.x << ", y: " << sstm.Center.y << endl;



			//Test Subpixel
			/*Point2d ptLT = vecAllResult[i].ptSubPixel;
			Point2d ptRT = Point2d (sstm.ptLT.x + iW * cos (dRAngle), sstm.ptLT.y - iW * sin (dRAngle));
			Point2d ptLB = Point2d (sstm.ptLT.x + iH * sin (dRAngle), sstm.ptLT.y + iH * cos (dRAngle));
			Point2d ptRB = Point2d (sstm.ptRT.x + iH * sin (dRAngle), sstm.ptRT.y + iH * cos (dRAngle));
			Point2d ptCenter = Point2d ((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
			CString strDiff;strDiff.Format (L"Diff(x, y):%.3f, %.3f", ptCenter.x - sstm.ptCenter.x, ptCenter.y - sstm.ptCenter.y);
			AfxMessageBox (strDiff);*/
			//Test Subpixel
		}

		return (int)matchResults.size();
	}

	int PatternMatcher::match(const cv::Mat& image, std::vector<MatchResult>& matchResults)
	{
		matchResults.clear();
		Mat maskForQuads;
		if (!templateMask_.empty())
			maskForQuads = templateMask_;
		else if (m_TemplData.bHasMask && !m_TemplData.vecMask.empty())
			maskForQuads = m_TemplData.vecMask[0];
		if (!scaleSearch_)
		{
			int n = matchPrepared(image, templateImage_, m_TemplData, matchResults);
			if (n < 0)
				return n;
			for (MatchResult& r : matchResults)
				r.Scale = 1.0;
			FinishPatternHits(matchResults, image, templateImage_, maskForQuads,
							  matchParam_.scoreThreshold, matchParam_.iouThreshold, matchParam_.maxCount);
			return (int)matchResults.size();
		}

		if (scaledPatterns_.empty())
			return -4;
		std::vector<MatchResult> all;
		int lastErr = -4;
		bool anyOk = false;
		for (ScaledPattern& sp : scaledPatterns_)
		{
			std::vector<MatchResult> part;
			int n = matchPrepared(image, sp.image, sp.data, part);
			if (n < 0)
			{
				lastErr = n;
				continue;
			}
			anyOk = true;
			for (MatchResult& r : part)
			{
				r.Scale = sp.scale;
				all.push_back(r);
			}
		}
		if (!anyOk)
			return lastErr;
		FinishPatternHits(all, image, templateImage_, maskForQuads,
						  matchParam_.scoreThreshold, matchParam_.iouThreshold, matchParam_.maxCount);
		matchResults.swap(all);
		return (int)matchResults.size();
	}

	int PatternMatcher::setTemplate(const cv::Mat& templateImage, const cv::Mat& mask)
	{
		scaledPatterns_.clear();
		templateMask_.release();
		scaleSearch_ = scaleSearchEnabled(matchParam_);
		if (templateImage.empty())
			return -1;
		if (templateImage.channels() > 1)
			return -2;
		m_TemplData.bIsPatternLearned = false;

		if (!scaleSearch_)
		{
			templateImage_ = templateImage.clone();
			LearnPattern(m_TemplData, templateImage_, matchParam_.minArea, mask);
			if (!m_TemplData.bIsPatternLearned)
			{
				logger_->error("LearnPattern failed (check mask size/channels).");
				return -3;
			}
			return 0;
		}

		templateImage_ = templateImage.clone();
		if (!mask.empty())
		{
			if (mask.size() != templateImage_.size() || mask.channels() != 1)
			{
				logger_->error("LearnPattern failed (check mask size/channels).");
				return -3;
			}
			templateMask_ = mask.clone();
		}

		const std::vector<double> scales = sampleMatchScales(matchParam_);
		double minArea = matchParam_.minArea;
		for (double s : scales)
		{
			ScaledPattern sp;
			sp.scale = s;
			cv::Mat layerMask;
			if (std::fabs(s - 1.0) < 1e-4)
			{
				sp.image = templateImage_.clone();
				layerMask = templateMask_;
			}
			else
			{
				const int w = std::max(1, (int)std::lround(templateImage_.cols * s));
				const int h = std::max(1, (int)std::lround(templateImage_.rows * s));
				if (w < 8 || h < 8)
					continue;
				const int interp = s < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR;
				cv::resize(templateImage_, sp.image, cv::Size(w, h), 0, 0, interp);
				if (!templateMask_.empty())
				{
					cv::resize(templateMask_, layerMask, sp.image.size(), 0, 0, cv::INTER_NEAREST);
					cv::threshold(layerMask, layerMask, 0, 255, cv::THRESH_BINARY);
				}
			}
			LearnPattern(sp.data, sp.image, minArea, layerMask);
			if (!sp.data.bIsPatternLearned)
				continue;
			scaledPatterns_.push_back(std::move(sp));
		}
		if (scaledPatterns_.empty())
		{
			logger_->error("LearnPattern failed for all scales (check mask size/channels).");
			return -3;
		}
		m_TemplData.bIsPatternLearned = true;
		return 0;
	}

}
