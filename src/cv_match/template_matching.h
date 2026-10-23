#ifndef _TEMPLATE_MATCHING_DEFINE_H 
#define _TEMPLATE_MATCHING_DEFINE_H 

#pragma once

#ifdef _WIN32
#pragma warning(disable: 4251)
#endif  // _WIN32

#ifdef LIB_SHARED_BUILD
#ifdef _WIN32
#ifdef LIB_EXPORTS
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __declspec(dllimport)
#endif  // MY_LIB_EXPORTS
#else
#define LIB_API
#endif  // _WIN32
#else
#define LIB_API
#endif  // MY_LIB_SHARED_BUILD


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

namespace template_matching
{
	// 匹配器类型
	enum LIB_API MatcherType { 
		PATTERN = 0,
		SHAPE = 1,
	};

	// 匹配器参数
	struct LIB_API MatcherParam
	{
		// 匹配器类型
		MatcherType matcherType = MatcherType::PATTERN;

		// 最大匹配目标数量
		int maxCount = 200;

		// 匹配得分阈值
		double scoreThreshold = 0.5;

		// 重叠框去重iou阈值
		double iouThreshold = 0.0;

		// 匹配角度范围
		double angle = 0;

		// 顶层金字塔最小面积
		double minArea = 256;

		// 旋转填充用模板均值（false 则恢复旧的黑/白）
		bool meanBorder = true;

		// 精搜停止层：0=原图，1=金字塔第 1 层（更快）
		int stopLayer = 0;

		// 形状匹配：最小对比度。低于这个梯度的点不进模型（对应 MinContrast）。
		double edgeMinMag = 18.0;

		// 形状匹配：对比度。高于这个梯度的点是种子；介于它和 edgeMinMag 之间的点，
		// 只有连到种子上才保留（对应 Contrast）。<= edgeMinMag 时退回单阈值，与旧行为一致。
		double edgeContrast = 0.0;

		// 形状匹配：每层最多保留的边缘点数
		int maxEdgePoints = 320;

		// 形状匹配：true=要求明暗方向一致（use_polarity）
		bool usePolarity = true;

		// 形状匹配：0~1，越大越早放弃不可能达到阈值的位置
		double greediness = 0.9;

		// 形状匹配：0=全程浮点余弦（默认，与旧行为一致）。
		// 16=顶层把方向量化成 16 档，先建成方向响应图（±3 扩散一次）再查表粗搜。
		// 金字塔对 143×183 这类模板补到 4 层（下标 3，大约 18×23）。再深一层，16px 外的强边会并进相邻粗像素。
		// 已经更深的大模板不变。多出来的层把贴边带宽按层缩小，原图上仍大约 64px。
		// 这一层步长是 4，角度步长仍是 atan(1/R)，不再加倍。四边一条带改回浮点分。
		// 贴边和内部都用整幅去重半径。内部种子在原图以上的每一层精修前比 θ 和 θ+180，对瓣高出 0.02 才改角度。贴边不翻。第一层 |角度|≥90° 的内部种子再留一份 +180°。贴边第一层精修是 ±4 px、±3 步；内部第一层是 ±10 px、大约 ±12°。贴边种子全部进入精修；其余内部种子只留粗分最高的 64 个，环带捞回的另算。
		// 内部环带用浮点分再挑一个峰，第一层位置窗是 ±4 px，中间层先留到末层再比分。普通内部种子 ±4 已过阈值就不再扩到 ±10。和贴边重叠时留下贴边。末层仍用浮点余弦。
		int oriBins = 0;

		// 尺度搜索（相对模板）。minScale=maxScale=1 时只搜原尺度，与旧行为一致。
		double minScale = 1.0;
		double maxScale = 1.0;
		// minScale!=maxScale 且 numScales<=1 时的步长；<=0 则按 0.05
		double scaleStep = 0.05;
		// >1 时在 [minScale, maxScale] 上均匀取这么多个（含端点），此时忽略 scaleStep
		int numScales = 0;

	};
	
	// 匹配结果
	struct LIB_API MatchResult
	{
		cv::Point2d LeftTop; // 左上角点
		cv::Point2d LeftBottom; // 左下角点
		cv::Point2d RightTop; // 右上角点
		cv::Point2d RightBottom; // 右下角点
		cv::Point2d Center; // 中心点

		double Angle; // 角度
		double Score; // 匹配得分
		double Scale; // 相对模板的尺度，1 为原大小

	// 默认构造函数，初始化所有成员
    MatchResult()
        : LeftTop(0,0), LeftBottom(0,0), RightTop(0,0), RightBottom(0,0), Center(0,0),
          Angle(0.0), Score(0.0), Scale(1.0)
    {};	
	};

}

#endif
