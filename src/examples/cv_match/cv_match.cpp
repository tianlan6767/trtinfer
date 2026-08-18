#include <iostream>
#include <opencv2/opencv.hpp>
#include "cv_match/matcher.h"   // 直接使用本地 matcher

void run_match_example()
{
    // ------------------------------
    // 1. 读取模板和测试图
    // ------------------------------
    std::string temp_path = R"(/home/ps/workspace/trt/cvter/workspace/match_img/161.jpg)";
    std::string test_path = R"(/home/ps/workspace/trt/cvter/workspace/match_img/images2/1_6_1 (3).jpg)";
    cv::Mat templ = cv::imread(temp_path, cv::IMREAD_GRAYSCALE);
    cv::Mat test  = cv::imread(test_path, cv::IMREAD_GRAYSCALE);

    if (templ.empty() || test.empty()) {
        std::cerr << "Failed to read template or test image.\n";
        // return -3;
    }

    // ------------------------------
    // 2. 初始化 matcher 参数
    // ------------------------------
    template_matching::MatcherParam param;
    param.angle = 0;
    param.iouThreshold = 0.5;
    param.matcherType = template_matching::MatcherType::PATTERN;
    param.maxCount = 10;
    param.minArea = 256;
    param.scoreThreshold = 0.8;

    // ------------------------------
    // 3. 直接创建 Matcher（不使用 DLL）
    // ------------------------------
    auto matcher = template_matching::GetMatcher(param);

    // 设置模板
    matcher->setTemplate(templ);

    // ------------------------------
    // 4. 执行匹配
    // ------------------------------
    std::vector<template_matching::MatchResult> results;
    matcher->match(test, results);

    // 输出结果
    if (results.empty()) {
        std::cout << "No match found\n";
    } else {
        for (auto& r : results) 
        {
            std::cout << "Match Found:" << std::endl;
            std::cout << "LeftTop:      " << r.LeftTop      << std::endl;
            std::cout << "RightTop:     " << r.RightTop     << std::endl;
            std::cout << "RightBottom:  " << r.RightBottom  << std::endl;
            std::cout << "LeftBottom:   " << r.LeftBottom   << std::endl;
            std::cout << "Score:        " << r.Score        << std::endl;
        }
    }

    // ------------------------------
    // 5. 保存可视化结果
    // ------------------------------
    cv::Mat vis;
    cv::cvtColor(test, vis, cv::COLOR_GRAY2BGR);

    for (auto& r : results) {
        std::vector<cv::Point> pts = {
            r.LeftTop, r.RightTop, r.RightBottom, r.LeftBottom
        };
        cv::polylines(vis, pts, true, cv::Scalar(0,0,255), 2);
        cv::putText(vis, cv::format("%.4f", r.Score), r.LeftTop, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
    }

    cv::imwrite("result.jpg", vis);
    std::cout << "Result saved to result.jpg\n";

    // ------------------------------
    // 6. 清理
    // ------------------------------
    // No need to delete matcher since it's managed by a smart pointer

    // return 0;
}
