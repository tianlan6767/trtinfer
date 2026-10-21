#pragma once

#include <pybind11/pybind11.h>

// 传统视觉（匹配 / 卡尺 / Blob / 金样比对）以及 bindExtra。完整 cvter.so 和无 CUDA 的 test-viz 共用。
void bindTrad(pybind11::module_& m);
