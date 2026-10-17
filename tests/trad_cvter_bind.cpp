#include <pybind11/pybind11.h>

#include "bind_trad.h"
#include "pybind_cvmat.hpp"

namespace py = pybind11;

// 不链 TensorRT / CUDA 的传统视觉模块，模块名仍是 cvter，供 make test-viz 使用。
PYBIND11_MODULE(cvter, m)
{
	m.doc() = "cvter traditional vision (no TensorRT)";
	bindTrad(m);
}
