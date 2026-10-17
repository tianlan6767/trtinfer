#pragma once

// numpy <-> cv::Mat。支持 8U / 32S / 32F / 64F，2 维或 3 维。
// 返回 numpy 时会拷贝，避免临时 Mat 释放后悬空。

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/core.hpp>

#include <cstdio>
#include <stdexcept>

namespace pybind11 { namespace detail {

template <>
struct type_caster<cv::Mat> {
public:
	PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

	bool load(handle obj, bool)
	{
		array b = reinterpret_borrow<array>(obj);
		buffer_info info = b.request();

		int nh = 1;
		int nw = 1;
		int nc = 1;
		int ndims = static_cast<int>(info.ndim);
		if (ndims == 2) {
			nh = static_cast<int>(info.shape[0]);
			nw = static_cast<int>(info.shape[1]);
		} else if (ndims == 3) {
			nh = static_cast<int>(info.shape[0]);
			nw = static_cast<int>(info.shape[1]);
			nc = static_cast<int>(info.shape[2]);
		} else {
			char msg[64];
			std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
			throw std::logic_error(msg);
		}

		int dtype;
		if (info.format == format_descriptor<unsigned char>::format())
			dtype = CV_8UC(nc);
		else if (info.format == format_descriptor<int>::format())
			dtype = CV_32SC(nc);
		else if (info.format == format_descriptor<float>::format())
			dtype = CV_32FC(nc);
		else if (info.format == format_descriptor<double>::format())
			dtype = CV_64FC(nc);
		else
			throw std::logic_error("Unsupported type, only support uchar, int32, float, double");

		value = cv::Mat(nh, nw, dtype, info.ptr).clone();
		return true;
	}

	static handle cast(const cv::Mat& matIn, return_value_policy, handle)
	{
		cv::Mat mat = matIn.isContinuous() ? matIn : matIn.clone();
		std::string format = format_descriptor<unsigned char>::format();
		size_t elemsize = sizeof(unsigned char);
		int nw = mat.cols;
		int nh = mat.rows;
		int nc = mat.channels();
		int depth = mat.depth();
		int type = mat.type();
		int dim = (depth == type) ? 2 : 3;

		if (depth == CV_8U) {
			format = format_descriptor<unsigned char>::format();
			elemsize = sizeof(unsigned char);
		} else if (depth == CV_32S) {
			format = format_descriptor<int>::format();
			elemsize = sizeof(int);
		} else if (depth == CV_32F) {
			format = format_descriptor<float>::format();
			elemsize = sizeof(float);
		} else if (depth == CV_64F) {
			format = format_descriptor<double>::format();
			elemsize = sizeof(double);
		} else {
			throw std::logic_error("Unsupported type, only support uchar, int32, float, double");
		}

		std::vector<size_t> bufferdim;
		std::vector<size_t> strides;
		if (dim == 2) {
			bufferdim = {(size_t)nh, (size_t)nw};
			strides = {elemsize * (size_t)nw, elemsize};
		} else {
			bufferdim = {(size_t)nh, (size_t)nw, (size_t)nc};
			strides = {(size_t)elemsize * nw * nc, (size_t)elemsize * nc, (size_t)elemsize};
		}
		return array(buffer_info(mat.data, elemsize, format, dim, bufferdim, strides)).release();
	}
};

}}  // namespace pybind11::detail
