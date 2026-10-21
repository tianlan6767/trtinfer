#pragma once

// numpy <-> cv::Mat。支持 8U / 32S / 32F / 64F，2 维或 3 维。
// 返回 numpy 时会拷贝，避免临时 Mat 释放后悬空。
// 读入时按 numpy 的步长拷贝。ROI 视图 scene[y0:y1, x0:x1] 不是 C 连续的，
// 若忽略行步长，Matcher 看到的是错位像素，会静默返回 0 个命中。

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

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

		if (nh <= 0 || nw <= 0) {
			value = cv::Mat(std::max(nh, 0), std::max(nw, 0), dtype);
			return true;
		}
		if (info.ptr == nullptr)
			throw std::logic_error("numpy array data pointer is null");
		if (static_cast<int>(info.strides.size()) < ndims || info.itemsize <= 0)
			throw std::logic_error("numpy array is missing strides");

		const ssize_t item = info.itemsize;
		const int packed_cols = (ndims == 2) ? nw : nw * nc;
		const ssize_t min_step = item * packed_cols;
		const bool inner_packed = (ndims == 2)
			? (info.strides[1] == item)
			: (info.strides[2] == item && info.strides[1] == item * nc);
		// 行内连续、行步长不小于一行：包含普通连续图和 ROI 切片。
		if (inner_packed && info.strides[0] >= min_step) {
			value = cv::Mat(nh, nw, dtype, info.ptr, static_cast<size_t>(info.strides[0])).clone();
			return true;
		}

		// 负步长、通道切片等，按元素拷进连续 Mat。
		cv::Mat packed(nh, nw, dtype);
		const auto* base = static_cast<const unsigned char*>(info.ptr);
		if (ndims == 2) {
			for (int y = 0; y < nh; ++y) {
				unsigned char* dst = packed.ptr<unsigned char>(y);
				for (int x = 0; x < nw; ++x) {
					const unsigned char* src = base
						+ static_cast<ptrdiff_t>(y) * info.strides[0]
						+ static_cast<ptrdiff_t>(x) * info.strides[1];
					std::memcpy(dst + static_cast<ptrdiff_t>(x) * item, src, static_cast<size_t>(item));
				}
			}
		} else {
			for (int y = 0; y < nh; ++y) {
				unsigned char* dst = packed.ptr<unsigned char>(y);
				for (int x = 0; x < nw; ++x) {
					for (int c = 0; c < nc; ++c) {
						const unsigned char* src = base
							+ static_cast<ptrdiff_t>(y) * info.strides[0]
							+ static_cast<ptrdiff_t>(x) * info.strides[1]
							+ static_cast<ptrdiff_t>(c) * info.strides[2];
						const ptrdiff_t di = (static_cast<ptrdiff_t>(x) * nc + c) * item;
						std::memcpy(dst + di, src, static_cast<size_t>(item));
					}
				}
			}
		}
		value = std::move(packed);
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
