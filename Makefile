cc        := g++
name      := cvter.so
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++17
cuda_home := /usr/local/cuda-12.6
cuda_arch := 8.9
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc) 


project_include_path := src
opencv_include_path  := /home/ps/workspace/trt/lean/cv4110/install/include/opencv4
trt_include_path     := /home/ps/workspace/trt/lean/TensorRT-10.11.0.33/include
cuda_include_path    := $(cuda_home)/include
ffmpeg_include_path  := 
spdlog_include_path := /home/ps/workspace/trt/cvter/src/3rdParty/spdlog/include
# python_include_path
exclude_path  := src/3rdParty/spdlog/tests src/3rdParty/spdlog/example src/3rdParty/spdlog/bench src/3rdParty/spdlog/src

FIND_EXCLUDES := $(foreach d,$(exclude_path),! -path "$(d)/*")

python_include_path  := /usr/local/include/python3.12


include_paths        := $(project_include_path) \
						$(opencv_include_path) \
						$(trt_include_path) \
						$(cuda_include_path) \
						$(python_include_path) \
						$(spdlog_include_path) \



opencv_library_path  := /home/ps/workspace/trt/lean/cv4110/install/lib
trt_library_path     := /home/ps/workspace/trt/lean/TensorRT-10.11.0.33/lib/
cuda_library_path    := $(cuda_home)/lib64/
# python_library_path  := /usr/lib/x86_64-linux-gnu
python_library_path  := /usr/local/lib/python3.12/config-3.12-x86_64-linux-gnu

library_paths        := $(opencv_library_path) \
						$(trt_library_path) \
						$(cuda_library_path) \
						$(python_library_path) \
						/usr/lib/wsl/lib \

link_opencv       := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs opencv_highgui \
					 opencv_features2d opencv_flann opencv_calib3d opencv_objdetect opencv_dnn opencv_aruco opencv_barcode
link_trt          := nvinfer nvinfer_plugin nvonnxparser
link_cuda         := cuda cublas cudart cudnn
link_sys          := stdc++ dl

link_librarys     := $(link_opencv) $(link_trt) $(link_cuda) $(link_sys)


empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread  $(include_paths) 
cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread -Wno-overloaded-virtual $(include_paths)
# cu_compile_flags  := -Xcompiler "$(cpp_compile_flags)"
cu_compile_flags := $(foreach flag,$(cpp_compile_flags),-Xcompiler $(flag)) -diag-suppress 611
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN' $(library_paths) $(link_librarys)


cpp_srcs := $(shell find $(srcdir) -name "*.cpp" $(FIND_EXCLUDES))
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

TRT_VERSION := 10

# 根据 TRT_VERSION 设置不同的编译选项
ifeq ($(TRT_VERSION), 8)
    CXXFLAGS = -DTRT8
    cpp_srcs := $(filter-out src/common/tensorrt.cpp, $(cpp_srcs))
    cpp_objs := $(filter-out objs/common/tensorrt.cpp.o, $(cpp_objs))
else
    CXXFLAGS = -DTRT10
    cpp_srcs := $(filter-out src/common/tensorrt8.cpp, $(cpp_srcs))
    cpp_objs := $(filter-out objs/common/tensorrt8.cpp.o, $(cpp_objs))
endif

# 本机有 tesseract 时编进 OCR；没有则 recognize_text 返回 backend=none。
TESS_CFLAGS := $(shell pkg-config --cflags tesseract 2>/dev/null)
TESS_LIBS   := $(shell pkg-config --libs tesseract 2>/dev/null)
ifneq ($(strip $(TESS_LIBS)),)
CXXFLAGS   += -DCVTER_WITH_TESSERACT $(TESS_CFLAGS)
link_flags += $(TESS_LIBS)
endif

# OpenCV barcode 在没有超分模型时经常解不出合成码。有 zbar.h 时用它做一维码后备。
ZBAR_LIBS :=
ifneq ($(wildcard /usr/include/zbar.h),)
ZBAR_LIBS := -lzbar
link_flags += $(ZBAR_LIBS)
endif

pro_cpp_objs := $(filter-out objs/interface.cpp.o, $(cpp_objs))

ifneq ($(MAKECMDGOALS), clean)
include $(mks)
endif


$(name)   : $(workdir)/$(name)

all       : $(name)

python    ?= /usr/local/bin/python3.12

run       : $(name)
	@cd $(workdir) && $(python) test.py

test-caliper : $(name)
	@PYTHONPATH=$(CURDIR)/$(workdir) $(python) $(CURDIR)/tests/test_caliper.py -v

test-match : $(name)
	@PYTHONPATH=$(CURDIR)/$(workdir) $(python) $(CURDIR)/tests/test_pattern_match.py -v

test-shape : $(name)
	@PYTHONPATH=$(CURDIR)/$(workdir) $(python) $(CURDIR)/tests/test_shape_match.py -v

# 尺度搜索：只编 cv_match，不链 TensorRT / CUDA。需要 pkg-config opencv4。
test-scale :
	@mkdir -p $(objdir)
	@$(cc) -std=$(stdcpp) -O2 -fopenmp -pthread -I$(srcdir) -I$(srcdir)/3rdParty/spdlog/include \
		$(CURDIR)/tests/test_scale_match.cpp \
		$(srcdir)/cv_match/PatternMatching.cpp \
		$(srcdir)/cv_match/ShapeMatching.cpp \
		$(srcdir)/cv_match/base_matcher.cpp \
		$(srcdir)/cv_match/matcher.cpp \
		-o $(objdir)/test_scale_match \
		$$(pkg-config --cflags --libs opencv4)
	@$(objdir)/test_scale_match

# 预处理 + Blob：只编 cv_blob，不链 TensorRT / CUDA。需要 pkg-config opencv4。
test-blob :
	@mkdir -p $(objdir)
	@$(cc) -std=$(stdcpp) -O2 -fopenmp -pthread -I$(srcdir) \
		$(CURDIR)/tests/test_blob.cpp \
		$(srcdir)/cv_blob/blob.cpp \
		-o $(objdir)/test_blob \
		$$(pkg-config --cflags --libs opencv4)
	@$(objdir)/test_blob

# 矩形卡尺 + 测量：只编 cv_caliper，不链 TensorRT / CUDA。
test-rect :
	@mkdir -p $(objdir)
	@$(cc) -std=$(stdcpp) -O2 -fopenmp -pthread -I$(srcdir) \
		$(CURDIR)/tests/test_rect_metrology.cpp \
		$(srcdir)/cv_caliper/caliper.cpp \
		$(srcdir)/cv_caliper/metrology.cpp \
		-o $(objdir)/test_rect \
		$$(pkg-config --cflags --libs opencv4)
	@$(objdir)/test_rect

# 金样比对：只编 cv_golden 及其复用的定位 / Blob / ORB，不链 TensorRT / CUDA。
test-golden :
	@mkdir -p $(objdir)
	@$(cc) -std=$(stdcpp) -O2 -fopenmp -pthread -I$(srcdir) \
		$(CURDIR)/tests/test_golden.cpp \
		$(srcdir)/cv_golden/golden.cpp \
		$(srcdir)/cv_blob/blob.cpp \
		$(srcdir)/cv_sim/similarity.cpp \
		$(srcdir)/cv_feature/feature.cpp \
		-o $(objdir)/test_golden \
		$$(pkg-config --cflags --libs opencv4)
	@$(objdir)/test_golden

# 传统视觉可视化：不链 TensorRT / CUDA，不弹窗，图片写到 tests/out_viz/。
# 已有 workspace/cvter.so 时仍编这份无 GPU 模块，保证新接口在没编过完整 so 时也能测。
viz_py_dir := $(objdir)/pytrad
viz_so     := $(viz_py_dir)/cvter.so
py_includes := $(shell python3-config --includes 2>/dev/null)
pyexec := $(if $(wildcard $(python)),$(python),python3)
viz_srcs := \
	$(CURDIR)/tests/trad_cvter_bind.cpp \
	$(srcdir)/bind_trad.cpp \
	$(srcdir)/bind_extra.cpp \
	$(srcdir)/cv_blob/blob.cpp \
	$(srcdir)/cv_golden/golden.cpp \
	$(srcdir)/cv_caliper/caliper.cpp \
	$(srcdir)/cv_caliper/metrology.cpp \
	$(srcdir)/cv_match/PatternMatching.cpp \
	$(srcdir)/cv_match/ShapeMatching.cpp \
	$(srcdir)/cv_match/base_matcher.cpp \
	$(srcdir)/cv_match/matcher.cpp \
	$(srcdir)/cv_sim/similarity.cpp \
	$(srcdir)/cv_seg/seg.cpp \
	$(srcdir)/cv_feature/feature.cpp \
	$(srcdir)/cv_code/code.cpp \
	$(srcdir)/cv_calib/calib.cpp

$(viz_so): $(viz_srcs)
	@mkdir -p $(viz_py_dir)
	@$(cc) -std=$(stdcpp) -O2 -fPIC -shared -fopenmp -pthread \
		-I$(srcdir) -I$(srcdir)/3rdParty/spdlog/include $(py_includes) \
		$(if $(strip $(TESS_LIBS)),-DCVTER_WITH_TESSERACT $(TESS_CFLAGS),) \
		$(viz_srcs) -o $@ \
		$$(pkg-config --cflags --libs opencv4) $(TESS_LIBS) $(ZBAR_LIBS)

test-viz : $(viz_so)
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_blob_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_pattern_match_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_shape_match_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_caliper_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_rect_metrology_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_similarity_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_golden_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_seg_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_feature_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_code_viz.py -v
	@PYTHONPATH=$(CURDIR)/$(viz_py_dir) $(pyexec) $(CURDIR)/tests/test_calib_viz.py -v

test-deeplab : $(name)
	@PYTHONPATH=$(CURDIR)/$(workdir) $(python) $(CURDIR)/tests/test_deeplabv3.py -v

pro       : $(workdir)/pro

runpro    : pro
	@export LD_LIBRARY_PATH=$(library_path_export)
	@cd $(workdir)/ && ./pro

$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

$(workdir)/pro : $(pro_cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) $(CXXFLAGS) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) $(CXXFLAGS) -c $< -o $@ $(cu_compile_flags)

$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)

$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)


clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/pro  $(workdir)/imgs

.PHONY : clean run $(name) runpro test-caliper test-match test-shape test-deeplab test-scale test-blob test-rect test-golden test-viz
