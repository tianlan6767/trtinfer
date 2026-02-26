cmake_minimum_required(VERSION 3.18)

# -------------------- 固定 CUDA 工具链路径 --------------------
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6" CACHE PATH "CUDA toolkit root" FORCE)

project(trtsahi LANGUAGES CXX CUDA)

# -------------------- 生成 compile_commands.json 给 clangd 使用 --------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# -------------------- C++17 --------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------- 源码目录 --------------------
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)  # 指向 src 子目录

# -------------------- 收集源码 --------------------
file(GLOB_RECURSE CPP_SRCS 
    ${SRC_DIR}/*.cpp 
    ${SRC_DIR}/*.h 
    ${SRC_DIR}/*.hpp)
file(GLOB_RECURSE CU_SRCS ${SRC_DIR}/*.cu)

# -------------------- 生成目标 --------------------
add_executable(trtsahi ${CPP_SRCS} ${CU_SRCS})

# -------------------- 保持 VS 层级结构 --------------------
foreach(FILE ${CPP_SRCS} ${CU_SRCS})
    get_filename_component(PARENT_DIR "${FILE}" DIRECTORY)
    file(RELATIVE_PATH REL_PATH "${SRC_DIR}" "${PARENT_DIR}")
    string(REPLACE "/" "\\" GROUP_PATH "${REL_PATH}")
    source_group("${GROUP_PATH}" FILES "${FILE}")
endforeach()

# -------------------- 依赖库路径 --------------------
set(TensorRT_DIR "/home/ps/workspace/trt/lean/TensorRT-10.11.0.33")
set(OpenCV_DIR   "/home/ps/workspace/trt/lean/cv4110/install")
set(CUDNN_DIR    "/home/ps/workspace/trt/lean/cudnn9.5.0")  # cuDNN 路径

# -------------------- 查找包 --------------------
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# -------------------- Include 路径 --------------------
target_include_directories(trtsahi PRIVATE
    ${SRC_DIR}               # 让 clangd 找到 trt/infer.hpp
    ${TensorRT_DIR}/include
    ${OpenCV_DIR}/include/opencv4
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDNN_DIR}/include
)

# -------------------- 库路径 --------------------
link_directories(
    ${TensorRT_DIR}/lib
    ${OpenCV_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_DIR}/lib
)

# -------------------- 链接库 --------------------
target_link_libraries(trtsahi PRIVATE
    # OpenCV 指定库
    ${OpenCV_DIR}/lib/libopencv_core.so
    ${OpenCV_DIR}/lib/libopencv_imgproc.so
    ${OpenCV_DIR}/lib/libopencv_imgcodecs.so
    ${OpenCV_DIR}/lib/libopencv_highgui.so

    # TensorRT
    nvinfer
    nvinfer_plugin
    nvonnxparser

    # CUDA / cuDNN
    cuda
    cublas
    cudart
    cudnn
)

# -------------------- CUDA 属性（可选） --------------------
set_target_properties(trtsahi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;90"
)