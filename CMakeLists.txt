cmake_minimum_required(VERSION 3.18)


if(UNIX)
    # Linux / Ubuntu 固定 CUDA 路径
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6" CACHE PATH "CUDA toolkit root" FORCE)
elseif(WIN32)
    # Windows 可以不写，或指定你的安装路径
    set(CMAKE_CUDA_COMPILER "D:/libs/cuda-12.6/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
    set(CUDA_TOOLKIT_ROOT_DIR "D:/libs/cuda-12.6" CACHE PATH "CUDA toolkit root" FORCE)
endif()

project(trtsahi LANGUAGES CXX CUDA)


# -------------------- 生成 compile_commands.json 给 clangd 使用 --------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall --compiler-options -fPIC")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

if(UNIX)
    # Linux / Ubuntu 特有的 nvcc 编译选项
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
elseif(WIN32)
    # Windows + VS2019/2022 特有选项（可选）
    # MSVC 默认开启大部分警告，-Wall 无需手动添加
    set(CMAKE_CUDA_FLAGS_RELEASE "/O2") # 优化级别
endif()

# -------------------- C++17 --------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------- 源码目录 --------------------
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# -------------------- 收集源码 --------------------
file(GLOB_RECURSE CPP_SRCS
    ${SRC_DIR}/*.cpp
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.hpp
)

file(GLOB_RECURSE CU_SRCS
    ${SRC_DIR}/*.cu
)

add_executable(trtsahi ${CPP_SRCS} ${CU_SRCS})

# -------------------- VS 层级结构 --------------------
foreach(FILE ${CPP_SRCS} ${CU_SRCS})
    get_filename_component(PARENT_DIR "${FILE}" DIRECTORY)
    file(RELATIVE_PATH REL_PATH "${SRC_DIR}" "${PARENT_DIR}")
    string(REPLACE "/" "\\" GROUP_PATH "${REL_PATH}")
    source_group("${GROUP_PATH}" FILES "${FILE}")
endforeach()

# -------------------- 手动指定依赖路径 --------------------
if(WIN32)
    set(CUDA_TOOLKIT_ROOT_DIR "D:/libs/cuda-12.6")
    set(TensorRT_DIR "D:/libs/TensorRT-10.11.0.33")
    set(OpenCV_DIR "D:/libs/opencv/build")
    set(CUDNN_DIR "D:/libs/cudnn-9.5")
else()
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
    set(TensorRT_DIR "/home/ps/workspace/trt/lean/TensorRT-10.11.0.33")
    set(OpenCV_DIR "/home/ps/workspace/trt/lean/cv4110/install")
    set(CUDNN_DIR "/home/ps/workspace/trt/lean/cudnn9.5.0")
endif()

# -------------------- Include --------------------
target_include_directories(trtsahi PRIVATE
    ${SRC_DIR}
    ${TensorRT_DIR}/include
    ${OpenCV_DIR}/include/opencv4
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CUDNN_DIR}/include
)

# -------------------- Link directories --------------------
if(WIN32)
    link_directories(
        ${SRC_DIR} 
        ${TensorRT_DIR}/lib
        ${OpenCV_DIR}/x64/vc16/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDNN_DIR}/lib/x64
    )
else()
    link_directories(
        ${SRC_DIR} 
        ${TensorRT_DIR}/lib
        ${OpenCV_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDNN_DIR}/lib
    )
endif()

# -------------------- Link libraries --------------------
if(WIN32)
    set(OPENCV_LIBS
        opencv_core411
        opencv_imgproc411
        opencv_imgcodecs411
        opencv_highgui411
    )
    set(TENSORRT_LIBS
        nvinfer
        nvinfer_plugin
        nvonnxparser
    )
    set(CUDA_LIBS
        cudart
        cublas
    )
    set(CUDNN_LIBS
        cudnn
    )
else()
    set(OPENCV_LIBS
        ${OpenCV_DIR}/lib/libopencv_core.so
        ${OpenCV_DIR}/lib/libopencv_imgproc.so
        ${OpenCV_DIR}/lib/libopencv_imgcodecs.so
        ${OpenCV_DIR}/lib/libopencv_highgui.so
    )
    set(TENSORRT_LIBS
        nvinfer
        nvinfer_plugin
        nvonnxparser
    )
    set(CUDA_LIBS
        cuda
        cublas
        cudart
    )
    set(CUDNN_LIBS
        cudnn
    )
endif()

target_link_libraries(trtsahi PRIVATE
    ${OPENCV_LIBS}
    ${TENSORRT_LIBS}
    ${CUDA_LIBS}
    ${CUDNN_LIBS}
)

# -------------------- CUDA properties --------------------
set_target_properties(trtsahi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89;90"
)
