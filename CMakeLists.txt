cmake_minimum_required(VERSION 3.18)

if(WIN32)
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe")
else()
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

project(cvter LANGUAGES CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# -------------------- 生成 compile_commands.json 给 clangd 使用 --------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(UNIX)

    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-fPIC"
    )

    set(CMAKE_CUDA_FLAGS_RELEASE
        "${CMAKE_CUDA_FLAGS_RELEASE} -O3"
    )

elseif(WIN32)

    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} --use-local-env -Xcompiler=/utf-8"
    )

endif()


if(MSVC)

    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
    )

else()

    add_compile_options(
        -Wall
        -Wextra
    )

endif()

# -------------------- C++17 --------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------- CUDA C++17 --------------------
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_OBJECT_PATH_MAX 260)

# -------------------- 源码目录 --------------------
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# -------------------- 收集源码 --------------------
file(GLOB_RECURSE CPP_SRCS CONFIGURE_DEPENDS
    ${SRC_DIR}/*.cpp 
)
file(GLOB_RECURSE CU_SRCS CONFIGURE_DEPENDS 
    ${SRC_DIR}/*.cu 
)
file(GLOB_RECURSE HEADER_SRCS CONFIGURE_DEPENDS
    ${SRC_DIR}/*.h 
    ${SRC_DIR}/*.hpp 
    ${SRC_DIR}/*.cuh 
)

# -------------------- 排除 3rdParty --------------------
list(FILTER CPP_SRCS EXCLUDE REGEX ".*/3rdParty/.*")
list(FILTER CU_SRCS EXCLUDE REGEX ".*/3rdParty/.*")
list(FILTER HEADER_SRCS EXCLUDE REGEX ".*/3rdParty/.*")

add_executable(cvter ${CPP_SRCS} ${CU_SRCS}  ${HEADER_SRCS})

source_group(TREE ${SRC_DIR} FILES
    ${CPP_SRCS}
    ${CU_SRCS}
    ${HEADER_SRCS}
)


set_source_files_properties(
    ${SRC_DIR}/common/tensorrt8.cpp
    PROPERTIES HEADER_FILE_ONLY TRUE
)

# -------------------- 手动指定依赖路径 --------------------
if(WIN32)
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" CACHE PATH "CUDA toolkit root" FORCE)
    set(TensorRT_DIR "D:/envs/lean/TensorRT-10.11.0.33")
    set(OpenCV_DIR "D:/envs/lean/opencv410-office/build")
    set(CUDNN_DIR "D:/envs/lean/cudnn-951_cuda12")
    set(PYTHON_DIR "D:/envs/py3117")
    set(SPDLOG_DIR "D:/work/workspace/projects/cvter/src/3rdParty/spdlog")
else()
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6" CACHE PATH "CUDA toolkit root" FORCE)
    set(TensorRT_DIR "/home/ps/workspace/trt/lean/TensorRT-10.11.0.33")
    set(OpenCV_DIR "/home/ps/workspace/trt/lean/cv4110/install")
    set(CUDNN_DIR "/home/ps/workspace/trt/lean/cudnn9.5.0")
    set(PYTHON_DIR "/usr")
    set(SPDLOG_DIR "${SRC_DIR}/3rdParty/spdlog")
endif()

# -------------------- Include --------------------
if(WIN32)
    target_include_directories(cvter PRIVATE
        "${SRC_DIR}"
        "${SRC_DIR}/common"
        "${SRC_DIR}/pybind11"
        "${SRC_DIR}/kernel"
        "${SRC_DIR}/trt"
        "${TensorRT_DIR}/include"
        "${OpenCV_DIR}/include"
        "${CUDA_TOOLKIT_ROOT_DIR}/include"
        "${CUDNN_DIR}/include"
        "${PYTHON_DIR}/include"
        "${SPDLOG_DIR}/include"
    )
else()
    target_include_directories(cvter PRIVATE
            "${SRC_DIR}"
            "${SRC_DIR}/common"
            "${SRC_DIR}/pybind11"
            "${SRC_DIR}/kernel"
            "${SRC_DIR}/trt"
            "${TensorRT_DIR}/include"
            "${OpenCV_DIR}/include/opencv4"
            "${CUDA_TOOLKIT_ROOT_DIR}/include"
            "${CUDNN_DIR}/include"
            "${SPDLOG_DIR}/include"
        )
endif()


# -------------------- Link directories --------------------
if(WIN32)
    target_link_directories(cvter PRIVATE
        "${SRC_DIR}"
        "${TensorRT_DIR}/lib"
        "${OpenCV_DIR}/x64/vc16/lib"
        "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
        "${CUDNN_DIR}/lib/x64"
        "${PYTHON_DIR}/libs"
    )
else()
    target_link_directories(cvter PRIVATE
        ${TensorRT_DIR}/lib
        ${OpenCV_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDNN_DIR}/lib
    )
endif()

# -------------------- Link libraries --------------------
if(WIN32)
    set(OPENCV_LIBS
        opencv_world4100
    )
    set(TENSORRT_LIBS
        nvinfer_10
        nvinfer_plugin_10
        nvonnxparser_10
    )
    set(CUDA_LIBS
        cudart
        cublas
    )
    set(CUDNN_LIBS
        cudnn
    )
    set(PYTHON_LIBS
        python311
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

target_link_libraries(cvter PRIVATE
    ${OPENCV_LIBS}
    ${TENSORRT_LIBS}
    ${CUDA_LIBS}
    ${CUDNN_LIBS}
    ${PYTHON_LIBS}
)

# -------------------- CUDA properties --------------------

set_target_properties(cvter PROPERTIES
    CUDA_ARCHITECTURES "86;89"
    CUDA_RUNTIME_LIBRARY Shared
)
