# 环境配置

本文档描述 **当前本机实际在用** 的编译与运行环境。路径以 [Makefile](../Makefile)（主构建）和 [CMakeLists.txt](../CMakeLists.txt) 为准。

## 当前环境一览

| 组件 | 版本 / 型号 | 路径或说明 |
|------|-------------|------------|
| OS | Ubuntu 22.04 (WSL2) | `5.15.153.1-microsoft-standard-WSL2` |
| GPU | RTX 4090 D | Compute Capability **8.9** |
| Driver | 576.57 | `nvidia-smi` 显示 CUDA Runtime 上限 12.9 |
| CUDA Toolkit | **12.6** | `/usr/local/cuda-12.6`（`nvcc` 12.6.20） |
| cuDNN | **9.5.0** | `/home/ps/workspace/trt/lean/cudnn9.5.0`（系统亦有 `/usr/lib/.../libcudnn.so.9`） |
| TensorRT | **10.11.0.33** | `/home/ps/workspace/trt/lean/TensorRT-10.11.0.33` |
| OpenCV | **4.11.0**（CUDA 12.6 + cuDNN 9.5） | `/home/ps/workspace/trt/lean/cv4110/install` |
| Python | **3.12.3** | `/usr/local/bin/python3.12` |
| 编译器 | g++ 11.4.0 / C++17 | `cc := g++` |
| CMake | 3.22.1 | 可选，clangd / IDE 用 |
| TensorRT API | **TRT10** | `Makefile` 中 `TRT_VERSION := 10` → `-DTRT10` |

依赖根目录统一放在：

```text
/home/ps/workspace/trt/lean/
├── TensorRT-10.11.0.33/
├── cudnn9.5.0/
└── cv4110/install/          # OpenCV 安装前缀
```

## Makefile 关键配置（日常编译）

主构建命令：`make -j`，产物为 `workspace/cvter.so`。

```makefile
cuda_home := /usr/local/cuda-12.6
cuda_arch := 8.9
TRT_VERSION := 10          # 8 → TRT8 API；10 → TRT10 API（当前）

opencv  include/lib : /home/ps/workspace/trt/lean/cv4110/install/...
trt     include/lib : /home/ps/workspace/trt/lean/TensorRT-10.11.0.33/...
python  include     : /usr/local/include/python3.12
python  lib         : /usr/local/lib/python3.12/config-3.12-x86_64-linux-gnu
python  解释器      : /usr/local/bin/python3.12
```

链接库：`opencv_core/imgproc/videoio/imgcodecs/highgui`、`nvinfer/nvinfer_plugin/nvonnxparser`、`cuda/cublas/cudart/cudnn`。

WSL 下额外加入 `/usr/lib/wsl/lib` 到 rpath。

### 常用目标

```bash
make -j
make test-caliper test-match test-shape test-deeplab
# 推理脚本
cd workspace && /usr/local/bin/python3.12 test.py
```

导入 `cvter` 时使用 **同一 Python 3.12**，并将 `workspace` 加入 `PYTHONPATH`。

切换 TensorRT API：

```bash
make clean
make -j TRT_VERSION=8    # 或默认 10
```

## CMakeLists.txt 路径（IDE / clangd）

Linux 段与 Makefile 对齐：

| 变量 | 值 |
|------|-----|
| `CUDA_TOOLKIT_ROOT_DIR` | `/usr/local/cuda-12.6` |
| `TensorRT_DIR` | `/home/ps/workspace/trt/lean/TensorRT-10.11.0.33` |
| `OpenCV_DIR` | `/home/ps/workspace/trt/lean/cv4110/install` |
| `CUDNN_DIR` | `/home/ps/workspace/trt/lean/cudnn9.5.0` |
| `CUDA_ARCHITECTURES` | `86;89` |

生成 `compile_commands.json`：

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

clangd 配置见仓库根目录 [.clangd](../.clangd) 与 README「配置 clangd 环境」。

## 运行时库路径

若找不到 `.so`，可导出：

```bash
export LD_LIBRARY_PATH=/home/ps/workspace/trt/lean/cv4110/install/lib:\
/home/ps/workspace/trt/lean/TensorRT-10.11.0.33/lib:\
/usr/local/cuda-12.6/lib64:\
/home/ps/workspace/trt/lean/cudnn9.5.0/lib:\
$LD_LIBRARY_PATH

export PYTHONPATH=/home/ps/workspace/trt/cvter/workspace:$PYTHONPATH
```

Makefile 链接时已对上述多数目录写入 `-Wl,-rpath=...`，通常无需再设。

## OpenCV 编译参考（当前机器曾用配置）

OpenCV 已装在 `cv4110/install`（4.11.0 + CUDA 12.6 + cuDNN 9.5）。若需在同类机器上重编，可参考：

```bash
cd /home/ps/workspace/trt/lean/cv4110/opencv/build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/home/ps/workspace/trt/lean/cv4110/install \
      -D OPENCV_EXTRA_MODULES_PATH=/home/ps/workspace/trt/lean/cv4110/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="8.6;8.9" \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDNN_VERSION=9.5 \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
      -D WITH_TENSORRT=ON \
      -D TensorRT_INCLUDE_DIRS=/home/ps/workspace/trt/lean/TensorRT-10.11.0.33/include/ \
      -D TensorRT_LIBRARIES="/home/ps/workspace/trt/lean/TensorRT-10.11.0.33/lib/libnvinfer.so;/home/ps/workspace/trt/lean/TensorRT-10.11.0.33/lib/libnvinfer_plugin.so;/home/ps/workspace/trt/lean/TensorRT-10.11.0.33/lib/libnvinfer_lean.so" \
      -D WITH_GTK=ON \
      -D BUILD_EXAMPLES=OFF ..
make -j$(nproc)
make install
```

换机时只需改 `CMAKE_INSTALL_PREFIX`、CUDA / TensorRT / cuDNN 路径，并同步修改 `Makefile` 与 `CMakeLists.txt` 中对应变量。

## 换机检查清单

1. 安装 CUDA 12.6，确认 `nvcc` 可用。
2. 安装 cuDNN 9.5，与 CUDA 12.x 匹配。
3. 解压 TensorRT 10.11.0.33，保证 `include/`、`lib/` 完整。
4. 安装 OpenCV 4.11（建议带 CUDA），或改 Makefile 指向本机 OpenCV。
5. Python ≥ 3.12 开发头文件：`/usr/local/include/python3.12`（或改 `python_include_path`）。
6. `cuda_arch` / `CUDA_ARCHITECTURES` 与目标 GPU 一致（4090 → 8.9；3090 → 8.6）。
7. `make -j` 后在 `workspace` 下用同一解释器 `import cvter`。

## Windows

Windows + **Visual Studio 2019** 的完整配置（路径、CMake、PATH、常见问题）见 **[env_windows.md](env_windows.md)**。日常 Linux/WSL 开发以本文为准。
