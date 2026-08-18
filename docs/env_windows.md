# Windows + Visual Studio 2019 配置

本文档描述仓库在 **Windows** 上用 **VS2019（MSVC v142 / vc16）+ CMake** 编译时的依赖与步骤。路径以 [CMakeLists.txt](../CMakeLists.txt) 中 `if(WIN32)` 段为准。

Linux / WSL 见 [env.md](env.md)。

## 当前 Windows 依赖一览

| 组件 | 版本 | CMakeLists 中的路径 |
|------|------|---------------------|
| IDE / 工具链 | **Visual Studio 2019** x64，C++17 | 生成器：`Visual Studio 16 2019` |
| CUDA Toolkit | **12.9** | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9` |
| cuDNN | **9.5.1**（CUDA 12） | `D:/envs/lean/cudnn-951_cuda12` |
| TensorRT | **10.11.0.33** | `D:/envs/lean/TensorRT-10.11.0.33` |
| OpenCV | **4.10.0**（`opencv_world4100`，`x64/vc16`） | `D:/envs/lean/opencv410-office/build` |
| Python | **3.11** | `D:/envs/py3117` |
| spdlog | 源码内置 | 建议改为 `${SRC_DIR}/3rdParty/spdlog`（见下文） |
| CUDA Arch | `86;89` | 对应 30 系 / 40 系 GPU |
| TensorRT 封装 | **TRT10**（`tensorrt.cpp`，`tensorrt8.cpp` 不参与编译） | 链接 `nvinfer_10` 等 |

依赖目录示意：

```text
D:/envs/
├── lean/
│   ├── TensorRT-10.11.0.33/     # include/, lib/, bin/
│   ├── cudnn-951_cuda12/        # include/, lib/x64/
│   └── opencv410-office/build/  # include/, x64/vc16/lib|bin/
└── py3117/                      # include/, libs/, python.exe
```

换机时改 `CMakeLists.txt` 里上述变量即可，**不要**用 Linux 的 `Makefile` 在 Windows 上构建。

## 安装清单

1. **Visual Studio 2019**  
   - 工作负载：**使用 C++ 的桌面开发**  
   - 组件：MSVC v142、Windows 10/11 SDK、**C++ CMake 工具**（或单独装 CMake ≥ 3.18）  
   - 平台目标：**x64**（不要用 Win32）

2. **CUDA 12.9**  
   - 安装时勾选 VS2019 集成  
   - 确认存在：  
     `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe`

3. **cuDNN 9.5.x（CUDA 12）**  
   - 解压到例如 `D:\envs\lean\cudnn-951_cuda12`  
   - 需有 `include\`、`lib\x64\`（及运行时 `bin` 若单独提供）

4. **TensorRT 10.11.0.33（Windows）**  
   - 解压到 `D:\envs\lean\TensorRT-10.11.0.33`  
   - 确认 `include\`、`lib\`、`bin\`（DLL）齐全

5. **OpenCV 4.10（VS2019 / vc16）**  
   - 当前配置期望：`opencv_world4100.lib` 位于  
     `...\opencv410-office\build\x64\vc16\lib`  
   - Debug 一般对应 `opencv_world4100d.lib`；若只有 Release 世界库，请用 **Release** 配置编译本工程

6. **Python 3.11**  
   - 安装到 `D:\envs\py3117`（或改 `PYTHON_DIR`）  
   - 需开发文件：`include\`、`libs\python311.lib`

## 修改本机路径

打开根目录 `CMakeLists.txt`，按机器修改 Windows 段：

```cmake
if(WIN32)
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" CACHE PATH "CUDA toolkit root" FORCE)
    set(TensorRT_DIR "D:/envs/lean/TensorRT-10.11.0.33")
    set(OpenCV_DIR "D:/envs/lean/opencv410-office/build")
    set(CUDNN_DIR "D:/envs/lean/cudnn-951_cuda12")
    set(PYTHON_DIR "D:/envs/py3117")
    # 建议改为仓库内路径，避免写死另一台机器的盘符：
    set(SPDLOG_DIR "${SRC_DIR}/3rdParty/spdlog")
endif()
```

库名对照（已在 CMakeLists 中写好）：

| 类别 | 链接名 |
|------|--------|
| OpenCV | `opencv_world4100` |
| TensorRT | `nvinfer_10`、`nvinfer_plugin_10`、`nvonnxparser_10` |
| CUDA | `cudart`、`cublas` |
| cuDNN | `cudnn` |
| Python | `python311` |

OpenCV 链接目录固定为 `${OpenCV_DIR}/x64/vc16/lib`（**vc16 = VS2019**）。若你用 VS2022 自编的 OpenCV（vc17），需同步改该路径与库名。

## 用 VS2019 打开并配置

### 方式 A：Visual Studio 打开文件夹（推荐）

1. VS2019 → **文件 → 打开 → 文件夹** → 选仓库根目录（含 `CMakeLists.txt`）。
2. 菜单 **项目 → CMake 设置**，或编辑 `.vs/CMakeSettings.json`，建议：  
   - Configuration：`x64-Release`（或 `x64-Debug`）  
   - Generator：`Visual Studio 16 2019` / 工具集 `v142`  
   - 确认 `CMAKE_CUDA_COMPILER` 指向 CUDA 12.9 的 `nvcc.exe`
3. **项目 → 配置缓存**（Configure），无报错后再 **生成 → 全部生成**。

命令行等价：

```bat
cd /d D:\work\workspace\projects\cvter
mkdir build-win & cd build-win
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release -j
```

产物默认在 `build-win\Release\cvter.exe`（当前 `CMakeLists.txt` 使用 `add_executable(cvter ...)`）。

### 方式 B：先生成 `.sln` 再打开

```bat
cmake -G "Visual Studio 16 2019" -A x64 -B build-win -S .
start build-win\cvter.sln
```

在解决方案里选 **Release | x64** 后生成。

## 运行时 PATH（DLL）

运行或调试前，把下列目录加入系统/用户 **PATH**，或在 VS 调试环境的 `PATH` 中前置：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin
D:\envs\lean\TensorRT-10.11.0.33\lib
D:\envs\lean\TensorRT-10.11.0.33\bin
D:\envs\lean\cudnn-951_cuda12\bin
D:\envs\lean\cudnn-951_cuda12\lib\x64
D:\envs\lean\opencv410-office\build\x64\vc16\bin
D:\envs\py3117
```

缺少 DLL 时常见报错：`nvinfer_10.dll` / `cudart64_*.dll` / `opencv_world4100.dll` / `cudnn64_9.dll` 找不到。

## Python 扩展说明

- Linux：`Makefile` 产出 `workspace/cvter.so`，解释器为 Python **3.12**。  
- Windows CMake：当前目标是 **可执行文件 `cvter`**，且链接的是 Python **3.11**。  
- 若要在 Windows 得到可 `import cvter` 的模块，需另行把目标改为 `SHARED` / pybind 扩展（如 `cvter.cp311-win_amd64.pyd`），并保证与 `PYTHON_DIR` 同一解释器；本仓库默认 CMake 尚未做成 `.pyd` 一键产物。

调试 exe 时，工作目录建议设为含模型与测试图的目录（可参考 Linux 的 `workspace/`）。

## 编译选项（已在 CMakeLists 中）

- C++ / CUDA：**C++17**
- MSVC：`/utf-8`、`/MP`
- CUDA：`--use-local-env -Xcompiler=/utf-8`
- `CMAKE_CUDA_SEPARABLE_COMPILATION ON`
- `tensorrt8.cpp` 标记为 `HEADER_FILE_ONLY`（走 TRT10 实现）
- `CUDA_ARCHITECTURES`：`86;89`（按 GPU 增删，如仅 4090 可只留 `89`）

## 常见问题

| 现象 | 处理 |
|------|------|
| `nvcc` / CUDA 语言未启用 | 重装 CUDA 并勾选 VS2019；用 x64 主机工具集重新 Configure |
| 找不到 `opencv_world4100` | 检查 `OpenCV_DIR` 与 `x64\vc16\lib`；Debug/Release 库名是否带 `d` |
| 找不到 `nvinfer_10` | 确认 TRT 为 10.x Windows 包；`lib` 目录在链接路径中 |
| 路径过长 / 对象文件失败 | 已设 `CMAKE_OBJECT_PATH_MAX 260`；仍失败则把工程放到较短盘符路径 |
| spdlog 头文件找不到 | 将 `SPDLOG_DIR` 改为 `"${SRC_DIR}/3rdParty/spdlog"` |
| GPU 架构不匹配 | 按显卡改 `CUDA_ARCHITECTURES`（如 3080→86，4090→89） |

## 与 Linux 环境对照

| 项 | Linux（Makefile） | Windows（CMake + VS2019） |
|----|-------------------|---------------------------|
| CUDA | 12.6 | 12.9 |
| OpenCV | 4.11（分模块 `.so`） | 4.10（`opencv_world4100`） |
| Python | 3.12 → `cvter.so` | 3.11 → 默认 `cvter.exe` |
| 主构建入口 | `make -j` | VS CMake / `cmake --build` |
| 依赖根 | `/home/ps/workspace/trt/lean` | `D:/envs/lean` |
