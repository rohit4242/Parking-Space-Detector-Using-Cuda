# Parking Space Detector (CUDA)

Real-time parking occupancy demo using CUDA and OpenCV. Each video frame is processed on the GPU: BGR → grayscale → Sobel edges → visual effect, plus per-spot variance inside fixed ROIs. OpenCV draws **green (free)** / **red (occupied)** boxes from a simple coordinate file.

Works on **Windows** and **Linux** (CMake + NVIDIA GPU).

---

## Table of contents

1. [Requirements](#requirements)
2. [GPU architecture](#gpu-architecture)
3. [Installation](#installation)
   - [Windows](#windows)
   - [Linux](#linux)
   - [Linux without admin (lab PC)](#linux-without-admin-lab-pc)
4. [Configure & build](#configure--build)
5. [Run](#run)
6. [Parking spot editor (Python)](#parking-spot-editor-python)
7. [Tuning detection](#tuning-detection)
8. [Project layout](#project-layout)
9. [Troubleshooting](#troubleshooting)
10. [Clean rebuild](#clean-rebuild)

---

## Requirements

| Component | Required | Notes |
|-----------|----------|--------|
| NVIDIA GPU | Yes | CUDA-capable; check with `nvidia-smi` |
| NVIDIA driver | Yes | Must match your CUDA toolkit |
| CUDA Toolkit | Yes | `nvcc --version`; used by CMake CUDA language |
| CMake | Yes | 3.18+ (`cmake --version`) |
| C++17 compiler | Yes | **Windows:** MSVC. **Linux:** GCC or Clang |
| OpenCV (C++) | Yes | Headers + libs; `find_package(OpenCV)` |
| Python 3 + opencv-python | Optional | Only for `tools/*.py` |

**Executables produced:**

| Binary | Purpose |
|--------|---------|
| `parking_detector` | Main app: video + parking spots |
| `image_filter` | Demo: grayscale / Sobel / effect on one image |

---

## GPU architecture

Kernels must be compiled for your GPU's **compute capability**. Set this in `CMakeLists.txt` or pass `-DCUDA_ARCH=XX` at configure time.

| GPU examples | `CUDA_ARCH` |
|--------------|-------------|
| GT 1030, GTX 1050 | `61` |
| GTX 1660, RTX 2060 | `75` |
| RTX 3050, RTX 3060 | `86` |
| RTX 4060 | `89` |

**Find your value:**

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Reference: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

Wrong architecture → build may succeed but runtime errors like *no kernel image is available*.

---

## Installation

### Windows

1. **NVIDIA driver** — from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) or GeForce Experience.
2. **CUDA Toolkit** — [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (match driver; e.g. CUDA 12.x).
3. **Visual Studio 2022** — workload "Desktop development with C++" (MSVC + CMake support).
4. **CMake** — via Visual Studio or [cmake.org](https://cmake.org/download/).
5. **OpenCV** — one of:
   - **vcpkg:** `vcpkg install opencv` and use `-DCMAKE_TOOLCHAIN_FILE=...`
   - Prebuilt / self-built OpenCV; note path to `OpenCVConfig.cmake`

Verify:

```powershell
nvcc --version
cmake --version
nvidia-smi
```

### Linux

**With admin (`sudo`):**

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
# CUDA: install from NVIDIA repo or: sudo apt install nvidia-cuda-toolkit
```

**Verify:**

```bash
nvidia-smi
nvcc --version
cmake --version
pkg-config --modversion opencv4
```

Typical OpenCV CMake path (Debian/Ubuntu):

```text
/usr/lib/x86_64-linux-gnu/cmake/opencv4
```

### Linux without admin (lab PC)

You can build if these already exist (no `sudo`):

```bash
which gcc g++ cmake make nvcc
pkg-config --modversion opencv4
find /usr -name "OpenCVConfig.cmake" 2>/dev/null
```

Use system OpenCV via `-DOpenCV_DIR=...`. Python tools only:

```bash
pip3 install --user opencv-python
```

---

## Configure & build

Always run from the **project root**. Create `build/` with `cmake -B build` first.

### Quick reference

| Platform | Configure | Build | Typical executable |
|----------|-----------|-------|-------------------|
| Linux | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=61` | `cmake --build build -j4` | `build/parking_detector` |
| Windows (VS) | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86` | `cmake --build build --config Release` | `build\Release\parking_detector.exe` |

Add `-DOpenCV_DIR=...` if `find_package(OpenCV)` fails.

---

### Linux (full example — GT 1030)

```bash
cd Parking-Space-Detector-Using-Cuda

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH=61 \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4

cmake --build build -j4
```

**GCC too new for nvcc** (e.g. GCC 14 + CUDA 12.4):

```bash
ls /usr/bin/g++-*
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=61 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
```

---

### Windows (Visual Studio generator)

```powershell
cd Parking-Space-Detector-Using-Cuda

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86

# If using vcpkg:
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86 `
#   -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake

# If OpenCV not found:
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86 `
#   -DOpenCV_DIR=C:\path\to\opencv\build

cmake --build build --config Release
```

Outputs: `build\Release\parking_detector.exe`, `build\Release\image_filter.exe`.

---

### Windows (Ninja)

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86
cmake --build build
```

Outputs often directly in `build\`.

---

### What to change in `CMakeLists.txt`

| Setting | Where | When |
|---------|--------|------|
| `CUDA_ARCH` default | Top of `CMakeLists.txt` | Default GPU for your dev machine |
| `OpenCV_DIR` | CMake command line | When `find_package` fails |
| `CMAKE_CUDA_HOST_COMPILER` | CMake command line | Linux: nvcc rejects GCC 14 |
| MSVC `/Zc:preprocessor` | Automatic (`if(MSVC)`) | Do not add on Linux |

Override architecture without editing the file:

```bash
cmake -B build -DCUDA_ARCH=61
```

---

## Run

Paths are relative to your **current working directory**, not the executable. **Run from the project root** or pass full paths.

### Main detector

**Linux:**

```bash
./build/parking_detector
./build/parking_detector data/sample /path/to/video.mp4
```

**Windows:**

```powershell
.\build\Release\parking_detector.exe
.\build\Release\parking_detector.exe data\sample data\sample.mp4
```

**Arguments:**

1. Positions file — one `x,y` per line (top-left of ROI). Default: `data/sample`
2. Video file. Default: `data/sample.mp4`

> `data/sample.mp4` may not be in the repo; add your own video under `data/` or pass a path.

**Keyboard:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save `output_parking.jpg` |

**Windows:** ensure OpenCV DLLs are on `PATH` (vcpkg / OpenCV `bin` folder).

**Linux:** if missing `.so` errors:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

---

### Image filter demo

**Linux:**

```bash
./build/image_filter path/to/image.jpg
```

**Windows:**

```powershell
.\build\Release\image_filter.exe path\to\image.jpg
```

Shows original, grayscale, edges, and effect. Press any key to close.

---

## Parking spot editor (Python)

Mark spots on a still frame; saves the same `x,y` format as `data/sample`.

1. **Python OpenCV** (user install OK on lab PCs):

   ```bash
   pip install opencv-python
   # or: pip3 install --user opencv-python
   ```

2. **Extract a frame** from video:

   ```bash
   python tools/extract_frame.py
   python tools/extract_frame.py --ms 5000
   python tools/extract_frame.py --video data/sample.mp4 --output data/frame.png
   ```

3. **Run editor:**

   ```bash
   python tools/parking_generator.py
   python tools/parking_generator.py --image data/frame.png --slots data/sample
   ```

**Controls:**

| Input | Action |
|-------|--------|
| Left click | Add spot (40×80 box from click) |
| Right click | Remove spot under cursor |
| `q` | Quit |
| `s` | Save preview `generator_preview.jpg` |

Spots save to the slots file after each edit.

---

## Tuning detection

Spot **size** is fixed in code (not in the spots file). Keep these in sync:

| File | Constants |
|------|-----------|
| `tools/parking_generator.py` | `ROI_W`, `ROI_H` |
| `src/main.cu` | `ROI_W`, `ROI_H`, `VARIANCE_THRESHOLD` |

- Change ROI size in **both** files, then **rebuild** `parking_detector`.
- **Sensitivity** (free vs occupied): only `VARIANCE_THRESHOLD` in `src/main.cu`; rebuild after changes.

---

## Project layout

```
├── CMakeLists.txt          # Build config (Windows + Linux)
├── README.md
├── data/
│   ├── sample              # Example spot coordinates (default)
│   ├── sample.mp4          # Your video (add locally)
│   └── frame.png           # Reference still (from extract_frame.py)
├── src/
│   ├── main.cu             # Parking demo entry
│   ├── kernels.cu
│   ├── kernels.cuh
│   └── image_filter/       # Single-image filter demo
└── tools/
    ├── extract_frame.py
    └── parking_generator.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `missing CMakeCache.txt` | Run `cmake -B build ...` before `cmake --build` |
| `find_package(OpenCV)` failed | Pass `-DOpenCV_DIR=...` (folder with `OpenCVConfig.cmake`) |
| Linux: `unknown argument '/Zc:preprocessor'` | Use updated `CMakeLists.txt` with `if(MSVC)` |
| `no kernel image is available` | Wrong `CUDA_ARCH`; use e.g. `61` for GT 1030 |
| nvcc: `unsupported GNU version` | `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13` |
| `undefined reference to __cxa_call_terminate@CXXABI_1.3.15` | OpenCV/GCC ABI mismatch; use latest `CMakeLists.txt` (links `libstdc++` from your `g++`), then `rm -rf build` and reconfigure |
| `Cannot open video` | Add `data/sample.mp4` or pass video path as arg 2 |
| `No parking spots loaded` | Check positions file path and `x,y` format |
| OpenCV windows don't show (SSH) | Need local display or X11 forwarding; app uses `cv::imshow` |
| Windows: DLL not found | Add OpenCV `bin` to `PATH` |

---

## Clean rebuild

```bash
# Linux
rm -rf build

# Windows PowerShell
Remove-Item -Recurse -Force build
```

Then configure and build again (see [Configure & build](#configure--build)).

---

## How it works (short)

1. OpenCV reads each video frame into host memory (`frame.data`).
2. `std::memcpy` copies pixels into CUDA managed memory (`d_bgr`).
3. GPU kernels: grayscale → Sobel → effect → ROI variance per parking spot.
4. CPU reads variances, classifies free/occupied, draws boxes with OpenCV.
