# Parking Space Detector (CUDA)

CUDA + OpenCV demo: video frames are processed on the GPU (grayscale, Sobel edges, effect filter, per-spot variance). Parking spots from a coordinate file are drawn **green (free)** or **red (occupied)**.

Builds on **Windows** and **Linux**. Outputs: `parking_detector` (video) and `image_filter` (single image).

---

## Requirements

| Component | Notes |
|-----------|--------|
| NVIDIA GPU + driver | `nvidia-smi` |
| CUDA Toolkit | `nvcc --version` |
| CMake 3.18+ | |
| C++17 compiler | MSVC (Windows) or GCC (Linux) |
| OpenCV (C++) | `find_package(OpenCV)` |
| Python + opencv-python | Optional — `tools/*.py` only |

---

## CMake settings

Edit [`CMakeLists.txt`](CMakeLists.txt) or pass flags at configure time.

| Setting | Windows (dev PC) | Linux lab PC |
|---------|------------------|--------------|
| `CUDA_ARCH` | `86` (RTX 3050) | `61` (GT 1030) |
| `OpenCV_DIR` | vcpkg or `C:\path\to\opencv\build` | `/usr/lib/x86_64-linux-gnu/cmake/opencv4` |
| Extra | Add OpenCV `bin` to `PATH` at runtime | Use `ssh -Y` to see GUI over SSH |

Find your GPU arch: `nvidia-smi --query-gpu=compute_cap --format=csv`

Wrong `CUDA_ARCH` → runtime error *no kernel image is available*.

---

## Windows — build and run

From the project root in PowerShell:

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86

# If OpenCV not found (pick one):
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86 `
#   -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=86 `
#   -DOpenCV_DIR=C:\path\to\opencv\build

cmake --build build --config Release
```

Run (from project root):

```powershell
.\build\Release\parking_detector.exe
.\build\Release\parking_detector.exe data\sample data\sample.mp4
```

Image filter demo:

```powershell
.\build\Release\image_filter.exe path\to\image.jpg
```

---

## Linux lab PC — build and run

No admin needed if `gcc`, `cmake`, `nvcc`, and `libopencv-dev` are already installed.

```bash
cd ~/Parking-Space-Detector-Using-Cuda

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH=61 \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4

cmake --build build -j4
```

Run:

```bash
./build/parking_detector
./build/parking_detector data/sample /path/to/video.mp4
```

If `nvcc` rejects GCC 14:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=61 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
```

---

## Run from Kali VM via SSH (lab PC)

Use this when you develop on a **Kali VM** and SSH into the lab machine. OpenCV windows appear on **Kali**, not on the lab monitor.

1. **Kali must have a desktop** — in a Kali terminal (before SSH):
   ```bash
   echo $DISPLAY    # should show :0 or similar
   ```

2. **SSH with X11 forwarding** (not plain `ssh`):
   ```bash
   ssh -Y rl22868@deggs206pc05
   echo $DISPLAY    # should show localhost:10.0 or similar
   ```

3. **Build and run** on the lab PC session:
   ```bash
   cd ~/Parking-Space-Detector-Using-Cuda
   cmake --build build -j4
   ./build/parking_detector
   ```

If you see `[ERROR] No DISPLAY`, reconnect with `ssh -Y`. X11 over the network may be slow on a GT 1030 — that is normal.

---

## Run arguments and keys

| Argument | Default | Description |
|----------|---------|-------------|
| 1 | `data/sample` | Spot file — one `x,y` per line (top-left of 40×80 ROI) |
| 2 | `data/sample.mp4` | Video file (add your own if missing) |

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save `output_parking.jpg` |

Run from the **project root** so default paths resolve.

---

## Python tools (optional)

Mark parking spots on a still frame:

```bash
pip3 install --user opencv-python
python3 tools/extract_frame.py
python3 tools/parking_generator.py --image data/frame.png --slots data/sample
```

Keep `ROI_W` / `ROI_H` in sync between `tools/parking_generator.py` and `src/main.cu`. Tune sensitivity with `VARIANCE_THRESHOLD` in `src/main.cu` (rebuild after changes).

---

## Project layout

```
├── CMakeLists.txt
├── data/sample          # spot coordinates
├── data/sample.mp4      # your video (add locally)
├── src/main.cu          # parking demo
├── src/kernels.cu
└── tools/               # extract_frame.py, parking_generator.py
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `missing CMakeCache.txt` | Run `cmake -B build ...` before `cmake --build` |
| `find_package(OpenCV)` failed | Pass `-DOpenCV_DIR=...` |
| `no kernel image is available` | Wrong `CUDA_ARCH` (lab: `61`) |
| `unsupported GNU version` | `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13` |
| `__cxa_call_terminate@CXXABI_1.3.15` | `rm -rf build`, reconfigure with current `CMakeLists.txt` |
| `[ERROR] No DISPLAY` | Use `ssh -Y` from Kali desktop |
| `Cannot open video` | Add video or pass path as arg 2 |
| Windows DLL not found | Add OpenCV `bin` to `PATH` |

---

## Clean rebuild

```bash
rm -rf build          # Linux
cmake -B build ...    # configure again
cmake --build build   # add -j4 on Linux, --config Release on Windows
```

```powershell
Remove-Item -Recurse -Force build   # Windows
```
