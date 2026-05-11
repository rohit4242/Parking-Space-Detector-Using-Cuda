# Parking Space Detector (CUDA)

Real-time parking occupancy demo: CUDA kernels convert frames to grayscale, run Sobel edges, apply an edge-based effect, and compute per-spot variance inside fixed ROIs. OpenCV loads video and draws **green (free)** / **red (occupied)** boxes from a simple coordinate file.

---

## What you need

| Requirement | Notes |
|-------------|--------|
| **NVIDIA GPU** + driver | CUDA-capable GPU |
| **CUDA Toolkit** | Match your driver (CMake uses the CUDA language) |
| **CMake** | 3.18 or newer |
| **C++17 compiler** | **Windows:** MSVC with CUDA support. **Linux:** GCC or Clang |
| **OpenCV** | Built or installed so CMake `find_package(OpenCV)` succeeds |
| **Python 3** + **OpenCV-Python** | Optional; for `tools/*.py` (frame extract + spot editor) |

---

## GPU architecture (important)

The build targets **compute capability 8.6** (e.g. RTX 3050 — Ampere). If your GPU is different, edit `CMakeLists.txt`:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)
```

Common values: `75` (Turing), `86` (Ampere consumer), `89` (Ada Lovelace), etc. See [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus).

---

## Build (CMake)

From the **project root**:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Always run **`cmake -B build ...`** first; otherwise you get *missing CMakeCache.txt*.

**First time OpenCV location (if `find_package` fails):** find the folder that contains `OpenCVConfig.cmake`, then configure with:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="C:/path/to/OpenCV/build"
```

With **vcpkg**, install `opencv`, then pass `-DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake` (with or instead of `OpenCV_DIR`, depending on your setup).

**Windows (Visual Studio):** use `--config Release`; executables are usually `build\Release\*.exe`. With **Ninja**, they are often in `build\`.

**Outputs:**

| Executable | Purpose |
|------------|---------|
| `parking_detector` | Main app: video + parking spots |
| `image_filter` | Standalone demo: grayscale / Sobel / effect on one image |

---

## Run

Defaults are **`data/sample`** (spots) and **`data/sample.mp4`** (video)—paths are relative to **your current folder**, not to the executable. Easiest fix: open a shell in the project root, then run `build/parking_detector` or `build\Release\parking_detector.exe`. Or pass two explicit paths:

```bash
build/parking_detector my_spots.txt my_video.mp4
```

### Main detector

```bash
build/parking_detector
.\build\Release\parking_detector.exe
```

**Arguments:**

1. **Positions file** — one `x,y` per line (top-left of **40×80** ROI). Default: `data/sample`.
2. **Video file.** Default: `data/sample.mp4`.

**Keyboard:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot `output_parking.jpg` |

Occupancy uses grayscale variance inside each ROI vs. an internal threshold (`VARIANCE_THRESHOLD` in `src/main.cu`). Tune threshold or ROI size there if needed.

### Image filter demo

```bash
build/image_filter path/to/image.jpg
build\Release\image_filter.exe path\to\image.jpg    # typical VS output on Windows
```

Shows original (color window uses the loaded Mat), grayscale, edges, and effect. Press any key to close.

---

## Mark parking spots (`parking_generator.py`)

Generate or edit the `x,y` list using a still frame from your video.

1. Install OpenCV for Python: `pip install opencv-python`
2. From the **project root**, save a reference frame (defaults: `data/sample.mp4` → `data/frame.png`):

```bash
python tools/extract_frame.py
python tools/extract_frame.py --ms 5000
```

3. Run the spot editor:

```bash
python tools/parking_generator.py
python tools/parking_generator.py --image data/frame.png --slots data/sample
```

**Controls:** left click = add spot (40×80 from click), right click = remove spot under cursor, `q` = quit, `s` = save overlay preview `generator_preview.jpg`. Positions are saved after each edit.

**Changing the spot box (width / height):** each line in the spots file is the **top-left corner** of a rectangle. The size is **not** stored in the file—it is fixed in code.

- Editor overlay: edit **`ROI_W`** and **`ROI_H`** at the top of `tools/parking_generator.py`, then re-run the tool so new clicks match the new box.
- Live detector: set the **same** **`ROI_W`** and **`ROI_H`** in `src/main.cu` (near `VARIANCE_THRESHOLD`), then **rebuild** `parking_detector`. If these two files disagree, on-screen boxes and variance math will not match what you marked in Python.
- **Sensitivity** (free vs occupied) is separate: adjust **`VARIANCE_THRESHOLD`** in `src/main.cu` only; rebuild after changing it.

---

## Project layout

```
├── CMakeLists.txt
├── data/
│   ├── sample            # example spots (default)
│   ├── sample.mp4        # example video (default)
│   └── frame.png         # optional reference still (from extract_frame or your own)
├── src/
│   ├── main.cu           # parking demo entry
│   ├── kernels.cu / kernels.cuh
│   └── image_filter/     # standalone filter demo
└── tools/
    ├── extract_frame.py   # video → PNG for the spot editor
    └── parking_generator.py
```

Place your **video** next to your coordinates file or pass full paths to `parking_detector`.

---

## Clean rebuild

Remove the `build` folder (Explorer, `rm -rf build`, or PowerShell: `Remove-Item -Recurse -Force build`), then:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## Notes

- **Windows + MSVC:** the project passes `/Zc:preprocessor` for some CUDA 13 + MSVC header setups. On **Linux**, if configure fails on unknown flags, adjust or remove the `target_compile_options(... CUDA ...)` blocks in `CMakeLists.txt` for your toolchain.
- Ensure OpenCV DLLs / `.so` paths are visible at runtime (e.g. `PATH` or install dir).
