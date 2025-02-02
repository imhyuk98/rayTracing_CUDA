# CUDA Ray Tracing Example

This repository showcases a **CUDA-based ray tracing** example, originally adapted from a CPU implementation. The code leverages GPU parallelism to render a scene with multiple spheres, various materials (lambertian, metal, dielectric), and a simple camera model.

**Table of Contents**  
1. [Overview](#overview)  
2. [Features](#features)  
3. [Dependencies](#dependencies)  
4. [Build Instructions](#build-instructions)  
5. [Usage](#usage)  
6. [Scene Description](#scene-description)  
7. [Performance Notes](#performance-notes)  
8. [License](#license)

---

## Overview

This project demonstrates a basic **Monte Carlo ray tracer** accelerated by CUDA. Each pixel in the output image is computed by a dedicated GPU thread (or small blocks of threads), sampling multiple rays per pixel to simulate soft shadows, reflection, refraction, and more.

**Key highlights**:
- **Ray generation** per pixel using a simple camera model.
- **Sphere-based geometry** with random placement and materials (diffuse, metal, glass).
- **Gamma-correction** and **anti-aliasing** via multiple samples per pixel.
- **Timing** measurement using CUDA events.

---

## Features

- **Parallel Rendering**: Each pixel is computed in parallel using CUDA kernels.  
- **Random State per Pixel**: Uses `curand` to generate pseudo-random values for Monte Carlo integration.  
- **Multiple Materials**: Lambertian (diffuse), metal (reflective), dielectric (refractive).  
- **Simple Scenes**: Large ground sphere plus many small spheres with randomized positions and materials.  
- **Configurable**: Resolution, number of samples, etc.

---

## Dependencies

1. **CUDA Toolkit** (e.g., version 10.0 or later).  
2. **C++ Compiler** supporting C++11 or higher (often bundled with CUDA).  
3. **CMake** (optional) if you want a more advanced build system, but you can also compile directly with `nvcc`.

---

## Build Instructions

1. **Clone** this repository:
   ```bash
   git clone https://github.com/YourUsername/cuda-raytracer.git
   cd cuda-raytracer
   ```
2. **Compile** using `nvcc` (or a custom build script). For instance:
   ```bash
   nvcc -o cuda_raytracer main.cu -O3 -arch=sm_60
   ```
   Adjust `-arch=sm_XX` to match your GPU architecture (e.g., `sm_75` for RTX 20-series, `sm_86` for RTX 30-series, etc.).
3. **Run** the executable:
   ```bash
   ./cuda_raytracer
   ```
   It will produce a PPM file (e.g., `image3.ppm`) containing the rendered scene.

---

## Usage

By default, the code:
- Uses an aspect ratio of **16:9**.
- Sets the image width to **1200** pixels (the height is derived).
- Uses **500** samples per pixel.
- Launches a CUDA kernel with a block size of **8×8** threads (you can adjust `tx` and `ty`).
- Writes the output to **`image3.ppm`**.

You can modify parameters in `main()`:
```cpp
int image_width = 1200;
int tx = 8;
int ty = 8;
int samples_per_pixel = 500;
```
- **`image_width`** and **`aspect_ratio`** control final resolution.  
- **`samples_per_pixel`** affects anti-aliasing / noise reduction.  
- **`tx`, `ty`** determine the CUDA block size (and thus concurrency).

---

## Scene Description

The default scene is generated in `create_world`:
- A large sphere forms the ground.
- Randomly placed small spheres with random materials (diffuse, metal, glass).
- Three bigger spheres with distinct materials:
  1. A glass sphere in the center.
  2. A lambertian sphere to the left.
  3. A metal sphere to the right.
- The camera is positioned at `(13, 2, 3)` looking at `(0, 0, 0)` with a field of view of **20°**.

You can edit these parameters in `create_world` if you’d like a different arrangement.

---

## Performance Notes

- **Block size**: `tx` and `ty` in `main()` control block dimensions. Try **32×32** or **16×16** on modern GPUs for potentially better performance.  
- **Number of samples**: `samples_per_pixel = 500` results in a smoother image but increases computation time. Reducing to 50–100 can speed up rendering at the cost of increased noise.  
- **Resolution**: The default 1200×675 can be changed by modifying `image_width` and `aspect_ratio`. Higher resolutions significantly increase render time.

---

## License

This code is provided for educational and demonstration purposes. You may adapt and reuse it freely. If you modify or distribute the code, maintaining a reference to this repository or the original authors is appreciated but not mandatory, unless specified by another license you impose.

---
