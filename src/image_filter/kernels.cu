#include "kernels.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: Grayscale conversion
// Each thread handles exactly one pixel — fully parallel across the image.
// ---------------------------------------------------------------------------
__global__ void grayscaleKernel(unsigned char* bgr, unsigned char* gray,
                                 int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx    = y * width + x;
    int bgrIdx = idx * 3;

    gray[idx] = (unsigned char)(
        0.0722f * (float)bgr[bgrIdx]     +  // B
        0.7152f * (float)bgr[bgrIdx + 1] +  // G
        0.2126f * (float)bgr[bgrIdx + 2]    // R
    );
}

// ---------------------------------------------------------------------------
// Kernel 2: Sobel edge detection
// Each thread handles exactly one pixel — fully parallel across the image.
//
// Convolves the grayscale image with:
//   Gx = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
//   Gy = [[-1,-2, -1], [ 0, 0,  0], [+1,+2, +1]]
//
// Output per pixel: clamp(sqrt(Gx^2 + Gy^2), 0, 255)
// Border pixels (1-pixel margin) are written as 0 — no out-of-bounds reads.
// ---------------------------------------------------------------------------
__global__ void sobelKernel(unsigned char* gray, unsigned char* edges,
                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Border pixels: no full 3x3 neighbourhood available — output 0.
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        edges[y * width + x] = 0;
        return;
    }

    // Fetch the 3x3 neighbourhood once.
    float p00 = (float)gray[(y-1)*width + (x-1)];
    float p01 = (float)gray[(y-1)*width +  x   ];
    float p02 = (float)gray[(y-1)*width + (x+1)];
    float p10 = (float)gray[ y   *width + (x-1)];
    float p12 = (float)gray[ y   *width + (x+1)];
    float p20 = (float)gray[(y+1)*width + (x-1)];
    float p21 = (float)gray[(y+1)*width +  x   ];
    float p22 = (float)gray[(y+1)*width + (x+1)];

    float gx = -p00 + p02
               - 2.0f*p10 + 2.0f*p12
               - p20 + p22;

    float gy = -p00 - 2.0f*p01 - p02
               + p20 + 2.0f*p21 + p22;

    float mag = sqrtf(gx*gx + gy*gy);
    edges[y * width + x] = (unsigned char)fminf(mag, 255.0f);
}

// ---------------------------------------------------------------------------
// Kernel 3: Effect filter
// Each thread handles exactly one pixel — fully parallel across the image.
//
// For pixels where Sobel edge magnitude > threshold:
//   all three BGR channels are darkened by 50%.
// All other pixels are copied unchanged from the original color frame.
// ---------------------------------------------------------------------------
__global__ void effectKernel(unsigned char* bgr, unsigned char* edges,
                              unsigned char* bgrOut,
                              int width, int height, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int bgrIdx  = (y * width + x) * 3;
    float mag   = (float)edges[y * width + x];

    if (mag > threshold) {
        bgrOut[bgrIdx    ] = (unsigned char)(bgr[bgrIdx    ] * 0.5f);
        bgrOut[bgrIdx + 1] = (unsigned char)(bgr[bgrIdx + 1] * 0.5f);
        bgrOut[bgrIdx + 2] = (unsigned char)(bgr[bgrIdx + 2] * 0.5f);
    } else {
        bgrOut[bgrIdx    ] = bgr[bgrIdx    ];
        bgrOut[bgrIdx + 1] = bgr[bgrIdx + 1];
        bgrOut[bgrIdx + 2] = bgr[bgrIdx + 2];
    }
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers
// ---------------------------------------------------------------------------

void launchGrayscale(unsigned char* bgr, unsigned char* gray,
                     int width, int height)
{
    dim3 block(32, 32);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    grayscaleKernel<<<grid, block>>>(bgr, gray, width, height);
}

void launchSobel(unsigned char* gray, unsigned char* edges,
                 int width, int height)
{
    dim3 block(32, 32);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    sobelKernel<<<grid, block>>>(gray, edges, width, height);
}

void launchEffect(unsigned char* bgr, unsigned char* edges,
                  unsigned char* bgrOut,
                  int width, int height, float threshold)
{
    dim3 block(32, 32);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    effectKernel<<<grid, block>>>(bgr, edges, bgrOut, width, height, threshold);
}
