#include "kernels.cuh"

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

__global__ void sobelKernel(unsigned char* gray, unsigned char* edges,
                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        edges[y * width + x] = 0;
        return;
    }

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

__global__ void roiVarianceKernel(unsigned char* gray, float* variances,
                                   int* posX, int* posY,
                                   int roiW, int roiH, int imgWidth)
{
    __shared__ float s_red[256];

    int spotIdx = blockIdx.x;
    int x0      = posX[spotIdx];
    int y0      = posY[spotIdx];
    int roiSize = roiW * roiH;

    float mySum = 0.0f;
    for (int i = threadIdx.x; i < roiSize; i += blockDim.x) {
        int px = x0 + (i % roiW);
        int py = y0 + (i / roiW);
        mySum += (float)gray[py * imgWidth + px];
    }

    s_red[threadIdx.x] = mySum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    
    float totalSum = s_red[0];
    float mean     = totalSum / (float)roiSize;

    float myVar = 0.0f;
    for (int i = threadIdx.x; i < roiSize; i += blockDim.x) {
        int px = x0 + (i % roiW);
        int py = y0 + (i / roiW);
        float diff = (float)gray[py * imgWidth + px] - mean;
        myVar += diff * diff;
    }

    s_red[threadIdx.x] = myVar;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        variances[spotIdx] = s_red[0] / (float)roiSize;
}

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

void launchROIVariance(unsigned char* gray, float* variances,
                       int* posX, int* posY, int numSpots,
                       int roiW, int roiH, int imgWidth)
{
   
    int threadsPerBlock = 256;
    roiVarianceKernel<<<numSpots, threadsPerBlock>>>(
        gray, variances, posX, posY, roiW, roiH, imgWidth
    );
}
