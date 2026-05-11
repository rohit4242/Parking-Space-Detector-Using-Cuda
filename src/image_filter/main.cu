#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#include "kernels.cuh"

static void checkCuda(cudaError_t err, const char* where)
{
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << where << ": "
                  << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image.png|jpg|...>\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // Load image
    // ------------------------------------------------------------------
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << argv[1] << "\n";
        return 1;
    }
    if (!img.isContinuous())
        img = img.clone();

    const int W = img.cols;
    const int H = img.rows;
    const int N = W * H;
    std::cerr << "[INFO] Image: " << W << "x" << H << "\n";

    // ------------------------------------------------------------------
    // Allocate Unified Memory — accessible by both CPU and GPU kernels.
    // ------------------------------------------------------------------
    unsigned char* d_bgr;
    unsigned char* d_gray;
    unsigned char* d_edges;
    unsigned char* d_bgrOut;

    checkCuda(cudaMallocManaged(&d_bgr,    N * 3 * sizeof(unsigned char)), "d_bgr");
    checkCuda(cudaMallocManaged(&d_gray,   N     * sizeof(unsigned char)), "d_gray");
    checkCuda(cudaMallocManaged(&d_edges,  N     * sizeof(unsigned char)), "d_edges");
    checkCuda(cudaMallocManaged(&d_bgrOut, N * 3 * sizeof(unsigned char)), "d_bgrOut");

    // Copy image pixels into unified memory
    std::memcpy(d_bgr, img.data, N * 3 * sizeof(unsigned char));

    // ------------------------------------------------------------------
    // Run GPU filters
    // ------------------------------------------------------------------
    launchGrayscale(d_bgr, d_gray,           W, H);
    launchSobel    (d_gray, d_edges,          W, H);
    launchEffect   (d_bgr, d_edges, d_bgrOut, W, H, 90.0f);

    checkCuda(cudaDeviceSynchronize(), "sync");

    // ------------------------------------------------------------------
    // Show all four windows (zero-copy — cv::Mat wraps unified memory)
    // ------------------------------------------------------------------
    cv::Mat viewOriginal(H, W, CV_8UC1, d_bgr);   // treat BGR as raw for display
    cv::Mat viewGray    (H, W, CV_8UC1, d_gray);
    cv::Mat viewEdge    (H, W, CV_8UC1, d_edges);
    cv::Mat viewEffect  (H, W, CV_8UC3, d_bgrOut);

    cv::namedWindow("Original",       cv::WINDOW_NORMAL);
    cv::namedWindow("Grayscale",      cv::WINDOW_NORMAL);
    cv::namedWindow("Edge Detection", cv::WINDOW_NORMAL);
    cv::namedWindow("Effect",         cv::WINDOW_NORMAL);

    cv::imshow("Original",       img);        // show original in color
    cv::imshow("Grayscale",      viewGray);
    cv::imshow("Edge Detection", viewEdge);
    cv::imshow("Effect",         viewEffect);

    std::cerr << "[INFO] Press any key to exit.\n";
    cv::waitKey(0);

    cv::destroyAllWindows();

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    cudaFree(d_bgr);
    cudaFree(d_gray);
    cudaFree(d_edges);
    cudaFree(d_bgrOut);

    return 0;
}
