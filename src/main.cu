#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#include "kernels.cuh"

static constexpr float VARIANCE_THRESHOLD = 1200.0f;
static constexpr int ROI_W = 40;
static constexpr int ROI_H = 80;

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
    std::string positionsPath = (argc > 1) ? argv[1] : "data/sample";
    std::string videoPath     = (argc > 2) ? argv[2] : "data/sample.mp4";

    std::vector<int> hPosX, hPosY;
    {
        std::ifstream file(positionsPath);
        if (!file.is_open()) {
            std::cerr << "Cannot open positions file: " << positionsPath << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            int x, y; char comma;
            if (ss >> x >> comma >> y)  {
                hPosX.push_back(x);
                hPosY.push_back(y);
            }
        }
    }

    int numSpots = (int)hPosX.size();
    if (numSpots == 0) {
        std::cerr << "No parking spots loaded from " << positionsPath << "\n";
        return 1;
    }
    std::cerr << "[INFO] Loaded " << numSpots << " parking spots.\n";

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << videoPath << "\n";
        return 1;
    }

    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cerr << "[INFO] Video: " << W << "x" << H << "\n";

    unsigned char* d_bgr;
    unsigned char* d_gray;
    unsigned char* d_edges;
    unsigned char* d_bgrOut;
    int*           d_posX;
    int*           d_posY;
    float*         d_variances;

    checkCuda(cudaMallocManaged(&d_bgr,       W * H * 3 * sizeof(unsigned char)), "d_bgr");
    checkCuda(cudaMallocManaged(&d_gray,      W * H     * sizeof(unsigned char)), "d_gray");
    checkCuda(cudaMallocManaged(&d_edges,     W * H     * sizeof(unsigned char)), "d_edges");
    checkCuda(cudaMallocManaged(&d_bgrOut,   W * H * 3 * sizeof(unsigned char)), "d_bgrOut");
    checkCuda(cudaMallocManaged(&d_posX,      numSpots  * sizeof(int)),           "d_posX");
    checkCuda(cudaMallocManaged(&d_posY,      numSpots  * sizeof(int)),           "d_posY");
    checkCuda(cudaMallocManaged(&d_variances, numSpots  * sizeof(float)),         "d_variances");

    std::copy(hPosX.begin(), hPosX.end(), d_posX);
    std::copy(hPosY.begin(), hPosY.end(), d_posY);

    cv::Mat frame;

    cv::namedWindow("Parking Lot",    cv::WINDOW_NORMAL);
    cv::namedWindow("Grayscale",      cv::WINDOW_NORMAL);
    cv::namedWindow("Edge Detection", cv::WINDOW_NORMAL);
    cv::namedWindow("Effect",         cv::WINDOW_NORMAL);

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        std::memcpy(d_bgr, frame.data, W * H * 3);

        // GPU: convert to grayscale
        launchGrayscale(d_bgr, d_gray, W, H);

        // GPU: Sobel edge detection on the grayscale image
        launchSobel(d_gray, d_edges, W, H);

        // GPU: effect filter — darken color pixels where edges are strong
        launchEffect(d_bgr, d_edges, d_bgrOut, W, H, 90.0f);

        // GPU: compute variance for every parking spot simultaneously
        launchROIVariance(d_gray, d_variances, d_posX, d_posY,
                          numSpots, ROI_W, ROI_H, W);

        // Wait for GPU to finish before reading results on CPU
        checkCuda(cudaDeviceSynchronize(), "sync");

        int freeCount = 0;
        for (int i = 0; i < numSpots; i++) {
            bool isFree   = d_variances[i] < VARIANCE_THRESHOLD;
            if (isFree) freeCount++;

            cv::Scalar color  = isFree ? cv::Scalar(0, 255, 0)   // green = Free
                                       : cv::Scalar(0, 0, 255);   // red   = Occupied
            int thickness     = isFree ? 2 : 1;

            cv::Point topLeft (d_posX[i],          d_posY[i]);
            cv::Point botRight(d_posX[i] + ROI_W,  d_posY[i] + ROI_H);
            cv::rectangle(frame, topLeft, botRight, color, thickness);

            cv::putText(frame,
                        std::to_string((int)d_variances[i]),
                        cv::Point(d_posX[i] + 2, d_posY[i] + 14),
                        cv::FONT_HERSHEY_PLAIN, 0.9, color, 1);
        }

        cv::putText(frame,
                    "Free: " + std::to_string(freeCount) + "/" + std::to_string(numSpots),
                    cv::Point(10, 50),
                    cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 200, 0), 3);

        cv::Mat grayDisplay(H, W, CV_8UC1, d_gray);
        cv::Mat edgeDisplay(H, W, CV_8UC1, d_edges);
        cv::Mat effectDisplay(H, W, CV_8UC3, d_bgrOut);
        cv::imshow("Parking Lot",    frame);
        cv::imshow("Grayscale",      grayDisplay);
        cv::imshow("Edge Detection", edgeDisplay);
        cv::imshow("Effect",         effectDisplay);

        int key = cv::waitKey(200) & 0xFF;
        if (key == 'q') break;
        if (key == 's') {
            cv::imwrite("output_parking.jpg", frame);
            std::cerr << "[INFO] Screenshot saved as output_parking.jpg\n";
        }
    }

    cv::destroyAllWindows();

    cudaFree(d_bgr);
    cudaFree(d_gray);
    cudaFree(d_edges);
    cudaFree(d_bgrOut);
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_variances);
    cap.release();

    return 0;
}
