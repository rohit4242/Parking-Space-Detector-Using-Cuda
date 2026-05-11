#pragma once

// Launches grayscale conversion kernel.
// Input : bgr  - packed BGR image in unified memory (W*H*3 bytes)
// Output: gray - single-channel grayscale image (W*H bytes)
void launchGrayscale(unsigned char* bgr, unsigned char* gray,
                     int width, int height);

// Launches Sobel edge detection kernel.
// Input : gray  - single-channel grayscale image (W*H bytes)
// Output: edges - single-channel edge magnitude image (W*H bytes)
// Border pixels are written as 0. Magnitude = clamp(sqrt(Gx^2+Gy^2), 0, 255).
void launchSobel(unsigned char* gray, unsigned char* edges,
                 int width, int height);

// Launches effect filter kernel.
// Reads bgr (original color) and edges (Sobel output).
// Pixels whose edge magnitude > threshold are darkened 50% in bgrOut;
// all other pixels are copied unchanged.
// bgrOut must be W*H*3 bytes (same layout as bgr).
void launchEffect(unsigned char* bgr, unsigned char* edges,
                  unsigned char* bgrOut,
                  int width, int height, float threshold);
