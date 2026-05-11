#pragma once

void launchGrayscale(unsigned char* bgr, unsigned char* gray,
                     int width, int height);

void launchSobel(unsigned char* gray, unsigned char* edges,
                 int width, int height);

void launchEffect(unsigned char* bgr, unsigned char* edges,
                  unsigned char* bgrOut,
                  int width, int height, float threshold);

void launchROIVariance(unsigned char* gray, float* variances,
                       int* posX, int* posY, int numSpots,
                       int roiW, int roiH, int imgWidth);
