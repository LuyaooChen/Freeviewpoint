#ifndef CALIB_DEAL_H
#define CALIB_DEAL_H

#include <opencv2/opencv.hpp>


typedef struct {
    double m_K[3][3]; // 3x3 intrinsic matrix
    double m_RotMatrix[3][3]; // rotation matrix
    double m_Trans[3]; // translation vector

    cv::Mat m_ProjMatrix; // projection matrix
} CalibStruct;

extern CalibStruct m_CalibParams[8];
int InitCalibParams(char *fileName);
cv::Mat Depth2Zmap(cv::Mat DepthMap);

#endif // CALIB_DEAL_H
