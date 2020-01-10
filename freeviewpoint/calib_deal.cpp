#include "calib_deal.h"
#include <iostream>

CalibStruct m_CalibParams[8];
static int m_NumCameras = 8;

#define MaxZ 130.0
#define MinZ 42.0

int readCalibrationFile(char *fileName)
{
    int i, j, k;
    FILE *pIn;
    double dIn; // dummy variable
    int camIdx;

    if(pIn = fopen(fileName, "r"))
    {
        for(k=0; k<m_NumCameras; k++)
        {
            // camera index
            fscanf(pIn, "%d", &camIdx);

            // camera intrinsics
            for (i=0; i<3; i++)
                fscanf(pIn, "%lf\t%lf\t%lf", &(m_CalibParams[camIdx].m_K[i][0]),
                    &(m_CalibParams[camIdx].m_K[i][1]), &(m_CalibParams[camIdx].m_K[i][2]));

            // read barrel distortion params (assume 0)
            fscanf(pIn, "%lf", &dIn);
            fscanf(pIn, "%lf", &dIn);

            // read extrinsics
            for(i=0;i<3;i++)
            {
                for(j=0;j<3;j++)
                {
                    fscanf(pIn, "%lf", &dIn);
                    m_CalibParams[camIdx].m_RotMatrix[i][j] = dIn;
                }

                fscanf(pIn, "%lf", &dIn);
                m_CalibParams[camIdx].m_Trans[i] = dIn;
            }

        }

        fclose(pIn);
        return 1;
    }
    else return 0;
} // readCalibrationFile

void computeProjectionMatrices()
{
    int i, j, k, camIdx;
    double (*inMat)[3];
    double exMat[3][4];

    for(camIdx=0; camIdx<m_NumCameras; camIdx++)
    {
        // The intrinsic matrix
        inMat = m_CalibParams[camIdx].m_K;

        // The extrinsic matrix
        for(i=0;i<3;i++)
        {
            for(j=0;j<3;j++)
            {
                exMat[i][j] = m_CalibParams[camIdx].m_RotMatrix[i][j];
            }
        }

        for(i=0;i<3;i++)
        {
            exMat[i][3] = m_CalibParams[camIdx].m_Trans[i];
        }

        // Multiply the intrinsic matrix by the extrinsic matrix to find our projection matrix
        for(i=0;i<3;i++)
        {
            for(j=0;j<4;j++)
            {
                m_CalibParams[camIdx].m_ProjMatrix.at<double>(i,j) = 0.0;

                for(k=0;k<3;k++)
                {
                    m_CalibParams[camIdx].m_ProjMatrix.at<double>(i,j) += inMat[i][k]*exMat[k][j];
                }
            }
        }

        m_CalibParams[camIdx].m_ProjMatrix.at<double>(3,0) = 0.0;
        m_CalibParams[camIdx].m_ProjMatrix.at<double>(3,1) = 0.0;
        m_CalibParams[camIdx].m_ProjMatrix.at<double>(3,2) = 0.0;
        m_CalibParams[camIdx].m_ProjMatrix.at<double>(3,3) = 1.0;
    }
}

int InitCalibParams(char *fileName)
{
    for (int camIdx=0; camIdx<m_NumCameras; camIdx++)
    {
        m_CalibParams[camIdx].m_ProjMatrix.create( 4, 4, CV_64FC1);
    }
    int res=readCalibrationFile(fileName);
    if(res==1)
    {
        computeProjectionMatrices();
        return 1;
    }
    else return 0;
}

double DepthLevelToZ( unsigned char d )
{
    double z;
//    double MinZ = 44.0, MaxZ = 120.0;

    z = 1.0/((d/255.0)*(1.0/MinZ - 1.0/MaxZ) + 1.0/MaxZ);
    return z;
}

cv::Mat Depth2Zmap(cv::Mat DepthMap)
{
    cv::Mat Zmap(DepthMap.rows, DepthMap.cols, CV_64FC1);
    for(int j=0; j< Zmap.rows; j++)
    {
        double *rowData = Zmap.ptr<double>(j);
        for(int i=0; i< Zmap.cols; i++)
        {
//            data[i]=i;
//            std::cout<<"col: "<<j<<" row: "<<i<<std::endl;
            rowData[i]=DepthLevelToZ(DepthMap.at<uchar>(j,i));
        }
    }
    return Zmap;
}
