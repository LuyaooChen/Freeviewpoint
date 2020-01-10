#include "kernel.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

using namespace cv;

#define TIMETEST 1
#define LOGWRITE 1
#if LOGWRITE
#include <fstream>
#endif

#define TPB1 32  //threads per block
#define TPB2 32  //threads per block

#if BREAKDANCER
#define MaxZ 120.0  //breakdancers
#define MinZ 44.0
#elif BALLET
#define MaxZ 130.0    //ballet
#define MinZ 42.0
#elif KENDO
#define MaxZ 11206.280350
#define MinZ 448.251214
#elif LOVEBIRD
#define MaxZ 156012.206895
#define MinZ 1418.292789
#elif BOOKARRIVAL
#define MaxZ 54.2
#define MinZ 23.2
#elif NEWSPAPER
#define MaxZ 9050.605493
#define MinZ 2715.181648
#elif POZNANSTREET
#define MaxZ 2760.510889
#define MinZ 34.506386
#elif POZNANHALL2
#define MaxZ 172.531931
#define MinZ 23.394160
#endif

#define GRAYLEVEL 255 //72

double Median(double n1, double n2, double n3, double n4, double n5,
    double n6, double n7, double n8,double n9) {
    double arr[9];
    double temp;
    arr[0] = n1;
    arr[1] = n2;
    arr[2] = n3;
    arr[3] = n4;
    arr[4] = n5;
    arr[5] = n6;
    arr[6] = n7;
    arr[7] = n8;
    arr[8] = n9;
    for (int gap = 9 / 2; gap > 0; gap /= 2)//希尔排序
        for (int i = gap; i < 9; ++i)
            for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
            {
//                swap(arr[j], arr[j + gap]);
                temp = arr[j];
                arr[j] = arr[j+gap];
                arr[j+gap] = temp;
            }
    return arr[4];//返回中值
}

int Median(int n1, int n2, int n3, int n4, int n5,
    int n6, int n7, int n8,int n9) {
    int arr[9];
    int temp;
    arr[0] = n1;
    arr[1] = n2;
    arr[2] = n3;
    arr[3] = n4;
    arr[4] = n5;
    arr[5] = n6;
    arr[6] = n7;
    arr[7] = n8;
    arr[8] = n9;
    for (int gap = 9 / 2; gap > 0; gap /= 2)//希尔排序
        for (int i = gap; i < 9; ++i)
            for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
            {
//                swap(arr[j], arr[j + gap]);
                temp = arr[j];
                arr[j] = arr[j+gap];
                arr[j+gap] = temp;
            }
    return arr[4];//返回中值
}

void projUVZtoXY(Mat projMatrix, double u, double v, double z, double *x, double *y, unsigned int rows)
{
    double c0, c1, c2;
    v = rows - v - 1.0;

    c0 = z*projMatrix.at<double>(0,2) + projMatrix.at<double>(0,3);
    c1 = z*projMatrix.at<double>(1,2) + projMatrix.at<double>(1,3);
    c2 = z*projMatrix.at<double>(2,2) + projMatrix.at<double>(2,3);

    *y = u*(c1*projMatrix.at<double>(2,0) - projMatrix.at<double>(1,0)*c2) +
            v*(c2*projMatrix.at<double>(0,0) - projMatrix.at<double>(2,0)*c0) +
            projMatrix.at<double>(1,0)*c0 - c1*projMatrix.at<double>(0,0);

    *y /= v*(projMatrix.at<double>(2,0)*projMatrix.at<double>(0,1) - projMatrix.at<double>(2,1)*projMatrix.at<double>(0,0)) +
        u*(projMatrix.at<double>(1,0)*projMatrix.at<double>(2,1) - projMatrix.at<double>(1,1)*projMatrix.at<double>(2,0)) +
        projMatrix.at<double>(0,0)*projMatrix.at<double>(1,1) - projMatrix.at<double>(1,0)*projMatrix.at<double>(0,1);

    *x = (*y)*(projMatrix.at<double>(0,1) - projMatrix.at<double>(2,1)*u) + c0 - c2*u;
    *x /= projMatrix.at<double>(2,0)*u - projMatrix.at<double>(0,0);
}

double projXYZtoUV(Mat projMatrix, double x, double y, double z, double *u, double *v, unsigned int rows)
{
    double w;

    *u = projMatrix.at<double>(0,0)*x +
         projMatrix.at<double>(0,1)*y +
         projMatrix.at<double>(0,2)*z +
         projMatrix.at<double>(0,3);

    *v = projMatrix.at<double>(1,0)*x +
         projMatrix.at<double>(1,1)*y +
         projMatrix.at<double>(1,2)*z +
         projMatrix.at<double>(1,3);

    w = projMatrix.at<double>(2,0)*x +
        projMatrix.at<double>(2,1)*y +
        projMatrix.at<double>(2,2)*z +
        projMatrix.at<double>(2,3);

    *u /= w;
    *v /= w;

    // image (0,0) is bottom lefthand corner
    *v = rows - *v - 1.0;

    return w;
} // projXYZtoUV

void ForwardWarping(Mat srcColor,
                    Mat srcDepth,
                    Mat dstColor,
                    Mat dstDepth,
                    Mat srcProjMat,
                    Mat dstProjMat,
                    unsigned int rows,
                    unsigned int columns)
{
    for(int i=0; i<columns; i++)
        for(int j=0; j<rows; j++)
        {
            double x, y, dstU=0, dstV=0;
    //        double z = 1.0/((srcDepth(j,i)/255.0)*(1.0/MinZ - 1.0/MaxZ) + 1.0/MaxZ);
            double z = MaxZ + srcDepth.at<uchar>(j,i) * (MinZ-MaxZ)/255.0;
    //        srcDepth(j, i) = z;

            projUVZtoXY(srcProjMat, i, j, z, &x, &y, rows);
            projXYZtoUV(dstProjMat, x, y, z, &dstU, &dstV, rows);

            if(0<=dstU && dstU< columns && 0<=dstV && dstV< rows)
            {

                if(dstDepth.at<double>((int)dstV, (int)dstU)==0.0 || z < dstDepth.at<double>((int)dstV, (int)dstU))
                {
                        dstDepth.at<double>((int)dstV, (int)dstU) = z;
#if DEBUG_MODE
    //                    dstColor((int)dstV, (int)dstU) = srcColor(j, i);
#endif
                }
            }

        }
}

void ForwardWarping(Mat srcColor,
                    Mat srcDepth,
                    Mat dstColor,
                    Mat dstDepth,
                    Mat srcProjMat,
                    Mat dstProjMat,
                    Mat edge_hole,
                    Mat dst_edge,
                    unsigned int rows,
                    unsigned int columns)
{

    for(int i=0; i<columns; i++)
        for(int j=0; j<rows; j++)
        {
            double x, y, dstU=0, dstV=0;
            double z = 1.0/((srcDepth.at<uchar>(j,i)/255.0)*(1.0/MinZ - 1.0/MaxZ) + 1.0/MaxZ);
//            double z = MaxZ + srcDepth.at<uchar>(j,i) * (MinZ-MaxZ)/255.0;

            projUVZtoXY(srcProjMat, i, j, z, &x, &y, rows);
            projXYZtoUV(dstProjMat, x, y, z, &dstU, &dstV, rows);

            if(0<=dstU && dstU< columns && 0<=dstV && dstV< rows)
            {
                if(edge_hole.at<uchar>(j,i)==255)
                {
    //                        edge_hole(j,i)= 0;
                    dst_edge.at<uchar>((int)dstV, (int)dstU)= 255;
                }

                if(dstDepth.at<double>((int)dstV, (int)dstU)==0.0 || z < dstDepth.at<double>((int)dstV, (int)dstU))
                {
                        dstDepth.at<double>((int)dstV, (int)dstU) = z;
#if DEBUG_MODE
    //                    dstColor((int)dstV, (int)dstU) = srcColor(j, i);
#endif
                }
            }

        }
}

void InverseWarping(Mat dstColor,
                    Mat dstDepth,
                    Mat srcColor,
                    Mat holeImg,
                    Mat srcProjMat,
                    Mat dstProjMat,
                    unsigned int rows,
                    unsigned int columns)
{
    for(int i=0; i<columns; i++)
        for(int j=0; j<rows; j++)
        {
            double x, y;
            double srcU = 0.0, srcV = 0.0;
            double z = dstDepth.at<double>(j, i);
            projUVZtoXY(dstProjMat, (double)i, (double)j, z, &x, &y, rows);
            projXYZtoUV(srcProjMat, x, y, z, &srcU, &srcV, rows);

            if(0<=srcU && srcU<columns && 0<=srcV && srcV<rows && z!=0)
            {
                dstColor.at<Vec3b>(j,i)=srcColor.at<Vec3b>((int)srcV, (int)srcU);
                holeImg.at<uchar>(j,i) = 255;
            }
        }
}

void calValueRatio(Mat srcColorL,
                   Mat holeImgL,
                   Mat srcColorR,
                   Mat holeImgR,
                   double *ratioV,
                   int *cntV)
{
    for(int i=0; i<srcColorL.cols; i++)
        for(int j=0; j<srcColorL.rows; j++)
        {
            //todo:
            //bgr to hsv
            //calculate average V
            //get the ratio
            if(holeImgL.at<uchar>(j, i)==255 && holeImgR.at<uchar>(j, i)==255)
            {
                double BGR_L[3];
                double BGR_R[3];
                double CmaxL, CmaxR;
                BGR_L[0] = srcColorL.at<Vec3b>(j, i)[0] / 255.0;
                BGR_L[1] = srcColorL.at<Vec3b>(j, i)[1] / 255.0;
                BGR_L[2] = srcColorL.at<Vec3b>(j, i)[2] / 255.0;

                CmaxL = (BGR_L[0] > BGR_L[1]) ? BGR_L[0] : BGR_L[1];
                CmaxL = (CmaxL > BGR_L[2]) ? CmaxL : BGR_L[2];

                BGR_R[0] = srcColorR.at<Vec3b>(j, i)[0] / 255.0;
                BGR_R[1] = srcColorR.at<Vec3b>(j, i)[1] / 255.0;
                BGR_R[2] = srcColorR.at<Vec3b>(j, i)[2] / 255.0;

                CmaxR = (BGR_R[0] > BGR_R[1]) ? BGR_R[0] : BGR_R[1];
                CmaxR = (CmaxR > BGR_R[2]) ? CmaxR : BGR_R[2];

                if(CmaxR<0.00001) *ratioV += 1;
                else *ratioV += CmaxL/CmaxR;
                *cntV += 1;
            }
        }
}

void adjValue(Vec3b &srcPix, double ratio)  //srcPix is in right  (ratio= left/right)
{
    double HSV_t[3];
    double BGR_t[3];
    double Cmax, Cmin, delta;
    BGR_t[0] = srcPix[0] / 255.0;
    BGR_t[1] = srcPix[1] / 255.0;
    BGR_t[2] = srcPix[2] / 255.0;

    Cmax = (BGR_t[0] > BGR_t[1]) ? BGR_t[0] : BGR_t[1];
    Cmax = (Cmax > BGR_t[2]) ? Cmax : BGR_t[2];
    Cmin = (BGR_t[0] < BGR_t[1]) ? BGR_t[0] : BGR_t[1];
    Cmin = (Cmin < BGR_t[2]) ? Cmin : BGR_t[2];
    delta = Cmax - Cmin;

    //H
    if(delta == 0) HSV_t[0] = 0;
    else if(BGR_t[0] == Cmax) HSV_t[0] = (BGR_t[2]-BGR_t[1]) / delta + 4;  //max == B
    else if(BGR_t[1] == Cmax) HSV_t[0] = (BGR_t[0]-BGR_t[2]) / delta + 2;  //max == G
    else if(BGR_t[2] == Cmax) HSV_t[0] = (BGR_t[1]-BGR_t[0]) / delta;  //max == R
    if(HSV_t[0]<0) HSV_t[0]+=6;
    //  HSV_t[0] *=60;
    //S
    if(Cmax == 0) HSV_t[1] = 0;
    else HSV_t[1] = delta/Cmax;
    //V
    HSV_t[2] = Cmax*ratio*255;   //V after adjustion

    if(HSV_t[2]>255) HSV_t[2]=255;

//    printf("H:%f S:%f V:%f\t",HSV_t[0],HSV_t[1],HSV_t[2]);

    //HSV to BGR
    if(HSV_t[1] < 0.000001)
    {
        srcPix = Vec3b(HSV_t[2],HSV_t[2],HSV_t[2]);
        return;
    }
    //  HSV_t[0] /=60;
    double f, a, b, c;
    int i = (int) HSV_t[0];
    f = HSV_t[0] - i;
    a = HSV_t[2] * ( 1 - HSV_t[1] );
    b = HSV_t[2] * ( 1 - HSV_t[1] * f );
    c = HSV_t[2] * ( 1 - HSV_t[1] * (1 - f ) );

    switch(i)
    {
        case 0: srcPix[2] = HSV_t[2]; srcPix[1] = c       ; srcPix[0] = a;  break;
        case 1: srcPix[2] = b       ; srcPix[1] = HSV_t[2]; srcPix[0] = a;  break;
        case 2: srcPix[2] = a       ; srcPix[1] = HSV_t[2]; srcPix[0] = c;  break;
        case 3: srcPix[2] = a       ; srcPix[1] = b       ; srcPix[0] = HSV_t[2];   break;
        case 4: srcPix[2] = c       ; srcPix[1] = a       ; srcPix[0] = HSV_t[2];   break;
        case 5: srcPix[2] = HSV_t[2]; srcPix[1] = a       ; srcPix[0] = b;  break;
    }
}

void ImageBlending( Mat srcColorL,
                    Mat srcColorR,
                    Mat holeImgL,
                    Mat holeImgR,
                    Mat holeImg,
                    Mat dstColor,
                    float alpha,
                    double ratioV,
                    unsigned int rows,
                    unsigned int columns)
{
    for(int i=0; i<columns; i++)
        for(int j=0; j<rows; j++)
        {
            if(holeImgL.at<uchar>(j,i)==255 && holeImgR.at<uchar>(j,i)==255)
            {
                //bgr to hsv
                //adjust V
                //hsv to bgr
//#if !BALLET
                adjValue(srcColorR.at<Vec3b>(j, i), ratioV);
//#endif
                dstColor.at<Vec3b>(j,i)[0] = srcColorL.at<Vec3b>(j,i)[0] * alpha + srcColorR.at<Vec3b>(j,i)[0] * (1-alpha);
                dstColor.at<Vec3b>(j,i)[1] = srcColorL.at<Vec3b>(j,i)[1] * alpha + srcColorR.at<Vec3b>(j,i)[1] * (1-alpha);
                dstColor.at<Vec3b>(j,i)[2] = srcColorL.at<Vec3b>(j,i)[2] * alpha + srcColorR.at<Vec3b>(j,i)[2] * (1-alpha);
            }
            else if(holeImgL.at<uchar>(j,i)==0 && holeImgR.at<uchar>(j,i)==255)
            {
//#if !BALLET
                adjValue(srcColorR.at<Vec3b>(j, i), ratioV);
//#endif
                dstColor.at<Vec3b>(j,i) = srcColorR.at<Vec3b>(j,i);
                srcColorL.at<Vec3b>(j,i) = Vec3b(0,0,0);
            }
            else if(holeImgL.at<uchar>(j,i)==255 && holeImgR.at<uchar>(j,i)==0)
            {
                dstColor.at<Vec3b>(j,i) = srcColorL.at<Vec3b>(j,i);
                srcColorR.at<Vec3b>(j,i) = Vec3b(0,0,0);
            }
            else
            {
                dstColor.at<Vec3b>(j,i) = Vec3b(0,0,0);     //clean the previous map
                srcColorL.at<Vec3b>(j,i) = Vec3b(0,0,0);
                srcColorR.at<Vec3b>(j,i) = Vec3b(0,0,0);
                holeImg.at<uchar>(j,i) = 0;
            }
        }
}

void MedianFilter(Mat srcDepth,
                  Mat dstDepth,
                  unsigned int rows,
                  unsigned int columns)
{

    for(int i=0; i<columns; i++)
        for(int j=0; j<rows; j++)
        {
            if(i>0 && i<columns-1 && j>0 && j<rows-1)
            {
    //            if(dstDepth(j,i)==0)
                    dstDepth.at<double>(j,i)=Median(srcDepth.at<double>(j-1,i-1),srcDepth.at<double>(j,i-1),srcDepth.at<double>(j+1,i-1),
                                         srcDepth.at<double>(j-1,i),srcDepth.at<double>(j,i),srcDepth.at<double>(j+1,i),
                                         srcDepth.at<double>(j-1,i+1),srcDepth.at<double>(j,i+1),srcDepth.at<double>(j+1,i+1));
            }
            else
            {
                dstDepth.at<double>(j,i)=srcDepth.at<double>(j,i);
            }
        }
}

//this is for binary img

int m_abs(int a, int b)
{
    int c = a - b;
    if(c>=0) return c;
    else return -c;
}


//just for compare
void m_erode2(Mat holeImg,
              Mat edge_holeImg,
              Mat dealed_edge)
{
    for(int i=1; i<holeImg.cols-1; i++)
        for(int j=1; j<holeImg.rows-1; j++)
        {
            if(holeImg.at<uchar>(j, i)==255)
            {
                if(holeImg.at<uchar>(j-1, i-1)== 0 ||             //is hole edge(value 255)
                    holeImg.at<uchar>(j-1, i  )== 0 ||
                    holeImg.at<uchar>(j-1, i+1)== 0 ||
                    holeImg.at<uchar>(j  , i-1)== 0 ||
                    holeImg.at<uchar>(j  , i+1)== 0 ||
                    holeImg.at<uchar>(j+1, i-1)== 0 ||
                    holeImg.at<uchar>(j+1, i  )== 0 ||
                    holeImg.at<uchar>(j+1, i+1)== 0 )
                {
                    if(edge_holeImg.at<uchar>(j-1, i-1)== 255 ||    //and is foreground edge
                        edge_holeImg.at<uchar>(j-1, i  )== 255 ||
                        edge_holeImg.at<uchar>(j-1, i+1)== 255 ||
                        edge_holeImg.at<uchar>(j  , i-1)== 255 ||
                        edge_holeImg.at<uchar>(j  , i+1)== 255 ||
                        edge_holeImg.at<uchar>(j+1, i-1)== 255 ||
                        edge_holeImg.at<uchar>(j+1, i  )== 255 ||
                        edge_holeImg.at<uchar>(j+1, i+1)== 255 )

                    {
                        for(int m=j-1; m<=j+1; m++)
                            for(int n=i-1; n<=i+1; n++)
                                dealed_edge.at<uchar>(m, n)=255;
                    }
                }
            }
        }
        for(int i=1; i<holeImg.cols-1; i++)
            for(int j=1; j<holeImg.rows-1; j++)
            {
                if(dealed_edge.at<uchar>(j, i)==255) holeImg.at<uchar>(j, i) = 0;
            }
}

typedef struct
{
    double depth;
    int distance;
    Vec3b rgb;
}samplePoint;


void Inpaint(Mat dstColor,          //there still are some bugs...    depth wrong?
             Mat VirDepthL,
             Mat VirDepthR,
             Mat holeImg)
{
    for(int i=0; i<dstColor.cols; i++)
        for(int j=0; j<dstColor.rows; j++)
        {
            int n1, n2, n3, n4;
            double d1=0, d2=0, d3=0, d4=0;
            if(holeImg.at<uchar>(j,i)==0)
            {
                for(n1=i; n1>=0; n1--)
                {
                    if(holeImg.at<uchar>(j,n1)==255) break;
                }
                for(n2=i; n2<dstColor.cols; n2++)
                {
                    if(holeImg.at<uchar>(j,n2)==255) break;
                }

                Vec3b temp;
                for(n3=j; n3>=0; n3--)
                {
                    if(holeImg.at<uchar>(n3,i)==255) break;
                }
                for(n4=j; n4<dstColor.rows; n4++)
                {
                    if(holeImg.at<uchar>(n4,i)==255) break;
                }

                if(n1<2) d1=0;
                else if(VirDepthL.at<double>(j, n1)!=0)    d1 = VirDepthL.at<double>(j, n1);
                else if(VirDepthR.at<double>(j, n1)!=0)    d1 = VirDepthR.at<double>(j, n1);
                if(n2>=dstColor.cols) d2=0;
                else if(VirDepthL.at<double>(j, n2)!=0)    d2 = VirDepthL.at<double>(j, n2);
                else if(VirDepthR.at<double>(j, n2)!=0)    d2 = VirDepthR.at<double>(j, n2);

                if(n3<2) d3=0;
                else if(VirDepthL.at<double>(n3, i)!=0)    d3 = VirDepthL.at<double>(n3, i);
                else if(VirDepthR.at<double>(n3, i)!=0)    d3 = VirDepthR.at<double>(n3, i);
                if(n4>=dstColor.rows) d4=0;
                else if(VirDepthL.at<double>(n4, i)!=0)    d4 = VirDepthL.at<double>(n4, i);
                else if(VirDepthR.at<double>(n4, i)!=0)    d4 = VirDepthR.at<double>(n4, i);

    #define THRESHOLD 10
                if(d2-d1>THRESHOLD)
                {
                    d1=0;
                    if(d4-d3>THRESHOLD)
                    {
                        d3=0;
                        if(d4-d2>THRESHOLD) d2=0;
                        else if(d2-d4>THRESHOLD) d4=0;
                    }
                    else if(d3-d4>THRESHOLD)
                    {
                        d4=0;
                        if(d3-d2>THRESHOLD) d2=0;
                        else if(d2-d3>THRESHOLD) d3=0;
                    }
                }
                else if(d1-d2>THRESHOLD)
                {
                    d2=0;
                    if(d4-d3>THRESHOLD)
                    {
                        d3=0;
                        if(d4-d1>THRESHOLD) d1=0;
                        else if(d1-d4>THRESHOLD) d4=0;
                    }
                    else if(d3-d4>THRESHOLD)
                    {
                        d4=0;
                        if(d3-d1>THRESHOLD) d1=0;
                        else if(d1-d3>THRESHOLD) d3=0;
                    }
                }
                else    //...... is incomplete ......
                {
                    if(d4-d3>THRESHOLD)
                    {
                        d3=0;
                        if(d4-d1>THRESHOLD || d4-d2>THRESHOLD)
                        {
                            d1=0;
                            d2=0;
                        }
                        else if(d1-d4>THRESHOLD || d2-d4>THRESHOLD) d4=0;
                    }
                    else if(d3-d4>THRESHOLD)
                    {
                        d4=0;
                        if(d3-d1>THRESHOLD || d3-d2>THRESHOLD)
                        {
                            d1=0;
                            d2=0;
                        }
                        else if(d1-d3>THRESHOLD || d2-d3>THRESHOLD) d3=0;
                    }
                }

                if(d1==0 && d2==0) dstColor.at<Vec3b>(j,i) = Vec3b(0,0,0);
                else if(d1==0) dstColor.at<Vec3b>(j,i) = dstColor.at<Vec3b>(j,n2);
                else if(d2==0) dstColor.at<Vec3b>(j,i) = dstColor.at<Vec3b>(j,n1);
                else
                {
                    dstColor.at<Vec3b>(j,i)[0] = ((n2-i)*dstColor.at<Vec3b>(j, n1)[0] + (i-n1)*dstColor.at<Vec3b>(j, n2)[0])/(n2-n1);
                    dstColor.at<Vec3b>(j,i)[1] = ((n2-i)*dstColor.at<Vec3b>(j, n1)[1] + (i-n1)*dstColor.at<Vec3b>(j, n2)[1])/(n2-n1);
                    dstColor.at<Vec3b>(j,i)[2] = ((n2-i)*dstColor.at<Vec3b>(j, n1)[2] + (i-n1)*dstColor.at<Vec3b>(j, n2)[2])/(n2-n1);
                }

                if(d3!=0 || d4!=0)
                {
                    if(d3==0) temp = dstColor.at<Vec3b>(n4,i);
                    else if(d4==0) temp = dstColor.at<Vec3b>(n3,i);
                    else
                    {
                        temp[0] = ((n4-j)*dstColor.at<Vec3b>(n3, i)[0] + (j-n3)*dstColor.at<Vec3b>(n4, i)[0])/(n4-n3);
                        temp[1] = ((n4-j)*dstColor.at<Vec3b>(n3, i)[1] + (j-n3)*dstColor.at<Vec3b>(n4, i)[1])/(n4-n3);
                        temp[2] = ((n4-j)*dstColor.at<Vec3b>(n3, i)[2] + (j-n3)*dstColor.at<Vec3b>(n4, i)[2])/(n4-n3);
                    }

                    if(d1==0 && d2==0)
                    {
                        dstColor.at<Vec3b>(j,i) = temp;
                    }
                    else
                    {
                        dstColor.at<Vec3b>(j,i)[0] = (dstColor.at<Vec3b>(j,i)[0] + temp[0])/2;
                        dstColor.at<Vec3b>(j,i)[1] = (dstColor.at<Vec3b>(j,i)[1] + temp[1])/2;
                        dstColor.at<Vec3b>(j,i)[2] = (dstColor.at<Vec3b>(j,i)[2] + temp[2])/2;
                    }
                }
            }
        }
}

/*
__global__
void Inpaint(cuda::PtrStepSz<uchar3> dstColor,
             cuda::PtrStep<uchar> holeImg)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i<dstColor.cols && j<dstColor.rows)
    {
        int n1, n2;
        if(holeImg(j,i)==0)
        {
            for(n1=i; n1>=0; n1--)
            {
                if(holeImg(j,n1)==255) break;
            }
            for(n2=i; n2<dstColor.cols; n2++)
            {
                if(holeImg(j,n2)==255) break;
            }

            if(n1<0) dstColor(j,i) = dstColor(j,n2);
            else if(n2>=dstColor.cols) dstColor(j,i) = dstColor(j,n1);
            else if(n1>=0 && n2<dstColor.cols)
            {
                dstColor(j,i).x = ((n2-i)*dstColor(j, n1).x + (i-n1)*dstColor(j, n2).x)/(n2-n1);
                dstColor(j,i).y = ((n2-i)*dstColor(j, n1).y + (i-n1)*dstColor(j, n2).y)/(n2-n1);
                dstColor(j,i).z = ((n2-i)*dstColor(j, n1).z + (i-n1)*dstColor(j, n2).z)/(n2-n1);
            }
            uchar3 temp;
            for(n1=j; n1>=0; n1--)
            {
                if(holeImg(n1,i)==255) break;
            }
            for(n2=j; n2<dstColor.rows; n2++)
            {
                if(holeImg(n2,i)==255) break;
            }

            if(n1<0) temp = dstColor(n2,i);
            else if(n2>=dstColor.rows) temp = dstColor(n1,i);
            else if(n1>=0 && n2<dstColor.rows)
            {
                temp.x = ((n2-j)*dstColor(n1, i).x + (j-n1)*dstColor(n2, i).x)/(n2-n1);
                temp.y = ((n2-j)*dstColor(n1, i).y + (j-n1)*dstColor(n2, i).y)/(n2-n1);
                temp.z = ((n2-j)*dstColor(n1, i).z + (j-n1)*dstColor(n2, i).z)/(n2-n1);
            }
            dstColor(j,i).x = (dstColor(j,i).x + temp.x)/2;
            dstColor(j,i).y = (dstColor(j,i).y + temp.y)/2;
            dstColor(j,i).z = (dstColor(j,i).z + temp.z)/2;
//            holeImg(j,i) = 255;
        }
    }
}
*/

void ImageWarping(Mat srcColorL,
                 Mat srcDepthL,
                 Mat srcProjMatL,
                 Mat srcColorR,
                 Mat srcDepthR,
                 Mat srcProjMatR,
                 Mat dstColorL,
                 Mat dstColorR,
                 Mat dstColor,
                 Mat dstProjMat,
                 float alpha)
{
    unsigned int rows= srcColorL.rows, columns= srcColorL.cols;

    Mat dstDepthL(rows,columns,CV_64FC1,Scalar(0));
    Mat dstDepthR(rows,columns,CV_64FC1,Scalar(0));
    Mat dstFiltDepthL(rows,columns,CV_64FC1,Scalar(0));
    Mat dstFiltDepthR(rows,columns,CV_64FC1,Scalar(0));
    Mat holeImgL(rows,columns,CV_8UC1,Scalar(0));
    Mat holeImgR(rows,columns,CV_8UC1,Scalar(0));
    Mat holeImg(rows,columns,CV_8UC1,Scalar(255));
    Mat edge_holeImgL(rows,columns,CV_8UC1,Scalar(0));
    Mat edge_holeImgR(rows,columns,CV_8UC1,Scalar(0));
    Mat edge_holeImgL_src;
    Mat edge_holeImgR_src;
    Mat srcDepth_dilate_L;
    Mat srcDepth_dilate_R;
    Mat dilate_edgeL(rows,columns,CV_8UC1,Scalar(0));
    Mat dilate_edgeR(rows,columns,CV_8UC1,Scalar(0));

    Mat elem = getStructuringElement(MORPH_RECT, Size(3,3));
//    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, CV_8U, elem);
//    Ptr<cuda::Filter> medianFilter = cuda::createMedianFilter(CV_64F, 3); //must CV_8UC1
//    Ptr<cuda::CannyEdgeDetector> canny_detecor = cuda::createCannyEdgeDetector(50,150,3);

    double ratioV=0;
    int cntV=0;
//    cudaMalloc(&ratioV, sizeof(double));
//    cudaMemset(ratioV, 0, sizeof(double));
//    cudaMalloc(&cntV, sizeof(int));
//    cudaMemset(cntV, 0, sizeof(int));


#if TIMETEST
    clock_t startTime = clock();
#endif

    dilate(srcDepthL, srcDepth_dilate_L, elem);
    dilate(srcDepthR, srcDepth_dilate_R, elem);

    Canny(srcDepth_dilate_L, edge_holeImgL_src, 50, 150, 3);
    Canny(srcDepth_dilate_R, edge_holeImgR_src, 50, 150, 3);
#if TIMETEST
    clock_t stopTime1 = clock();
#endif

    ForwardWarping(srcColorL,
                   srcDepthL,
                   dstColorL,
                   dstDepthL,
                   srcProjMatL,
                   dstProjMat,
                   edge_holeImgL_src,
                   edge_holeImgL,
                   rows,
                   columns);
    ForwardWarping(srcColorR,
                   srcDepthR,
                   dstColorR,
                   dstDepthR,
                   srcProjMatR,
                   dstProjMat,
                   edge_holeImgR_src,
                   edge_holeImgR,
                   rows,
                   columns);

    MedianFilter(dstDepthL,dstFiltDepthL,rows,columns);
    MedianFilter(dstDepthR,dstFiltDepthR,rows,columns);

    InverseWarping(dstColorL,
                   dstFiltDepthL,
                   srcColorL,
                   holeImgL,
                   srcProjMatL,
                   dstProjMat,
                   rows,
                   columns);
    InverseWarping(dstColorR,
                   dstFiltDepthR,
                   srcColorR,
                   holeImgR,
                   srcProjMatR,
                   dstProjMat,
                   rows,
                   columns);

#if TIMETEST
    clock_t stopTime2 = clock();
#endif

#if DEBUG_MODE
    Mat tempMap;
//    edge_holeImgL_src.download(tempMap);
//    imshow("eholeL_src",tempMap);
//    edge_holeImgR_src.download(tempMap);
//    imshow("eholeR_src",tempMap);
//    srcDepth_dilate_L.download(tempMap);
//    imshow("dilate_dep_L",tempMap);
//    srcDepth_dilate_R.download(tempMap);
//    imshow("dilate_dep_R",tempMap);
//    dstFiltDepthR.download(tempMap);
//    imwrite("/home/cly/Desktop/FDR.png", tempMap);
//    dstFiltDepthL.download(tempMap);
//    imwrite("/home/cly/Desktop/FDL.png", tempMap);
//    srcColorR.download(tempMap);

#endif
    m_erode2(holeImgL, edge_holeImgL, dilate_edgeL);
    m_erode2(holeImgR, edge_holeImgR, dilate_edgeR);

#if TIMETEST
    clock_t stopTime3 = clock();
#endif

//#if !BALLET
    calValueRatio(dstColorL,holeImgL,dstColorR,holeImgR,&ratioV,&cntV);

    ratioV /= cntV;

//    std::cout<< ratioV << cntV << std::endl;
//#endif
    ImageBlending(dstColorL,
                  dstColorR,
                  holeImgL,
                  holeImgR,
                  holeImg,
                  dstColor,
                  alpha,
                  ratioV,
                  rows,
                  columns);

#if TIMETEST
    clock_t stopTime4 = clock();
#endif

#if DEBUG_MODE
    dstColor.download(tempMap);
    imshow("dst_without_inpaint",tempMap);
#endif

//    Inpaint <<< gridSize, blockSize >>>(dstColor,holeImg);
    Inpaint(dstColor,dstFiltDepthL,dstFiltDepthR,holeImg);

#if TIMETEST
    clock_t stopTime5 = clock();

    double time1 = ((double)(stopTime1-startTime))/CLOCKS_PER_SEC;
    double time2 = ((double)(stopTime2-stopTime1))/CLOCKS_PER_SEC;
    double time3 = ((double)(stopTime3-stopTime2))/CLOCKS_PER_SEC;
    double time4 = ((double)(stopTime4-stopTime3))/CLOCKS_PER_SEC;
    double time5 = ((double)(stopTime5-stopTime4))/CLOCKS_PER_SEC;
//    std::cout<< "gpu:time1:"<<time1<<" time2:"<<time2<<" time3:"<<time3<<" time4:"<<time4<<std::endl;

#if LOGWRITE
    std::ofstream out("/home/cly/Desktop/time.txt",std::ios::app);
    out <<time1<<"\t"<<time2<<"\t"<<time3<<"\t"<<time4<<"\t"<<time5<<std::endl;
    out.close();
#endif
#endif


#if DEBUG_MODE
//    Mat tempMap;
//    dilate_edgeL.download(tempMap);
//    imshow("edge-holeL",tempMap);
//    dilate_edgeR.download(tempMap);
//    imshow("edge-holeR",tempMap);
//    holeImgL.download(tempMap);
//    imshow("holeL",tempMap);
//    holeImgR.download(tempMap);
//    imshow("holeR",tempMap);
    holeImg.download(tempMap);
    imshow("hole",tempMap);
//    edge_holeImgL.download(tempMap);
//    imshow("eholeL",tempMap);
//    edge_holeImgR.download(tempMap);
//    imshow("eholeR",tempMap);
//    dstDepthL.download(tempMap);
//    imshow("dstDL",tempMap/255);
//    dstDepthR.download(tempMap);
//    imshow("dstDR",tempMap/255);
//    dstFiltDepthL.download(tempMap);
//    imshow("dstDFL",tempMap/255);
//    dstFiltDepthR.download(tempMap);
//    imshow("dstDFR",tempMap/255);
//    srcDepthL.download(tempMap);
//    imshow("srcDL",tempMap);
//    srcDepthR.download(tempMap);
//    imshow("srcDR",tempMap);
#endif
}

