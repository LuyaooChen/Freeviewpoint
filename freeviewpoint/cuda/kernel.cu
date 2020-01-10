#include "kernel.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include <iostream>
#include "device_launch_parameters.h"
#include "cuda.h"
using namespace cv;

#define TIMETEST 0
#define LOGWRITE 0
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

__device__
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

__device__
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

__device__
void projUVZtoXY(cuda::PtrStep<double> projMatrix, double u, double v, double z, double *x, double *y, unsigned int rows)
{
    double c0, c1, c2;
    v = rows - v - 1.0;

    c0 = z*projMatrix(0,2) + projMatrix(0,3);
    c1 = z*projMatrix(1,2) + projMatrix(1,3);
    c2 = z*projMatrix(2,2) + projMatrix(2,3);

    *y = u*(c1*projMatrix(2,0) - projMatrix(1,0)*c2) +
            v*(c2*projMatrix(0,0) - projMatrix(2,0)*c0) +
            projMatrix(1,0)*c0 - c1*projMatrix(0,0);

    *y /= v*(projMatrix(2,0)*projMatrix(0,1) - projMatrix(2,1)*projMatrix(0,0)) +
        u*(projMatrix(1,0)*projMatrix(2,1) - projMatrix(1,1)*projMatrix(2,0)) +
        projMatrix(0,0)*projMatrix(1,1) - projMatrix(1,0)*projMatrix(0,1);

    *x = (*y)*(projMatrix(0,1) - projMatrix(2,1)*u) + c0 - c2*u;
    *x /= projMatrix(2,0)*u - projMatrix(0,0);
}

__device__
double projXYZtoUV(cuda::PtrStep<double> projMatrix, double x, double y, double z, double *u, double *v, unsigned int rows)
{
    double w;

    *u = projMatrix(0,0)*x +
         projMatrix(0,1)*y +
         projMatrix(0,2)*z +
         projMatrix(0,3);

    *v = projMatrix(1,0)*x +
         projMatrix(1,1)*y +
         projMatrix(1,2)*z +
         projMatrix(1,3);

    w = projMatrix(2,0)*x +
        projMatrix(2,1)*y +
        projMatrix(2,2)*z +
        projMatrix(2,3);

    *u /= w;
    *v /= w;

    // image (0,0) is bottom lefthand corner
    *v = rows - *v - 1.0;

    return w;
} // projXYZtoUV

__global__
void ForwardWarping(cuda::PtrStep<uchar3> srcColor,
                    cuda::PtrStep<uchar> srcDepth,
                    cuda::PtrStep<uchar3> dstColor,
                    cuda::PtrStep<double> dstDepth,
                    cuda::PtrStep<double> srcProjMat,
                    cuda::PtrStep<double> dstProjMat,
                    unsigned int rows,
                    unsigned int columns)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i<columns && j<rows)
    {
        double x, y, dstU=0, dstV=0;
//        double z = 1.0/((srcDepth(j,i)/255.0)*(1.0/MinZ - 1.0/MaxZ) + 1.0/MaxZ);
        double z = MaxZ + srcDepth(j,i) * (MinZ-MaxZ)/255.0;
//        srcDepth(j, i) = z;

        projUVZtoXY(srcProjMat, i, j, z, &x, &y, rows);
        projXYZtoUV(dstProjMat, x, y, z, &dstU, &dstV, rows);

        if(0<=dstU && dstU< columns && 0<=dstV && dstV< rows)
        {

            if(dstDepth((int)dstV, (int)dstU) ==0.0)
            {
                    dstDepth((int)dstV, (int)dstU) = z;
#if DEBUG_MODE
//                    dstColor((int)dstV, (int)dstU) = srcColor(j, i);
#endif
            }

//            atomicMin(&dstDepth((int)dstV, (int)dstU),(int)z);    //It's not useful even if I change the GpuMat and PtrStep's type to 'int'

            /* z-buffer
            syncthreads several times to compare,promising the min depth pixel be showed.
            Is there a better way to do this?
            atomicMin must need value with 'int' type */
            for(int cnt= 0; cnt<7; cnt++)
            {
                __syncthreads();
                if(z < dstDepth((int)dstV, (int)dstU))
                {
                    dstDepth((int)dstV, (int)dstU) = z;
                }
            }
        }

    }
//    __syncthreads();
}


__global__
void ForwardWarping(cuda::PtrStep<uchar3> srcColor,
                    cuda::PtrStep<uchar> srcDepth,
                    cuda::PtrStep<uchar3> dstColor,
                    cuda::PtrStep<double> dstDepth,
                    cuda::PtrStep<double> srcProjMat,
                    cuda::PtrStep<double> dstProjMat,
                    cuda::PtrStep<uchar> edge_hole,
                    cuda::PtrStep<uchar> dst_edge,
                    unsigned int rows,
                    unsigned int columns)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i<columns && j<rows)
    {
        double x, y, dstU=0, dstV=0;
        double z = 1.0/((srcDepth(j,i)/255.0)*(1.0/MinZ - 1.0/MaxZ) + 1.0/MaxZ);
//        double z = MaxZ + srcDepth(j,i) * (MinZ-MaxZ)/255.0;
//        srcDepth(j, i) = z;


        projUVZtoXY(srcProjMat, i, j, z, &x, &y, rows);
        projXYZtoUV(dstProjMat, x, y, z, &dstU, &dstV, rows);

        if(0<=dstU && dstU< columns && 0<=dstV && dstV< rows)
        {
            if(edge_hole(j,i)==255)
            {
//                        edge_hole(j,i)= 0;
                dst_edge((int)dstV, (int)dstU)= 255;
            }

            if(dstDepth((int)dstV, (int)dstU) ==0.0)
            {
                    dstDepth((int)dstV, (int)dstU) = z;
#if DEBUG_MODE
//                    dstColor((int)dstV, (int)dstU) = srcColor(j, i);
#endif
            }

//            atomicMin(&dstDepth((int)dstV, (int)dstU),(int)z);    //It's not useful even if I change the GpuMat and PtrStep's type to 'int'

            /* z-buffer
            syncthreads several times to compare,promising the min depth pixel be showed.
            Is there a better way to do this?
            atomicMin must need value with 'int' type */
            for(int cnt= 0; cnt<5; cnt++)
            {
                __syncthreads();
                if(z < dstDepth((int)dstV, (int)dstU))
                {
                    dstDepth((int)dstV, (int)dstU) = z;
                }
            }
        }

    }
//    __syncthreads();
}

__global__
void InverseWarping(cuda::PtrStep<uchar3> dstColor,
                    cuda::PtrStep<double> dstDepth,
                    cuda::PtrStep<uchar3> srcColor,
//                    cuda::PtrStep<uchar3> srcDepth,
                    cuda::PtrStep<uchar>  holeImg,
                    cuda::PtrStep<double> srcProjMat,
                    cuda::PtrStep<double> dstProjMat,
                    unsigned int rows,
                    unsigned int columns)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i<columns && j<rows)
    {
        double x, y;
        double srcU = 0.0, srcV = 0.0;
        double z = dstDepth(j, i);
        projUVZtoXY(dstProjMat, (double)i, (double)j, z, &x, &y, rows);
        projXYZtoUV(srcProjMat, x, y, z, &srcU, &srcV, rows);

        if(0<=srcU && srcU<columns && 0<=srcV && srcV<rows && z!=0)
        {
            dstColor(j,i)=srcColor((int)srcV, (int)srcU);
            holeImg(j,i) = 255;
        }
    }
//    __syncthreads();
}

__global__
void calValueRatio(cuda::PtrStepSz<uchar3> srcColorL,
                   cuda::PtrStep<uchar>    holeImgL,
                   cuda::PtrStep<uchar3>   srcColorR,
                   cuda::PtrStep<uchar>    holeImgR,
                   double *ratioV,
                   int *cntV)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i<srcColorL.cols && j<srcColorL.rows)
    {
        //todo:
        //bgr to hsv
        //calculate average V
        //get the ratio

        if(holeImgL(j, i)==255 && holeImgR(j, i)==255)
        {
            double BGR_L[3];
            double BGR_R[3];
            double CmaxL, CmaxR;
            BGR_L[0] = srcColorL(j, i).x / 255.0;
            BGR_L[1] = srcColorL(j, i).y / 255.0;
            BGR_L[2] = srcColorL(j, i).z / 255.0;

            CmaxL = (BGR_L[0] > BGR_L[1]) ? BGR_L[0] : BGR_L[1];
            CmaxL = (CmaxL > BGR_L[2]) ? CmaxL : BGR_L[2];

            BGR_R[0] = srcColorR(j, i).x / 255.0;
            BGR_R[1] = srcColorR(j, i).y / 255.0;
            BGR_R[2] = srcColorR(j, i).z / 255.0;

            CmaxR = (BGR_R[0] > BGR_R[1]) ? BGR_R[0] : BGR_R[1];
            CmaxR = (CmaxR > BGR_R[2]) ? CmaxR : BGR_R[2];

            if(CmaxR<0.00001) atomicAdd(ratioV, 1);
            else atomicAdd(ratioV, CmaxL/CmaxR);
            atomicAdd(cntV, 1);
        }
    }
}

__global__
void m_divide(double *src1, int *src2)
{
    *src1 /= *src2;
//    printf("ratio:%f", *src1);
}

__device__
void adjValue(uchar3 *srcPix, double ratio)  //srcPix is in right  (ratio= left/right)
{
    double HSV_t[3];
    double BGR_t[3];
    double Cmax, Cmin, delta;
    BGR_t[0] = srcPix->x / 255.0;
    BGR_t[1] = srcPix->y / 255.0;
    BGR_t[2] = srcPix->z / 255.0;

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

    //HSV to BGR
    if(HSV_t[1] < 0.000001)
    {
        *srcPix = make_uchar3(HSV_t[2],HSV_t[2],HSV_t[2]);
        return;
    }
    //  HSV_t[0] /=60;
    float f, a, b, c;
    int i = (int) HSV_t[0];
    f = HSV_t[0] - i;
    a = HSV_t[2] * ( 1 - HSV_t[1] );
    b = HSV_t[2] * ( 1 - HSV_t[1] * f );
    c = HSV_t[2] * ( 1 - HSV_t[1] * (1 - f ) );

    switch(i)
    {
        case 0: srcPix->z = HSV_t[2]; srcPix->y = c       ; srcPix->x = a;break;
        case 1: srcPix->z = b       ; srcPix->y = HSV_t[2]; srcPix->x = a;break;
        case 2: srcPix->z = a       ; srcPix->y = HSV_t[2]; srcPix->x = c;break;
        case 3: srcPix->z = a       ; srcPix->y = b       ; srcPix->x = HSV_t[2];break;
        case 4: srcPix->z = c       ; srcPix->y = a       ; srcPix->x = HSV_t[2];break;
        case 5: srcPix->z = HSV_t[2]; srcPix->y = a       ; srcPix->x = b;break;
    }
}

__global__
void ImageBlending( cuda::PtrStep<uchar3> srcColorL,
                    cuda::PtrStep<uchar3> srcColorR,
                    cuda::PtrStep<uchar>  holeImgL,
                    cuda::PtrStep<uchar>  holeImgR,
                    cuda::PtrStep<uchar>  holeImg,
                    cuda::PtrStep<uchar3> dstColor,
                    float alpha,
                    double *ratioV,
                    unsigned int rows,
                    unsigned int columns)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i<columns && j<rows)
    {
        if(holeImgL(j,i)==255 && holeImgR(j,i)==255)
        {
            //bgr to hsv
            //adjust V
            //hsv to bgr
//#if BALLET
            adjValue(&srcColorR(j, i), *ratioV);
//#endif
            dstColor(j,i).x = srcColorL(j,i).x * alpha + srcColorR(j,i).x * (1-alpha);
            dstColor(j,i).y = srcColorL(j,i).y * alpha + srcColorR(j,i).y * (1-alpha);
            dstColor(j,i).z = srcColorL(j,i).z * alpha + srcColorR(j,i).z * (1-alpha);
        }
        else if(holeImgL(j,i)==0 && holeImgR(j,i)==255)
        {
//#if BALLET
            adjValue(&srcColorR(j, i), *ratioV);
//#endif
            dstColor(j,i) = srcColorR(j,i);
            srcColorL(j,i) = make_uchar3(0,0,0);
        }
        else if(holeImgL(j,i)==255 && holeImgR(j,i)==0)
        {
            dstColor(j,i) = srcColorL(j,i);
            srcColorR(j,i) = make_uchar3(0,0,0);
        }
        else
        {
            dstColor(j,i) = make_uchar3(0,0,0);     //clean the previous map
            srcColorL(j,i) = make_uchar3(0,0,0);
            srcColorR(j,i) = make_uchar3(0,0,0);
            holeImg(j,i) = 0;
        }
    }
}

__global__
void ImageBlending2(cuda::PtrStep<uchar3> srcColorL,
                    cuda::PtrStep<uchar3> srcColorR,
                    cuda::PtrStep<uchar>  holeImgL,
                    cuda::PtrStep<uchar>  holeImgR,
                    cuda::PtrStep<uchar>  holeImg,
                    cuda::PtrStep<uchar3> dstColor,
                    float alpha,
                    unsigned int rows,
                    unsigned int columns)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i<columns && j<rows)
    {
        if(holeImgL(j,i)==255 && holeImgR(j,i)==255)
        {
            dstColor(j,i).x = srcColorL(j,i).x * alpha + srcColorR(j,i).x * (1-alpha);
            dstColor(j,i).y = srcColorL(j,i).y * alpha + srcColorR(j,i).y * (1-alpha);
            dstColor(j,i).z = srcColorL(j,i).z * alpha + srcColorR(j,i).z * (1-alpha);
        }
        else if(holeImgL(j,i)==0 && holeImgR(j,i)==255)
        {
            dstColor(j,i) = srcColorR(j,i);
            srcColorL(j,i) = make_uchar3(0,0,0);
        }
        else if(holeImgL(j,i)==255 && holeImgR(j,i)==0)
        {
            dstColor(j,i) = srcColorL(j,i);
            srcColorR(j,i) = make_uchar3(0,0,0);
        }
        else
        {
            dstColor(j,i) = make_uchar3(0,0,0);     //clean the previous map
            srcColorL(j,i) = make_uchar3(0,0,0);
            srcColorR(j,i) = make_uchar3(0,0,0);
            holeImg(j,i) = 0;
        }
    }
}

__global__
void MedianFilter(cuda::PtrStep<double> srcDepth,
                  cuda::PtrStep<double> dstDepth,
                  unsigned int rows,
                  unsigned int columns)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i<columns && j<rows)
    {
        if(i>0 && i<columns-1 && j>0 && j<rows-1)
        {
//            if(dstDepth(j,i)==0)
                dstDepth(j,i)=Median(srcDepth(j-1,i-1),srcDepth(j,i-1),srcDepth(j+1,i-1),
                                     srcDepth(j-1,i),srcDepth(j,i),srcDepth(j+1,i),
                                     srcDepth(j-1,i+1),srcDepth(j,i+1),srcDepth(j+1,i+1));
        }
        else
        {
            dstDepth(j,i)=srcDepth(j,i);
        }
    }
}

//this is for binary img
__global__
void m_erode(cuda::PtrStepSz<uchar> holeImg)    //, cuda::PtrStep<uchar> holeImg2)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i>0 && i<holeImg.cols-1 && j>0 && j<holeImg.rows-1)
    {
        if(holeImg(j, i)==255)  // && holeImg2(j, i)==255)
        {   //don't erode pixel which is hole in the other img(L/R),or it will expand the hole.
            //because it make holeimgL and R both equal 0.

            if(holeImg(j-1, i-1)== 0 ||
                holeImg(j-1, i  )== 0 ||
                holeImg(j-1, i+1)== 0 ||
                holeImg(j  , i-1)== 0 ||
                holeImg(j  , i+1)== 0 ||
                holeImg(j+1, i-1)== 0 ||
                holeImg(j+1, i  )== 0 ||
                holeImg(j+1, i+1)== 0 )
                holeImg(j, i) = 127;    //127 is temp
        }
        __syncthreads();
        if(holeImg(j, i)==127) holeImg(j, i) = 0;
    }
}

__global__
void m_dilate(cuda::PtrStepSz<uchar> holeImg, cuda::PtrStep<uchar> holeImg2)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i>0 && i<holeImg.cols-1 && j>0 && j<holeImg.rows-1)
    {
        if(holeImg(j, i)==0)
        {   //don't erode pixel which is hole in the other img(L/R),or it will expand the hole.
            //because it make holeimgL and R both equal 0.

            if(holeImg(j-1, i-1)== 255 ||
                holeImg(j-1, i  )== 255 ||
                holeImg(j-1, i+1)== 255 ||
                holeImg(j  , i-1)== 255 ||
                holeImg(j  , i+1)== 255 ||
                holeImg(j+1, i-1)== 255 ||
                holeImg(j+1, i  )== 255 ||
                holeImg(j+1, i+1)== 255 )
                holeImg(j, i) = 127;    //127 is temp
        }
        __syncthreads();
        if(holeImg(j, i)==127) holeImg(j, i) = 255;
    }
}

__device__
int m_abs(int a, int b)
{
    int c = a - b;
    if(c>=0) return c;
    else return -c;
}


//just for compare
__global__
void m_erode2(cuda::PtrStepSz<uchar> holeImg,
              cuda::PtrStep<uchar> edge_holeImg,
              cuda::PtrStep<uchar> dealed_edge)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    if(i>0 && i<holeImg.cols-1 && j>0 && j<holeImg.rows-1)
    {
//        if(holeImg(j, i)==255)
//        {
//            if(holeImg(j-1, i-1)== 0 ||
//                holeImg(j-1, i  )== 0 ||
//                holeImg(j-1, i+1)== 0 ||
//                holeImg(j  , i-1)== 0 ||
//                holeImg(j  , i+1)== 0 ||
//                holeImg(j+1, i-1)== 0 ||
//                holeImg(j+1, i  )== 0 ||
//                holeImg(j+1, i+1)== 0 )
//            {
//                for(int m=j-1; m<=j+1; m++)
//                    for(int n=i-1; n<=i+1; n++)
//                        edge_holeImg(m, n)=0;
//            }
//        }
//        __syncthreads();
//        if(edge_holeImg(j, i)==0) holeImg(j, i) = 0;

        if(holeImg(j, i)==255)
        {
            if(holeImg(j-1, i-1)== 0 ||             //is hole edge(value 255)
                holeImg(j-1, i  )== 0 ||
                holeImg(j-1, i+1)== 0 ||
                holeImg(j  , i-1)== 0 ||
                holeImg(j  , i+1)== 0 ||
                holeImg(j+1, i-1)== 0 ||
                holeImg(j+1, i  )== 0 ||
                holeImg(j+1, i+1)== 0 )
            {
                if(edge_holeImg(j-1, i-1)== 255 ||    //and is foreground edge
                    edge_holeImg(j-1, i  )== 255 ||
                    edge_holeImg(j-1, i+1)== 255 ||
                    edge_holeImg(j  , i-1)== 255 ||
                    edge_holeImg(j  , i+1)== 255 ||
                    edge_holeImg(j+1, i-1)== 255 ||
                    edge_holeImg(j+1, i  )== 255 ||
                    edge_holeImg(j+1, i+1)== 255 )

                {
                    for(int m=j-1; m<=j+1; m++)
                        for(int n=i-1; n<=i+1; n++)
                            dealed_edge(m, n)=255;
                }
            }
        }
        __syncthreads();
        if(dealed_edge(j, i)==255) holeImg(j, i) = 0;
    }
}

typedef struct
{
    double depth;
    int distance;
    uchar3 rgb;
}samplePoint;

__global__
void Inpaint(cuda::PtrStepSz<uchar3> dstColor,          //there still are some bugs...    depth wrong?
             cuda::PtrStep<double> VirDepthL,
             cuda::PtrStep<double> VirDepthR,
             cuda::PtrStep<uchar> holeImg)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;    //column
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;    //row

    /*
#define N_SAMPLE 4

    if(i<dstColor.cols && j<dstColor.rows )
    {
        if(holeImg(j,i)==0)
        {
            samplePoint p[N_SAMPLE];   //up right down left

            for(int n=j; n>=0; n--)
            {
                if(holeImg(n,i)==255)
                {
                    p[0].rgb = dstColor(n, i);
                    p[0].distance = j-n;
                    if(VirDepthL(n, i)!=0)    p[0].depth=VirDepthL(n, i);
                    else    p[0].depth=VirDepthR(n, i);
                    break;
                }
            }
            for(int n=j; n<dstColor.rows; n++)
            {
                if(holeImg(n,i)==255)
                {
                    p[1].rgb = dstColor(n, i);
                    p[1].distance = n-j;
                    if(VirDepthL(n, i)!=0)    p[0].depth=VirDepthL(n, i);
                    else    p[1].depth=VirDepthR(n, i);
                    break;
                }
            }
            for(int n=i; n>=0; n--)
            {
                if(holeImg(j,n)==255)
                {
                    p[2].rgb = dstColor(j, n);
                    p[2].distance = i-n;
                    if(VirDepthL(j, n)!=0)    p[2].depth=VirDepthL(j, n);
                    else    p[2].depth=VirDepthR(j, n);
                    break;
                }
            }
            for(int n=i; n<dstColor.cols; n++)
            {
                if(holeImg(j,n)==255)
                {
                    p[3].rgb = dstColor(j, n);
                    p[3].distance = n-i;
                    if(VirDepthL(j, n)!=0)    p[3].depth=VirDepthL(j, n);
                    else    p[3].depth=VirDepthR(j, n);
                    break;
                }
            }

            samplePoint temp;   //sort
            for (int m = 0; m < N_SAMPLE; m++)
            {
                for (int n = 0; n<N_SAMPLE-1; n++)
                {
                    if (p[n].depth>p[n + 1].depth)
                    {
                        temp = p[n];
                        p[n] = p[n + 1];
                        p[n + 1] = temp;
                    }
                }
            }

#define DEPTH_THESHOLD 10
            int record=0;
            for(int m=N_SAMPLE-1; m>0; m--)     //divite background and foreground
            {
//                printf("%f ", p[m].depth-p[m-1].depth);
                if(p[m].depth - p[m-1].depth > DEPTH_THESHOLD || p[m-1].depth==0)
                {
                    record = m;
                    break;
                }
            }
//            printf("\n");

            double weight_sum = 0;
            double3 weighted_mean = make_double3(0,0,0);
            for(int m=record; m<N_SAMPLE; m++)
            {
                double weight = 1.0 / p[m].distance;
                weight_sum += weight;
                weighted_mean.x += p[m].rgb.x * weight;
                weighted_mean.y += p[m].rgb.y * weight;
                weighted_mean.z += p[m].rgb.z * weight;
            }
            uchar3 tmp;
            tmp.x = weighted_mean.x / weight_sum;
            tmp.y = weighted_mean.y / weight_sum;
            tmp.z = weighted_mean.z / weight_sum;

            __syncthreads();
            dstColor(j,i) = tmp;
        }
    }*/


    if(i<dstColor.cols && j<dstColor.rows )
    {
        int n1, n2, n3, n4;
        double d1=0, d2=0, d3=0, d4=0;
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

            uchar3 temp;
            for(n3=j; n3>=0; n3--)
            {
                if(holeImg(n3,i)==255) break;
            }
            for(n4=j; n4<dstColor.rows; n4++)
            {
                if(holeImg(n4,i)==255) break;
            }

            if(n1<2) d1=0;
            else if(VirDepthL(j, n1)!=0)    d1 = VirDepthL(j, n1);
            else if(VirDepthR(j, n1)!=0)    d1 = VirDepthR(j, n1);
            if(n2>=dstColor.cols) d2=0;
            else if(VirDepthL(j, n2)!=0)    d2 = VirDepthL(j, n2);
            else if(VirDepthR(j, n2)!=0)    d2 = VirDepthR(j, n2);

            if(n3<2) d3=0;
            else if(VirDepthL(n3, i)!=0)    d3 = VirDepthL(n3, i);
            else if(VirDepthR(n3, i)!=0)    d3 = VirDepthR(n3, i);
            if(n4>=dstColor.rows) d4=0;
            else if(VirDepthL(n4, i)!=0)    d4 = VirDepthL(n4, i);
            else if(VirDepthR(n4, i)!=0)    d4 = VirDepthR(n4, i);

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

            if(d1==0 && d2==0) dstColor(j,i) = make_uchar3(0,0,0);
            else if(d1==0) dstColor(j,i) = dstColor(j,n2);
            else if(d2==0) dstColor(j,i) = dstColor(j,n1);
            else
            {
                dstColor(j,i).x = ((n2-i)*dstColor(j, n1).x + (i-n1)*dstColor(j, n2).x)/(n2-n1);
                dstColor(j,i).y = ((n2-i)*dstColor(j, n1).y + (i-n1)*dstColor(j, n2).y)/(n2-n1);
                dstColor(j,i).z = ((n2-i)*dstColor(j, n1).z + (i-n1)*dstColor(j, n2).z)/(n2-n1);
            }

            if(d3!=0 || d4!=0)
            {
                if(d3==0) temp = dstColor(n4,i);
                else if(d4==0) temp = dstColor(n3,i);
                else
                {
                    temp.x = ((n4-j)*dstColor(n3, i).x + (j-n3)*dstColor(n4, i).x)/(n4-n3);
                    temp.y = ((n4-j)*dstColor(n3, i).y + (j-n3)*dstColor(n4, i).y)/(n4-n3);
                    temp.z = ((n4-j)*dstColor(n3, i).z + (j-n3)*dstColor(n4, i).z)/(n4-n3);
                }

                if(d1==0 && d2==0)
                {
                    dstColor(j,i) = temp;
                }
                else
                {
                    dstColor(j,i).x = (dstColor(j,i).x + temp.x)/2;
                    dstColor(j,i).y = (dstColor(j,i).y + temp.y)/2;
                    dstColor(j,i).z = (dstColor(j,i).z + temp.z)/2;
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


__global__
void test_kernel()
{
    double sum =0.0;
    for(int i=0 ; i<65536; i++)
    {
        for(int j=0; j<65536;j++)
            sum = sum+tan(0.1)*tan(0.1);
    }
}

__global__
void test_kernel2()
{
    double sum =0.0;
    for(int i=0 ; i<65536; i++)
    {
        for(int j=0; j<65536;j++)
            sum = sum+tan(0.1)*tan(0.1);
    }
}

void ImageWarping(cuda::GpuMat srcColorL,
                 cuda::GpuMat srcDepthL,
                 cuda::GpuMat srcProjMatL,
                 cuda::GpuMat srcColorR,
                 cuda::GpuMat srcDepthR,
                 cuda::GpuMat srcProjMatR,
                 cuda::GpuMat dstColorL,
                 cuda::GpuMat dstColorR,
                 cuda::GpuMat dstColor,
                 cuda::GpuMat dstProjMat,
                 float alpha)
{
    unsigned int rows= srcColorL.rows, columns= srcColorL.cols;

    cuda::GpuMat dstDepthL(rows,columns,CV_64FC1,Scalar(0));
    cuda::GpuMat dstDepthR(rows,columns,CV_64FC1,Scalar(0));
    cuda::GpuMat dstFiltDepthL(rows,columns,CV_64FC1,Scalar(0));
    cuda::GpuMat dstFiltDepthR(rows,columns,CV_64FC1,Scalar(0));
    cuda::GpuMat holeImgL(rows,columns,CV_8UC1,Scalar(0));
    cuda::GpuMat holeImgR(rows,columns,CV_8UC1,Scalar(0));
    cuda::GpuMat holeImg(rows,columns,CV_8UC1,Scalar(255));
    cuda::GpuMat edge_holeImgL(rows,columns,CV_8UC1,Scalar(0));
    cuda::GpuMat edge_holeImgR(rows,columns,CV_8UC1,Scalar(0));
    cuda::GpuMat edge_holeImgL_src;
    cuda::GpuMat edge_holeImgR_src;
    cuda::GpuMat srcDepth_dilate_L;
    cuda::GpuMat srcDepth_dilate_R;
    cuda::GpuMat dilate_edgeL(rows,columns,CV_8UC1,Scalar(0));
    cuda::GpuMat dilate_edgeR(rows,columns,CV_8UC1,Scalar(0));

    Mat elem = getStructuringElement(MORPH_RECT, Size(3,3));
    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, CV_8U, elem);
//    Ptr<cuda::Filter> medianFilter = cuda::createMedianFilter(CV_64F, 3); //must CV_8UC1
    Ptr<cuda::CannyEdgeDetector> canny_detecor = cuda::createCannyEdgeDetector(50,150,3);

    double *ratioV;
    int *cntV;
    cudaMalloc(&ratioV, sizeof(double));
    cudaMemset(ratioV, 0, sizeof(double));
    cudaMalloc(&cntV, sizeof(int));
    cudaMemset(cntV, 0, sizeof(int));

    cudaStream_t stream0,stream1;       //it's not helpful...
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

#if TIMETEST
    cudaEvent_t startTime, stopTime1, stopTime2, stopTime3, stopTime4, stopTime5;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime1);
    cudaEventCreate(&stopTime2);
    cudaEventCreate(&stopTime3);
    cudaEventCreate(&stopTime4);
    cudaEventCreate(&stopTime5);

//    int _blockdize, _mingridsize;
//    cudaOccupancyMaxPotentialBlockSize(&_mingridsize, &_blockdize, MedianFilter, 0, 1024*768);
//    std::cout<<_blockdize<<std::endl;

#endif

    const dim3 blockSize(32, 32);
    unsigned int bx = (columns + blockSize.x - 1)/blockSize.x;
    unsigned int by = (rows + blockSize.y - 1)/blockSize.y;
    const dim3 gridSize = dim3(bx, by);

    const dim3 blockSize1(32, 20);
    bx = (columns + blockSize1.x - 1)/blockSize1.x;
    by = (rows + blockSize1.y - 1)/blockSize1.y;
    const dim3 gridSize1 = dim3(bx, by);

    const dim3 blockSize2(32, 24);
    bx = (columns + blockSize2.x - 1)/blockSize2.x;
    by = (rows + blockSize2.y - 1)/blockSize2.y;
    const dim3 gridSize2 = dim3(bx, by);

#if TIMETEST
    cudaEventRecord(startTime);
#endif

    dilateFilter->apply(srcDepthL, srcDepth_dilate_L);
    dilateFilter->apply(srcDepthR, srcDepth_dilate_R);

    canny_detecor->detect(srcDepth_dilate_L, edge_holeImgL_src);
    canny_detecor->detect(srcDepth_dilate_R, edge_holeImgR_src);

#if TIMETEST
    cudaEventRecord(stopTime1);
#endif

    ForwardWarping <<< gridSize1, blockSize1, 0, stream0 >>>(srcColorL,
                                                           srcDepthL,
                                                           dstColorL,
                                                           dstDepthL,
                                                           srcProjMatL,
                                                           dstProjMat,
                                                           edge_holeImgL_src,
                                                           edge_holeImgL,
                                                           rows,
                                                           columns);
    ForwardWarping <<< gridSize1, blockSize1, 0, stream1 >>>(srcColorR,
                                                           srcDepthR,
                                                           dstColorR,
                                                           dstDepthR,
                                                           srcProjMatR,
                                                           dstProjMat,
                                                           edge_holeImgR_src,
                                                           edge_holeImgR,
                                                           rows,
                                                           columns);
//    cudaDeviceSynchronize();
//    medianFilter->apply(dstDepthL, dstFiltDepthL);
//    medianFilter->apply(dstDepthR, dstFiltDepthR);
    MedianFilter <<< gridSize, blockSize, 0, stream0 >>>(dstDepthL,dstFiltDepthL,rows,columns);
    MedianFilter <<< gridSize, blockSize, 0, stream1 >>>(dstDepthR,dstFiltDepthR,rows,columns);

    InverseWarping <<< gridSize1, blockSize1, 0, stream0 >>>(dstColorL,
                                                           dstFiltDepthL,
                                                           srcColorL,
//                                                           srcDepthL,
                                                           holeImgL,
                                                           srcProjMatL,
                                                           dstProjMat,
                                                           rows,
                                                           columns);
    InverseWarping <<< gridSize1, blockSize1, 0, stream1 >>>(dstColorR,
                                                           dstFiltDepthR,
                                                           srcColorR,
//                                                           srcDepthR,
                                                           holeImgR,
                                                           srcProjMatR,
                                                           dstProjMat,
                                                           rows,
                                                           columns);
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

#if TIMETEST
    cudaEventRecord(stopTime2);
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


//    m_erode <<< gridSize, blockSize, 0, stream0 >>>(holeImgL);
//    m_erode <<< gridSize, blockSize, 0, stream1 >>>(holeImgR);
//    m_erode <<< gridSize, blockSize, 0, stream0 >>>(holeImgL);
//    m_erode <<< gridSize, blockSize, 0, stream1 >>>(holeImgR);

    m_erode2 <<< gridSize, blockSize, 0, stream0 >>>(holeImgL, edge_holeImgL, dilate_edgeL);
    m_erode2 <<< gridSize, blockSize, 0, stream1 >>>(holeImgR, edge_holeImgR, dilate_edgeR);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

#if TIMETEST
    cudaEventRecord(stopTime3);
#endif

//#if BALLET
    calValueRatio <<< gridSize, blockSize >>>(dstColorL,holeImgL,dstColorR,holeImgR,ratioV,cntV);

    m_divide <<< 1, 1 >>>(ratioV, cntV);
//#endif

    ImageBlending <<< gridSize2, blockSize2 >>>(dstColorL,
                                              dstColorR,
                                              holeImgL,
                                              holeImgR,
                                              holeImg,
                                              dstColor,
                                              alpha,
                                              ratioV,
                                              rows,
                                              columns);
    cudaDeviceSynchronize();

#if TIMETEST
    cudaEventRecord(stopTime4);
#endif

#if DEBUG_MODE
    dstColor.download(tempMap);
    imshow("dst_without_inpaint",tempMap);
#endif

//    Inpaint <<< gridSize, blockSize >>>(dstColor,holeImg);
    Inpaint <<< gridSize, blockSize >>>(dstColor,dstFiltDepthL,dstFiltDepthR,holeImg);
    cudaDeviceSynchronize();

#if TIMETEST
    cudaEventRecord(stopTime5);
#endif

    cudaFree(ratioV);
    cudaFree(cntV);



#if TIMETEST
    float time1, time2, time3, time4, time5;
    cudaEventElapsedTime(&time1, startTime, stopTime1);
    cudaEventElapsedTime(&time2, stopTime1, stopTime2);
    cudaEventElapsedTime(&time3, stopTime2, stopTime3);
    cudaEventElapsedTime(&time4, stopTime3, stopTime4);
    cudaEventElapsedTime(&time5, stopTime4, stopTime5);
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

#if DEBUG_MODE

void ImageWarping2(cuda::GpuMat srcColorL,
                 cuda::GpuMat srcDepthL,
                 cuda::GpuMat srcProjMatL,
                 cuda::GpuMat srcColorR,
                 cuda::GpuMat srcDepthR,
                 cuda::GpuMat srcProjMatR,
                 cuda::GpuMat dstColorL,
                 cuda::GpuMat dstColorR,
                 cuda::GpuMat dstColor,
                 cuda::GpuMat dstProjMat,
                 float alpha)
{
    unsigned int rows= srcColorL.rows, columns= srcColorL.cols;

    cuda::GpuMat dstDepthL(rows,columns,CV_64FC1,Scalar(0));
    cuda::GpuMat dstDepthR(rows,columns,CV_64FC1,Scalar(0));
    cuda::GpuMat dstFiltDepthL(rows,columns,CV_64FC1,Scalar(0));
    cuda::GpuMat dstFiltDepthR(rows,columns,CV_64FC1,Scalar(0));
    cuda::GpuMat holeImgL(rows,columns,CV_8UC1,Scalar(0));
    cuda::GpuMat holeImgR(rows,columns,CV_8UC1,Scalar(0));
    cuda::GpuMat holeImg(rows,columns,CV_8UC1,Scalar(255));


    double *ratioV;
    int *cntV;
    cudaMalloc(&ratioV, sizeof(double));
    cudaMemset(ratioV, 0, sizeof(double));
    cudaMalloc(&cntV, sizeof(int));
    cudaMemset(cntV, 0, sizeof(int));

    cudaStream_t stream0,stream1;       //it's not helpful...
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

#if DEBUG_MODE
    cudaEvent_t startTime, stopTime1, stopTime2, stopTime3, stopTime4;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime1);
    cudaEventCreate(&stopTime2);
    cudaEventCreate(&stopTime3);
    cudaEventCreate(&stopTime4);
#endif

    const dim3 blockSize(TPB1, TPB2);
    const unsigned int bx = (columns + blockSize.x - 1)/blockSize.x;
    const unsigned int by = (rows + blockSize.y - 1)/blockSize.y;
    const dim3 gridSize = dim3(bx, by);

#if DEBUG_MODE
    cudaEventRecord(startTime);
#endif
    ForwardWarping <<< gridSize, blockSize, 0, stream0 >>>(srcColorL,
                                                           srcDepthL,
                                                           dstColorL,
                                                           dstDepthL,
                                                           srcProjMatL,
                                                           dstProjMat,
                                                           rows,
                                                           columns);
    ForwardWarping <<< gridSize, blockSize, 0, stream1 >>>(srcColorR,
                                                           srcDepthR,
                                                           dstColorR,
                                                           dstDepthR,
                                                           srcProjMatR,
                                                           dstProjMat,
                                                           rows,
                                                           columns);
//    cudaDeviceSynchronize();
    MedianFilter <<< gridSize, blockSize, 0, stream0 >>>(dstDepthL,dstFiltDepthL,rows,columns);
    MedianFilter <<< gridSize, blockSize, 0, stream1 >>>(dstDepthR,dstFiltDepthR,rows,columns);


    InverseWarping <<< gridSize, blockSize, 0, stream0 >>>(dstColorL,
                                                           dstFiltDepthL,
                                                           srcColorL,
//                                                           srcDepthL,
                                                           holeImgL,
                                                           srcProjMatL,
                                                           dstProjMat,
                                                           rows,
                                                           columns);
    InverseWarping <<< gridSize, blockSize, 0, stream1 >>>(dstColorR,
                                                           dstFiltDepthR,
                                                           srcColorR,
//                                                           srcDepthR,
                                                           holeImgR,
                                                           srcProjMatR,
                                                           dstProjMat,
                                                           rows,
                                                           columns);
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    m_erode <<< gridSize, blockSize, 0, stream0 >>>(holeImgL);
    m_erode <<< gridSize, blockSize, 0, stream1 >>>(holeImgR);
    m_erode <<< gridSize, blockSize, 0, stream0 >>>(holeImgL);
    m_erode <<< gridSize, blockSize, 0, stream1 >>>(holeImgR);
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    calValueRatio <<< gridSize, blockSize >>>(dstColorL,holeImgL,dstColorR,holeImgR,ratioV,cntV);

    m_divide <<< 1, 1 >>>(ratioV, cntV);

    ImageBlending <<< gridSize, blockSize >>>(dstColorL,
                                              dstColorR,
                                              holeImgL,
                                              holeImgR,
                                              holeImg,
                                              dstColor,
                                              alpha,
                                              ratioV,
                                              rows,
                                              columns);
    cudaDeviceSynchronize();

//    Inpaint <<< gridSize, blockSize >>>(dstColor,holeImg);
    Inpaint <<< gridSize, blockSize >>>(dstColor,dstFiltDepthL,dstFiltDepthR,holeImg);
    cudaDeviceSynchronize();


    cudaFree(ratioV);
    cudaFree(cntV);

}

#endif
