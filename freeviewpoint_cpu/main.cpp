#include <iostream>
#include <opencv2/opencv.hpp>
//#include <boost/program_options.hpp>
#include <time.h>
#include <string>

#include "calib_deal.h"
#include "kernel.h"

#define CAM_INTERVAL 40
#define HAND_GESTURE 0
#define ONE_VIEW 1
#define LOGWRITE 0
#define TIMETEST 0

#if LOGWRITE
#include <fstream>
#endif


#if POZNANSTREET || POZNANHALL2
#define IMG_COLS 1920
#define IMG_ROWS 1088
#define FRAME_NUM 200
#else
#define IMG_COLS 1024
#define IMG_ROWS 768
#define FRAME_NUM 100
#endif

#if BALLET
#define IMG_PATH "/home/cly/Documents/3DVideos-distrib/MSR3DVideo-Ballet/cam"
#define CALI_PATH "/home/cly/Documents/3DVideos-distrib/MSR3DVideo-Ballet/calibParams-ballet.txt"
#elif BREAKDANCER
#define IMG_PATH "/home/cly/Documents/3DVideos-distrib/MSR3DVideo-Breakdancers/cam"
#define CALI_PATH "/home/cly/Documents/3DVideos-distrib/MSR3DVideo-Breakdancers/calibParams-breakdancers.txt"
#elif KENDO
#define IMG_PATH "/home/cly/Documents/Kendo/cam"
#define CALI_PATH "/home/cly/Documents/Kendo/cam_param_kendo.txt"
#elif LOVEBIRD
#define IMG_PATH "/home/cly/Documents/lovebird/cam"
#define CALI_PATH "/home/cly/Documents/lovebird/cam_param_lovebird1.txt"
#elif BOOKARRIVAL
#define IMG_PATH "/home/cly/Documents/BookArrival/cam"
#define CALI_PATH "/home/cly/Documents/BookArrival/cam_param_bookarrival.txt"
#elif NEWSPAPER
#define IMG_PATH "/home/cly/Documents/Newspaper/cam"
#define CALI_PATH "/home/cly/Documents/Newspaper/cam_param_news.txt"
#elif POZNANSTREET
#define IMG_PATH "/home/cly/Documents/poznanstreet/cam"
#define CALI_PATH "/home/cly/Documents/poznanstreet/cam_param_poznan_street.txt"
#elif POZNANHALL2
#define IMG_PATH "/home/cly/Documents/poznanhall2/cam"
#define CALI_PATH "/home/cly/Documents/poznanhall2/cam_param_poznan_hall2.txt"
#endif

#define IMG_COLOR_NAME "color-cam"

#if HAND_GESTURE
#define FIFO "/tmp/FIFO" /*有名管道的名字*/
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace std;
using namespace cv;
//namespace po = boost::program_options;

extern void ImageWarping(Mat srcColorL,
                             Mat srcDepthL,
                             Mat srcProjMatL,
                             Mat srcColorR,
                             Mat srcDepthR,
                             Mat srcProjMatR,
                             Mat dstColorL,
                             Mat dstColorR,
                             Mat dstColor,
                             Mat dstProjMat,
                             float alpha);


//connect the 8 referenced viewpoint consecutively with virtual viewpoint.It's linear.
void computeVisualProjMat(Mat visualProjMat, int shiftValue)
{
    int camR = shiftValue / CAM_INTERVAL;
    int offset = shiftValue % CAM_INTERVAL;
    visualProjMat.at<double>(0,0)=m_CalibParams[camR].m_ProjMatrix.at<double>(0,0)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(0,0) - m_CalibParams[camR].m_ProjMatrix.at<double>(0,0))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(0,1)=m_CalibParams[camR].m_ProjMatrix.at<double>(0,1)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(0,1) - m_CalibParams[camR].m_ProjMatrix.at<double>(0,1))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(0,2)=m_CalibParams[camR].m_ProjMatrix.at<double>(0,2)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(0,2) - m_CalibParams[camR].m_ProjMatrix.at<double>(0,2))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(0,3)=m_CalibParams[camR].m_ProjMatrix.at<double>(0,3)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(0,3) - m_CalibParams[camR].m_ProjMatrix.at<double>(0,3))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(1,0)=m_CalibParams[camR].m_ProjMatrix.at<double>(1,0)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(1,0) - m_CalibParams[camR].m_ProjMatrix.at<double>(1,0))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(1,1)=m_CalibParams[camR].m_ProjMatrix.at<double>(1,1)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(1,1) - m_CalibParams[camR].m_ProjMatrix.at<double>(1,1))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(1,2)=m_CalibParams[camR].m_ProjMatrix.at<double>(1,2)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(1,2) - m_CalibParams[camR].m_ProjMatrix.at<double>(1,2))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(1,3)=m_CalibParams[camR].m_ProjMatrix.at<double>(1,3)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(1,3) - m_CalibParams[camR].m_ProjMatrix.at<double>(1,3))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(2,0)=m_CalibParams[camR].m_ProjMatrix.at<double>(2,0)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(2,0) - m_CalibParams[camR].m_ProjMatrix.at<double>(2,0))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(2,1)=m_CalibParams[camR].m_ProjMatrix.at<double>(2,1)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(2,1) - m_CalibParams[camR].m_ProjMatrix.at<double>(2,1))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(2,2)=m_CalibParams[camR].m_ProjMatrix.at<double>(2,2)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(2,2) - m_CalibParams[camR].m_ProjMatrix.at<double>(2,2))/CAM_INTERVAL*offset;
    visualProjMat.at<double>(2,3)=m_CalibParams[camR].m_ProjMatrix.at<double>(2,3)+
            (m_CalibParams[camR+1].m_ProjMatrix.at<double>(2,3) - m_CalibParams[camR].m_ProjMatrix.at<double>(2,3))/CAM_INTERVAL*offset;
}

double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|AbsDiff函数是 OpenCV 中计算两个数组差的绝对值的函数
    s1.convertTo(s1, CV_32F);  // 这里我们使用的CV_32F来计算，因为8位无符号char是不能进行平方计算
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         //对每一个通道进行加和

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // 对于非常小的值我们将约等于0
        return 0;
    else
    {
        double mse =sse /(double)(I1.channels() * I1.total());//计算MSE
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;//返回PSNR
    }
}

Scalar getSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2


     /***********************PRELIMINARY COMPUTING ******************************/

    Mat mu1, mu2;   //
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}


int main(int argc,char *argv[])
{

#if HAND_GESTURE
    if(access(FIFO,F_OK))//判断是否已经创建了有名管道，如果已经创建，则返回0否则返回非0的数
    {
        int ret = mkfifo(FIFO,0777); /*创建有名管道,成功返回0,失败返回-1*/
        if(ret == -1) /*创建有名管道失败*/
        {
            std::cout<<"mkfifo failed"<<std::endl;
            exit(1);
        }
    }
    int fd = open(FIFO, O_RDONLY);
    if(fd  == -1) /*打开失败*/
    {
        cout<<"open FIFO failed"<<endl;
//        exit(1);
    }
#endif

    if(0 == InitCalibParams(CALI_PATH))
    {
        cout<<"open calibparams file failed"<<endl;
        return -1;
    }

    //print camera param matrix
//    for(int n=0;n<8;n++)
//    {
//        for(int i=0;i<3;i++)
//        {
//            for(int j=0;j<3;j++)
//                cout<<m_CalibParams[n].m_K[i][j]<<ends;
//            cout<<""<<endl;
//        }

//        for(int i=0;i<3;i++)
//        {
//            for(int j=0;j<3;j++)
//                cout<<m_CalibParams[n].m_RotMatrix[i][j]<<ends;
//            cout<<""<<endl;
//        }

//        for(int i=0;i<3;i++)
//            cout<<m_CalibParams[n].m_Trans[i]<<ends;
//        cout<<""<<endl;
//    }

    if(cuda::getCudaEnabledDeviceCount()==0){
        cerr<<"此OpenCV编译的时候没有启用CUDA模块"<<endl;
        return -1;
    }

    Mat colorImgL, colorImgR, depthImgL, depthImgR, srcProjMat[8], dstProjMat;
    Mat dstL(IMG_ROWS, IMG_COLS, CV_8UC3);
    Mat dstR(IMG_ROWS, IMG_COLS, CV_8UC3);
    Mat dstImg(IMG_ROWS, IMG_COLS, CV_8UC3);

    for(int i=0; i<8;i++)
    {
        srcProjMat[i] = m_CalibParams[i].m_ProjMatrix;
    }
    int shiftValue_t=CAM_INTERVAL*7/2;
    namedWindow("dstImg",WINDOW_AUTOSIZE);

#if !HAND_GESTURE && !ONE_VIEW
    createTrackbar("shift","dstImg",&shiftValue_t,CAM_INTERVAL*7-1);
#endif

#if !ONE_VIEW
    int camR = shiftValue_t / CAM_INTERVAL;
    int camL = camR +1;
    float alpha = (shiftValue_t % CAM_INTERVAL) / (float)CAM_INTERVAL;
    Mat visualProjMat = m_CalibParams[4].m_ProjMatrix.clone();
    computeVisualProjMat(visualProjMat,shiftValue_t);
    d_dstProjMat.upload(visualProjMat);
#else
#if BALLET || BREAKDANCER || POZNANSTREET || BOOKARRIVAL
    int camR = 3;
    int camL = 5;
    Mat visualProjMat = m_CalibParams[4].m_ProjMatrix.clone();
#elif NEWSPAPER
    int camR = 2;
    int camL = 6;
    Mat visualProjMat = m_CalibParams[4].m_ProjMatrix.clone();
#elif KENDO
    int camR = 1;
    int camL = 5;
    Mat visualProjMat = m_CalibParams[3].m_ProjMatrix.clone();
#elif LOVEBIRD
    int camR = 3;
    int camL = 7;
    Mat visualProjMat = m_CalibParams[5].m_ProjMatrix.clone();
#elif POZNANHALL2
    int camR = 5;
    int camL = 7;
    Mat visualProjMat = m_CalibParams[6].m_ProjMatrix.clone();
#endif
    float alpha = 0.5;
    dstProjMat = visualProjMat;
#endif

    unsigned int frameCnt=0;
    int shiftValue = shiftValue_t;
    bool pause=false;
    bool update = true;
    double m_time;
#if DEBUG_MODE
//    int lowthreshold=50, highthreshold=150;
//    namedWindow("ddfad",WINDOW_AUTOSIZE);
//    createTrackbar("low","ddfad",&lowthreshold,255);
//    createTrackbar("high","ddfad",&highthreshold,255);
#endif
#if LOGWRITE
    ofstream out("/home/cly/Desktop/log",ios::app);
#endif
    while(1)
    {
        clock_t startTime = clock();
        clock_t stopTime1;

        if(shiftValue!=shiftValue_t && update==false) update = true;  //don't update when pause and shift is not be changed
        if(update)
        {
#if !ONE_VIEW
            shiftValue = shiftValue_t;

            if(camR != shiftValue / CAM_INTERVAL)
            {
                camR = shiftValue / CAM_INTERVAL;
                camL = camR +1;
            }

            alpha = (shiftValue % CAM_INTERVAL) / (float)CAM_INTERVAL;
            computeVisualProjMat(visualProjMat,shiftValue);
            d_dstProjMat.upload(visualProjMat);
#endif

            stringstream f_num;
            f_num << setw(3) << setfill('0') << frameCnt;
            colorImgL = imread(IMG_PATH+to_string(camL)+"/color-cam"+to_string(camL)+"-f"+f_num.str()+".jpg");

            colorImgR = imread(IMG_PATH+to_string(camR)+"/color-cam"+to_string(camR)+"-f"+f_num.str()+".jpg");

            depthImgL = imread(IMG_PATH+to_string(camL)+"/depth-cam"+to_string(camL)+"-f"+f_num.str()+".png", IMREAD_GRAYSCALE);

            depthImgR = imread(IMG_PATH+to_string(camR)+"/depth-cam"+to_string(camR)+"-f"+f_num.str()+".png", IMREAD_GRAYSCALE);

            ImageWarping(colorImgL,
                         depthImgL,
                         srcProjMat[camL],
                         colorImgR,
                         depthImgR,
                         srcProjMat[camR],
                         dstL,
                         dstR,
                         dstImg,
                         dstProjMat,
                         alpha);

            imshow("dstImg",dstImg);
//            imshow("RImg",dstR);
//            imshow("LImg",dstL);
#if DEBUG_MODE
//            Mat tempmap;
//            Canny(depthImg,tempmap,lowthreshold,highthreshold);

//            imshow("ddfad", tempmap);

//            ImageWarping2(d_colorImgL,
//                         d_depthImgL,
//                         d_srcProjMat[camL],
//                         d_colorImgR,
//                         d_depthImgR,
//                         d_srcProjMat[camR],
//                         d_dstL,
//                         d_dstR,
//                         d_dst,
//                         d_dstProjMat,
//                         alpha);

//////            d_dstL.download(dstImg);
//////            imshow("dstL2",dstImg);
//////            d_dstR.download(dstImg);
//////            imshow("dstR2",dstImg);
//            d_dst.download(dstImg);
//            imshow("dstImg2",dstImg);
#endif
#if TIMETEST
//        double m_time1 = ((double)(stopTime1-startTime))/CLOCKS_PER_SEC;
//            Mat trueMidImg = imread(IMG_PATH+to_string(camL-1)+"/color-cam"+to_string(camL-1)+"-f"+f_num.str()+".jpg");
//            double psnr = getPSNR(trueMidImg,dstImg);
//            Scalar ssim = getSSIM(trueMidImg,dstImg);
//            cout << frameCnt << "psnr:" << psnr << endl;
//            cout << frameCnt << "ssim:" << (ssim[0]+ssim[1]+ssim[2])/3 << endl;
//            Mat elem = getStructuringElement(MORPH_RECT, Size(3,3));
//            Mat tempppp;
//            resize(depthImg,depthImg,Size(IMG_COLS*4,IMG_ROWS*4));
//            startTime = clock();
//            dilate(depthImg, tempppp, elem);
//            Canny(tempppp,tempppp,50,150);
//            dilate(depthImg, tempppp, elem);
//            Canny(tempppp,tempppp,50,150);
//            imshow("xxxxxxx", tempppp);
//            stopTime1 = clock();
//            double m_time1 = ((double)(stopTime1-startTime))/CLOCKS_PER_SEC;
//            cout << m_time1*1000 <<endl;
#if LOGWRITE
//            out << m_time1*1000 <<endl;
//            out << psnr << "\t" << (ssim[0]+ssim[1]+ssim[2])/3 << endl;
#endif
#endif
            update = false;
        }

        clock_t stopTime = clock();
        m_time = ((double)(stopTime-startTime))/CLOCKS_PER_SEC;

#if HAND_GESTURE
        char rx[30];
        int len = read(fd,rx,sizeof(rx));
//        cout<<rx<<endl;
        if(len>0 && rx[0]=='A')
        {
            shiftValue_t += rx[1];
            if(shiftValue_t > CAM_INTERVAL*7-1) shiftValue_t = CAM_INTERVAL*7-1;
            else if(shiftValue_t < 0) shiftValue_t = 0;
            if(rx[2]==1) pause=true;
            else pause=false;
        }
#endif

        char delayTime;
        if(m_time*1000 < 64) delayTime = 65 - m_time*1000;
        else delayTime = 1;

//        delayTime= 0 ;
        char key = waitKey(delayTime);
        if(key=='p')
        {
            pause=!pause;
        }
        else if(key == 27) break;
        else if(key == 's') imwrite("/home/cly/Desktop/dstImg.jpg",dstImg);

        if(pause)
        {
            update = false;
            if(key=='a' && frameCnt>0)
            {
                frameCnt -= 1;
                update = true;
            }
            else if(key=='d' && frameCnt<FRAME_NUM-1)
            {
                frameCnt += 1;
                update = true;
            }
        }
        else
        {
            frameCnt++;
            update = true;
        }
        if(frameCnt >= FRAME_NUM)      //loop playing video
        {
            frameCnt = 0;
//            cout << "time:" << m_time*10 <<"ms" <<endl;
//            m_time=0;
        }
    }

#if HAND_GESTURE
    close(fd);
#endif
    cout<< "success!" <<endl;
    waitKey();
#if LOGWRITE
    out.close();
#endif
    return 0;
}
