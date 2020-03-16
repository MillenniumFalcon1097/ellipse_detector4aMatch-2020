#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;

const double f=82.41;
const double r=9.1;
#define PI 3.14159


Mat blueChannel(Mat img)
{
    Mat src=img;
    Mat schannel[3];
    split(src,schannel);
    return schannel[0];
}
Mat greenChannel(Mat img)
{
    Mat src=img;
    Mat schannel[3];
    split(src,schannel);
    return schannel[1];
}
Mat redChannel(Mat img)
{
    Mat src=img;
    Mat schannel[3];
    split(src,schannel);
    return schannel[2];
}
Mat HChannel(Mat img)
{
    Mat src=img.clone();
    Mat schannel[3];
    cvtColor(img,src,COLOR_BGR2HSV);
    split(src,schannel);
    return schannel[0];
}
Mat SChannel(Mat img)
{
    Mat src=img.clone();
    Mat schannel[3];
    cvtColor(img,src,COLOR_BGR2HSV);
    split(src,schannel);
    return schannel[1];
}
Mat VChannel(Mat img)
{
    Mat src=img.clone();
    Mat schannel[3];
    cvtColor(img,src,COLOR_BGR2HSV);
    split(src,schannel);
    return schannel[2];
}
double getDistance(Point2f point1, Point2f point2)
// calc distance between two points
{
    double distance;
    distance = powf((point1.x-point2.x),2) + powf((point1.y-point2.y),2);
    distance = sqrtf(distance);
    return distance;
}

double dist=0,dist_temp=0;
double radius=0, angle_temp_h=0, angle_temp_v=0;
const int lowthreshold=100;
const int maxthreshold=900;
int ratio=3;
Mat after_weighted;
int sliderH,sliderL;
double x_delay=0,y_delay=0;
double radius_delay=0,dist_delay=0;
double x_contemp=0,y_contemp=0;

void cannyFunc(int, void*)
{
    Mat dst,edges;
    blur(after_weighted,edges,Size(3,3));
    Canny(edges,dst,sliderL,sliderH,3);
    imshow("cann",dst);
}


//CheckMode: 0代表去除黑区域，1代表去除白区域; NeihborMode：0代表4邻域，1代表8邻域;
// cited by ellipse_detect func
void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit=100, int CheckMode=1, int NeihborMode=0)  
{     
    int RemoveCount=0;       //记录除去的个数  
    //记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
    Mat Pointlabel = Mat::zeros( Src.size(), CV_8UC1 );  
      
    if(CheckMode==1)  
    {  
        cout<<"Mode: 去除小区域. ";  
        for(int i = 0; i < Src.rows; ++i)    
        {    
            uchar* iData = Src.ptr<uchar>(i);  
            uchar* iLabel = Pointlabel.ptr<uchar>(i);  
            for(int j = 0; j < Src.cols; ++j)    
            {    
                if (iData[j] < 10)    
                {    
                    iLabel[j] = 3;   
                }    
            }    
        }    
    }  
    else  
    {  
        cout<<"Mode: 去除孔洞. ";  
        for(int i = 0; i < Src.rows; ++i)    
        {    
            uchar* iData = Src.ptr<uchar>(i);  
            uchar* iLabel = Pointlabel.ptr<uchar>(i);  
            for(int j = 0; j < Src.cols; ++j)    
            {    
                if (iData[j] > 10)    
                {    
                    iLabel[j] = 3;   
                }    
            }    
        }    
    }  
  
    vector<Point2i> NeihborPos;  //记录邻域点位置  
    NeihborPos.push_back(Point2i(-1, 0));  
    NeihborPos.push_back(Point2i(1, 0));  
    NeihborPos.push_back(Point2i(0, -1));  
    NeihborPos.push_back(Point2i(0, 1));  
    if (NeihborMode==1)  
    {  
        cout<<"Neighbor mode: 8邻域."<<endl;  
        NeihborPos.push_back(Point2i(-1, -1));  
        NeihborPos.push_back(Point2i(-1, 1));  
        NeihborPos.push_back(Point2i(1, -1));  
        NeihborPos.push_back(Point2i(1, 1));  
    }  
    else cout<<"Neighbor mode: 4邻域."<<endl;  
    int NeihborCount=4+4*NeihborMode;  
    int CurrX=0, CurrY=0;  
    //开始检测  
    for(int i = 0; i < Src.rows; ++i)    
    {    
        uchar* iLabel = Pointlabel.ptr<uchar>(i);  
        for(int j = 0; j < Src.cols; ++j)    
        {    
            if (iLabel[j] == 0)    
            {    
                //********开始该点处的检查**********  
                vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点  
                GrowBuffer.push_back( Point2i(j, i) );  
                Pointlabel.at<uchar>(i, j)=1;  
                int CheckResult=0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出  
  
                for ( int z=0; z<GrowBuffer.size(); z++ )  
                {  
  
                    for (int q=0; q<NeihborCount; q++)                                      //检查四个邻域点  
                    {  
                        CurrX=GrowBuffer.at(z).x+NeihborPos.at(q).x;  
                        CurrY=GrowBuffer.at(z).y+NeihborPos.at(q).y;  
                        if (CurrX>=0&&CurrX<Src.cols&&CurrY>=0&&CurrY<Src.rows)  //防止越界  
                        {  
                            if ( Pointlabel.at<uchar>(CurrY, CurrX)==0 )  
                            {  
                                GrowBuffer.push_back( Point2i(CurrX, CurrY) );  //邻域点加入buffer  
                                Pointlabel.at<uchar>(CurrY, CurrX)=1;           //更新邻域点的检查标签，避免重复检查  
                            }  
                        }  
                    }  
  
                }  
                if (GrowBuffer.size()>AreaLimit) CheckResult=2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出  
                else {CheckResult=1;   RemoveCount++;}  
                for (int z=0; z<GrowBuffer.size(); z++)                         //更新Label记录  
                {  
                    CurrX=GrowBuffer.at(z).x;   
                    CurrY=GrowBuffer.at(z).y;  
                    Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;  
                }  
                //********结束该点处的检查**********  
  
  
            }    
        }    
    }    
  
    CheckMode=255*(1-CheckMode);  
    //开始反转面积过小的区域  
    for(int i = 0; i < Src.rows; ++i)    
    {    
        uchar* iData = Src.ptr<uchar>(i);  
        uchar* iDstData = Dst.ptr<uchar>(i);  
        uchar* iLabel = Pointlabel.ptr<uchar>(i);  
        for(int j = 0; j < Src.cols; ++j)    
        {    
            if (iLabel[j] == 2)    
            {    
                iDstData[j] = CheckMode;   
            }    
            else if(iLabel[j] == 3)  
            {  
                iDstData[j] = iData[j];  
            }  
        }    
    }   
      
    cout<<RemoveCount<<" objects removed."<<endl;  
}


// detect function, reffering to removeSmallRegion func.
Mat ellipse_detect_pre(Mat frame)
{
    int count=0;
    Mat result=frame.clone();
    Mat S=SChannel(frame);
    Mat V=VChannel(frame);
    Mat after_blur,after_can,after_filt;
    after_filt=Mat::zeros(frame.size(),CV_8UC1);
    //addWeighted(S,0.45,V,0.55,1,after_weighted);
    cvtColor(frame,after_weighted,COLOR_BGR2GRAY);
    blur(after_weighted,after_weighted,Size(3,3));
    //GaussianBlur(after_weighted,after_blur,Size(3,3),0,0,4);
    // cv::imshow("after_blur",after_blur);
    // contours
    vector<vector<Point> > contours;
    vector<vector<Point> > contours_final;
    vector<Vec4i> hier;
    //namedWindow("cann",CV_WINDOW_AUTOSIZE);
    Canny(after_weighted,after_can,80,120,3);
    Mat kernel_d=getStructuringElement(MORPH_ELLIPSE,Size(3,3));
    dilate(after_can,after_can,kernel_d,Point(-1,-1),2);
    RemoveSmallRegion(after_can,after_filt,300,1,0);
    
    //Mat kernel_e=getStructuringElement(MORPH_ELLIPSE,Size(3,3));
    //erode(after_can,after_can,kernel_e,Point(-1,-1),1);
    
    //imshow("cann",after_filt);
    findContours(after_filt,contours,hier,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
    contours_final=contours;
    //imshow("cann",);
    vector<Moments> mu(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<RotatedRect> box(contours.size());
    vector<RotatedRect> box_final(contours.size());
    vector<Point> min_rectangle;
    Point2f rect[4];
    double r_temp=0;
    double maxR=0;
    vector<Point2f> mc(contours.size());
    double temp_dist_ratio[contours.size()]={0};
    double temp_dist_1[contours.size()]={0};
    double temp_dist_2[contours.size()]={0};
    double score[contours.size()]={0};
    Point2f center[contours.size()];
    double temp_dist_1_final[contours.size()]={0};
    double temp_dist_2_final[contours.size()]={0};
    Point2f center_final[contours.size()];

    for(int i=0;i<contours.size();i++)
    {
        mu[i]=moments(contours[i],false);
        mc[i]=Point2d(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
        
        if(contours[i].size()<=5)
            continue;
        else
        {
            if(contourArea(contours[i])>5000&&contourArea(contours[i])<200)
                continue;
            else
            {

                boundRect[i] = boundingRect(Mat(contours[i]));
                double recOrNot=abs(boundRect[i].width-boundRect[i].height)/(0.5*(boundRect[i].width+boundRect[i].height));
                //box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
                //rectangle(final, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + 
                //boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);
                if(recOrNot>0.55)  // 1st rect check
                    continue;
                else  // boundingbox is rect
                {
                    box[i]=fitEllipse(Mat(contours[i]));
                    Point2f* vertices=new Point2f[4];
                    box[i].points(vertices);
                    temp_dist_1[i]=sqrtf(powf((vertices[0].x-vertices[1].x),2)+powf(vertices[0].y-vertices[1].y,2));
                    temp_dist_2[i]=sqrtf(powf((vertices[2].x-vertices[1].x),2)+powf(vertices[2].y-vertices[1].y,2));
                    temp_dist_ratio[i]=temp_dist_1[i]/temp_dist_2[i];
                    if(temp_dist_ratio[i]>1.4||temp_dist_ratio[i]<0.6)  // 2nd rect check
                        continue;
                    else  // rect for sure
                    {
                        //ellipse(result,box[i],Scalar(0,0,255),2,8);
                        count++;
                        double centerX=0.5*(vertices[0].x+vertices[2].x);
                        double centerY=0.5*(vertices[0].y+vertices[2].y);
                        center[i]=Point2f(centerX,centerY);
                        contours_final[count-1]=contours[i];
                        box_final[count-1]=box[i];
                        temp_dist_1_final[count-1]=temp_dist_1[i];
                        temp_dist_2_final[count-1]=temp_dist_2[i];
                        center_final[count-1]=center[i];
                    }
                }
            }
        }

    }
    // score items
    for(int i=0;i<count;i++)
    {
        if(contourArea(contours_final[i]))
            score[i]=0.8*contourArea(contours_final[i]);
        else
            score[i]=0;
    }
        
    // find max score
    double max=0;
    int idx=0;
    for(int i=0;i<count;i++)
    {
        if(max<score[i])
        {
            max=score[i];
            idx=i;
        }
    }
    // plot
    ellipse(result,box_final[idx],Scalar(0,0,255),2);
    cv::circle(result,center_final[idx],2,Scalar(255,0,0),1,8);

    // calc
    radius=0.5*sqrtf(powf(temp_dist_1_final[idx],2)+powf(temp_dist_2_final[idx],2));
    dist=f*r/radius*6*1.5;
    x_delay=center_final[idx].x;
    y_delay=center_final[idx].y;
    radius_delay=radius;
    dist_delay=dist;

    return result;
}



// detect function, reffering to removeSmallRegion func.
Mat ellipse_detect(Mat frame)
{
    int count=0;
    Mat result=frame.clone();
    Mat S=SChannel(frame);
    Mat V=VChannel(frame);
    Mat after_blur,after_can,after_filt;
    after_filt=Mat::zeros(frame.size(),CV_8UC1);
    //addWeighted(S,0.45,V,0.55,1,after_weighted);
    cvtColor(frame,after_weighted,COLOR_BGR2GRAY);
    blur(after_weighted,after_weighted,Size(3,3));
    //GaussianBlur(after_weighted,after_blur,Size(3,3),0,0,4);
    // cv::imshow("after_blur",after_blur);
    // contours
    vector<vector<Point> > contours;
    vector<vector<Point> > contours_final;
    vector<Vec4i> hier;
    namedWindow("cann",CV_WINDOW_AUTOSIZE);
    Canny(after_weighted,after_can,80,120,3);
    Mat kernel_d=getStructuringElement(MORPH_ELLIPSE,Size(3,3));
    dilate(after_can,after_can,kernel_d,Point(-1,-1),2);
    RemoveSmallRegion(after_can,after_filt,300,1,0);
    
    //Mat kernel_e=getStructuringElement(MORPH_ELLIPSE,Size(3,3));
    //erode(after_can,after_can,kernel_e,Point(-1,-1),1);
    
    imshow("cann",after_filt);
    findContours(after_filt,contours,hier,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
    contours_final=contours;
    //imshow("cann",);
    vector<Moments> mu(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<RotatedRect> box(contours.size());
    vector<RotatedRect> box_final(contours.size());
    vector<Point> min_rectangle;
    Point2f rect[4];
    double r_temp=0;
    double maxR=0;
    vector<Point2f> mc(contours.size());
    double temp_dist_ratio[contours.size()]={0};
    double temp_dist_1[contours.size()]={0};
    double temp_dist_2[contours.size()]={0};
    double score[contours.size()]={0};
    Point2f center[contours.size()];
    double temp_dist_1_final[contours.size()]={0};
    double temp_dist_2_final[contours.size()]={0};
    Point2f center_final[contours.size()];

    for(int i=0;i<contours.size();i++)
    {
        mu[i]=moments(contours[i],false);
        mc[i]=Point2d(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
        
        if(contours[i].size()<=5)
            continue;
        else
        {
            if(contourArea(contours[i])>5000&&contourArea(contours[i])<200)
                continue;
            else
            {

                boundRect[i] = boundingRect(Mat(contours[i]));
                double recOrNot=abs(boundRect[i].width-boundRect[i].height)/(0.5*(boundRect[i].width+boundRect[i].height));
                //box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
                //rectangle(final, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + 
                //boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);
                if(recOrNot>0.55)  // 1st rect check
                    continue;
                else  // boundingbox is rect
                {
                    box[i]=fitEllipse(Mat(contours[i]));
                    Point2f* vertices=new Point2f[4];
                    box[i].points(vertices);
                    temp_dist_1[i]=sqrtf(powf((vertices[0].x-vertices[1].x),2)+powf(vertices[0].y-vertices[1].y,2));
                    temp_dist_2[i]=sqrtf(powf((vertices[2].x-vertices[1].x),2)+powf(vertices[2].y-vertices[1].y,2));
                    temp_dist_ratio[i]=temp_dist_1[i]/temp_dist_2[i];
                    if(temp_dist_ratio[i]>1.4||temp_dist_ratio[i]<0.6)  // 2nd rect check
                        continue;
                    else  // rect for sure
                    {
                        //ellipse(result,box[i],Scalar(0,0,255),2,8);
                        count++;
                        double centerX=0.5*(vertices[0].x+vertices[2].x);
                        double centerY=0.5*(vertices[0].y+vertices[2].y);
                        center[i]=Point2f(centerX,centerY);
                        contours_final[count-1]=contours[i];
                        box_final[count-1]=box[i];
                        temp_dist_1_final[count-1]=temp_dist_1[i];
                        temp_dist_2_final[count-1]=temp_dist_2[i];
                        center_final[count-1]=center[i];
                    }
                }
            }
        }

    }
    // score items
    for(int i=0;i<count;i++)
    {
        if(contourArea(contours_final[i]))
            score[i]=0.8*contourArea(contours_final[i]);
        else
            score[i]=0;
    }
        
    // find max score
    double max=0;
    int idx=0;

    for(int i=0;i<count;i++)
    {
        if(max<score[i])
        {
            max=score[i];
            idx=i;
        }
    }

    // calc
    //radius=0.5*sqrtf(powf(temp_dist_1_final[idx],2)+powf(temp_dist_2_final[idx],2));
    //dist=f*r/radius*6*1.5;
    //x_contemp=center_final[idx].x;
    //y_contemp=center_final[idx].y;

    // for misdetect
    // compare
    Point2f point_temp=Point2f(x_contemp,y_contemp);
    Point2f point_delay=Point2f(x_delay,y_delay);
    if(getDistance(point_delay,point_temp)/frame.cols>0.4)
    {
        radius=radius_delay;
        x_contemp=x_delay;
        y_contemp=y_delay;
        dist=dist_delay;
    }
    else
    {
        radius=0.5*sqrtf(powf(temp_dist_1_final[idx],2)+powf(temp_dist_2_final[idx],2));
        x_contemp=center_final[idx].x;
        y_contemp=center_final[idx].y;
        dist=f*r/radius*6*1.5;

        // refresh
        x_delay=x_contemp;
        y_delay=y_contemp;
        radius_delay=radius;
        dist_delay=dist;
    }

    // plot
    ellipse(result,box_final[idx],Scalar(0,0,255),2);
    cv::circle(result,Point2f(x_contemp,y_contemp),2,Scalar(255,0,0),1,8);
    
    return result;
}



int main()
{
    Mat frame, final;
    Mat temp, ttemp;
    VideoCapture cap;
    cap.open(0);
    int cnt=100;
    //namedWindow("pre_detect",CV_WINDOW_AUTOSIZE);
    while (cnt>0)
    {
        cap>>ttemp;
        temp=ellipse_detect_pre(ttemp);
        //cv::imshow("pre_detect",temp);
        cnt--;
    }
    //destroyWindow("pre_detect");
    cout<<""<<endl;
    cout<<"begin detect!!!"<<endl;
    cout<<""<<endl;
    while(1)
    {
        cap>>frame;

        final=ellipse_detect(frame);
        // write distance
        stringstream distText;
        distText<<dist*0.8+dist_temp*0.2;
        string strDist=distText.str();
        string dist_text_stream="Distance: "+strDist+" cm";
        cv::putText(final,dist_text_stream,Point(50,60),CV_FONT_HERSHEY_TRIPLEX,0.6,(255,155,0),2,8);
        // write angle
        stringstream angleText1,angleText2;
        double xrange=frame.cols/2-x_contemp;
        double yrange=frame.rows/2-y_contemp;
        angleText1<<(atan((r*xrange/radius)/dist)*180/PI)*1.5*0.8+angle_temp_h*0.2;
        angleText2<<(atan((r*yrange/radius)/dist)*180/PI)*1.5*0.8+angle_temp_v*0.2;
        string strAngle1=angleText1.str();
        string strAngle2=angleText2.str();
        string angle_text_stream1="Angle_horizontal: "+strAngle1+" degree";
        string angle_text_stream2="Angle_vertical: "+strAngle2+" degree";
        cv::putText(final,angle_text_stream1,Point(50,80),CV_FONT_HERSHEY_TRIPLEX,0.6,(155,25,0),2,8);
        cv::putText(final,angle_text_stream2,Point(50,100),CV_FONT_HERSHEY_TRIPLEX,0.6,(155,25,0),2,8);
        // refresh data
        dist_temp=dist;
        angle_temp_h=atan((r*xrange/radius)/dist)*180/PI;
        angle_temp_v=atan((r*yrange/radius)/dist)*180/PI;
        cv::imshow("final_result",final);
        if(cvWaitKey(20)>0)
            break;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
