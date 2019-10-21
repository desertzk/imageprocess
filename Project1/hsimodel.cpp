#include"imageprocess.h"
#include<cmath>


using namespace cv;
using namespace std;

//提取一副彩色图像中红色，用HIS模型处理(HSI分割)

#define PI 3.14159265

//may error
Mat HSIToRGB(const Mat& src)
{
	Mat hsi(src.rows, src.cols, src.type());
	double up, down = 0.0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			double r = 0, g = 0, b = 0;
			int huepix = src.at<Vec3b>(i, j)[0];
			double huedegree = static_cast<double>(huepix) / 255 * 360;
			double hue = huedegree / 360;
			int saturationpix = src.at<Vec3b>(i, j)[1];
			double saturation = static_cast<double>(saturationpix) / 255;
			int intensitypix = src.at<Vec3b>(i, j)[2];
			double intensity = static_cast<double>(intensitypix) / 255;
			if (0 <= huedegree && huedegree < 120)
			{
				b = intensity*(1- saturation);
				
				r = intensity*(1+ saturation*cos(hue)/cos(PI/3-hue));
				g = 3 * intensity - (r + b);
			}
			else if (120 <= huedegree && huedegree < 240)
			{
				hue = (huedegree - 120) / 360;
				r= intensity * (1 - saturation);
				g = intensity * (1 + saturation * cos(hue) / cos(PI / 3 - hue));
				b = 3 * intensity - (r + g);
			}
			else if (240 <= huedegree && huedegree < 360)
			{
				hue = (huedegree - 240) / 360;
				g = intensity * (1 - saturation);
				b = intensity * (1 + saturation * cos(hue) / cos(PI / 3 - hue));
				r= 3 * intensity - (g + b);
			}
			
		

			hsi.at<Vec3b>(i, j)[0] = b* 255;
			hsi.at<Vec3b>(i, j)[1] = g * 255;
			hsi.at<Vec3b>(i, j)[2] = r * 255;

		}
	}

	return hsi;
}


//cv::cvtColor(dark, darkHSV, cv::COLOR_BGR2HSV);
//3chanel
int main1021()
{
	//testhsi();
	Mat src = imread("cartoon.png", 1);
	//Mat src = imread("splite.png", 1);
	
	//auto hsvImg = img.clone();
	if (src.empty())
		cerr << "Error: Loading image" << endl;
	Mat hsi(src.rows, src.cols, src.type());
	/*Mat hsvImg(img.rows, img.cols, img.type());
	Mat hsvImgbgr(img.rows, img.cols, img.type());
	cvtColor(img, hsvImg, cv::COLOR_RGB2HSV);
	cvtColor(img, hsvImgbgr, cv::COLOR_BGR2HSV);*/
	
	double up, down = 0.0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			double r=0, g=0, b=0, hue=0, saturation=0, intensity=0;
			int blue = src.at<Vec3b>(i, j)[0];
			int green = src.at<Vec3b>(i, j)[1];
			int red = src.at<Vec3b>(i, j)[2];
			b = static_cast<double>(blue) / 255;
			g = static_cast<double>(green) / 255;
			r = static_cast<double>(red) / 255;
			intensity = (b + g + r) / 3;

			saturation = 1 - 3 / (b + g + r) * min({ b,g,r });
			up = ((r - g) + (r - b)) / 2;
			down = pow((r-g)*(r - g) + (r - b) * (g - b), 0.5);
			hue = acos(up / down);
			if (b <= g)
			{
				hue = hue;
			}
			else {
				hue = ((360 * PI) / 180.0) - hue;
			}
			
			//当saturation为0时对应的是无色彩的中心点，此时hue没有意义也定义为0
			if (saturation == 0)
				hue = 0;

			double degree = (hue * 180) / PI;

			hsi.at<Vec3b>(i, j)[0] = degree/360*255;
			hsi.at<Vec3b>(i, j)[1] = saturation * 255;
			hsi.at<Vec3b>(i, j)[2] = intensity * 255;

		}
	}

	imshow("RGB image", src);
	imshow("HSI image", hsi);
	vector <Mat> channels;
	split(hsi, channels);
	//hue 色调图
	Mat Hvalue = channels.at(0);
	//saturation饱和度图
	Mat Svalue = channels.at(1);
	//intensity亮度
	Mat Ivalue = channels.at(2);
	Mat srcSvalue = Svalue.clone();
	//srcSvalue二值图 二值饱和度模板（以最大饱和度的10%为门限）
	for (int i = 0; i < srcSvalue.rows; i++)
	{
		auto row = srcSvalue.ptr<uchar>(i);
		for (int j = 0; j < srcSvalue.cols; j++)
		{
			auto pix = row[j];
			if(pix>25)
				row[j] = 255;
			else
				row[j] = 0;
		}
	}
	//色调图和srcSvalue二值图相乘
	Mat mutiSvalue = srcSvalue.clone();
	for (int i = 0; i < mutiSvalue.rows; i++)
	{
		auto row = mutiSvalue.ptr<uchar>(i);
		auto Hrow= Hvalue.ptr<uchar>(i);
		for (int j = 0; j < mutiSvalue.cols; j++)
		{
			auto pix = row[j];
			if (pix == 255)
				row[j] = 1* Hrow[j];
			else
				row[j] = 0* Hrow[j];
		}
	}

	int Channels[] = { 0 };
	int nHistSize[] = { 256 };
	float range[] = { 0, 256 };
	const float* fHistRanges[] = { range };
	Mat histnum, histpr, hissumpr;
	//cv标准库算出来的就好比我算出来的 grayscalemap
	calcHist(&mutiSvalue, 1, Channels, Mat(), histnum, 1, nHistSize, fHistRanges, true, false);
	//cv标准库算出来的就好比我算出来的 pr 对应概率
	normalize(histnum, histpr, 1.0, 0.0, NORM_L1);
	
	//取出大于90%的像素
	double threshold = 0.0;
	int thresholdpix = 0;
	for (int i = 0; i < histpr.rows; i++)
	{
		auto row = histpr.ptr<float>(i);
		
		threshold += row[0];
		if (threshold > 0.9)
		{
			thresholdpix = i;
			break;
		}
		
	}

	vector<pair<int, int>> notred;
	//红色分量的分割
	Mat retvalue = mutiSvalue.clone();
	for (int i = 0; i < retvalue.rows; i++)
	{
		auto row = retvalue.ptr<uchar>(i);
		for (int j = 0; j < retvalue.cols; j++)
		{
			auto curpix = row[j];
			
			if (curpix < thresholdpix)
			{
				row[j] = 0;
				notred.push_back(make_pair(i, j));
			}
		}
	}
	Mat nowhsi = src.clone();
	for (auto index : notred)
	{
		int i = index.first;
		int j = index.second;
		src.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
	}

	//Mat nowhsi;
	//channels[0] = retvalue;
	//merge(channels, nowhsi);
	//
	////hsi反运算转RGB
	//Mat retrgb = HSIToRGB(nowhsi);
	return 0;
}




