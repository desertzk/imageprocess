#include"imageprocess.h"
#include<cmath>


using namespace cv;
using namespace std;






//cv::cvtColor(dark, darkHSV, cv::COLOR_BGR2HSV);
//3chanel
int main()
{
	//testhsi();
	Mat src = imread("space_splite.png", 1);
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
				hue = ((360 * 3.14159265) / 180.0) - hue;
			}
			
			//当saturation为0时对应的是无色彩的中心点，此时hue没有意义也定义为0
			if (saturation == 0)
				hue = 0;

			double degree = (hue * 180) / 3.14159265;

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
				
			}
		}
	}

	return 0;
}


