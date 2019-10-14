#include"imageprocess.h"
#include<cmath>


using namespace cv;
using namespace std;

int main20191014()
{
	Mat src = imread("threecircle.png");
	Mat dst = Mat(Size(src.rows, src.cols), CV_8UC3);
	vector <Mat> channels; split(src, channels);
	Mat Hvalue = channels.at(0); Mat Svalue = channels.at(1);
	Mat Ivalue = channels.at(2);
	for (int i = 0; i < src.rows; ++i)for (int j = 0; j < src.cols; ++j) {
		double H, S, I;
		int Bvalue = src.at<Vec3b>(i, j)(0);
		int Gvalue = src.at<Vec3b>(i, j)(1);
		int Rvalue = src.at<Vec3b>(i, j)(2);//求Theta =acos((((Rvalue - Gvalue) + (Rvalue - Bvalue)) / 2) / sqrt(pow((Rvalue - Gvalue), 2) + (Rvalue - Bvalue)*                           
											//(Gvalue - Bvalue)));
		double numerator = ((Rvalue - Gvalue) + (Rvalue - Bvalue)) / 2;
		double denominator = sqrt(pow((Rvalue - Gvalue), 2) + (Rvalue - Bvalue) * (Gvalue - Bvalue)); 
		if (denominator == 0) H = 0; else { double Theta = acos(numerator / denominator) * 180 / 3.14; if (Bvalue <= Gvalue)H = Theta; else  H = 360 - Theta; }Hvalue.at<uchar>(i, j) = (int)(H * 255 / 360); //为了显示将[0~360]映射到[0~255]  //求S = 1-3*min(Bvalue,Gvalue,Rvalue)/(Rvalue+Gvalue+Bvalue);
		int minvalue = Bvalue; if (minvalue > Gvalue) minvalue = Gvalue; if (minvalue > Rvalue) minvalue = Rvalue; numerator = 3 * minvalue; denominator = Rvalue + Gvalue + Bvalue; if (denominator == 0)  S = 0; else { S = 1 - numerator / denominator; }Svalue.at<uchar>(i, j) = (int)(S * 255);//为了显示将[0~1]映射到[0~255]
		I = (Rvalue + Gvalue + Bvalue) / 3;
		Ivalue.at<uchar>(i, j) = (int)(I);
	}
	merge(channels, dst);
	namedWindow("HSI");
	imshow("HSI", dst);
	imwrite("HSI空间图像.jpg", dst);
}



void BGR2HSI(const Mat& src, Mat& channelH, Mat& channelS, Mat& channelI) {

	Mat matBGR[3];

	split(src, matBGR);

	Mat channelB, channelG, channelR;

	matBGR[0].convertTo(channelB, CV_32FC1);

	matBGR[1].convertTo(channelG, CV_32FC1);

	matBGR[2].convertTo(channelR, CV_32FC1);



	Mat matMin, matSum;

	add(channelB, channelG, matSum); // R G B 之和

	add(matSum, channelR, matSum);

	divide(channelB, matSum, channelB);  // 求解 b g r

	divide(channelG, matSum, channelG);

	divide(channelR, matSum, channelR);



	// 计算饱和度 s

	channelS.create(src.rows, src.cols, CV_32FC1);

	min(channelB, channelG, matMin);

	min(matMin, channelR, matMin);

	subtract(Mat(src.rows, src.cols, CV_32FC1, Scalar(1)), matMin * 3, channelS);



	// 计算 h

	channelH.create(src.rows, src.cols, CV_32FC1);

	float* bData = channelB.ptr<float>(0);

	float* gData = channelG.ptr<float>(0);

	float* rData = channelR.ptr<float>(0);

	float* hData = channelH.ptr<float>(0);

	float r, g, b, temp;

	for (int i = 0; i < src.rows * src.cols; i++) {

		b = bData[i]; g = gData[i]; r = rData[i];



		// 单独处理 灰度图像

		if (b == g && b == r) {

			hData[i] = 0.0f;

			continue;

		}



		temp = 0.5 * ((r - g) + (r - b)) / sqrt((r - g) * (r - g) + (r - b) * (g - b));

		if (b <= g) {

			hData[i] = acos(temp);

		}
		else {

			hData[i] = 2 * 3.1415926 - acos(temp);

		}

	}



	// 计算强度 I

	divide(matSum, 3, channelI);

}



int main23414()
{

	Mat src = imread("threecircle.png", 1);
	auto img = src.clone();
	if (src.empty())
		cerr << "Error: Loading image" << endl;
	Mat hsi(src.rows, src.cols, src.type());
	//Mat hsvImg(img.rows, img.cols, img.type());
	//Mat hsvImgbgr(img.rows, img.cols, img.type());
	//cvtColor(img, hsvImg, cv::COLOR_RGB2HSV);
	//cvtColor(img, hsvImgbgr, cv::COLOR_BGR2HSV);
	float r, g, b, h, s, in;

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			b = static_cast<double>(src.at<Vec3b>(i, j)[0]) / 255;
			g = static_cast<double>(src.at<Vec3b>(i, j)[1]) / 255;
			r = static_cast<double>(src.at<Vec3b>(i, j)[2]) / 255;

			in = (b + g + r) / 3;

			int min_val = 0;
			min_val = std::min(r, std::min(b, g));

			s = 1 - 3 * (min_val / (b + g + r));
			if (s < 0.00001)
			{
				s = 0;
			}
			else if (s > 0.99999) {
				s = 1;
			}

			if (s != 0)
			{
				h = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g) * (r - g)) + ((r - b) * (g - b)));
				h = acos(h);

				if (b <= g)
				{
					h = h;
				}
				else {
					h = ((360 * 3.14159265) / 180.0) - h;
				}
			}

			hsi.at<Vec3b>(i, j)[0] = (h * 180) / 3.14159265;
			hsi.at<Vec3b>(i, j)[1] = s * 255;
			hsi.at<Vec3b>(i, j)[2] = in * 255;
		}
	}

	/*namedWindow("RGB image", CV_WINDOW_AUTOSIZE);
	namedWindow("HSI image", CV_WINDOW_AUTOSIZE);
*/
	imshow("RGB image", src);
	imshow("HSI image", hsi);
	vector <Mat> channels;
	split(hsi, channels);
	Mat Hvalue = channels.at(0);
	Mat Svalue = channels.at(1);
	Mat Ivalue = channels.at(2);
	waitKey(0);
	return 0;
}





//cv::cvtColor(dark, darkHSV, cv::COLOR_BGR2HSV);
//3chanel
int main()
{
	//testhsi();
	Mat src = imread("threecircle.png", 1);
	//Mat src = imread("splite.png", 1);
	Mat img = src.clone();
	//auto hsvImg = img.clone();
	if (src.empty())
		cerr << "Error: Loading image" << endl;
	Mat hsi(src.rows, src.cols, src.type());
	/*Mat hsvImg(img.rows, img.cols, img.type());
	Mat hsvImgbgr(img.rows, img.cols, img.type());
	cvtColor(img, hsvImg, cv::COLOR_RGB2HSV);
	cvtColor(img, hsvImgbgr, cv::COLOR_BGR2HSV);*/
	double r, g, b, hue, saturation, intensity;
	double up, down = 0.0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			b = static_cast<double>(src.at<Vec3b>(i, j)[0]) / 255;
			g = static_cast<double>(src.at<Vec3b>(i, j)[1]) / 255;
			r = static_cast<double>(src.at<Vec3b>(i, j)[2]) / 255;
			intensity = (b + g + r) / 3;

			saturation = 1 - 3 / (b + g + r) * min({ b,g,r });
			up = ((r - g) + (r - b)) / 2;
			down = pow(pow(r - g, 2.0) + (r - b) * (g - b), 0.5);
			hue = acos(up / down);
			if (b <= g)
			{
				hue = hue;
			}
			else {
				hue = ((360 * 3.14159265) / 180.0) - hue;
			}
			hsi.at<Vec3b>(i, j)[0] = (hue * 180) / 3.14159265;
			hsi.at<Vec3b>(i, j)[1] = saturation * 255;
			hsi.at<Vec3b>(i, j)[2] = intensity * 255;

		}
	}

	imshow("RGB image", src);
	imshow("HSI image", hsi);
	vector <Mat> channels;
	split(hsi, channels);
	Mat Hvalue = channels.at(0);
	Mat Svalue = channels.at(1);
	Mat Ivalue = channels.at(2);
	return 0;
}


int main1014()
{
	Mat src = imread("threecircle.png");
	Mat dst = Mat(Size(src.rows, src.cols), CV_8UC3);
	Mat hsv(src.rows, src.cols, src.type());
	cvtColor(src, hsv, COLOR_BGR2HSV);
	vector <Mat> channels;
	split(hsv, channels);
	Mat Hvalue = channels.at(0);
	Mat Svalue = channels.at(1);
	Mat Ivalue = channels.at(2);

	double r, g, b, hue, saturation, intensity;
	double up, down = 0.0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			b = src.at<Vec3b>(i, j)[0];
			g = src.at<Vec3b>(i, j)[1];
			r = src.at<Vec3b>(i, j)[2];
			intensity = (b + g + r) / 3;

			saturation = intensity - 3 / (b + g + r) * min({ b,g,r });
			up = ((r - g) + (r - b)) / 2;
			down = pow(pow(r - g, 2.0) + (r - b) * (g - b), 0.5);
			hue = acos(up / down);
			if (b <= g)
			{
				hue = hue;
			}
			else {
				hue = ((360 * 3.14159265) / 180.0) - hue;
			}
			Hvalue.at<uchar>(i, j) = (hue * 180) / 3.14159265;
			Svalue.at<uchar>(i, j) = saturation;
			Ivalue.at<uchar>(i, j) = intensity;

		}
	}

	imshow("RGB image", src);
	imshow("HSI image", Hvalue);
	imshow("HSI image", Svalue);
	imshow("HSI image", Ivalue);
	return 0;
}