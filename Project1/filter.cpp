#include"imageprocess.h"

using namespace cv;
using namespace std;

//5.对一副图像加噪，进行几何均值，算术均值，谐波，逆谐波处理
class ImageRecovery {
private:
	//累加/rows*cols
	double filter_aver(Mat src)
	{
		//算术均值滤波
		double sum = 0;
		for (int i = 0; i < src.rows; i++) {
			uchar* data = src.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++) {
				sum += double(data[j]);
			}
		}
		return sum / double(src.cols * src.rows);
	}
	double filter_geo(Mat src)
	{
		//几何均值滤波
		double geo = 1;
		for (int i = 0; i < src.rows; i++) {
			uchar* data = src.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++) {
				if (data[j] != 0) geo *= data[j];
			}
		}
		double power = 1.0 / double(src.cols * src.rows);
		return pow(geo, power);
	}
	double filter_har(Mat src)
	{
		//谐波滤波
		double har = 0;
		for (int i = 0; i < src.rows; i++) {
			uchar* data = src.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++) {
				if (data[j] != 0) har += 1 / (double)(data[j]);
			}
		}
		return (src.cols * src.rows) / har;
	}
	//逆谐波
	double filter_inversehar(Mat src)
	{
		//谐波滤波
		double harq1 = 0;
		double harq = 0;
		for (int i = 0; i < src.rows; i++) {
			uchar* data = src.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++) {
				if (data[j] != 0) harq1 += pow((double)data[j],2.0);
				if (data[j] != 0) harq += pow((double)data[j], 1.0);
			}
		}
		return harq1 / harq;
	}

	void BubbleSort(float* pData, int count)
	{
		//冒泡排序，用于中值滤波
		float tData;
		for (int i = 1; i < count; i++) {
			for (int j = count - 1; j >= i; j--) {
				if (pData[j] < pData[j - 1]) {
					tData = pData[j - 1];
					pData[j - 1] = pData[j];
					pData[j] = tData;
				}
			}
		}
	}
	double filter_median(Mat src)
	{
		//中值滤波
		int index = 0;
		int bubble_len = (src.cols) * (src.rows);
		float* bubble = new float[bubble_len];
		for (int i = 0; i < src.rows; i++) {
			uchar* data = src.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++) {
				bubble[index] = data[j];
				index++;
			}
		}
		BubbleSort(bubble, bubble_len);
		double median = bubble[bubble_len / 2];
		return median;
	}
public:
	void salt(Mat& image, float salt_ratio,int saltval)
	{
		//salt_ratio为加入椒盐噪声的比例
		int n = image.rows * image.cols * salt_ratio;
		for (int k = 0; k < n; k++)
		{
			int i = rand() % image.cols;
			int j = rand() % image.rows;
			
				image.at<Vec3b>(j, i)[0] = saltval;
				image.at<Vec3b>(j, i)[1] = saltval;
				image.at<Vec3b>(j, i)[2] = saltval;
			
		}
	}
	Mat filter(Mat image, string filter_type, Size size)
	{
		//image为输入待滤波图像，filter_tpye为滤波器类型，size为滤波器的尺寸
		Mat result;
		image.copyTo(result);
		Mat channel[3];
		split(image, channel);
		int l = (size.height - 1) / 2;
		int w = (size.width - 1) / 2;
		for (int i = l; i < result.rows - l; i++) {
			for (int j = w; j < result.cols - w; j++) {
				for (int ii = 0; ii < 3; ii++) {
					if (filter_type == "aver")    result.at<Vec3b>(i, j)[ii] = saturate_cast<uchar>(filter_aver(channel[ii](Rect(j - w, i - l, size.width, size.height))));
					if (filter_type == "geo")    result.at<Vec3b>(i, j)[ii] = saturate_cast<uchar>(filter_geo(channel[ii](Rect(j - w, i - l, size.width, size.height))));
					if (filter_type == "har")    result.at<Vec3b>(i, j)[ii] = saturate_cast<uchar>(filter_har(channel[ii](Rect(j - w, i - l, size.width, size.height))));
					if (filter_type == "ivshar") result.at<Vec3b>(i, j)[ii] = saturate_cast<uchar>(filter_inversehar(channel[ii](Rect(j - w, i - l, size.width, size.height))));
					if (filter_type == "median") result.at<Vec3b>(i, j)[ii] = saturate_cast<uchar>(filter_median(channel[ii](Rect(j - w, i - l, size.width, size.height))));
				}
			}
		}
		return result;
	}
};



int main1007()
{
	Mat mSource = imread("lena_top.jpg", 1);
	imshow("Source Image", mSource);

	Mat mColorNoise(mSource.size(), mSource.type());
	mSource.copyTo(mColorNoise);
	Mat dstNoise(mSource.size(), mSource.type());
	//高斯噪声
	//AddGaussianNoise(mSource, dstNoise, 0, 10.0);

	//imshow("Source + Color Noise", mColorNoise);
	//addnoise(mSource, 255);

	//AddGaussianNoise_Opencv(mSource, mColorNoise, 0, 10.0);//I recommend to use this way!

	imshow("Source + Color Noise OpenCV", mColorNoise);
	//初始化IR类
	ImageRecovery IR;
	//加入椒盐噪声
	IR.salt(mColorNoise, 0.1,0);
	//imshow("salt", img);
	//对噪声图片进行滤波
	Mat resultaver = IR.filter(mColorNoise, "aver", Size(3, 3));
	Mat result = IR.filter(mColorNoise, "geo", Size(3, 3));
	Mat resulthar = IR.filter(mColorNoise, "har", Size(3, 3));
	//逆谐波对于消除胡椒噪声（充满黑点） 效果很好 但是不能消除盐噪声
	Mat resultivshar = IR.filter(mColorNoise, "ivshar", Size(3, 3));
	imshow("result", result);
	waitKey();
	return 0;
}