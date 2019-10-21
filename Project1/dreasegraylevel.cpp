#include"imageprocess.h"
using namespace std;
using namespace cv;
//降低灰度分辨率
int mainhomework1(int argc, char** argv)
{

	Mat image, _dst;
	//打开图像
	image = imread("lena_top.jpg", IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	namedWindow("orgin scale", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("orgin scale", image); // Show our image inside it.
	Mat outImg;
	cv::resize(image, outImg, cv::Size(), 2.45, 2.45);
	namedWindow("Display scale", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display scale", outImg); // Show our image inside it.
	_dst.create(image.size(), image.type());
	int RowsNum = image.rows;
	int nc = image.cols * image.channels();
	int ColsNum = image.cols;
	uchar* output = _dst.ptr<uchar>(0);
	int dstRownum = RowsNum / 4;
	for (int i = 0; i < RowsNum; i++)
	{

		if (i % 4 != 0)
			continue;
		output = _dst.ptr<uchar>(i / 4);  // 输出行
		auto pix = image.ptr<uchar>(i);
		//灰度分辨率降低四倍
		for (int j = 0; j < nc; j++)
		{
			if (j % 4 == 0)
				* output++ = pix[j];


		}
	}
	namedWindow("Display scale", WINDOW_AUTOSIZE);
	//显示图像
	imshow("01", _dst);
	//图片写入
	imwrite("gray_image1021.jpg", _dst);
	//imwrite("D:/1.jpg",);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}
