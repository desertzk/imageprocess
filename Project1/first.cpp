#include"imageprocess.h"
using namespace std;
using namespace cv;


inline uchar Clamp(int n)
{
	n = n > 255 ? 255 : n;
	return n < 0 ? 0 : n;
}
//高斯噪声
bool AddGaussianNoise(const Mat mSrc, Mat& mDst, double Mean, double StdDev)
{
	if (mSrc.empty())
	{
		cout << "[Error]! Input Image Empty!";
		return 0;
	}

	Mat mGaussian_noise = Mat(mSrc.size(), CV_16SC3);
	randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));

	for (int Rows = 0; Rows < mSrc.rows; Rows++)
	{
		for (int Cols = 0; Cols < mSrc.cols; Cols++)
		{
			Vec3b Source_Pixel = mSrc.at<Vec3b>(Rows, Cols);
			Vec3b& Des_Pixel = mDst.at<Vec3b>(Rows, Cols);
			Vec3s Noise_Pixel = mGaussian_noise.at<Vec3s>(Rows, Cols);

			for (int i = 0; i < 3; i++)
			{
				int Dest_Pixel = Source_Pixel.val[i] + Noise_Pixel.val[i];
				Des_Pixel.val[i] = Clamp(Dest_Pixel);
			}
		}
	}

	return true;
}

bool AddGaussianNoise_Opencv(const Mat mSrc, Mat& mDst, double Mean, double StdDev)
{
	if (mSrc.empty())
	{
		cout << "[Error]! Input Image Empty!";
		return 0;
	}
	Mat mSrc_16SC;
	Mat mGaussian_noise = Mat(mSrc.size(), CV_16SC3);
	randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));

	mSrc.convertTo(mSrc_16SC, CV_16SC3);
	addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
	mSrc_16SC.convertTo(mDst, mSrc.type());

	return true;
}


//几何均值滤波
void GeoAverFliter(const Mat& src, Mat& dst) {
	Mat _dst(src.size(), CV_32FC1);
	double power = 1.0 / 9;
	cout << "power:" << power << endl;
	double geo = 1;
	int step = src.channels();
	if (src.channels() == 1) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols* step; j+= step) {
				if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
					if (src.at<uchar>(i, j) != 0) geo = geo * src.at<uchar>(i, j);
					if (src.at<uchar>(i + 1, j + 1) != 0) geo = geo * src.at<uchar>(i + 1, j + 1);
					if (src.at<uchar>(i + 1, j) != 0) geo = geo * src.at<uchar>(i + 1, j);
					if (src.at<uchar>(i, j + 1) != 0) geo = geo * src.at<uchar>(i, j + 1);
					if (src.at<uchar>(i + 1, j - 1) != 0) geo = geo * src.at<uchar>(i + 1, j - 1);
					if (src.at<uchar>(i - 1, j + 1) != 0) geo = geo * src.at<uchar>(i - 1, j + 1);
					if (src.at<uchar>(i - 1, j) != 0) geo = geo * src.at<uchar>(i - 1, j);
					if (src.at<uchar>(i, j - 1) != 0) geo = geo * src.at<uchar>(i, j - 1);
					if (src.at<uchar>(i - 1, j - 1) != 0) geo = geo * src.at<uchar>(i - 1, j - 1);
					/*geo = src.at<uchar>(i, j)* src.at<uchar>(i + 1, j + 1)* src.at<uchar>(i + 1, j)* src.at<uchar>(i, j + 1)*
						src.at<uchar>(i + 1, j - 1)* src.at<uchar>(i - 1, j + 1)* src.at<uchar>(i - 1, j)*
						src.at<uchar>(i, j - 1)* src.at<uchar>(i - 1, j - 1);*/
					_dst.at<float>(i, j) = pow(geo, power);
					geo = 1;
					//if (i % 10 == 0&&j%10==0)
						//printf("_dst.at<float>(%d, %d)=%f\n", i, j, _dst.at<float>(i, j));


				}
				else
					_dst.at<float>(i, j) = src.at<uchar>(i, j);
			}
		}
	}
	_dst.convertTo(dst, CV_8UC1);

	//_dst.copyTo(dst);//拷贝
	imshow("geoAverFilter", dst);
}

//
int main20191005()
{
	//Mat srcImage = imread("lena_top.jpg", 0);//灰度
	Mat mSource = imread("lena_top.jpg", 1);
	imshow("Source Image", mSource);

	Mat mColorNoise(mSource.size(), mSource.type());
	Mat dstNoise(mSource.size(), mSource.type());

	//AddGaussianNoise(mSource, dstNoise, 0, 10.0);

	//imshow("Source + Color Noise", mColorNoise);


	AddGaussianNoise_Opencv(mSource, mColorNoise, 0, 10.0);//I recommend to use this way!

	imshow("Source + Color Noise OpenCV", mColorNoise);

	Mat geofliter;
	GeoAverFliter(mColorNoise, geofliter);
	waitKey();
	return(0);
}



//主函数
int mainfourierfreq(void)
{
	//读取原始图像
	Mat srcImage = imread("lena_top.jpg", 0);//灰度
	imshow("original", srcImage);

	//将输入图像延扩至最佳尺寸，边界用0填充
	int m = getOptimalDFTSize(srcImage.rows);
	int n = getOptimalDFTSize(srcImage.cols);

	Mat padded;//定义填充后的图像
	copyMakeBorder(srcImage, padded, 0, m - srcImage.rows, 0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));

	imshow("padded image", padded);



	//为傅里叶变换的结果分配存储空间
	Mat planes[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) };
	Mat complexI;//将planes数组组合合并成一个多通道的数组complexI
	merge(planes, 2, complexI);

	//进行离散傅里叶变换
	dft(complexI, complexI);

	//将复数转换为幅值magitude
	split(complexI, planes);//将多通道数组complexI分离为几个单通道数组,planes[0]=Re(DFT(I),planes[1]=Im(DFT(I)));re是实数，Im是复数吧

	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];



	//进行对数尺度缩放
	magnitudeImage += Scalar::all(1);
	log(magnitudeImage, magnitudeImage);//求自然对数

	//剪切和重分布幅度图象限（若有奇数行或奇数列，进行频谱裁剪）
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));

	//重新排列傅里叶图像中的象限，使得原点位于图像中心
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	Mat q0(magnitudeImage, Rect(0, 0, cx, cy));//ROI区域的左上
	Mat q1(magnitudeImage, Rect(cx, 0, cx, cy));//ROI区域的右上
	Mat q2(magnitudeImage, Rect(0, cy, cx, cy));//ROI区域的左下
	Mat q3(magnitudeImage, Rect(cx, cy, cx, cy));//ROI区域的左上

	Mat tmp;//交换象限(左上与右下进行交换)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);


	q1.copyTo(tmp);//右上与左下进行交换
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//归一化，用0到1之间的浮点值，将矩阵变幻为可视的图像格式

	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
	imshow("MAGNITUDE", magnitudeImage);

	waitKey(0); // Wait for a keystroke in the window
	return(0);



}


//傅立叶变换，显示频谱fourierTransform
int mainfstackoverflow()
{
	// Load an image
	cv::Mat inputImage = cv::imread("lena_top.jpg", IMREAD_GRAYSCALE);

	// Go float
	cv::Mat fImage;
	inputImage.convertTo(fImage, CV_32F);

	// FFT
	std::cout << "Direct transform...\n";
	cv::Mat fourierTransform;
	cv::dft(fImage, fourierTransform, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);

	// Some processing
	//doSomethingWithTheSpectrum();

	// IFFT
	std::cout << "Inverse transform...\n";
	cv::Mat inverseTransform;
	cv::dft(fourierTransform, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

	// Back to 8-bits
	cv::Mat finalImage;
	inverseTransform.convertTo(finalImage, CV_8U);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}







/*
	laplace算子
	 0    -1    0
	 -1    4    -1
	 0    -1    0
 */
void sharpen(const Mat& image, Mat& result)
{
	//CV_Assert(image.depth() == CV_8U);

	result.create(image.size(), image.type());

	const int channels = image.channels();
	for (int j = 1; j < image.rows - 1; j++) {
		const uchar* previous = image.ptr<const uchar>(j - 1); // 当前行的上一行
		const uchar* current = image.ptr<const uchar>(j); //当前行
		const uchar* next = image.ptr<const uchar>(j + 1); //当前行的下一行

		uchar* output = result.ptr<uchar>(j);  // 输出行
		for (int i = channels; i < channels * (image.cols - 1); i++) {
			auto pix = 4 * current[i] - previous[i] - next[i] - current[i - channels] - current[i + channels];
			*output++ = saturate_cast<uchar>((255- pix));

		}

	}

	//对图像边界进行处理
	 //边界像素设置为0
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}









//add noise 锐化
int mainsharp(int argc, char** argv)
{

	Mat image;
	image = imread("lena_top.jpg", IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image); // Show our image inside it.
	
	Mat result;
	sharpen(image, result);

	cv::imwrite("shape.jpg", result);
	//imwrite("D:/1.jpg",);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}


/*

	 0    1    0
	 1    4    1
	 0    1    0
 */
void midvalue(const Mat& image, Mat& result)
{
	//CV_Assert(image.depth() == CV_8U);

	result.create(image.size(), image.type());

	const int channels = image.channels();
	for (int j = 1; j < image.rows - 1; j++) {
		const uchar* previous = image.ptr<const uchar>(j - 1); // 当前行的上一行
		const uchar* current = image.ptr<const uchar>(j); //当前行
		const uchar* next = image.ptr<const uchar>(j + 1); //当前行的下一行

		uchar* output = result.ptr<uchar>(j);  // 输出行


		for (int i = channels; i < channels * (image.cols - 1); i++) {
			vector<uchar> fivepix{ current[i] ,previous[i] ,next[i] ,current[i - channels] ,current[i + channels] };
			sort(fivepix.begin(), fivepix.end());
			*output++ = saturate_cast<uchar>(fivepix[2]);

		}

	}

	//对图像边界进行处理
	 //边界像素设置为0
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}


//椒盐噪声
void addnoise(Mat &image,int noisevalue)
{
	int RowsNum = image.rows;
	int nc = image.cols * image.channels();
	int ColsNum = image.cols;
	vector<pair<int, int>> vnoise;
	for (int i = 0; i < 300; ++i)
	{
		int x = rand() % RowsNum;
		int y = rand() % nc;
		//vnoise.push_back(make_pair(x,y));
		auto pix = image.ptr<uchar>(x);
		pix[y] = noisevalue;
	}
}

//加椒盐噪声然后平滑
int mainaddnoisemid(int argc, char** argv)
{

	Mat image;
	image = imread("lena_top.jpg", IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	//增加椒盐噪声
	addnoise(image,255);
	Mat result;
	midvalue(image, result);

	cv::imwrite("lena_top_noise.jpg", image);
	//imwrite("D:/1.jpg",);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}


//直方图均衡化   equalizeHist
int mainequalizeHist(int argc, char** argv)
{

	Mat image, dst, uselibdst;

	image = imread("handlelow.png", IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	dst.rows = image.rows;
	dst.cols = image.cols;
	dst.flags = image.flags;
	cv::resize(dst, dst, image.size());

	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image); // Show our image inside it.
	int RowsNum = image.rows;
	int nc = image.cols * image.channels();
	int ColsNum = image.cols;
	map<int, int> grayscalemap;
	//直方图均衡化
	//equalizeHist(image, uselibdst);
	for (int i = 0; i < RowsNum; i++)
	{
		auto pix = image.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			auto data = pix[j];
			//（结果最大值-结果最小值）/（源最大值-源最小值）*（当前像素值-源最小）+结果最小值
			grayscalemap[data]++;


		}
	}
	int sum = 0;
	for (auto item : grayscalemap)
	{
		sum += item.second;

	}
	double totalpix = RowsNum * nc;
	//灰度所占的比例
	map<int, double> pr;
	for (auto it = grayscalemap.begin(); it != grayscalemap.end(); ++it)
	{
		double probability = it->second / totalpix;
		pr[it->first] = probability;
	}

	
	//sk
	map<int, double> sk;
	auto itnext = ++pr.begin();
	auto it = pr.begin();
	for (; itnext != pr.end(); )
	{
		const double result =
			accumulate(pr.begin(), itnext, 0.0,
				[](const double previous, const std::pair<int, double>& p)
				{
					double next = p.second;
					return previous + next;
				});
		sk[it->first] = result;
		itnext++;
		it++;
	}
	auto penult = sk.crbegin()->second;
	sk.insert(make_pair(it->first, it->second + penult));

	int Channels[] = { 0 };
	int nHistSize[] = { 256 };
	float range[] = { 0, 256 };
	const float* fHistRanges[] = { range };
	Mat histnum, histpr, hissumpr;
	//cv标准库算出来的就好比我算出来的 grayscalemap
	calcHist(&image, 1, Channels, Mat(), histnum, 1, nHistSize, fHistRanges, true, false);
	//cv标准库算出来的就好比我算出来的 pr 对应概率
	normalize(histnum, histpr, 1.0, 0.0, NORM_L1);
	

	//这个不是现有像素的最小值最大值 ，而是你要变化的范围
	int maxpix = 255;//pr.rbegin()->first;
	int minpix = 0;//pr.begin()->first;
	//像素映射表
	map<int, int> pixmap;
	for (auto it = sk.begin(); it != sk.end(); ++it)
	{
		pixmap[it->first] = (maxpix - minpix) * it->second + 0.5;
	}


	for (int i = 0; i < RowsNum; i++)
	{
		auto pix = image.ptr<uchar>(i);
		auto pixdst = dst.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			auto data = pix[j];
			pixdst[j] = pixmap[data];

		}
	}

	//cv::imwrite("gray_image.jpg", gray_image);
	//imwrite("D:/1.jpg",);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}




//homework2
//对比度拉伸  图片增强技术Contrast_stretch
int mainhomework2(int argc, char** argv)
{

	Mat imageorgin;
	imageorgin = imread("handlelow.png", IMREAD_COLOR); // Read the file
	if (imageorgin.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	Mat image=imageorgin.clone();
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image); // Show our image inside it.
	int RowsNum = image.rows;
	int nc = image.cols * image.channels();
	int ColsNum = image.cols;

	int maxpix = 0;
	int minpix = 255;
	for (int i = 0; i < RowsNum; i++)
	{
		auto pix = image.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			auto data = pix[j];
			if (data > maxpix)
				maxpix = data;

			if (data < minpix)
				minpix = data;

		}
	}



	for (int i = 0; i < RowsNum; i++)
	{
		auto pix = image.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			auto data = pix[j];
			//（结果最大值-结果最小值）/（源最大值-源最小值）*（当前像素值-源最小）+结果最小值
			auto newpix = (255 - 0) / (maxpix - minpix) * (data - minpix) + 0;

			pix[j] = newpix;
		}
	}


	//cv::imwrite("gray_image.jpg", gray_image);
	//imwrite("D:/1.jpg",);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}







//make it lighter
int main3(int argc, char** argv)
{

	Mat image;
	image = imread("low.png", IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image); // Show our image inside it.
	int RowsNum = image.rows;
	int nc = image.cols * image.channels();
	int ColsNum = image.cols;
	for (int i = 0; i < RowsNum; i++)
	{
		auto pix = image.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			auto data = pix[j];
			auto newdata = 0.98 * pow(data, 1.05);
			pix[j] = newdata;
			//cout << "row:" << RowsNum << "col:" << ColsNum << "pix:" << (int)pix[j];
			//c为遍历图像的三个通道			
			//for (int c = 0; c < 3; c++)			
			//{				
			//	//使用at操作符，防止越界				
			//	dstImg.at<Vec3b>(i, j)[c] = saturate_cast<uchar>					
			//		(k* (srcImg.at<Vec3b>(i, j)[c]) + b); 			
			//}		
		}
	}



	//cv::imwrite("gray_image.jpg", gray_image);
	//imwrite("D:/1.jpg",);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}









int main2(int argc, char** argv)
{

	Mat image;
	image = imread("gray01.jpg", IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image); // Show our image inside it.
	int RowsNum = image.rows;
	int nc = image.cols * image.channels();
	int ColsNum = image.cols;
	for (int i = 0; i < RowsNum; i++)
	{
		auto pix = image.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			auto data = pix[j];
			auto newdata = 38 * log(data);
			pix[j] = newdata;
			//cout << "row:" << RowsNum << "col:" << ColsNum << "pix:" << (int)pix[j];
			//c为遍历图像的三个通道			
			//for (int c = 0; c < 3; c++)			
			//{				
			//	//使用at操作符，防止越界				
			//	dstImg.at<Vec3b>(i, j)[c] = saturate_cast<uchar>					
			//		(k* (srcImg.at<Vec3b>(i, j)[c]) + b); 			
			//}		
		}
	}



	//cv::imwrite("gray_image.jpg", gray_image);
	//imwrite("D:/1.jpg",);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}



int main1(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: " << argv[0] << " ImageToLoadAndDisplay" << endl;
		return -1;
	}
	Mat image;
	image = imread(argv[1], IMREAD_COLOR); // Read the file
	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image); // Show our image inside it.

	Mat outImg;
	cv::resize(image, outImg, cv::Size(), 2.45, 2.45);
	namedWindow("Display scale", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display scale", outImg); // Show our image inside it.

	Mat gray_image;
	cvtColor(image, gray_image, COLOR_BGR2GRAY);
	int RowsNum = gray_image.rows;

	int ColsNum = gray_image.cols;
	for (int i = 0; i < RowsNum; i++)
	{
		for (int j = 0; j < ColsNum; j++)
		{

			//c为遍历图像的三个通道			
			//for (int c = 0; c < 3; c++)			
			//{				
			//	//使用at操作符，防止越界				
			//	dstImg.at<Vec3b>(i, j)[c] = saturate_cast<uchar>					
			//		(k* (srcImg.at<Vec3b>(i, j)[c]) + b); 			
			//}		
		}
	}





	namedWindow("Display gray_image", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display gray_image", gray_image); // Show our image inside it.

	cv::imwrite("gray_image.jpg", gray_image);
	//imwrite("D:/1.jpg",);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}


 