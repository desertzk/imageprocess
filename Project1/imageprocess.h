#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"

#include "opencv2/highgui/highgui.hpp"
// C++ Standard Libraries
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath> 
#include <numeric>
#include<algorithm>

void addnoise(cv::Mat& image);
bool AddGaussianNoise(const cv::Mat mSrc, cv::Mat& mDst, double Mean = 0.0, double StdDev = 10.0);
bool AddGaussianNoise_Opencv(const cv::Mat mSrc, cv::Mat& mDst, double Mean = 0.0, double StdDev = 10.0);
/*
	laplace算子
	 0    -1    0
	 -1    4    -1
	 0    -1    0
 */
void sharpen(const cv::Mat& image, cv::Mat& result);
//椒盐噪声
void addnoise(cv::Mat& image, int noisevalue);