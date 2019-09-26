#include "segmentation.cuh"
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

int main()
{
	//int level_num = 256;	

	Mat img = cv::imread("mri_brain_tumor.jpg");
	//Mat img = cv::imread("brain.jpeg");

	Mat greyMat;
	cvtColor(img, greyMat, COLOR_BGR2GRAY);



	//create an output image
	bool histogram_in_parallel=true;
	bool membership_in_parallel=true;
	int number_clusters = 5;

	segmentation NewSeg(histogram_in_parallel, membership_in_parallel,number_clusters);
	NewSeg.fcm(greyMat);

	Mat outImg = Mat::zeros(greyMat.rows, greyMat.cols, CV_8UC1);;
	NewSeg.genOutImage(outImg, greyMat);
	namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", outImg);

	imwrite("tumor_5_cluster.jpg", outImg);

	
	Mat outImg2 = Mat::zeros(greyMat.rows, greyMat.cols, CV_8UC1);;
	NewSeg.genOutImage(outImg2, greyMat,0.7);
	namedWindow("image_threshold", WINDOW_AUTOSIZE);
	imshow("image_threshold", outImg2);
	imwrite("tumor_3_cluster2.jpg", outImg2);
	
	waitKey(0);
	return 0;
}

