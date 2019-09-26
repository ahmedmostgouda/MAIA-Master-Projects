#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif


#define SEGMENTATION_H
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;


class segmentation
{
private:
#define MAX_DATA_POINTS 1000
#define MAX_CLUSTER 100
#define FUZZINESS 2
	// ATRIBUTES
	// Parameters
	int histSize = 256;
	int num_clusters;
	double epsilon;
	bool histogram_in_parallel;
	bool membership_in_parallel;
	// Image
	int imgHeight;
	int imgWidth;
	// Variables
	int *hist;
	float **degree_of_memb;
	int *data_point;
	float *cluster_center;
	// METHODS
	// Interface with CUDA
	int *histogram2device(Mat &);
	float calculate_membership2device();
	// General methods
	int initialize(Mat &inputImageName);
	int calculate_centre_vectors();
	double get_norm(int i, int j);
	double get_new_value(int i, int j);
	double update_degree_of_membership();

public:
	segmentation(bool=false,bool=true,int=5);
	~segmentation();
	// Segmentation using Fuzzy C-Means
	int fcm(Mat &inputImage);
	// Generate Output image
	void genOutImage(Mat &outputImage, Mat &inputImage,float min_membership=0);
};


