#include<opencv2/opencv.hpp>
#include<iostream>
#include "segmentation.cuh"
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <cstring>


using namespace std;
using namespace cv;

////////////////////// CUDA KERNELS ///////////////////////////////////////////
__constant__ int const_hist_d[256];

__global__ void device_hist(int *hist_d_out, const int *img_d_in, const int BIN_COUNT)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int myItem = img_d_in[y * gridDim.y + x];
	int myBin = myItem % BIN_COUNT;
	atomicAdd(&(hist_d_out[myBin]), 1);
}


__constant__ float sum_arr[256];
__global__ void device_membership_cost(const int BIN_COUNT, int CLUSTERS_NUMBER, float * membership_d, float * cost_d, float * randcenters_d)
{
	__shared__ float shared_cost_pixel;

	int x = blockIdx.x; //Histogram Bins
	int y = threadIdx.x; // Realtion btw center and bin
	float sum = 0;
	const int threath_position=x * blockDim.x  + y;
	// 1- Calculate membership
	float p = 2 / (FUZZINESS - 1);
	float numerator = fabs(float(x - randcenters_d[y]));
		if (numerator < 0.00001)
			membership_d[threath_position]= 1;
		else {
			for (int k = 0; k < CLUSTERS_NUMBER; k++) {
				float denominator = fabs(float(x - randcenters_d[k]));
				if (denominator < 0.00001) {
					denominator = 0.000001;
				}
				float t = numerator / denominator;
				t = pow(t, p);
				sum += t;
			}
			membership_d[threath_position] = 1.0 / sum;
		}

		// 2. Calculate distance
		if (y == 0) {
			sum_arr[x] = 0;
		}
		__syncthreads();

		shared_cost_pixel = pow(x - randcenters_d[y], 2) * membership_d[x * blockDim.x + y] * const_hist_d[x];
		atomicAdd(&(sum_arr[x]), shared_cost_pixel);
		if (y == 0) {

			for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
			{
				if (x < offset)
				{
					// add a partial sum upstream to our own
					sum_arr[x] += sum_arr[x + offset];
				}
				__syncthreads();
			}
			__syncthreads();
			cost_d[0] = sum_arr[0];
			
		}
}

////////////////////// METHODS OF SEGMENTATION --- INTERFACE WITH CUDA  ///////////////////////////////////////////
int *segmentation::histogram2device(Mat &inputImage) {
	//Create a dynamic array on CPU side (host) to and convert the Mat data type to it, check this link: http://www.trevorsimonton.com/blog/2016/11/16/transfer-2d-array-memory-to-cuda.html
	cout << imgHeight << "\n";
	cout << imgWidth << "\n";

	cout << histSize << "\n";
	
	int** img_h_in = new int*[imgHeight];
	img_h_in[0] = new int[imgHeight * imgWidth];
	for (int i = 1; i < imgHeight; ++i)
		img_h_in[i] = img_h_in[i - 1] + imgWidth;
	for (int i = 0; i < imgHeight; ++i) {
		for (int j = 0; j < imgWidth; ++j) {
			img_h_in[i][j] = (int)inputImage.at<uchar>(i, j);
		}
	}

	//Allocate a dynamic memory on GPU side (device) for image
	int *img_d_in;
	cudaMalloc((void **)&img_d_in, imgHeight * imgWidth * sizeof(int));

	//Allocate a dynamic memory on GPU side (device) for histogram
	int *hist_d_out;
	cudaMalloc((void **)&hist_d_out, histSize * sizeof(int));

	//Allocate a dynamic memory on CPU side (host) for histogram
	int *hist_h_out = new int[histSize];

	//Copy the image from the host to the device
	cudaMemcpy(img_d_in, *img_h_in, imgHeight * imgWidth * sizeof(int), cudaMemcpyHostToDevice);

	//Start the kernel function with imgWidth*imgHeight block and 1 threath
	dim3 hist_dimGrid(imgWidth, imgHeight, 1);
	dim3 hist_dimBlock(1, 1, 1);
	device_hist<< < hist_dimGrid, hist_dimBlock >> > (hist_d_out, img_d_in, histSize);

	//Copy the histogram result from the device to the host
	cudaMemcpy(hist_h_out, hist_d_out, histSize * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(hist_d_out);
	cudaFree(img_d_in);
	delete[] img_h_in;
	return hist_h_out;
}

float segmentation::calculate_membership2device() {
	// Allocate randcenter
	float *centers_d;
	cudaMalloc((void **)&centers_d, num_clusters * sizeof(float));
	// Send center to device
	cudaMemcpy(centers_d, cluster_center, num_clusters * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate membership
	float *membership_d;
	cudaMalloc((void **)&membership_d, num_clusters * histSize * sizeof(float));
	//cudaMemcpy(membership_d, *membership_h, ARRAY_IN_BYTES, cudaMemcpyHostToDevice);

	float *cost_d;
	cudaMalloc((void **)&cost_d, 1 * sizeof(float));

	float *cost_h = new float[1];	

	float** membership_h = new float*[histSize];
	membership_h[0] = new float[histSize  * num_clusters];
	for (int i = 1; i < histSize; ++i)
		membership_h[i] = membership_h[i - 1] + num_clusters;

	dim3 fuzzy_dimGrid(histSize, 1, 1);
	dim3 fuzzy_dimBlock(num_clusters, 1, 1);
	device_membership_cost << <fuzzy_dimGrid, fuzzy_dimBlock >> > (histSize, num_clusters, membership_d, cost_d, centers_d);

	// Output of Kernel, need to be copy in the Host
	cudaMemcpy(*membership_h, membership_d, num_clusters * histSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cost_h, cost_d, 1 * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < histSize; ++i) {
		for (int j = 0; j < num_clusters; ++j) {
			degree_of_memb[i][j] =membership_h[i][j];
		}
	}
	float final_cost = cost_h[0];

	//cout << "\n Host = " << final_cost << "\n";

	//Clear the declared dynamic memories
	delete[] cost_h;
	delete[] membership_h;

	cudaFree(centers_d);
	cudaFree(membership_d);
	cudaFree(cost_d);
	//cudaFree(img_d_in);
	return final_cost;
}



////////////////////// METHODS OF SEGMENTATION --- CONSTRUCTOR & DESTRUCTOR ///////////////////////////////////////////

segmentation::segmentation(bool in_histogram_in_parallel, bool in_membership_in_parallel, int in_num_clusters) {
	char answer;
	cout << "Do you want it to enter the values? (y/n): ";
	cin >> answer;
	if (answer == 'y') {
		cout << "\nDo you want to calculate the histogram in parallel? (y/n): ";
		char answer_parallel;
		cin >> answer_parallel;
		if (answer_parallel == 'y')
			histogram_in_parallel = true;
		else
			histogram_in_parallel = false;
		cout << "\nDo you want to calculate the histogram in parallel? (y=1/n=0): ";
		char answer_membership;
		cin >> answer_membership;
		if (answer_parallel == 'y')
			membership_in_parallel = true;
		else
			membership_in_parallel = false;
		cout << "\nEnter the number of clusters (take in consideration your GPU specifications): ";
		cin >> num_clusters;

		cout << "\nEnter the value of epsilon, termination criterion (between 0 and 1) : ";
		cin >> epsilon;
	}
	else {
		num_clusters = in_num_clusters;
		epsilon = 0.01;
		histogram_in_parallel = in_histogram_in_parallel;
		membership_in_parallel = in_membership_in_parallel;
	}
	
	cout << "\n \nStarting segmentation with FUZZY C-MEANS\n";
	if (histogram_in_parallel )
		cout << "Histogram  in Parallel \n";
	else
	{
		cout << "Histogram in Serial \n";
	}
	if (membership_in_parallel)
		cout << "Membership  in Parallel \n";
	else
	{
		cout << "Membership in Serial \n";
	}

	hist = new int[histSize];
	degree_of_memb = new float *[MAX_DATA_POINTS];
	for (int i = 0; i < MAX_DATA_POINTS; i++) {
		degree_of_memb[i] = new float[MAX_CLUSTER];
	}
	data_point = new int[MAX_DATA_POINTS];
	cluster_center = new float[MAX_CLUSTER];

	for (int i = 0; i < MAX_CLUSTER; i++) {
		cluster_center[i] = rand() % 10 *256; // Random Initialization 
	}
}

segmentation::~segmentation() {
	delete  hist;
	delete[] degree_of_memb;
	delete data_point;
	delete cluster_center;
}

////////////////////// METHODS OF SEGMENTATION --- GENERAL FUNCTIONS ///////////////////////////////////////////

int segmentation::initialize(Mat &inputImage) {
	// CALCULATION OF THE HISTOGRAM
	imgHeight = inputImage.rows;
	imgWidth = inputImage.cols;
	if (histogram_in_parallel) {
		hist = histogram2device(inputImage);
	}
	else {
		//imgData_type = typeToString(inputImage.type());
		for (int i = 0; i < histSize; i++) {
			hist[i] = 0;
		}
		for (int i = 0; i < inputImage.rows; i++) {
			for (int j = 0; j < inputImage.cols; j++) {
				hist[(int)inputImage.at<uchar>(i, j)] = hist[(int)inputImage.at<uchar>(i, j)] + 1;
			}
		}

	}
	//initializing the matrix (data_point) with intensity values that the histogram represent [0,255]
	for (int j = 0; j < histSize; j++) {
		data_point[j] = j;
	}
	int  r, rval;
	double s;
	for (int i = 0; i < histSize; i++) {
		s = 0.0;
		r = 100;
		for (int j = 1; j < num_clusters; j++) {
			rval = rand() % (r + 1);
			r -= rval;
			degree_of_memb[i][j] = rval / 100.0;
			s += degree_of_memb[i][j];
		}
		degree_of_memb[i][0] = 1.0 - s;
	}
	return 0;
}

// to initialize the matrix(cluster_center)
int segmentation::calculate_centre_vectors() {
	int i, j;
	double numerator, denominator;
	double t[MAX_DATA_POINTS][MAX_CLUSTER];
	for (i = 0; i < histSize; i++) {
		for (j = 0; j < num_clusters; j++) {
			t[i][j] = pow(degree_of_memb[i][j], FUZZINESS);
		}
	}
	for (j = 0; j < num_clusters; j++) {
		numerator = 0.0;
		denominator = 0.0;
		for (i = 0; i < histSize; i++) {
			numerator += t[i][j] * data_point[i] * hist[i]; //here we have to multiply by the num. of pixels of that intensity
			denominator += t[i][j] * hist[i]; 
		}
		cluster_center[j] = numerator / denominator;
	}
	return 0;
}

double segmentation::get_norm(int i, int j) {
	return  fabs(data_point[i] - cluster_center[j]);
}

// new value of membership of point i to class j
double segmentation::get_new_value(int i, int j) {
	int k;
	double t, p, sum;
	sum = 0.0;
	p = 2 / (FUZZINESS - 1);
	double numerator = get_norm(i, j) ;
	if (numerator < 0.00001)
		return 1;
	for (k = 0; k < num_clusters; k++) {
		t = numerator / get_norm(i, k);
		if (get_norm(i, k) < 0.00001)
			return 0;
		t = pow(t, p);
		sum += t;
	}
	return 1.0 / sum;
}

//updating each poit's membership to each class
double segmentation::update_degree_of_membership() {
	int i, j;
	double new_uij;
	double diff;
	double max_diff = 0.0;
	for (j = 0; j < num_clusters; j++) {
		for (i = 0; i < histSize; i++) {
			new_uij = get_new_value(i, j);
			diff = new_uij - degree_of_memb[i][j];
			if (diff > max_diff)
				max_diff = diff;
			degree_of_memb[i][j] = new_uij;
		}
	}
	return max_diff;
}

//generate the output image
void segmentation::genOutImage(Mat &outputImage, Mat &inputImage,float min_membership) {
	for (int i = 0; i < histSize; i++) {
		//find the cluster index of max. membership value
		double max_membership = 0.0;
		int max_memb_index = 0;
		for (int c = 0; c < num_clusters; c++) {
			if (degree_of_memb[i][c] > max_membership) {
				max_membership = degree_of_memb[i][c];
				max_memb_index = c;
			}
		}
		int new_intensity;
		if(max_membership<min_membership)
			new_intensity = 255;
		else
			new_intensity = cluster_center[max_memb_index];//the new pixel value in the output image
		for (int j = 0; j < imgHeight; j++) {
			for (int k = 0; k < imgWidth; k++) {
				if ((int)inputImage.at<uchar>(j, k) == data_point[i])
					outputImage.at<uchar>(j, k) = new_intensity;
			}
		}
	}
}

// Fuzzy C-Means
int segmentation::fcm(Mat &inputImage) {
	initialize(inputImage);
	double cost_old=0;
	double cost = 0;
	if (membership_in_parallel)
		// Send the histogram to the constant memory of CUDA
		cudaMemcpyToSymbol(const_hist_d, hist, 256 * sizeof(int));
	do {
		if (membership_in_parallel) {
			cost_old = cost;
			cost = calculate_membership2device();
		}
		else {
			cost = update_degree_of_membership();
		}
		calculate_centre_vectors();
	} while (abs(cost-cost_old) > epsilon);
	if (true) {
		cout << "\nFuzzy Segmentation Done, the clusters centers are: \n" ;
		for (int i = 0; i < num_clusters; i++)
		{
			printf("%f\n", cluster_center[i]);
		}
	}
	return 0;
}


