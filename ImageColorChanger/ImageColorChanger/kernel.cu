//Carlos Miguel Negrete
//A01208733

//Programming Languajes Final Project

//Dependencies:
//	- Opencv2 --Version 4.5.2
//	- Some images for testing. I provide a bounch of  images for usage.
//I used:
//	- Nvidia GeForcce GTX 960m compute_50,sm_50

#include <iostream>
#include <stdio.h>
#include "string"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#define N 32 //Threads per block

using namespace std;
using namespace cv;

__global__ void filter1(uint8_t* image, int width, int height, int Channels) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int ind = (x + y * gridDim.x * blockDim.x) * Channels;

	//Changes the color  from "Black" to something kind of red
	if ((image[ind + 0] <= 70) && (image[ind + 1] <= 70) && (image[ind + 2] <= 70)) {
		image[ind + 2] = image[ind + 2] + 80;
	}
	
}

//Adds vertical black lines to the image, if it is PNG it adds a different pattern since the number of channels. 
__global__ void filter2(uint8_t* image, int width, int height, int Channels) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int ind = (x + y * gridDim.x * blockDim.x) * Channels;

	if (Channels < 4) {
		if ((ind % 2) == 0) {
			for (int i = 0; i < Channels; i++) {
				image[ind + i] = 0;
			}
		}
	}
	else {
		if ((ind % 3) == 0) {
			for (int i = 0; i < Channels; i++) {
				image[ind + i] = 0;
			}
		}
	}
	
}

//Set tranparenecy from white color. 
__global__ void filter3(uint8_t* image, int width, int height, int Channels) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int ind = (x + y * gridDim.x * blockDim.x) * Channels;

	if ((image[ind + 0] >= 91) && (image[ind + 1] >= 91) && (image[ind + 2] >= 91)) {
		image[ind + 3] = 0;
	}

}

int main() {
	//List actual directory
	system("dir"); //For Windows users
	//system("ls"); //Uncomment this for Unix based SO's

	//Timestamp purposes
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



	//Instructions for the user
	cout << endl;
	cout << "------------- Welcome to my Image Color and More changer program.-------------" << endl << "- Please write the name of the image you want to transform" << endl << "- (include filename extension), you have your actual directory above. -" << endl;
	
	//ASking for the original Image
	//string imageName;

	//cin >> imageName;

	//Variables creation
	//Mat image = imread(imageName, IMREAD_UNCHANGED);

	string imageName = "img4v3.jpg";
	cout << imageName << endl;
	Mat image = imread(imageName, IMREAD_UNCHANGED);

	uint8_t* d_image, *d_image2, *d_image3;
	int pixelNumValues = image.step;
	int h = image.rows;
	int w = image.cols;

	//Displaying some windows for preview the images.
	namedWindow(imageName + " ORIGINAL", WINDOW_NORMAL);
	resizeWindow(imageName + " ORIGINAL", 400, 400);
	imshow(imageName + " ORIGINAL", image);

	//Creating the blocks and the threads.
	dim3 blocks(h / N, w / N);
	dim3 threads(N, N);
	
	//Memory allocation in Device and copy of info for filter 1
	cudaMalloc((void**)&d_image, sizeof(uint8_t) * (pixelNumValues * h));
	cudaMemcpy(d_image, image.data, sizeof(uint8_t) * (pixelNumValues * h), cudaMemcpyHostToDevice);

	//Memory allocation in Device and copy of info for filter 2
	cudaMalloc((uint8_t**)&d_image2, sizeof(uint8_t) * (pixelNumValues * h));
	cudaMemcpy(d_image2, image.data, sizeof(uint8_t) * (pixelNumValues * h), cudaMemcpyHostToDevice);

	//Memory allocation in Device and copy of info for filter 3 only if it is a PNG image
	if (image.channels() == 4) {
		cudaMalloc((uint8_t**)&d_image3, sizeof(uint8_t) * (pixelNumValues * h));
		cudaMemcpy(d_image3, image.data, sizeof(uint8_t) * (pixelNumValues * h), cudaMemcpyHostToDevice);
	}
	

	//Used only to show the number of channels (4 if it has tranparency capacity or 3 if normal image)
	//cout << image.channels();

	//filter1 changes blackish colours to redish tones. 
	cudaEventRecord(start, 0);
	filter1 << <blocks, threads >> > (d_image, w, h, image.channels());
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaMemcpy(image.data, d_image, sizeof(uint8_t) * (pixelNumValues * h), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Time elapsed for filter 1: " << milliseconds << " ms" << endl;
	cudaFree(d_image);

	namedWindow(imageName + "RED", WINDOW_NORMAL);
	resizeWindow(imageName + "RED", 400, 400);
	imshow(imageName + "RED", image);
	imwrite(imageName + "RED.png", image);

	//Filter 2 adds black pattern
	cudaEventRecord(start, 0);
	filter2 << <blocks, threads >> > (d_image2, w, h, image.channels());
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaMemcpy(image.data, d_image2, sizeof(uint8_t) * (pixelNumValues * h), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float milliseconds2 = 0;
	cudaEventElapsedTime(&milliseconds2, start, stop);
	cout << "Time elapsed for filter 2: " << milliseconds2 << " ms" << endl;
	cudaFree(d_image2);

	namedWindow(imageName + "LINES", WINDOW_NORMAL);
	resizeWindow(imageName + "LINES", 400, 400);
	imshow(imageName + "LINES", image);
	imwrite(imageName + "LINES.png", image);

	//Filter 3 only if image is a png
	if (image.channels() == 4) {
		cudaEventRecord(start, 0);
		filter3 << <blocks, threads >> > (d_image3, w, h, image.channels());
		cudaEventRecord(stop, 0);
		cudaDeviceSynchronize();
		cudaMemcpy(image.data, d_image3, sizeof(uint8_t) * (pixelNumValues * h), cudaMemcpyDeviceToHost);
		cudaEventSynchronize(stop);
		float milliseconds3 = 0;
		cudaEventElapsedTime(&milliseconds3, start, stop);
		cout << "Time elapsed for filter 3: " << milliseconds3 << " ms" << endl;
		cudaFree(d_image3);

		namedWindow(imageName + "TRANPARENCY", WINDOW_NORMAL);
		resizeWindow(imageName + "TRANPARENCY", 400, 400);
		imshow(imageName + "TRANPARENCY", image);
		imwrite(imageName + "TRANPARENCY.png", image);
	}

	waitKey(0);

	return 0;
}