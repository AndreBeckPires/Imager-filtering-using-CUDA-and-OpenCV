#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>



#define BLOCK_SIZE      16
#define FILTER_WIDTH    3     
#define FILTER_HEIGHT   3      

using namespace std;
using namespace cv;

__global__ void blurFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float kernel[FILTER_WIDTH][FILTER_HEIGHT] = { 0.0, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0 };

	if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
	{
		for (int c = 0; c < channel; c++)
		{
		
			float sum = 0;
		
			for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
				for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
					float fl = srcImage[((y + ky) * width + (x + kx)) * channel + c];
					sum += fl * kernel[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
				}
			}
			dstImage[(y * width + x) * channel + c] = sum;
		}
	}
}

__global__ void embossFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float kernel[FILTER_WIDTH][FILTER_HEIGHT] = { -1, -1, 0, -1, 0, 1, 0.0, 1, 1 };
	
	if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
	{
		for (int c = 0; c < channel; c++)
		{
		
			float sum = 0;
	
			for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
				for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
					float fl = srcImage[((y + ky) * width + (x + kx)) * channel + c];
					sum += fl * kernel[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
				}
			}
			dstImage[(y * width + x) * channel + c] = sum + 128;
		}
	}
}


__global__ void sharpenFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float kernel[FILTER_WIDTH][FILTER_HEIGHT] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
	
	if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
	{
		for (int c = 0; c < channel; c++)
		{
	
			float sum = 0;
			
			for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
				for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
					float fl = srcImage[((y + ky) * width + (x + kx)) * channel + c];
					sum += fl * kernel[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
				}
			}
			dstImage[(y * width + x) * channel + c] = sum;
		}
	}
}

extern "C" void Filter_wrapper_blur(const Mat& input, Mat& output)
{


	
	int channel = input.step / input.cols;

   //seta o numero de bytes usando o cols e rows
	const int inputSize = input.cols * input.rows * channel;
	const int outputSize = output.cols * output.rows * channel;
	unsigned char* d_input, * d_output;

	//aloca memoria na grpu
	cudaMalloc<unsigned char>(&d_input, inputSize);
	cudaMalloc<unsigned char>(&d_output, outputSize);

	// passa a memoria pra gpu
	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

	//setta o grid da imegem
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);



	// roda o filtro
	blurFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);


 // memoria de volta pro host
	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	//limpa
	cudaFree(d_input);
	cudaFree(d_output);

}
extern "C" void Filter_wrapper_sharpen(const Mat& input, Mat& output)
{


	
	int channel = input.step / input.cols;

	
	const int inputSize = input.cols * input.rows * channel;
	const int outputSize = output.cols * output.rows * channel;
	unsigned char* d_input, * d_output;

	cudaMalloc<unsigned char>(&d_input, inputSize);
	cudaMalloc<unsigned char>(&d_output, outputSize);


	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);


	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);



 
	sharpenFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);




	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

}

extern "C" void Filter_wrapper_emboss(const Mat& input, Mat& output)
{


	int channel = input.step / input.cols;


	const int inputSize = input.cols * input.rows * channel;
	const int outputSize = output.cols * output.rows * channel;
	unsigned char* d_input, * d_output;


	cudaMalloc<unsigned char>(&d_input, inputSize);
	cudaMalloc<unsigned char>(&d_output, outputSize);


	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);


	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);



 
	embossFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);




	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);


	cudaFree(d_input);
	cudaFree(d_output);

}



