#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
//#include "Filters.cu"


using namespace std;
using namespace cv;

extern "C" bool Filter_wrapper_blur(const Mat& input,Mat& output);
extern "C" bool Filter_wrapper_sharpen(const Mat& input, Mat& output);
extern "C" bool Filter_wrapper_emboss(const Mat& input, Mat& output);
int main() {
	string image_name = "lena";
	string input_file = image_name + ".png";
	string output_blur_file = image_name + "_blur.png";
	string output_sharpen_file = image_name + "_sharpen.png";
	string output_emboss_file = image_name + "_emboss.png";

	Mat srcImage = imread(input_file, 1);
	cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

	Mat dstImage(srcImage.size(), srcImage.type());
	Mat dstImage2(srcImage.size(), srcImage.type());
	Mat dstImage3(srcImage.size(), srcImage.type());

	Filter_wrapper_blur(srcImage, dstImage);
	Filter_wrapper_sharpen(srcImage, dstImage2);
	Filter_wrapper_emboss(srcImage, dstImage3);
	imwrite(output_blur_file, dstImage);
	imwrite(output_sharpen_file, dstImage2);
	imwrite(output_emboss_file, dstImage3);

	return 0;
}