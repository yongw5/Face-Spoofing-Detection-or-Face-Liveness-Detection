#ifndef FEATURE_HPP_
#define FEATURE_HPP_

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

const int SUCCESS = 0;
const int FAILURE = 1;
const Size2f kOffsetImage(0.25, 0.25);
const Size2i kOutputImageSize(128, 128);

int CropFaceBasedOnLeftEye(const Mat& src_image, const vector<float>& src_landmarks,
                           const Size2f& offset_image, const Size2i& size_output);
int SingalChannleImageDoG(const Mat& input_image, const Vec2d& vec_sigma, Mat& dog_image);
int MultiChannalImageDoG(const Mat& input_image, const Vec2d& vec_sigma, vector<Mat>& vector_dog);
int LBP(const vector<Mat>& vector_dog, vector<Mat>& vector_lbp);
int LBP2Histogram(const vector<Mat>& vector_lbp, Mat& output_histogram);
#endif
