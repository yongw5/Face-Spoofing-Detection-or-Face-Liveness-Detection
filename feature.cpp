#include "feature.h"

int CropFaceBasedOnLeftEye(const Mat& src_image, const vector<float>& src_landmarks,
                           const Size2f& offset_image, const Size2i& size_output,
                           Mat &output_image) {
  if (src_image.empty()) {
	cout << "the input image is empty" << endl;
	return FAILURE;
  }
  Point2f left_eye = Point2f(src_landmarks[0], src_landmarks[1]);
  Point2f right_eye = Point2f(src_landmarks[2], src_landmarks[3]);
  // calculate offsets in original image
  float offset_width = floor(float(offset_image.width * size_output.width));
  float offset_height = floor(float(offset_image.height * size_output.height));
  // get the direction
  Point2f eye_direction = right_eye - left_eye;
  // calculate the distance between two eyes
  float eye_distance = sqrt(pow(eye_direction.y, 2) + pow(eye_direction.x, 2));
  // calculate rotation angle in radians
  float rotation = atan2f(eye_direction.y, eye_direction.x);
  // calculate the reference eye_width
  float reference = size_output.width - 2 * offset_width;
  // scale factor
  float scale = eye_distance / reference;
  // transfrom the angle in radians to degrees
  double angle = rotation * 180 / CV_PI;
  // get the rotation matrix based on left eye
  Mat rotation_matrix = getRotationMatrix2D(left_eye, angle, 1.0);
  //rotate the image by the center of left eye
  Mat rotated_image;
  warpAffine(src_image, rotated_image, rotation_matrix, src_image.size());
  //crop the image to the target size
  Point2f crop_xy(left_eye.x - scale*offset_width, left_eye.y - scale*offset_height);
  Size2f crop_size(size_output.width*scale, size_output.height*scale);
  Rect crop_area;
  crop_area.x = crop_xy.x;
  crop_area.y = crop_xy.y;
  crop_area.width = crop_size.width;
  crop_area.height = crop_size.height;
  //crop area is in image area?
  Point2f right_down;
  right_down.x = crop_area.x + crop_area.width;
  right_down.y = crop_area.y + crop_area.height;
  if (crop_area.x < 0 || right_down.x >= rotated_image.cols || crop_area.y < 0 || right_down.y >= rotated_image.rows) {
	cout << "the crop area is beyong original image" << endl;
	return FAILURE;
  } else {
	output_image = rotated_image(crop_area);
	resize(output_image, output_image, kOutputImageSize);
	return SUCCESS;
  }
}
int SingalChannleImageDoG(const Mat& input_image, const Vec2d& vec_sigma, Mat& dog_image) {
  if (input_image.empty()) {
    cout << "the input image is empty"<<endl;
    return FAILURE;
  }
  if (input_image.channels() != 1) {
    cout << "the input image is not singal channal image"<<endl;
    return FAILURE;
  }
  Mat gaussian_image1, gaussian_image2;
  int size1, size2;
  // Filter Sizes
  size1 = 2 * (int)(3 * vec_sigma[0]) + 3;
  size2 = 2 * (int)(3 * vec_sigma[1]) + 3;
  // Gaussian Filter
  GaussianBlur(input_image, gaussian_image1, Size(size1, size1), vec_sigma[0], vec_sigma[0], BORDER_REPLICATE);
  GaussianBlur(input_image, gaussian_image2, Size(size2, size2), vec_sigma[1], vec_sigma[1], BORDER_REPLICATE);
  // Difference
  Mat difference_image = Mat::zeros(gaussian_image1.rows, gaussian_image1.cols, CV_32FC1);
  for (int i = 0; i < difference_image.rows; i++) {
    for (int j = 0; j < difference_image.cols; j++) {
      difference_image.at<float>(i, j) = (float)abs((gaussian_image1.at<uchar>(i, j) - gaussian_image2.at<uchar>(i, j)));
    }
  }
  difference_image.copyTo(dog_image);
  return SUCCESS;
}
int MultiChannalImageDoG(const Mat& input_image, const Vec2d& vec_sigma, vector<Mat>& vector_dog) {
  if (input_image.empty()) {
    cout << "the input image is empty"<<endl;
    return FAILURE;
  }
  vector<Mat> vector_channel;
  split(input_image, vector_channel);
  for (int i = 0; i < vector_channel.size(); ++i) {
    Mat dog_image;
    SingalChannleImageDoG(vector_channel[i],vec_sigma,dog_image);
    vector_dog.push_back(dog_image);
  }
  return SUCCESS;
}
int LBP(const vector<Mat>& vector_dog, vector<Mat>& vector_lbp){
  if (vector_dog.empty()) {
	cout << "the input vector of DoG is empty" << endl;
	return FAILURE;
  }
  if (vector_dog[0].channels() != 1) {
    cout << "the channel of DoG is larger than one"<<endl;
    return FAILURE;
  }
  for (int i = 0; i < vector_dog.size(); ++i) {
    Mat output_temp = Mat(vector_dog[i].rows - 2, vector_dog[i].cols - 2, CV_8UC1);
    float center_value = 0;
    uchar code = 0;
    for (int row = 1; row < vector_dog[i].rows - 1; ++row) {
      for (int col = 1; col < vector_dog[i].cols - 1; ++col) {
        center_value = vector_dog[i].at<float>(row, col);
        code = 0;
        code |= (vector_dog[i].at<float>(row - 1, col - 1) > center_value) << 7;
        code |= (vector_dog[i].at<float>(row - 1, col + 0) > center_value) << 6;
        code |= (vector_dog[i].at<float>(row - 1, col + 1) > center_value) << 5;
        code |= (vector_dog[i].at<float>(row + 0, col + 1) > center_value) << 4;
        code |= (vector_dog[i].at<float>(row + 1, col + 1) > center_value) << 3;
        code |= (vector_dog[i].at<float>(row + 1, col + 0) > center_value) << 2;
        code |= (vector_dog[i].at<float>(row + 1, col - 1) > center_value) << 1;
        code |= (vector_dog[i].at<float>(row + 0, col - 1) > center_value) << 0;
        output_temp.at<uchar>(row - 1, col - 1) = code;
      }
    }
    vector_lbp.push_back(output_temp);
  }
  return SUCCESS;
}
int LBP2Histogram(const vector<Mat>& vector_lbp, Mat& output_histogram) {
  if (vector_lbp.empty()) {
	cout << "the inout image is empty" << endl;
	return FAILURE;
  }
  Mat hist = Mat::zeros(1, 256 * vector_lbp.size(), CV_32FC1);
  int bin = 0;
  for (int i = 0; i < vector_lbp.size(); ++i) {
    for (int row = 0; row < vector_lbp[i].rows; ++row) {
      for (int col = 0; col < vector_lbp[i].cols; ++col) {
        bin = vector_lbp[i].at<uchar>(row, col);
        hist.at<float>(0, i * 256 + bin) += 1.0;
      }
    }
  }
  normalize(hist, hist, 0, 1, NORM_MINMAX, CV_32FC1);//nor sure whether it is right here
  hist.copyTo(output_histogram);
  return SUCCESS;
}
