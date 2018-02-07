#include"feature.h"
#include<iostream>
#include<fstream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

void CreateFeature(const string& image_list, const string& src_folder,
                   vector<Mat>& hist_temp, vector<Mat>& label_temp);
int VectorOfMat2Mat(vector<Mat> vector_mat, Mat& output_mat);
const vector<Vec2d> vector_sigma = { Vec2d(0.5, 1), Vec2d(1, 2), Vec2d(0.5,2)};
int main() {
  const string src_folder = "F:/VS2013_Project/DOG_LBP_SVM";

  const string train_image_list = src_folder + "/train_image_list.txt";
  vector<Mat> hist_temp;
  vector<Mat> label_temp;
  CreateFeature(train_image_list, src_folder, hist_temp, label_temp);
  // create SVM Data
  Mat train_feature, train_label;
  if (VectorOfMat2Mat(hist_temp, train_feature)) {
	cout << "cannot convert hist_temp to train feature" << endl;
  }
  if (VectorOfMat2Mat(label_temp, train_label)) {
	cout << "cannot convert label_temp to train label" << endl;
  }
  //save feature and label
  FileStorage fs;
  fs.open("train_feature.xml", FileStorage::WRITE);
  fs << "TrainFeature" << train_feature;
  fs.release();
  fs.open("train_label.xml", FileStorage::WRITE);
  fs << "TrainLabel" << train_label;
  fs.release();
  SVM::Params params;
  params.svmType = SVM::C_SVC;
  params.kernelType = SVM::LINEAR;
  params.C = 10;
  params.termCrit = TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);
  cout << "training SVM......" << endl;
  Ptr<SVM> svm;
  try {
	svm = StatModel::train<SVM>(train_feature, ROW_SAMPLE, train_label, params);//train SVM
  cout << "training finished......" << endl;
  svm->save("color_texture_svm.xml");
  } catch (cv::Exception& e) {
	cout << e.msg;
  }
  return 0;
}

void CreateFeature(const string& image_list, const string& src_folder,
				   vector<Mat>& hist_temp, vector<Mat>& label_temp) {
  
  ifstream fid_image_list;
  string line, image_path;
  Mat read_image;
  fid_image_list.open(image_list);
  if (fid_image_list.is_open()) {
	while (!fid_image_list.eof()) {
	  fid_image_list >> line;
	  image_path = src_folder + '/' + line;
	  cout << "reading......" << line.substr(line.rfind('\\') + 1, line.length()) << endl;
	  try {
		read_image = imread(image_path, IMREAD_COLOR);
		if (!read_image.empty()) {
		  normalize(read_image, read_image, 0, 255, NORM_MINMAX, CV_8UC3);
          Mat hsv_image;
          cvtColor(read_image, hsv_image, COLOR_RGB2HSV);
          vector<Mat> vector_dog;
          for (int i = 0; i < vector_sigma.size(); ++i)
            MultiChannalImageDoG(hsv_image, vector_sigma[i],vector_dog);
          Mat ycbcr_image;
          cvtColor(read_image, ycbcr_image, COLOR_RGB2YCrCb);
          for (int i = 0; i<vector_sigma.size();++i)
            MultiChannalImageDoG(ycbcr_image, vector_sigma[i],vector_dog);
          vector<Mat> vector_lbp;
          LBP(vector_dog, vector_lbp);
		  Mat hist;
		  LBP2Histogram(vector_lbp, hist);
		  hist_temp.push_back(hist);
		  if (line.find("attack") != string::npos) {
			label_temp.push_back(Mat(1, 1, CV_32SC1, Scalar_<int>(-1)));//attack is labeled with -1
		  } else if (line.find("real") != string::npos) {
			label_temp.push_back(Mat(1, 1, CV_32SC1, Scalar_<int>(1)));// real is labeled with 1
		  }
		}//if (!read_image.empty())
	  } catch (cv::Exception& e) {
		cerr << e.msg << endl;
	  }
	}//while
  } else {
	cout << "cannot open the file:" << image_list << endl;
  }
  fid_image_list.close();
}
int VectorOfMat2Mat(vector<Mat> vector_mat, Mat& output_mat) {
  if (vector_mat.empty()) {
	cout << "the input vector of mat is empty" << endl;
	return 1;
  }
  Mat temp(vector_mat.size(), vector_mat[0].cols, vector_mat[0].type());
  for (int i = 0; i < vector_mat.size(); ++i) {
	vector_mat[i].copyTo(temp.row(i));
  }
  temp.copyTo(output_mat);
  return 0;
}
