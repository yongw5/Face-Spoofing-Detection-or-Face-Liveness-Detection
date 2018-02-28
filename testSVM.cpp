#include"feature.h"
#include<iostream>
#include<fstream>
#include<time.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;
void TestSVM(const string& image_list, const string& src_folder, const Ptr<SVM>& svm,
			 int& attack_num, int& real_num,
			 int& false_reject_num, int& false_accept_num);
const vector<Vec2d> vector_sigma = { Vec2d(0.5, 1), Vec2d(1, 2), Vec2d(0.5, 2) };
int main() {
  const string src_folder  = "X:/SpoofData/MSU_MFSD/cropped_face_224/scene01";
  const string test_image_list = src_folder + "/test_image_list.txt";
  int total_test_num = 0, attack_num = 0, real_num = 0, false_reject_num = 0, false_accept_num = 0;
  Ptr<SVM> svm = StatModel::load<SVM>("color_texture_svm.xml");
  //test image
  TestSVM(test_image_list, src_folder, svm,
		  attack_num,real_num, false_reject_num,false_accept_num);
  //cout the result
  float accuracy = float(attack_num + real_num - false_accept_num - false_reject_num) / (attack_num + real_num);
  float false_reject_rate = float(false_reject_num) / real_num;
  float false_accept_rate = float(false_accept_num) / attack_num;
  cout << "the test result: " << endl;
  cout << "the accuracy: " << accuracy << endl;
  cout << "the false reject rate: " << false_reject_rate << endl;
  cout << "the false accept rate: " << false_accept_rate << endl;
  // write the reuslt in txt 
  time_t tt = time(NULL);
  struct tm local_time;
  localtime_s(&local_time, &tt);
  ofstream fid("test_result.txt", ios::app);
  fid
	<< "***testing date: "
	<< local_time.tm_year + 1900 << "-" << local_time.tm_mon + 1 << "-" << local_time.tm_mday << " "
	<< local_time.tm_hour << ":" << local_time.tm_min << ":" << local_time.tm_sec << endl
	<< "the test result: " << endl
	<< "num of test samples: " << attack_num + real_num << endl
	<< "false reject num: " << false_reject_num << endl
	<< "false accept num: " << false_accept_num << endl
    << "the accuracy: " << accuracy << endl
    << "the false reject rate: " << false_reject_rate << endl
    << "the false accept rate: " << false_accept_rate << endl;
  fid.close();
  return 0;
}

void TestSVM(const string& image_list, const string& src_folder, const Ptr<SVM>& svm,
			 int& attack_num, int& real_num, 
			 int& false_reject_num, int& false_accept_num) {
  ifstream fid_image_list;
  ofstream fid_error_result("predict_error_list.txt");
  string line, image_path;
  Mat read_image, hist;
  int response_svm;
  vector<string>result;
  result.push_back("attack");
  result.push_back("real");
  fid_image_list.open(image_list);
  if (fid_image_list.is_open()) {
	while (!fid_image_list.eof()) {
	  fid_image_list >> line;
	  image_path = src_folder + '/' + line;
	  try {
		read_image = imread(image_path, IMREAD_COLOR);
		if (!read_image.empty()) {
		  normalize(read_image, read_image, 0, 255, NORM_MINMAX, CV_8UC3);
          Mat hsv_image;
          cvtColor(read_image, hsv_image, COLOR_RGB2HSV);
          vector<Mat> vector_dog;
          for (int i = 0; i < vector_sigma.size(); ++i)
            MultiChannalImageDoG(hsv_image, vector_sigma[i], vector_dog);
          Mat ycbcr_image;
          cvtColor(read_image, ycbcr_image, COLOR_RGB2YCrCb);
          for (int i = 0; i<vector_sigma.size(); ++i)
            MultiChannalImageDoG(ycbcr_image, vector_sigma[i], vector_dog);
          vector<Mat> vector_lbp;
          LBP(vector_dog, vector_lbp);
          Mat hist;
          LBP2Histogram(vector_lbp, hist);
		  response_svm = svm->predict(hist);
		  cout << "testing......"
			   << line.substr(line.rfind('\\') + 1, line.length()) 
			   << "--svm_predict->" << result[int((response_svm + 1)/2)] << endl;
		  if (line.find("attack") != string::npos) {
			++attack_num;
            if (response_svm == 1) {
              ++false_accept_num;
              fid_error_result << line.substr(line.rfind('\\') + 1, line.length())
                               << "--svm_predict->" << result[int((response_svm + 1) / 2)] << endl;
            }//(response_svm == 1)
		  } else if (line.find("real") != string::npos) {
			++real_num;
            if (response_svm == -1) {
              ++false_reject_num;
              fid_error_result << line.substr(line.rfind('\\') + 1, line.length())
                << "--svm_predict->" << result[int((response_svm + 1) / 2)] << endl;
            }//if (response_svm == -1)
		  }
		}
	  } catch (cv::Exception& e) {
		cerr << e.msg << endl;
	  }
	}//while
  } else {
	cout << "cannot open the file:" << image_list << endl;
  }
  fid_image_list.close();
}
