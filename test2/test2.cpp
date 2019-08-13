#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include <boost/system/config.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/svm_threaded.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/matrix.h>
//#include <dlib/matrix/matrix_utilities.h>

namespace fs = boost::filesystem;

#include "lbp.hpp"
#include "histogram.hpp" 

#include <opencv/cv.h>
#include<opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
// #include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

#define totalTrainingImages 327
#define totalLBPHistogram 1652 
#define totalGeometricFeatures 6
#define FACE_DOWNSAMPLE_RATIO 4
#define SKIP_FRAMES 4

typedef dlib::matrix<double, totalGeometricFeatures, 1> sample_geometry;
typedef dlib::matrix<double, totalLBPHistogram, 1> sample_lbp;
typedef dlib::matrix<double, totalLBPHistogram + totalGeometricFeatures, 1> sample_concatenate;

std::vector<double> mouthareas;
std::vector<double> mouthperimeters;
std::vector<double> lefteyeareas;
std::vector<double> lefteyeperimeters;
std::vector<double> righteyeareas;
std::vector<double> righteyeperimeters;

std::vector<string> labelpaths;
std::vector<Mat> lbp_histogram;

void open_webcam();
string find_emotion(dlib::full_object_detection);

void cal_GeometricFeatures(sample_geometry& sample_g);
void cal_LBPFeatures(sample_geometry& sample_l, cv::Mat image, dlib::rectangle rect, int lbp_operator = 0);
void cal_ConFeatures(sample_concatenate& sample_c, dlib::full_object_detection shape, cv::Mat image, dlib::rectangle rect);
string classify_LBP(cv::Mat image, dlib::rectangle rect);
string classify_Concatenated(dlib::full_object_detection shape, cv::Mat image, dlib::rectangle rect);

void train_SVM_LBPClassifier(std::vector<sample_lbp> samples, std::vector<string>& labels);
void generate_LBP_histogram(int lbp_operator = 0);
void readSampleLBP(std::vector<sample_lbp>& samples, std::vector<string>& labels);
void extractFaceROI();
cv::Mat convertLBPHistogramToSingleUniform(cv::Mat histogram, int numCells, int numPatterns);
cv::Mat convertLBPHistogramToSpatialUniform(cv::Mat histogram, int numCells, int numPatterns);

void findGeometricalFeatures();
void generate_GeometricSamples(std::vector<sample_geometry>& samples, std::vector<string>& labels);
void concatenate_Features(std::vector<sample_concatenate>& samples_c, std::vector<sample_geometry> samples_g, std::vector<sample_lbp> samples_l);

void train_SVM_ConcatenatedClassifier(std::vector<sample_concatenate> samples_c, std::vector<string>& labels);
void train_SVM_GeometricClassifier(std::vector<sample_geometry> samples_g, std::vector<string>& labels);

void checkFiles();
void generate_data(std::vector<sample_geometry>& samples, std::vector<string>& labels);
void classify();

//#define FACE_DETECTION // enable this to get the face ROI from training images
//#define LBP // enable this to generate LBP histogram from training images
//#define TRAIN_GEO // enable this to train Geometric-based feature SVM classifier
//#define TRAIN_LBP // enable this to train LBP-based feature SVM classifier
//#define TRAIN_CON // enable this to train Concatenated feature SVM classifier
#define TEST // enable this to do emotion recognition in live webcam

int main(int argc, const char *argv[]) 
{
	try
	{
		#ifdef FACE_DETECTION
			extractFaceROI();
		#endif // FACE_DETECTION

		#ifdef LBP
			generate_LBP_histogram();
		#endif // LBP

		#ifdef TRAIN_LBP
			std::vector<sample_lbp> samples_l;
			std::vector<string> labels;
			readSampleLBP(samples_l, labels);
			train_SVM_LBPClassifier(samples_l, labels);
		#endif

		#ifdef TRAIN_GEO
			std::vector<sample_geometry> samples_g;
			std::vector<string> labels2;
			findGeometricalFeatures();
			generate_GeometricSamples(samples_g, labels2);
			train_SVM_GeometricClassifier(samples_g, labels2);
		#endif

		#ifdef TRAIN_CON
			std::vector<sample_lbp> samples_l2;
			std::vector<sample_geometry> samples_g2;
			std::vector<string> labels3,labels4;
			findGeometricalFeatures();
			generate_GeometricSamples(samples_g2, labels3);
			readSampleLBP(samples_l2, labels4);
			std::vector<sample_concatenate> samples_c;
			concatenate_Features(samples_c, samples_g2, samples_l2);
			train_SVM_ConcatenatedClassifier(samples_c, labels3);
		#endif

		#ifdef TEST
			open_webcam();
		#endif // TEST

	}
	catch (exception& e)
	{
		std::cout << "\nexception thrown!" << endl;
		std::cout << e.what() << endl;
	}

	return 0; // success
}

void open_webcam()
{
	cv::VideoCapture cap(0); // 0 = webcam
	cv::Mat im;
	cv::Mat im_small, im_display;

	// get face detector
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	std::cout << "detector detected" << endl;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	std::cout << "deserialized" << endl;

	int count = 0;
	std::vector<dlib::rectangle> faces;

	string window = "window";
	namedWindow("window", CV_WINDOW_AUTOSIZE);

	if (!cap.isOpened()) {
		std::cout << "fail" << endl;
		return;
	}

	while (true) {
		count++;
		// Grab a frame
		cap.read(im);
		cout << im.size().height << endl;
		// Resize image for face detection
		cv::resize(im, im_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO); // resize to make detection faster

		// Change to dlib's image format. No memory is copied.
		dlib::cv_image<dlib::bgr_pixel> cimg_small(im_small);
		dlib::cv_image<dlib::bgr_pixel> cimg(im);

		// Detect faces on resize image
		if (count % SKIP_FRAMES == 0) // skip some frames to make detection faster
		{
			faces = detector(cimg);
		}

		// Find the pose of each face.
		std::vector<dlib::full_object_detection> shapes;
		if (faces.size() > 0) {
			for (unsigned long i = 0; i < faces.size(); ++i)
			{
				std::cout << "iteration " << i << endl;
				// Resize obtained rectangle for full resolution image.
				dlib::rectangle r(
					(long)(faces[i].left()),
					(long)(faces[i].top()),
					(long)(faces[i].right()),
					(long)(faces[i].bottom())
				);

				// draw rectangle on face
				int width = faces[i].width();
				int heigth = faces[i].height();
				int x = faces[i].left();
				int y = faces[i].top();
				Rect rec(x, y, width, heigth);
				cv::rectangle(im, rec, Scalar(0, 255, 0), 2);

				// Landmark detection on full sized image
				dlib::full_object_detection shape = pose_model(cimg, r);

				//find emotion of current face
				string emotion = find_emotion(shape);	
				//string emotion = classify_LBP(im, faces[i]);
				//string emotion = classify_Concatenated(shape, im, faces[i]);
				std::cout << emotion << endl;

				// write text on image
				cv::putText(im, emotion, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
			}
		}
		else {
			cv::putText(im, "no face recognized", cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
		}

		cv::imshow(window, im);

		if (waitKey(30) >= 0) {
			break;
		}
	}
}

string find_emotion(dlib::full_object_detection shape) 
{
	//calculate features of face detected by webcam
	std::vector<dlib::point> mouthlist;
	std::vector<dlib::point> righteyelist;
	std::vector<dlib::point> lefteyelist;

	for (int i = 1; i < shape.num_parts() + 1; i++) {
		if (i > 36 && i <= 42) {
			lefteyelist.push_back(shape.part(i - 1));
		}
		else if (i > 42 && i <= 48) {
			righteyelist.push_back(shape.part(i - 1));
		}
		else if (i > 48) {
			mouthlist.push_back(shape.part(i - 1));
		}
	}

	int mouthxmin = 20000;
	int mouthxmax = -1;
	int mouthymin = 20000;
	int mouthymax = -1;

	int lefteyexmin = 20000;
	int lefteyexmax = -1;
	int lefteyeymin = 20000;
	int lefteyeymax = -1;

	int righteyexmin = 20000;
	int righteyexmax = -1;
	int righteyeymin = 20000;
	int righteyeymax = -1;

	for (int i = 0; i < mouthlist.size(); i++) {
		if (mouthlist.at(i).x() < mouthxmin) {
			mouthxmin = mouthlist.at(i).x();
		}
		if (mouthlist.at(i).x() > mouthxmax) {
			mouthxmax = mouthlist.at(i).x();
		}
		if (mouthlist.at(i).y() < mouthymin) {
			mouthymin = mouthlist.at(i).y();
		}
		if (mouthlist.at(i).y() > mouthymax) {
			mouthymax = mouthlist.at(i).y();
		}
	}

	for (int i = 0; i < lefteyelist.size(); i++) {
		if (lefteyelist.at(i).x() < lefteyexmin) {
			lefteyexmin = lefteyelist.at(i).x();
		}
		if (lefteyelist.at(i).x() > lefteyexmax) {
			lefteyexmax = lefteyelist.at(i).x();
		}
		if (lefteyelist.at(i).y() < lefteyeymin) {
			lefteyeymin = lefteyelist.at(i).y();
		}
		if (lefteyelist.at(i).y() > lefteyeymax) {
			lefteyeymax = lefteyelist.at(i).y();
		}
	}

	for (int i = 0; i < righteyelist.size(); i++) {
		if (righteyelist.at(i).x() < righteyexmin) {
			righteyexmin = righteyelist.at(i).x();
		}
		if (righteyelist.at(i).x() > righteyexmax) {
			righteyexmax = righteyelist.at(i).x();
		}
		if (righteyelist.at(i).y() < righteyeymin) {
			righteyeymin = righteyelist.at(i).y();
		}
		if (righteyelist.at(i).y() > righteyeymax) {
			righteyeymax = righteyelist.at(i).y();
		}
	}

	double lefteyewidth = lefteyexmax - lefteyexmin;
	double lefteyelength = lefteyeymax - lefteyeymin;
	double lefteyeA = lefteyewidth * lefteyelength / 1000;
	double lefteyeP = 2 * (1 + lefteyewidth) / 10;

	double righteyewidth = righteyexmax - righteyexmin;
	double righteyelength = righteyeymax - righteyeymin;
	double righteyeA = righteyewidth * righteyelength / 1000;
	double righteyeP = 2 * (1 + righteyewidth) / 10;

	double mouthwidth = mouthxmax - mouthxmin;
	double mouthlength = mouthymax - mouthymin;
	double mouthA = mouthwidth * mouthlength / 1000;
	double mouthP = 2 * (1 + mouthwidth) / 10;

	//retreive trained classifier
	typedef dlib::linear_kernel<sample_geometry> lin_kernel;
	dlib::multiclass_linear_decision_function<lin_kernel, string> df;
	dlib::deserialize("svm_geometric.dat") >> df;

	// sample_type m;
	sample_geometry m;
	m = { mouthA, mouthP, lefteyeA, lefteyeP, righteyeA, righteyeP };
	string label = df(m);
	int labelvalue = std::stoi(label);
	string emotion = "no emotion";

	if (labelvalue <= 1.5) {
		emotion = "anger";
	}
	else if (labelvalue <= 2.5) {
		emotion = "neutral";
	}
	else if (labelvalue <= 3.5) {
		emotion = "disgust";
	}
	else if (labelvalue <= 4.5) {
		emotion = "fear";
	}
	else if (labelvalue <= 5.5) {
		emotion = "happy";
	}
	else if (labelvalue <= 6.5) {
		emotion = "sad";
	}
	else if (labelvalue <= 7.5) {
		emotion = "surprise";
	}

	return emotion;
}

void cal_GeometricFeatures(sample_geometry& sample_g, dlib::full_object_detection shape)
{
	//calculate features of face detected by webcam
	std::vector<dlib::point> mouthlist;
	std::vector<dlib::point> righteyelist;
	std::vector<dlib::point> lefteyelist;

	for (int i = 1; i < shape.num_parts() + 1; i++) {
		if (i > 36 && i <= 42) {
			lefteyelist.push_back(shape.part(i - 1));
		}
		else if (i > 42 && i <= 48) {
			righteyelist.push_back(shape.part(i - 1));
		}
		else if (i > 48) {
			mouthlist.push_back(shape.part(i - 1));
		}
	}

	int mouthxmin = 20000;
	int mouthxmax = -1;
	int mouthymin = 20000;
	int mouthymax = -1;

	int lefteyexmin = 20000;
	int lefteyexmax = -1;
	int lefteyeymin = 20000;
	int lefteyeymax = -1;

	int righteyexmin = 20000;
	int righteyexmax = -1;
	int righteyeymin = 20000;
	int righteyeymax = -1;

	for (int i = 0; i < mouthlist.size(); i++) {
		if (mouthlist.at(i).x() < mouthxmin) {
			mouthxmin = mouthlist.at(i).x();
		}
		if (mouthlist.at(i).x() > mouthxmax) {
			mouthxmax = mouthlist.at(i).x();
		}
		if (mouthlist.at(i).y() < mouthymin) {
			mouthymin = mouthlist.at(i).y();
		}
		if (mouthlist.at(i).y() > mouthymax) {
			mouthymax = mouthlist.at(i).y();
		}
	}

	for (int i = 0; i < lefteyelist.size(); i++) {
		if (lefteyelist.at(i).x() < lefteyexmin) {
			lefteyexmin = lefteyelist.at(i).x();
		}
		if (lefteyelist.at(i).x() > lefteyexmax) {
			lefteyexmax = lefteyelist.at(i).x();
		}
		if (lefteyelist.at(i).y() < lefteyeymin) {
			lefteyeymin = lefteyelist.at(i).y();
		}
		if (lefteyelist.at(i).y() > lefteyeymax) {
			lefteyeymax = lefteyelist.at(i).y();
		}
	}

	for (int i = 0; i < righteyelist.size(); i++) {
		if (righteyelist.at(i).x() < righteyexmin) {
			righteyexmin = righteyelist.at(i).x();
		}
		if (righteyelist.at(i).x() > righteyexmax) {
			righteyexmax = righteyelist.at(i).x();
		}
		if (righteyelist.at(i).y() < righteyeymin) {
			righteyeymin = righteyelist.at(i).y();
		}
		if (righteyelist.at(i).y() > righteyeymax) {
			righteyeymax = righteyelist.at(i).y();
		}
	}

	double lefteyewidth = lefteyexmax - lefteyexmin;
	double lefteyelength = lefteyeymax - lefteyeymin;
	double lefteyeA = lefteyewidth * lefteyelength / 1000;
	double lefteyeP = 2 * (1 + lefteyewidth) / 10;

	double righteyewidth = righteyexmax - righteyexmin;
	double righteyelength = righteyeymax - righteyeymin;
	double righteyeA = righteyewidth * righteyelength / 1000;
	double righteyeP = 2 * (1 + righteyewidth) / 10;

	double mouthwidth = mouthxmax - mouthxmin;
	double mouthlength = mouthymax - mouthymin;
	double mouthA = mouthwidth * mouthlength / 1000;
	double mouthP = 2 * (1 + mouthwidth) / 10;

	sample_g = { mouthA, mouthP, lefteyeA, lefteyeP, righteyeA, righteyeP };
}

void cal_LBPFeatures(sample_lbp& sample_l, cv::Mat image, dlib::rectangle rect, int lbp_operator = 0)
{
	// initial values
	int radius = 1;
	int neighbors = 8;
	int numPatterns = 256;
	int gridx = 7;  
	int gridy = 4; 
	int overlap = 0;
	int numCells = gridx * gridy;

	// matrices used
	Mat dst;			// image after preprocessing
	Mat lbp;			// lbp image
	Mat hist_spat;		// spatial LBP histogram
	Mat uni_spat_hist;	// uniform spatial LBP histogram 

	int width = rect.width() - rect.width() % gridx + gridx;
	int heigth = rect.height() - rect.height() % gridy + gridy;
	int x = rect.left();
	int y = rect.top();

	cv::Rect roi(x, y, width, heigth);
	cv::Mat temp(image, roi);

	cvtColor(temp, dst, CV_BGR2GRAY);

	switch (lbp_operator) {
	case 0:
		lbp::ELBP(dst, lbp, radius, neighbors); // use the extended operator
		break;
	case 1:
		lbp::OLBP(dst, lbp); // use the original operator
		break;
	case 2:
		lbp::VARLBP(dst, lbp, radius, neighbors);
		break;
	}

	// spatial histogram, 256 pattern
	lbp::spatial_histogram(lbp, hist_spat, numPatterns, gridx, gridy, overlap);

	// uniform spatial histogram
	uni_spat_hist = convertLBPHistogramToSpatialUniform(hist_spat, numCells, numPatterns);
	
	for (int i = 0; i<uni_spat_hist.size().width; i++)
	{
		sample_l(i, 0) = (double)uni_spat_hist.at<int>(0,i) / 100;// 
	}
}

void cal_ConFeatures(sample_concatenate& sample_c, dlib::full_object_detection shape, cv::Mat image, dlib::rectangle rect)
{
	sample_geometry g;
	sample_lbp l;

	cal_GeometricFeatures(g, shape);
	cal_LBPFeatures(l, image, rect);

	sample_c = dlib::join_cols(g, l);
}

string classify_LBP(cv::Mat image, dlib::rectangle rect)
{
	std::cout << "classify using LBP features" << endl;

	sample_lbp sample_l;

	cal_LBPFeatures(sample_l, image, rect);

	//retreive trained classifier
	typedef dlib::linear_kernel<sample_lbp> lin_kernel;
	dlib::multiclass_linear_decision_function<lin_kernel, string> df;
	dlib::deserialize("svm_lbp.dat") >> df;

	string label = df(sample_l);
	int labelvalue = std::stoi(label);
	// cout << "label " << labelvalue << endl;
	string emotion = "no emotion";

	if (labelvalue <= 1.5) {
		emotion = "anger";
	}
	else if (labelvalue <= 2.5) {
		emotion = "neutral";
	}
	else if (labelvalue <= 3.5) {
		emotion = "disgust";
	}
	else if (labelvalue <= 4.5) {
		emotion = "fear";
	}
	else if (labelvalue <= 5.5) {
		emotion = "happy";
	}
	else if (labelvalue <= 6.5) {
		emotion = "sad";
	}
	else if (labelvalue <= 7.5) {
		emotion = "surprise";
	}

	return emotion;
}

string classify_Concatenated(dlib::full_object_detection shape, cv::Mat image, dlib::rectangle rect)
{
	std::cout << "classify using concatenated features" << endl;

	sample_concatenate sample_c;

	cal_ConFeatures(sample_c, shape, image, rect);

	//retreive trained classifier
	typedef dlib::linear_kernel<sample_concatenate> lin_kernel;
	dlib::multiclass_linear_decision_function<lin_kernel, string> df;
	dlib::deserialize("svm_concatenated.dat") >> df;

	string label = df(sample_c);
	int labelvalue = std::stoi(label);
	// cout << "label " << labelvalue << endl;
	string emotion = "no emotion";

	if (labelvalue <= 1.5) {
		emotion = "anger";
	}
	else if (labelvalue <= 2.5) {
		emotion = "neutral";
	}
	else if (labelvalue <= 3.5) {
		emotion = "disgust";
	}
	else if (labelvalue <= 4.5) {
		emotion = "fear";
	}
	else if (labelvalue <= 5.5) {
		emotion = "happy";
	}
	else if (labelvalue <= 6.5) {
		emotion = "sad";
	}
	else if (labelvalue <= 7.5) {
		emotion = "surprise";
	}

	return emotion;
}

void train_SVM_LBPClassifier(std::vector<sample_lbp> samples, std::vector<string>& labels)
{
	std::cout << "\ntrain SVM Classifier for LBP" << endl;

	std::cout << "size sample " << samples.size() << endl;
	std::cout << "size lable " << labels.size() << endl;

	typedef dlib::linear_kernel<sample_lbp> lin_kernel;
	typedef dlib::svm_multiclass_linear_trainer <lin_kernel, string> svm_mc_trainer;

	svm_mc_trainer trainer;

	dlib::multiclass_linear_decision_function<lin_kernel, string> svm_lbp = trainer.train(samples, labels);

	dlib::randomize_samples(samples, labels);
	std::cout << "randomized" << endl;
	std::cout << "cross validation: \n" << dlib::cross_validate_multiclass_trainer(trainer, samples, labels, 2) << endl;

	//save classifier, so classification does not have to be done every time program is runned
	dlib::serialize("svm_lbp.dat") << svm_lbp;
}

void readSampleLBP(std::vector<sample_lbp>& samples, std::vector<string>& labels)
{
	std::cout << "generate LBP samples" << endl;

	sample_lbp sample;
	string label;

	fs::path targetDirIm("histogram");
	fs::directory_iterator itim(targetDirIm), eodim;

	string namefile;
	string hist_path;
	hist_path = "histogram/";

	int i = 0;
	BOOST_FOREACH(fs::path const &p, std::make_pair(itim, eodim))
	{
		if (fs::is_regular_file(p))
		{
			// std::cout << "iteration: " << i << endl;

			// reading LBP histogram from .txt files
			ifstream myfile(p.string());
			string line;
			if (myfile.is_open())
			{
				int j = 0; 
				while (getline(myfile, line))
				{
					sample(j, 0) = (double)stoi(line) / 100;// 
					j++;
					//cout << sample(j, 0) << endl;
				}
				myfile.close();
			}

			// getting the emotion .txt files name
			namefile = p.string();
			namefile = namefile.substr(hist_path.size(), namefile.size() - hist_path.size() - 14);
			string path_label = "labels/" + namefile + "_emotion" + ".txt";

			ifstream myfile1(path_label);
			if (myfile1.is_open())	
			{
				while (getline(myfile1, line))
				{
					label = line; // line.substr(3, 3);
					// cout << label << endl;
				}
				myfile1.close();
			}

			i++;
			samples.push_back(sample);
			labels.push_back(label);
			//cout << "size sample " << samples[0](0, 0) << endl;
			//cout << "size sample " << samples[0](1, 0) << endl;
			//cout << "size sample " << samples[0](2, 0) << endl;
			//if (i==1)
			//	break;
		}
	}
}

void extractFaceROI()
{
	/******************************* Face Detection *******************************/
	// extract the Region of Interest (face) of training images
	// and store in each .txt files with order like this :
	// left (x position) 
	// top (y position)
	// width
	// height

	fs::path targetDirIm("training_images");
	fs::directory_iterator itim(targetDirIm), eodim;

	string namefile;
	string image_path;
	image_path = "training_images/";

	// face detector
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	std::cout << "face detection \n";

	int i = 0;
	BOOST_FOREACH(fs::path const &p, std::make_pair(itim, eodim))
	{
		if (fs::is_regular_file(p))
		{
			std::cout << "iteration: " << i << endl;

			// load image
			dlib::array2d<unsigned char> img;
			dlib::load_image(img, p.string());

			// extracting ROI from the image	
			std::vector<dlib::rectangle> dets = detector(img);

			// getting the files'name
			namefile = p.string(); 
			namefile = namefile.substr(image_path.size(), namefile.size() - image_path.size() - 4);
			namefile = namefile + ".txt";
			std::cout << namefile << endl;

			// writing the data to txt files
			string path = "ROI/" + namefile; 
			ofstream myfile(path);
			if (myfile.is_open())
			{
				myfile << dets[0].left() << endl;
				myfile << dets[0].top() << endl;
				myfile << dets[0].width() << endl;
				myfile << dets[0].height() << endl;
				myfile.close();
				std::cout << "writing success \n";
			}

			i++;
			//if (i)
			//	break;
		}
	}
	
	std::cout << "storing the ROI (faces) is done " << endl;
}

void generate_LBP_histogram(int lbp_operator)
{
	/******************************* LBP Histogram *******************************/
	// extract LBP features of training images
	// and store the results to .txt files

	// initial values
	int radius = 1;
	int neighbors = 8;
	int numPatterns = pow(2, neighbors);// 256;
	int gridx = 7;  
	int gridy = 4;  
	int overlap = 0;
	int numCells = gridx * gridy;
	
	cout << gridx << endl;
	cout << gridy << endl;

	// matrices used
	Mat dst;			// image after preprocessing
	Mat lbp;			// lbp image
	Mat hist_spat;		// spatial LBP histogram
	Mat uni_spat_hist;	// uniform spatial LBP histogram 

	fs::path targetDirIm("training_images");
	fs::directory_iterator itim(targetDirIm), eodim;

	string namefile;
	string image_path;
	image_path = "training_images/";

	int i = 0;
	BOOST_FOREACH(fs::path const &p, std::make_pair(itim, eodim))
	{
		if (fs::is_regular_file(p))
		{
			std::cout << "iteration: " << i << endl;

			// getting the image name
			namefile = p.string();
			namefile = namefile.substr(image_path.size(), namefile.size() - image_path.size() - 4);

			string path_histogram = "histogram/" + namefile + "_histogram" + ".txt";

			namefile = namefile + ".txt";
			std::cout << namefile << endl;

			// read ROI from .txt files
			string path_ROI = "ROI/" + namefile;
			ifstream myfile(path_ROI);
			string line;
			int roi_rect[4];
			if (myfile.is_open())
			{
				int j = 0;
				while (getline(myfile, line))
				{
					roi_rect[j] = std::stoi(line);
					j++;
				}
				myfile.close();
			}
			
			cv::Mat image = imread(p.string());

			int width = roi_rect[2] -roi_rect[2] % gridx + gridx;
			int heigth = roi_rect[3] -roi_rect[3] % gridy + gridy;
			int x = roi_rect[0];
			int y = roi_rect[1];
			
			Rect roi(x, y, width, heigth); 
			cv::Mat temp(image, roi); 

			cvtColor(temp, dst, CV_BGR2GRAY);

			switch (lbp_operator) {
			case 0:
				lbp::ELBP(dst, lbp, radius, neighbors); // use the extended operator
				break;
			case 1:
				lbp::OLBP(dst, lbp); // use the original operator
				break;
			case 2:
				lbp::VARLBP(dst, lbp, radius, neighbors);
				break;
			}

			// spatial histogram, 256 pattern
			lbp::spatial_histogram(lbp, hist_spat, numPatterns, gridx, gridy, overlap);

			// uniform spatial histogram
			uni_spat_hist = convertLBPHistogramToSpatialUniform(hist_spat, numCells, numPatterns);

			// write LBP uniform spatioal histogram to .txt files
			ofstream myfile1(path_histogram);
			if (myfile1.is_open())
			{
				for (int j=0; j<uni_spat_hist.size().width; j++)
				{
					myfile1 << uni_spat_hist.at<int>(0,j) << endl;
				}
				
				myfile1.close();
				std::cout << "writing histogram success \n";
			}

			i++;
		}
	}
	std::cout << "calculating LBP histogram is done " << endl;
}

uint8_t uniform_pattern[58] = { 0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31,
32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128,
129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223,
224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
251, 252, 253, 254, 255};

cv::Mat convertLBPHistogramToSpatialUniform(cv::Mat histogram, int numCells, int numPatterns)
{

	int uni_pattern = 59;
	cv::Mat uni_hist(1, numCells * uni_pattern, CV_32S, Scalar(0));

	for (int i = 0; i<numCells; i++)
	{
		int k = 0;
		for (int j = 0; j < numPatterns; j++)
		{
			if (uniform_pattern[k] == j && k != 58)
			{
				uni_hist.at<int>(0, i* uni_pattern + k) = histogram.at<int>(0, i * numPatterns + j);
				k++;
			}
			else
			{
				uni_hist.at<int>(0, i* uni_pattern + 58) += histogram.at<int>(0, i * numPatterns + j);
			}
		}
	}

	return uni_hist;
}

cv::Mat convertLBPHistogramToSingleUniform(cv::Mat histogram, int numCells, int numPatterns)
{

	cv::Mat uni_hist(1,59, CV_32S, Scalar(0));

	for (int i = 0; i < numCells; i++)
	{
		int k = 0;
		for (int j = 0; j < numPatterns; j++)
		{
			if (uniform_pattern[k] == j && k != 58)
			{
				uni_hist.at<int>(0,k) += histogram.at<int>(0, i * numPatterns + j);
				k++;
			}
			else
			{
				uni_hist.at<int>(0,58) += histogram.at<int>(0, i * numPatterns + j);
			}
		}
	}
	return uni_hist;
}

//save geometrical features of test set and save to samples with corresponding labels.
void findGeometricalFeatures()
{
	fs::path targetDirIm("textfaces");
	fs::directory_iterator itim(targetDirIm), eodim;

	string emotionname;

	// Loop over all the images provided in folder.
	BOOST_FOREACH(fs::path const &p, std::make_pair(itim, eodim))
	{

		if (fs::is_regular_file(p))
		{

			emotionname = p.string();
			emotionname = emotionname.substr(10, emotionname.size() - 14);
			emotionname = "labels/" + emotionname + "_emotion.txt";

			if (fs::is_regular_file(emotionname)) {
				
				labelpaths.push_back(emotionname);

				string line;
				string labelname;
				std::vector<string> featurepoints;
				ifstream myfile(p.string());
				if (myfile.is_open())
				{
					while (getline(myfile, line))
					{
						featurepoints.push_back(line);
					}
					myfile.close();
				}

				else {
					std::cout << "Unable to open file";
				}

				std::vector<string> mouthlist;
				std::vector<string >leftbrowlist;
				std::vector<string > rightbrowlist;
				std::vector<string > righteyelist;
				std::vector<string > lefteyelist;
				std::vector<string> noselist;
				std::vector<string> jawlist;

				for (int i = 1; i < featurepoints.size() + 1; i++) {
					if (i <= 17) {
						jawlist.push_back(featurepoints.at(i - 1));
					}
					else if (i <= 22) {
						leftbrowlist.push_back(featurepoints.at(i - 1));
					}
					else if (i <= 27) {
						rightbrowlist.push_back(featurepoints.at(i - 1));
					}
					else if (i <= 36) {
						noselist.push_back(featurepoints.at(i - 1));
					}
					else if (i <= 42) {
						lefteyelist.push_back(featurepoints.at(i - 1));
					}
					else if (i <= 48) {
						righteyelist.push_back(featurepoints.at(i - 1));
					}
					else {
						mouthlist.push_back(featurepoints.at(i - 1));
					}
				}

				//compute geometrical features

				int mouthxmin = 20000;
				int mouthxmax = -1;
				int mouthymin = 20000;
				int mouthymax = -1;

				int lefteyexmin = 20000;
				int lefteyexmax = -1;
				int lefteyeymin = 20000;
				int lefteyeymax = -1;

				int righteyexmin = 20000;
				int righteyexmax = -1;
				int righteyeymin = 20000;
				int righteyeymax = -1;


				for (int ii = 0; ii < mouthlist.size(); ii++) {
					int xvalue = std::stoi(mouthlist.at(ii).substr(1, 3));
					int yvalue = std::stoi(mouthlist.at(ii).substr(6, 3));
					if (xvalue > mouthxmax) {
						mouthxmax = xvalue;
					}
					if (xvalue < mouthxmin) {
						mouthxmin = xvalue;
					}
					if (yvalue > mouthymax) {
						mouthymax = yvalue;
					}
					if (yvalue < mouthymin) {
						mouthymin = yvalue;
					}
				}

				int mouthwidth = mouthxmax - mouthxmin;
				int mouthlength = mouthymax - mouthymin;
				int mouthA = mouthwidth * mouthlength;
				int mouthP = 2 * (1 + mouthwidth);
				mouthareas.push_back(mouthA / 1000);
				mouthperimeters.push_back(mouthP / 10);

				//cout << mouthareas.size() << endl;


				for (int ii = 0; ii < lefteyelist.size(); ii++) {
					int xvalue = std::stoi(lefteyelist.at(ii).substr(1, 3));
					int yvalue = std::stoi(lefteyelist.at(ii).substr(6, 3));
					if (xvalue> lefteyexmax) {
						lefteyexmax = xvalue;
					}
					if (xvalue < lefteyexmin) {
						lefteyexmin = xvalue;
					}
					if (yvalue > lefteyeymax) {
						lefteyeymax = yvalue;
					}
					if (yvalue < lefteyeymin) {
						lefteyeymin = yvalue;
					}
				}


				int lefteyewidth = lefteyexmax - lefteyexmin;
				int lefteyelength = lefteyeymax - lefteyeymin;
				int lefteyeA = lefteyewidth * lefteyelength;


				int lefteyeP = 2 * (1 + lefteyewidth);
				lefteyeareas.push_back(lefteyeA / 1000);
				lefteyeperimeters.push_back(lefteyeP / 10);

				//  cout << lefteyeareas.size() << endl;

				for (int ii = 0; ii < righteyelist.size(); ii++) {
					int xvalue = std::stoi(righteyelist.at(ii).substr(1, 3));
					int yvalue = std::stoi(righteyelist.at(ii).substr(6, 3));
					if (xvalue > righteyexmax) {
						righteyexmax = xvalue;
					}
					if (xvalue < righteyexmin) {
						righteyexmin = xvalue;
					}
					if (yvalue > righteyeymax) {
						righteyeymax = yvalue;
					}
					if (yvalue < righteyeymin) {
						righteyeymin = yvalue;
					}
				}


				int righteyewidth = righteyexmax - righteyexmin;
				int righteyelength = righteyeymax - righteyeymin;
				int righteyeA = righteyewidth * righteyelength;
				int righteyeP = 2 * (1 + righteyewidth);
				//  cout << righteyeP << endl;
				righteyeareas.push_back(righteyeA / 1000);
				righteyeperimeters.push_back(righteyeP / 10);


			}
		}
	}
}

void generate_GeometricSamples(std::vector<sample_geometry>& samples, std::vector<string>& labels)
{
	std::cout << "Generate Geometric Samples" << endl;

	//make data with all features + labels in order to train classifier
	//cout << "generating data" << endl;
	sample_geometry m;

	for (int i = 0; i < labelpaths.size(); i++) 
	{
		string emotionname = labelpaths.at(i);

		m = { mouthareas.at(i), mouthperimeters.at(i), lefteyeareas.at(i), lefteyeperimeters.at(i), righteyeareas.at(i), righteyeperimeters.at(i) }; //features
		samples.push_back(m);

		string line;
		string labelname;
		ifstream myfile(emotionname);
		if (myfile.is_open())
		{
			while (getline(myfile, line))
			{
				labelname = line;
			}
			myfile.close();
		}

		else {
			std::cout << "Unable to open file";
		}

		labels.push_back(labelname);
		// cout << labelname << endl;
	}
}

void concatenate_Features(std::vector<sample_concatenate>& samples_c, std::vector<sample_geometry> samples_g, std::vector<sample_lbp> samples_l)
{
	std::cout << "Concatenating LBP Histogram and Geometrical Features" << endl;
	std::cout << "LBP sample size " << samples_l.size() << endl;
	std::cout << "Geometrical sample size " << samples_g.size() << endl;

	sample_concatenate sample;

	int total_samples = samples_g.size();

	for (int i=0; i<total_samples; i++)
	{
		sample = dlib::join_cols(samples_g[i],samples_l[i]);
		samples_c.push_back(sample);
	}
}

void train_SVM_ConcatenatedClassifier(std::vector<sample_concatenate> samples_c, std::vector<string>& labels)
{
	std::cout << "\nSVM Concatenated Features" << endl;

	std::cout << "size sample " << samples_c.size() << endl;
	std::cout << "size label " << labels.size() << endl;

	typedef dlib::linear_kernel<sample_concatenate> lin_kernel;
	typedef dlib::svm_multiclass_linear_trainer <lin_kernel, string> svm_mc_trainer;

	svm_mc_trainer trainer;

	dlib::multiclass_linear_decision_function<lin_kernel, string> svm_concatenated = trainer.train(samples_c, labels);

	dlib::randomize_samples(samples_c, labels);
	std::cout << "randomized" << endl;
	std::cout << "cross validation: \n" << dlib::cross_validate_multiclass_trainer(trainer, samples_c, labels, 2) << endl;

	//save classifier, so classification does not have to be done every time program is runned
	dlib::serialize("svm_concatenated.dat") << svm_concatenated;
}

void train_SVM_GeometricClassifier(std::vector<sample_geometry> samples_g, std::vector<string>& labels)
{
	std::cout << "\nSVM Geometric Features" << endl;

	std::cout << "size sample " << samples_g.size() << endl;
	std::cout << "size label " << labels.size() << endl;

	typedef dlib::linear_kernel<sample_geometry> lin_kernel;
	typedef dlib::svm_multiclass_linear_trainer <lin_kernel, string> svm_mc_trainer;

	svm_mc_trainer trainer;

	dlib::multiclass_linear_decision_function<lin_kernel, string> svm_geometric = trainer.train(samples_g, labels);

	dlib::randomize_samples(samples_g, labels);
	std::cout << "randomized" << endl;
	std::cout << "cross validation: \n" << dlib::cross_validate_multiclass_trainer(trainer, samples_g, labels, 2) << endl;

	//save classifier, so classification does not have to be done every time program is runned
	dlib::serialize("svm_geometric.dat") << svm_geometric;

}

void checkFiles()
{
	std::cout << "check files in textfaces and label" << endl;

	fs::path targetDirIm("textfaces");
	fs::directory_iterator itim(targetDirIm), eodim;

	std::vector<string> textfaces_path;
	std::vector<string> label_path;
	string namefile;

	BOOST_FOREACH(fs::path const &p, std::make_pair(itim, eodim))
	{
		if (fs::is_regular_file(p))
		{
			namefile = p.string();
			namefile = namefile.substr(10, namefile.size() - 14);
			textfaces_path.push_back(namefile);
		}
	}

	fs::path targetDirIm1("labels");
	fs::directory_iterator itim1(targetDirIm1), eodim1;
	BOOST_FOREACH(fs::path const &p, std::make_pair(itim1, eodim1))
	{
		if (fs::is_regular_file(p))
		{
			namefile = p.string();
			namefile = namefile.substr(7, namefile.size() - 19);
			label_path.push_back(namefile);
		}
	}

	for (int i = 0; i < label_path.size(); i++)
	{
		
		if (textfaces_path[i].compare(label_path[i]) != 0)
		{
			std::cout << "NOT SAME" << endl;
			std::cout << "iteration " << i << endl;
			std::cout << "textfaces " << textfaces_path[i] << endl;
			std::cout << "label_path " << label_path[i] << endl;
		}
	}
	std::cout << "compare textfaces and labels DONE" << endl;



	std::vector<string> histogram_path;
	fs::path targetDirIm2("histogram");
	fs::directory_iterator itim2(targetDirIm2), eodim2;
	BOOST_FOREACH(fs::path const &p, std::make_pair(itim2, eodim2))
	{
		if (fs::is_regular_file(p))
		{
			namefile = p.string();
			namefile = namefile.substr(10, namefile.size() - 24);
			histogram_path.push_back(namefile);
		}
	}

	for (int i = 0; i < histogram_path.size(); i++)
	{

		if (textfaces_path[i].compare(histogram_path[i]) != 0)
		{
			std::cout << "NOT SAME" << endl;
			std::cout << "iteration " << i << endl;
			std::cout << "textfaces " << textfaces_path[i] << endl;
			std::cout << "histogram_path " << histogram_path[i] << endl;
		}
	}
	std::cout << "compare textfaces and histogram DONE" << endl;

}

//svm classifier using cross-fold classification
void classify()
{
	std::cout << "classifying" << endl;
	std::vector<sample_geometry> samples;
	std::vector<string> labels;

	generate_data(samples, labels);

	typedef dlib::linear_kernel<sample_geometry> lin_kernel;

	typedef dlib::svm_multiclass_linear_trainer <lin_kernel, string> svm_mc_trainer;
	svm_mc_trainer trainer;

	//multiclass_linear_decision_function<lin_kernel, string> df = trainer.train(samples, labels);

	randomize_samples(samples, labels);
	std::cout << "randomized" << endl;
	std::cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 2) << endl;
}

void generate_data(std::vector<sample_geometry>& samples, std::vector<string>& labels)
{
	//make data with all features + labels in order to train classifier
	std::cout << "generating data" << endl;
	sample_geometry m;

	for (int i = 0; i < labelpaths.size(); i++) {
		string emotionname = labelpaths.at(i);
		m = (mouthareas.at(i), lefteyeareas.at(i), righteyeareas.at(i));
		samples.push_back(m);

		string line;
		string labelname;
		ifstream myfile(emotionname);
		if (myfile.is_open())
		{
			while (getline(myfile, line))
			{
				labelname = line;
			}
			myfile.close();
		}

		else {
			std::cout << "Unable to open file";
		}

		labels.push_back(labelname);
	}
}

// ----------------------------------------------------------------------------------------


