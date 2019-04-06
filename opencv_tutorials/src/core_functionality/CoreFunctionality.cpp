/*
 * CoreFunctionalityMain.cpp
 *
 *  Created on: Apr 1, 2019
 *      Author: kmcgill
 */


#include "CoreFunctionality.h"

int CoreFunctionality::run(std::string filename){
	printf("Starting CoreFunctionality tutorial...\n");

	printf("Using filename: %s\n", filename.c_str());
	cv::Mat A, C;
	A = cv::imread(filename, cv::IMREAD_COLOR);

	cv::Mat B(A); // copies the header of A but not the image

	C = A;  //again copies the header but not the image

	//All the headers are different but A,B,C all point to the same
	//underlying data.

	//This creates a Region Of Interest (ROI)
	cv::Mat D(A, cv::Rect(10,10,100,100)); //Rect(x, y, width, height)
	cv::Mat E = A(cv::Range::all(), cv::Range(1,3)); // All rows and from columns [1,3]

	cv::Mat F = A.clone(); //replicates the image data a well as the header
	cv::Mat G;
	A.copyTo(G); //Also, replicates the image data along with the header

	//Creating a cv::Mat object explicitly
	//Macro expansion -> CV_[bits per item][<S>igned or<U>nsigned][type prefix][channel number]
	//# channels = number of 'colors'
	cv::Mat M(2,2, CV_8UC3, cv::Scalar(0,0,255));
	std::cout << "M = " << std::endl << " " << M << std::endl;
	printf("-----------------------------------------------\n");


	M.create(4,4, CV_8UC(2));
	std::cout << "M = " << std::endl << " " << M << std::endl;
	printf("-----------------------------------------------\n");

	cv::Mat eye = cv::Mat::eye(4,4, CV_64F);
	std::cout << "eye = " << std::endl << " " << eye << std::endl;
	printf("-----------------------------------------------\n");

	cv::Mat O = cv::Mat::ones(2,2, CV_32F);
	std::cout << "O = " << std::endl << " " << O << std::endl;
	printf("-----------------------------------------------\n");

	cv::Mat Z = cv::Mat::zeros(3,3, CV_8UC1) ;
	std::cout << "Z = " << std::endl << " " << Z << std::endl;
	printf("-----------------------------------------------\n");

	cv::Mat thing = (cv::Mat_<double>(3,3) << -1, 0, 1, 0 ,1 , -1, 1, -1, 0) ;
	std::cout << "thing = " << std::endl << " " << thing << std::endl;
	printf("-----------------------------------------------\n");
	thing = (cv::Mat_<double>({0,1,-2, 1,-2,0, -2,0,1})).reshape(3);
	std::cout << "H = " << std::endl << " " << thing << std::endl;
	printf("-----------------------------------------------\n");

	//create a new header for the existing Mat obj
	cv::Mat rowClone = thing.row(1).clone();
	std::cout << "rowClone = " << std::endl << " " << rowClone << std::endl;
	printf("-----------------------------------------------\n");

	//random matrix
	cv::Mat rand = cv::Mat(3,2, CV_8UC3);
	cv::randu(rand, cv::Scalar::all(0), cv::Scalar::all(255));


	//OUTPUT FORMATTING
	//Default
	std::cout << "R(defualt) " << std::endl << rand << std::endl << std::endl;

	//Python
	std::cout << "R(Python) " << std::endl << cv::format(rand, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

	//CSV
	std::cout << "R(CSV) " << std::endl << cv::format(rand, cv::Formatter::FMT_CSV) << std::endl << std::endl;

	//Numpy
	std::cout << "R(Numpy) " << std::endl << cv::format(rand, cv::Formatter::FMT_NUMPY) << std::endl << std::endl;

	//C
	std::cout << "R(C) " << std::endl << cv::format(rand, cv::Formatter::FMT_C) << std::endl << std::endl;
	printf("-----------------------------------------------\n");

	//OTHER STRUCTURES
	//2d pt
	cv::Point2f p(5,1);
	std::cout << "Point2D = " << p << std::endl << std::endl;

	//3d pt
	cv::Point3f p3(1,2,3);
	std::cout << "Point3D = " << p3 << std::endl << std::endl;

	//c++ vectors
	std::vector<float> v;
	v.push_back((float) CV_PI);
	v.push_back(2);
	v.push_back(3.01f);
	std::cout << "Vector of Floats via Mat: " << cv::Mat(v) << std::endl << std::endl;

	//c++ vector of points
	std::vector<cv::Point2f> vPoints(20);
	for(int i = 0; i < 20; ++i){
		vPoints[i] = cv::Point2f((float)(i*5), (float)(i%7));
	}
	std::cout <<  "A Vector of 2d points: " << vPoints << std::endl << std::endl;
	printf("-----------------------------------------------\n");









	if(A.empty()){
		printf("Could not open load image from file: %s\n", filename.c_str());
		return -1;
	}else{
		printf("Read file successfully!\n");
	}



	return 0;
}

