/*
 * CoreFunctionality2.cpp
 *
 *  Created on: Apr 5, 2019
 *      Author: kmcgill
 */

#include "CoreFunctionality2.h"

int CoreFunctionality2::run(std::string filename){
	printf("Starting CoreFunctionality 2...\n");
	printf("Using filename: %s\n", filename.c_str());
	cv::Mat I = cv::imread(filename, cv::IMREAD_COLOR);

	if(I.empty()){
		printf("Unable to open and read image! quiting...\n");
		return -1;
	}

	//Measure the time it takes to run this
	double totalTime = (double)cv::getTickCount();

	/**
	 * Even with 3 channels each of 3 bits, we still get
	 * 16 million colors.  For computer vision we can usually
	 * still get the job done with fewer colors.  That is why
	 * we need to know how to REDUCE THE COLOR SPACE.
	 *
	 * We do this by doing integer division on the color value:
	 * 		new_val = (old_val / 10) * 10; //new_val and old_val -> int
	 *
	 */

	int divisor = 8; //for now keep power of 2 to see how things work
	uint8_t table[256];

	//gives values which are multiples of the divisor
	/*
	 * Ex. if divisor is 4 then table looks like
	 * 		table = {0,0,0,0, 1,1,1,1, 2,2,2,2, ... };
	 */
	for(int i = 0; i < 256; ++i){
		table[i] = (uint8_t)(divisor * (i/divisor));
	}

	//Number of times for bench marking
	cv::Mat J; //don't want to keep reallocating this matrix
	int times = 100; //according to the tutorial, this is a good number
	int millisPerSec = 1000;
	double t;
	t = (double)cv::getTickCount();

	for(int i = 0; i < times; ++i){
		//don't want to mess up the picture so clone it
		cv::Mat clone = I.clone();
		J = scanImageAndReduceC(clone, table);
	}

	t = millisPerSec * ((double)cv::getTickCount() - t) / cv::getTickFrequency(); //total run time
	t /= times;

	printf("Completed C color space reduction method with average of %.3f millis per reduction\n", t);

	t = (double)cv::getTickCount();

	for(int i = 0; i < times; ++i){
		cv::Mat clone = I.clone();
		J = scanImageAndReduceIterator(clone, table);
	}

	t = millisPerSec * ((double)cv::getTickCount() - t) / cv::getTickFrequency(); //total run time
	t /= times;

	printf("Completed Iterator color space reduction method with average of %3f millis per reduction\n", t);

	t = (double)cv::getTickCount();

	for(int i = 0; i < times; ++i){
		cv::Mat clone = I.clone();
		J = scanImageAndReduceRandomAccess(clone, table);
	}

	t = millisPerSec * ((double)cv::getTickCount() - t) / cv::getTickFrequency(); //total run time
	t /= times;

	printf("Completed Random Access color space reduction method with average of %3f millis per reduction\n", t);


	//Preferred method since it is taken care of in the cv lib
	cv::Mat lookUpTable(1,256,CV_8U);
	uchar* p = lookUpTable.ptr();
	for(int i = 0; i < 256; ++i){
		p[i] = table[i]; //copy the table to this new struct
	}

	t = cv::getTickCount();
	for(int i = 0; i < times; ++ i){
		cv::LUT(I, lookUpTable, J); //uses multi-threading with Intel Building Blocks
	}

	t = millisPerSec * ((double)cv::getTickCount() - t) / cv::getTickFrequency(); //total run time
	t /= times;

	printf("Completed LUT color space reduction method with average of %3f millis per reduction\n", t);

	//print out the time this example took to run
	//Idea: countthe number of ticks the CPU makes and then divide by number
	//of ticks per second to get seconds of processing time
	totalTime = ((double)cv::getTickCount() - totalTime)/cv::getTickFrequency();
	printf("Time taken to run CoreFunctionality2: %.3f seconds\n", t);


	return 0;
}

cv::Mat& CoreFunctionality2::scanImageAndReduceC(cv::Mat& I, const uchar* const table){
	CV_Assert(I.depth() == CV_8U);

	int channels = I.channels();

	int nRows = I.rows;
	int nCols = I.cols * channels;

	if(I.isContinuous()){
		nCols *= nRows; //nCols is now the number of elements
		nRows = 1; //basically make matrix 1D
	}

	int i,j;
	uchar* p;
	for(i = 0; i < nRows; ++i){
		p = I.ptr<uchar>(i); //get the beginning of the cv::Mat as a pointer
		for(j = 0; j < nCols; ++j){
			//replace by reading off value from the table
			//i.e. if real value is 5 an divisor=4 then p[5]=1
			// so substitute 1 for 5
			p[j] = table[p[j]];
		}
	}
	return I;
}

cv::Mat& CoreFunctionality2::scanImageAndReduceIterator(cv::Mat& I, const uchar* const table){
	CV_Assert(I.depth() == CV_8U);

	const int channels = I.channels();

	switch(channels){
	case 1:{
		//greyscale
		cv::MatIterator_<uchar> it, end;
		for(it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it){
			*it = table[*it];
		}
		break;
	}
	case 3:{
		//3 channel color
		cv::MatIterator_<cv::Vec3b> it, end;
		for(it = I.begin<cv::Vec3b>(), end = I.end<cv::Vec3b>(); it != end; ++it){
			(*it)[0] = table[(*it)[0]]; //dref the iterator to Vector and then match the color channel
			(*it)[1] = table[(*it)[1]];
			(*it)[2] = table[(*it)[2]];
		}
		break;
	}
	}

	return I;
}

cv::Mat& CoreFunctionality2::scanImageAndReduceRandomAccess(cv::Mat& I, const uchar* const table){
	CV_Assert(I.depth() == CV_8U);

	const int channels = I.channels();
	switch(channels){
	case 1:{
		//greyscale
		for(int i = 0; i < I.rows; ++i){
			for(int j = 0; j < I.cols; ++i){
				I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];
			}
		}
		break;
	}
	case 3:{
		//3 color channel
		cv::Mat_<cv::Vec3b> _I = I; //returns the matrix as an array of Vec3b elements
		for(int i = 0; i < I.rows; ++i){
			for(int j = 0; j < I.cols; ++j){
				_I(i,j)[0] = table[_I(i,j)[0]];
				_I(i,j)[1] = table[_I(i,j)[1]];
				_I(i,j)[2] = table[_I(i,j)[2]];
			}
		}
		I = _I;

		break;
	}
	}
	return I;
}
