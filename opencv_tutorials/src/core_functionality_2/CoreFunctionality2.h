/*
 * CoreFunctionality2.h
 *
 *  Created on: Apr 5, 2019
 *      Author: kmcgill
 */

#ifndef CORE_FUNCTIONALITY_2_COREFUNCTIONALITY2_H_
#define CORE_FUNCTIONALITY_2_COREFUNCTIONALITY2_H_

#include "../StandardHeaders.h"
#include <opencv2/opencv.hpp>


class CoreFunctionality2{
public:
	static int run(std::string filename);
	static cv::Mat& scanImageAndReduceC(cv::Mat& I, const uchar* const table);
	static cv::Mat& scanImageAndReduceIterator(cv::Mat& I, const uchar* const table);
	static cv::Mat& scanImageAndReduceRandomAccess(cv::Mat& I, const uchar* const table);
};



#endif /* CORE_FUNCTIONALITY_2_COREFUNCTIONALITY2_H_ */
