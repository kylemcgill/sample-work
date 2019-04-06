/*
 * ParseUtils.h
 *
 *  Created on: Mar 12, 2019
 *      Author: kmcgill
 */

#ifndef PARSEUTILS_CPP_
#define PARSEUTILS_CPP_

#include "StandardHeaders.h"

inline void printHex(void* p, uint32_t len){
	uint8_t* ptr = (uint8_t*)p;
	uint32_t count = 0;
	uint8_t space = 0;
	while(count < len){
		if(space % 2 == 0){
			std::cout << " ";
		}
		if(*ptr < 16){
			std::cout << "0";
		}
		std::cout << std::hex << uint32_t(*ptr);
		++ptr; ++space; ++count;
	}
	std::cout << "\n";
}


#endif /* PARSEUTILS_CPP_ */
