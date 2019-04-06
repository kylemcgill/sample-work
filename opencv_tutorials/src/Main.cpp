/*
 * Main.cpp
 *
 *  Created on: Apr 5, 2019
 *      Author: kmcgill
 */


#include "core_functionality/CoreFunctionality.h"
#include "core_functionality_2/CoreFunctionality2.h"

enum Tutorial{
	NONE,
	CORE_FUNCTIONALITY,
	CORE_FUNCTIONALITY_2
};


Tutorial parseTutorial(std::string str){
	if(strcmp("core_functionality", str.c_str()) == 0){
		return Tutorial::CORE_FUNCTIONALITY;
	}else if(strcmp("core_functionality_2", str.c_str()) == 0){
		return Tutorial::CORE_FUNCTIONALITY_2;
	}

	return Tutorial::NONE;
}

void printTutorialOptions(){
	printf("Valid Tutorial Options are:\n");
	printf("\tcore_functionality\n");
	printf("\tcore_functionality_2\n");
}

int main(int argc, char* argv[]){
	printf("Beginning opencv...\n");

	if(argc < 3){
		printf("Too few arguments given!\n");
		printf("Usage: -f <path_to_file> -t <tutorial>\n");
	}

	// Parse the arguments
	std::string filename = "";
	Tutorial useTutorial = Tutorial::NONE;
	for(int i = 0; i < argc; ++i){
		if(strcmp("-f", argv[i]) == 0){
			filename = argv[i+1];
		}else if(strcmp("-t", argv[i]) == 0){
			if((useTutorial = parseTutorial(argv[i+1])) == Tutorial::NONE){
				printf("Unable to find tutorial from %s", argv[i+1]);
				return -1;
			}
		}
	}

	printf("Using Tutorial: %i", useTutorial);
	int error = 0;
	switch(useTutorial){
	case CORE_FUNCTIONALITY:{
		error = CoreFunctionality().run(filename);
		break;
	}
	case CORE_FUNCTIONALITY_2:{
		error = CoreFunctionality2().run(filename);
		break;
	}
	case NONE:{
		printf("Choose NONE as the Tutorial to use.\n");
		break;
	}
	default:{
		printf("UseTutorial doesn't have a valid value.\n");
	}
	}

	printf("Returned demo with error code: %i", error);
	return 0;
}
