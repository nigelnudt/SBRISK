/*
    BRISK - Binary Robust Invariant Scalable Keypoints
    Reference implementation of
    [1] Stefan Leutenegger,Margarita Chli and Roland Siegwart, BRISK:
    	Binary Robust Invariant Scalable Keypoints, in Proceedings of
    	the IEEE International Conference on Computer Vision (ICCV2011).

    Copyright (C) 2011  The Autonomous Systems Lab (ASL), ETH Zurich,
    Stefan Leutenegger, Simon Lynen and Margarita Chli.

    This file is part of BRISK.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
       * Neither the name of the ASL nor the names of its contributors may be 
         used to endorse or promote products derived from this software without 
         specific prior written permission.
   
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "brisk.h"
#include <fstream>
#include <iostream>
#include <list>
#include <time.h>

//standard configuration for the case of no file given
const int n=12;
const float r=2.5; // found 8-9-11, r=3.6, exponent 1.5

void help(char** argv){
	std::cout << "This command line tool lets you evaluate different keypoint "
			<< "detectors, descriptors and matchers." << std::endl
			<< "usage:" << std::endl
			<< argv[0] << " <dataset> <2nd> <detector> <descriptor> [descFile1 descFile1]" << std::endl
			<< "    " << "dataset:    Folder containing the images. The images must be of .ppm "<< std::endl
			<< "    " << "            format. They must be named img#.ppm, and there must be "<< std::endl
			<< "    " << "            corresponding homographies named H1to#." << std::endl
			<< "    " << "            You can also use the prefix rot-, then 2nd will be the" << std::endl
			<< "    " << "            rotation in degrees." << std::endl
			<< "    " << "2nd:        Number of the 2nd image (the 1st one is always img1.ppm)"<< std::endl
			<< "    " << "            or the rotation in degrees, if rot- used." << std::endl
			<< "    " << "detector:   Feature detector, e.g. AGAST, or BRISK. You can add the "<< std::endl
			<< "    " << "            threshold, e.g. BRISK80 or SURF2000"<< std::endl
			<< "    " << "descriptor: Feature descriptor, e.g. SURF, BRIEF, BRISK or U-BRISK."<< std::endl
			<< "    " << "[descFile]: Optional: files with descriptors to act as detected points."<< std::endl;
}

int main(int argc, char ** argv) 
{

	//std::cout<<sizeof(cv::Point2i)<<" "<<sizeof(CvPoint)<<std::endl;

	// process command line args
	if(argc != 6 && argc != 7 && argc != 1){
		help(argv);
		return 1;
	}

	// names of the two image files
	std::string fname1;
	std::string fname2;
	cv::Mat imgRGB1;
	cv::Mat imgRGB2;
	cv::Mat imgRGB3;
	bool do_rot=false;
	// standard file extensions
	std::vector<std::string> fextensions;
	fextensions.push_back(".bmp");
	fextensions.push_back(".jpeg");
	fextensions.push_back(".jpg");
	fextensions.push_back(".jpe");
	fextensions.push_back(".jp2");
	fextensions.push_back(".png");
	fextensions.push_back(".pgm");
	fextensions.push_back(".ppm");
	fextensions.push_back(".sr");
	fextensions.push_back(".ras");
	fextensions.push_back(".tiff");
	fextensions.push_back(".tif");

	// if no arguments are passed: 
	if(argc==1){
		int i=0;
		int fextensions_size=fextensions.size();
		while(imgRGB1.empty()||imgRGB2.empty()){
			fname1 = "../../images/img1"+fextensions[i];
			fname2 = "../../images/img2"+fextensions[i];
			imgRGB1 = cv::imread(fname1);
			imgRGB2 = cv::imread(fname2);
			i++;
			if(i>=fextensions_size) break;
		}
		if (imgRGB2.empty()||imgRGB2.empty())
		{
			std::cout<<"image(s) "<<fname1<<", "<<fname2<<" not found." << std::endl;
			return 2;
		}
	}
	else{
		if(strncmp("rot-", argv[1], 4)==0){
			do_rot=true;
			int i=0;
			int fextensions_size=fextensions.size();
			while(imgRGB1.empty()){
				fname1 = std::string(argv[1]+4)+"/img1"+fextensions[i];
				imgRGB1 = cv::imread(fname1);
				i++;
				if(i>=fextensions_size) break;
			}
			if (imgRGB2.empty())
			{
				std::cout<<"image not found." << std::endl;
				return 2;
			}
		}
		else{
			int i=0;
			int fextensions_size=fextensions.size();
			while(imgRGB1.empty()||imgRGB2.empty()){
				fname1 = std::string(argv[1])+"/img1"+fextensions[i];
				fname2 = std::string(argv[1])+"/img"+std::string(argv[2])+fextensions[i];
				imgRGB1 = cv::imread(fname1);
				imgRGB2 = cv::imread(fname2);
				i++;
				if(i>=fextensions_size) break;
			}
			if (imgRGB2.empty()||imgRGB2.empty())
			{
				std::cout<<"image(s)"<<fname1<<", "<<fname2<<" not found." << std::endl;
				return 2;
			}
		}
		//unsigned int N=atoi(argv[3]);
		if (imgRGB1.empty())
		{
			fname1 = std::string(argv[1]+4)+"/img1.pgm";
			imgRGB1 = cv::imread(fname1);
			if (imgRGB1.empty()){
				std::cout<<"image not found at " << fname1 << std::endl;
				return 2;
			}
		}
	}

	// convert to grayscale
	cv::Mat imgGray1;
	cv::cvtColor(imgRGB1, imgGray1, CV_BGR2GRAY);
	cv::Mat imgGray2;
	if(!do_rot){
		cv::cvtColor(imgRGB2, imgGray2, CV_BGR2GRAY);
	}

	// run FAST in first image
	std::vector<cv::KeyPoint> keypoints, keypoints2;
	int threshold;

	// create the detector:
	cv::Ptr<cv::FeatureDetector> detector;
	if(argc==1){
		detector = new cv::BriskFeatureDetector(60,4);
	}
	else{
		if(strncmp("FAST", argv[3], 4 )==0){
			threshold = atoi(argv[3]+4);
			if(threshold==0)
				threshold = 30;
			detector = new cv::FastFeatureDetector(threshold,true);
		}
		else if(strncmp("AGAST", argv[3], 5 )==0){
			threshold = atoi(argv[3]+5);
			if(threshold==0)
				threshold = 30;
			detector = new cv::BriskFeatureDetector(threshold,0);
		}
		else if(strncmp("BRISK", argv[3], 5 )==0){
			threshold = atoi(argv[3]+5);
			if(threshold==0)
				threshold = 30;
			detector = new cv::BriskFeatureDetector(threshold,4);
		}
		else if(strncmp("SURF", argv[3], 4 )==0){
			threshold = atoi(argv[3]+4);
			if(threshold==0)
				threshold = 400;
			detector = new cv::SurfFeatureDetector(threshold);
		}
		else if(strncmp("SIFT", argv[3], 4 )==0){
			float thresh = 0.04 / 3 / 2.0;
			float edgeThreshold=atof(argv[3]+4);
			if(edgeThreshold==0)
				thresh = 10.0;
			detector = new cv::SiftFeatureDetector(thresh,edgeThreshold);
		}
		else{
			detector = cv::FeatureDetector::create( argv[3] );
		}
		if (detector.empty()){
			std::cout << "Detector " << argv[3] << " not recognized. Check spelling!" << std::endl;
			return 3;
		}
	}

	int repeat_cnt = atoi(argv[6]);
	{
		clock_t start = clock();
		for( int i=0; i<repeat_cnt; i++ ){
			keypoints.clear();
			keypoints2.clear();
			detector->detect(imgGray1,keypoints);
			detector->detect(imgGray2,keypoints2);	
		}
		clock_t end = clock();
		double time_total = (double)(end - start) / repeat_cnt / (double)CLOCKS_PER_SEC;
		printf("Time consumed in keypoint detection: %lfs\n", time_total );
	}

	// now the extractor:
	bool hamming=true;
	bool is_black_white = false;
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
	// now the extractor:
	if(argc==1){
		descriptorExtractor = new cv::BriskDescriptorExtractor();
	}
	else{
		if(std::string(argv[4])=="SBRISK"){
			descriptorExtractor = new cv::ZKBriskDescriptorExtractor();
			is_black_white      = true;
		}
		else if(std::string(argv[4])=="BRISK"){
			descriptorExtractor = new cv::BriskDescriptorExtractor();
		}
		else if(std::string(argv[4])=="U-BRISK"){
			descriptorExtractor = new cv::BriskDescriptorExtractor(false);
		}
		else if(std::string(argv[4])=="SU-BRISK"){
			descriptorExtractor = new cv::BriskDescriptorExtractor(false,false);
		}
		else if(std::string(argv[4])=="S-BRISK"){
			descriptorExtractor = new cv::BriskDescriptorExtractor(true,false);
		}
		else if(std::string(argv[4])=="BRIEF"){
			descriptorExtractor = new cv::BriefDescriptorExtractor(64);
		}
		else if(std::string(argv[4])=="CALONDER"){
			descriptorExtractor = new cv::CalonderDescriptorExtractor<float>("current.rtc");
			hamming=false;
		}
		else if(std::string(argv[4])=="SURF"){
			descriptorExtractor = new cv::SurfDescriptorExtractor();
			hamming=false;
		}
		else if(std::string(argv[4])=="SIFT"){
			descriptorExtractor = new cv::SiftDescriptorExtractor();
			hamming=false;
		}
		else{
			descriptorExtractor = cv::DescriptorExtractor::create( argv[4] );
		}
		if (descriptorExtractor.empty()){
			hamming=false;
			std::cout << "Descriptor " << argv[4] << " not recognized. Check spelling!" << std::endl;
			return 4;
		}
	}

	// get the descriptors
	cv::Mat descriptors, descriptors2;
	std::vector<cv::DMatch> indices;

	std::vector<cv::KeyPoint> black_keypoints, black_keypoints2;
	cv::Mat black_descriptors, black_descriptors2;
	std::vector<cv::DMatch> black_indices;

	std::vector<cv::KeyPoint> white_keypoints, white_keypoints2;
	cv::Mat white_descriptors, white_descriptors2;
	std::vector<cv::DMatch> white_indices;
	
	clock_t start = clock();
	for( int i=0; i<repeat_cnt; i++ ){
		// first image
		descriptorExtractor->compute(imgGray2,keypoints2,descriptors2);
		// and the second one
		descriptorExtractor->compute(imgGray1,keypoints,descriptors);
	}
	clock_t end = clock();
	double time_total = (double)(end-start) / repeat_cnt / (double)CLOCKS_PER_SEC;
	printf("Time consumed in feature extraction£º%lfs\n", time_total );

	// matching
	std::vector<std::vector<cv::DMatch> > matches;
	std::vector<std::vector<cv::DMatch> > white_matches, black_matches;
	cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;
	if(hamming)
		descriptorMatcher = new cv::BFMatcher(cv::NORM_HAMMING);//new cv::BruteForceMatcher<cv::HammingSse>();
	else
		descriptorMatcher = new cv::BruteForceMatcher<cv::L2<float> >();

	float hamming_thresh = atoi(argv[5]);
	if( is_black_white ){
		unsigned char * psrc   = NULL;
		unsigned char * pwhite = NULL;
		unsigned char * pblack = NULL;
		int nwhite=0,nblack=0;

		// the first key set ------------------------------------------------------------------------------------
		nwhite=nblack=0;
		for( std::vector<cv::KeyPoint>::iterator iter = keypoints.begin(); iter != keypoints.end(); iter ++ ){
			if( iter->class_id > 0 ){
				nwhite ++;
			}
			else{
				nblack ++;
			}
		}
		white_descriptors = cv::Mat::zeros(nwhite,64, CV_8U);
		black_descriptors = cv::Mat::zeros(nblack,64, CV_8U);
		psrc   = descriptors.data;
		pwhite = white_descriptors.data;
		pblack = black_descriptors.data;
		for( std::vector<cv::KeyPoint>::iterator iter = keypoints.begin(); iter != keypoints.end(); iter ++ ){
			if( iter->class_id > 0 ){
				white_keypoints.push_back(*iter);
				memcpy( pwhite, psrc, 64 );
				pwhite += 64;
			}
			else{
				black_keypoints.push_back(*iter);
				memcpy( pblack, psrc, 64 );
				pblack += 64;
			}
			psrc += 64;
		}

		// the second key set ------------------------------------------------------------------------------------
		nwhite=nblack=0;
		for( std::vector<cv::KeyPoint>::iterator iter = keypoints2.begin(); iter != keypoints2.end(); iter ++ ){
			if( iter->class_id > 0 ){
				nwhite ++;
			}
			else{
				nblack ++;
			}
		}
		white_descriptors2 = cv::Mat::zeros(nwhite,64, CV_8U);
		black_descriptors2 = cv::Mat::zeros(nblack,64, CV_8U);
		psrc   = descriptors2.data;
		pwhite = white_descriptors2.data;
		pblack = black_descriptors2.data;
		for( std::vector<cv::KeyPoint>::iterator iter = keypoints2.begin(); iter != keypoints2.end(); iter ++ ){
			if( iter->class_id > 0 ){
				white_keypoints2.push_back(*iter);
				memcpy( pwhite, psrc, 64 );
				pwhite += 64;
			}
			else{
				black_keypoints2.push_back(*iter);
				memcpy( pblack, psrc, 64 );
				pblack += 64;
			}
			psrc += 64;
		}
		start = clock();
		for( int i=0; i<repeat_cnt; i++ ){
			white_matches.clear();
			black_matches.clear();
			if(hamming){
				descriptorMatcher->radiusMatch(white_descriptors2,white_descriptors,white_matches,hamming_thresh);
				descriptorMatcher->radiusMatch(black_descriptors2,black_descriptors,black_matches,hamming_thresh);
				//descriptorMatcher->knnMatch(white_descriptors2,white_descriptors,white_matches,hamming_thresh);
				//descriptorMatcher->knnMatch(black_descriptors2,black_descriptors,black_matches,hamming_thresh);
			}
			else{
				descriptorMatcher->radiusMatch(white_descriptors2,white_descriptors,white_matches,0.21);
				descriptorMatcher->radiusMatch(black_descriptors2,black_descriptors,black_matches,0.21);
			}
		}
		end = clock();
		time_total = (double)(end-start) / repeat_cnt / (double)CLOCKS_PER_SEC;
		printf("Time consumed in keypoint matching£º%lfs\n", time_total );
	}
	else{
		start = clock();
		for( int i=0; i<repeat_cnt; i++ ){
			matches.clear();
			if(hamming){
				descriptorMatcher->radiusMatch(descriptors2,descriptors,matches,hamming_thresh);
				//descriptorMatcher->knnMatch(descriptors2,descriptors,matches,hamming_thresh);
			}
			else
				descriptorMatcher->radiusMatch(descriptors2,descriptors,matches,0.21);
		}
		end = clock();
		time_total = (double)(end-start) / repeat_cnt / (double)CLOCKS_PER_SEC;
		printf("Time consumed in keypoint matching£º%lfs\n", time_total );
	}
	fgetc(stdin);

	// drawing-----------------------------------------------------------------------------------------
	cv::Mat outimg;
	if( is_black_white ){
		// save the white keypoints----------------------------------------
		{
		std::string desc1 = std::string(std::string(argv[1])+"/img1_white.txt");
		std::string desc2 = std::string(std::string(argv[1])+"/img2_white.txt");
		std::ofstream descf1(desc1.c_str());
		if(!descf1.good()){
			std::cout<<"Descriptor file not found at " << desc1 <<std::endl;
			return 3;
		}
		std::ofstream descf2(desc2.c_str());
		if(!descf2.good()){
			std::cout<<"Descriptor file not found at " << desc2 <<std::endl;
			return 3;
		}
		unsigned char * pwhite2 = white_descriptors2.data;
		unsigned char * pwhite  = white_descriptors.data;
		for( std::vector<cv::KeyPoint>::iterator iter = white_keypoints.begin(); iter != white_keypoints.end(); iter ++ ){
			descf1 << iter->pt.x << " " << iter->pt.y << " " << iter->size << " " << iter->class_id;
			for( int i=0; i<64; i++ )
				descf1 << " " << (unsigned)pwhite[i];
			descf1 << std::endl;
			pwhite += 64;
		}
		for( std::vector<cv::KeyPoint>::iterator iter = white_keypoints2.begin(); iter != white_keypoints2.end(); iter ++ ){
			descf2 << iter->pt.x << " " << iter->pt.y << " " << iter->size << " " << iter->class_id;
			for( int i=0; i<64; i++ )
				descf2 << " " << (unsigned)pwhite2[i];
			descf2 << std::endl;
			pwhite2 += 64;
		}
		// clean up
		descf1.close();
		descf2.close();
		}

		// save the black keypoints----------------------------------------
		{
		std::string desc1 = std::string(std::string(argv[1])+"/img1_black.txt");
		std::string desc2 = std::string(std::string(argv[1])+"/img2_black.txt");
		std::ofstream descf1(desc1.c_str());
		if(!descf1.good()){
			std::cout<<"Descriptor file not found at " << desc1 <<std::endl;
			return 3;
		}
		std::ofstream descf2(desc2.c_str());
		if(!descf2.good()){
			std::cout<<"Descriptor file not found at " << desc2 <<std::endl;
			return 3;
		}
		unsigned char * pblack2 = black_descriptors2.data;
		unsigned char * pblack  = black_descriptors.data;
		int cnt = 0;
		for( std::vector<cv::KeyPoint>::iterator iter = black_keypoints.begin(); iter != black_keypoints.end(); iter ++ ){
			descf1 << iter->pt.x << " " << iter->pt.y << " " << iter->size << " " << iter->class_id;
			for( int i=0; i<64; i++ )
				descf1 << " " << (unsigned)pblack[i]; 
			descf1 << std::endl;
			pblack += 64;
			cnt ++;
		}
		for( std::vector<cv::KeyPoint>::iterator iter = black_keypoints2.begin(); iter != black_keypoints2.end(); iter ++ ){
			descf2 << iter->pt.x << " " << iter->pt.y << " " << iter->size << " " << iter->class_id;
			for( int i=0; i<64; i++ )
				descf2 << " " << (unsigned)pblack2[i];
			descf2 << std::endl;
			pblack2 += 64;
		}
		// clean up
		descf1.close();
		descf2.close();
		}

		{
		// save the white matches-------------------------------------------------------------------------------
		std::string desc1 = std::string(std::string(argv[1])+"/img1_img2_white.txt");
		std::ofstream descf1(desc1.c_str());
		if(!descf1.good()){
			std::cout<<"Cannot open file: " << desc1 <<std::endl;
			return 3;
		}
		for( std::vector<std::vector<cv::DMatch> >::iterator iter = white_matches.begin(); iter != white_matches.end(); iter ++ ){
			for( std::vector<cv::DMatch>::iterator iter2 = iter->begin(); iter2 != iter->end(); iter2 ++ ){
				descf1 << iter2->queryIdx << " " << iter2->trainIdx << " " << iter2->distance << std::endl;
			}
		}
		// clean up
		descf1.close();
		}

		{
		// save the black matches-------------------------------------------------------------------------------
		std::string desc1 = std::string(std::string(argv[1])+"/img1_img2_black.txt");
		std::ofstream descf1(desc1.c_str());
		if(!descf1.good()){
			std::cout<<"Cannot open file: " << desc1 <<std::endl;
			return 3;
		}
		for( std::vector<std::vector<cv::DMatch> >::iterator iter = black_matches.begin(); iter != black_matches.end(); iter ++ ){
			for( std::vector<cv::DMatch>::iterator iter2 = iter->begin(); iter2 != iter->end(); iter2 ++ ){
				descf1 << iter2->queryIdx << " " << iter2->trainIdx << " " << iter2->distance << std::endl;
			}
		}
		// clean up
		descf1.close();
		}
		drawMatches(imgRGB2, white_keypoints2, imgRGB1, white_keypoints,white_matches,outimg,
				 cv::Scalar(0,255,0), cv::Scalar(0,0,255),
				std::vector<std::vector<char> >(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
		cv::namedWindow("White Matches");
		cv::imshow("White Matches", outimg);

		drawMatches(imgRGB2, black_keypoints2, imgRGB1, black_keypoints,black_matches,outimg,
				cv::Scalar(0,255,0), cv::Scalar(0,0,255),
				std::vector<std::vector<char> >(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
		cv::namedWindow("Black Matches");
		cv::imshow("Black Matches", outimg);
	}
	else{
		// save the keypoints----------------------------------------
		{
		std::string desc1 = std::string(std::string(argv[1])+"/img1.txt");
		std::string desc2 = std::string(std::string(argv[1])+"/img2.txt");
		std::ofstream descf1(desc1.c_str());
		if(!descf1.good()){
			std::cout<<"Descriptor file not found at " << desc1 <<std::endl;
			return 3;
		}
		std::ofstream descf2(desc2.c_str());
		if(!descf2.good()){
			std::cout<<"Descriptor file not found at " << desc2 <<std::endl;
			return 3;
		}
		unsigned char * pdat2 = descriptors2.data;
		unsigned char * pdat  = descriptors.data;
		int cnt = 0;
		for( std::vector<cv::KeyPoint>::iterator iter = keypoints.begin(); iter != keypoints.end(); iter ++ ){
			descf1 << iter->pt.x << " " << iter->pt.y << " " << iter->size << " " << iter->class_id;
			for( int i=0; i<64; i++ )
				descf1 << " " << (unsigned)pdat[i];
			pdat += 64;
			descf1 << std::endl;
			cnt ++;
		}
		for( std::vector<cv::KeyPoint>::iterator iter = keypoints2.begin(); iter != keypoints2.end(); iter ++ ){
			descf2 << iter->pt.x << " " << iter->pt.y << " " << iter->size << " " << iter->class_id;
			for( int i=0; i<64; i++ )
				descf2 << " " << (unsigned)pdat2[i];
			pdat2 += 64;
			descf2 << std::endl;
		}
		// clean up
		descf1.close();
		descf2.close();
		}
		{
		// save the matches-------------------------------------------------------------------------------
		std::string desc1 = std::string(std::string(argv[1])+"/img1_img2.txt");
		std::ofstream descf1(desc1.c_str());
		if(!descf1.good()){
			std::cout<<"Cannot open file: " << desc1 <<std::endl;
			return 3;
		}
		for( std::vector<std::vector<cv::DMatch> >::iterator iter = matches.begin(); iter != matches.end(); iter ++ ){
			for( std::vector<cv::DMatch>::iterator iter2 = iter->begin(); iter2 != iter->end(); iter2 ++ ){
				descf1 << iter2->queryIdx << " " << iter2->trainIdx << " " << iter2->distance << std::endl;
			}
		}
		// clean up
		descf1.close();
		}
		drawMatches(imgRGB2, keypoints2, imgRGB1, keypoints,matches,outimg,
				 cv::Scalar(0,255,0), cv::Scalar(0,0,255),
				std::vector<std::vector<char> >(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
		cv::namedWindow("Matches");
		cv::imshow("Matches", outimg);
	}
	cv::waitKey();

	return 0;
}
