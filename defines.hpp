/*
 * defines.hpp
 *
 *  Created on: 6/2/2017
 *      Author: julio
 */

#ifndef DEFINES_HPP_
#define DEFINES_HPP_

#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <cuda.h>
#include <dirent.h>
#include <vector>
#define DEMO_IMAGE	"/home/julio/universidad/proyecto/proyecto/HarrisCornerRAwareEval/corridor/0001.bmp"
#define SRC_IMAGE		"/home/julio/universidad/proyecto/proyecto/HarrisCornerRAwareEval/corridor/"
#define HARRIS_WINDOW_SIZE	3
#define dimOfBlock 16

typedef struct{
	int x;
	int y;
}punto;

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#ifndef MAX
#define MAX(x,y) ((x > y) ? x : y)
#endif

__global__ void GradientCalculation(int width,int height, const unsigned char *input,
		unsigned int *pixHist_, const int stop, int *d_p, int *d_q, int *d_pq,
		int BINS, int BINSp, int R);

__global__ void harrisResponseFunction(int diff, int width, int height, int *R,int *p,int *q, int *pq,int maskThreshold, unsigned char *pCovImagePixels, int despl);

void QuicksortInverse(int *pOffsets, const int *pValues, int nLow, int nHigh);

__global__ void goodPixels(int *d_data, int *d_pCandidateOffsets, int *d_aux, int *d_nCandidates, int width,int d_max);


__device__ void warpReduce(volatile int *sdata, unsigned int tid, int blockSize);
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n, int blockSize);
unsigned int nextPow2(unsigned int x);
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

#endif /* DEFINES_HPP_ */
