#include "defines.hpp"

__global__ void goodPixels(int *d_data, int *d_pCandidateOffsets, int *d_aux, int *d_nCandidates, int width,int d_max){

	int row= blockIdx.y*dimOfBlock+threadIdx.y;
	int col= blockIdx.x*dimOfBlock+threadIdx.x;
	int id=row*width+col;

	if(d_data[id]>=d_max){
		int temp = atomicAdd(&d_nCandidates[0],1);
		//d_aux[temp]=d_data[id];
		d_pCandidateOffsets[temp]=id;

		}
}
