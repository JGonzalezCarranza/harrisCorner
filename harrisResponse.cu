#include "defines.hpp"

__global__ void harrisResponseFunction(int diff, int width, int height, int *R,int *p,int *q, int *pq,int maskThreshold, unsigned char *pCovImagePixels, int despl){

	int row= blockIdx.y*dimOfBlock+threadIdx.y;
	int col= blockIdx.x*dimOfBlock+threadIdx.x;

	int id=row*width+col;
	int *d_R;
	d_R=R;
	d_R[id]=0;
	d_R=d_R+despl;
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	__shared__ int s_p[dimOfBlock+3][dimOfBlock+3];
	__shared__ int s_q[dimOfBlock+3][dimOfBlock+3];
	__shared__ int s_pq[dimOfBlock+3][dimOfBlock+3];

	s_p[idy][idx]=p[id];
	s_q[idy][idx]=q[id];
	s_pq[idy][idx]=pq[id];

	if(threadIdx.x==dimOfBlock-1){
		s_p[idy][idx+1]=p[id+1];
		s_q[idy][idx+1]=q[id+1];
		s_pq[idy][idx+1]=pq[id+1];

		s_p[idy][idx+2]=p[id+2];
		s_q[idy][idx+2]=q[id+2];
		s_pq[idy][idx+2]=pq[id+2];
	}
	if(threadIdx.y==dimOfBlock-1){
		s_p[idy+1][idx]=p[id+width];
		s_q[idy+1][idx]=q[id+width];
		s_pq[idy+1][idx]=pq[id+width];

		s_p[idy+2][idx]=p[id+width*2];
		s_q[idy+2][idx]=q[id+width*2];
		s_pq[idy+2][idx]=pq[id+width*2];

		if(threadIdx.x==dimOfBlock-1){
			s_p[idy+1][idx+1]=p[id+width+1];
			s_q[idy+1][idx+1]=q[id+width+1];
			s_pq[idy+1][idx+1]=pq[id+width+1];

			s_p[idy+1][idx+2]=p[id+width+2];
			s_q[idy+1][idx+2]=q[id+width+2];
			s_pq[idy+1][idx+2]=pq[id+width+2];

			s_p[idy+2][idx+1]=p[id+width*2+1];
			s_q[idy+2][idx+1]=q[id+width*2+1];
			s_pq[idy+2][idx+1]=pq[id+width*2+1];

			s_p[idy+2][idx+2]=p[id+width*2+2];
			s_q[idy+2][idx+2]=q[id+width*2+2];
			s_pq[idy+2][idx+2]=pq[id+width*2+2];
		}
	}

	__syncthreads();

	//if(id>=width+1 && col <=width-diff-1 && row <=height-diff-1 && col>=1){
	if(row >= 2 && row < height-2 && col >= 2 && col < width-2 ){

		int temp;
		temp=(abs(s_pq[idy][idx]));
		int qqsum=0;
		int ppsum=0;
		int pqsum=0;
		/*if(temp < maskThreshold){
			pCovImagePixels[id]=0;
			return;
			}
		else pCovImagePixels[id]=255;*/

		for (int i=0;i<3;i++){
			for(int j=0;j<3;j++){
				ppsum+=s_p[idy+i][idx+j];
				qqsum+=s_q[idy+i][idx+j];
				pqsum+=s_pq[idy+i][idx+j];
			}
		}

		pqsum=abs(pqsum);
		ppsum >>= 4;
		qqsum >>= 4;
		pqsum >>= 4;
		d_R[id] = (ppsum * qqsum - pqsum * pqsum) - (((ppsum + qqsum) >> 2) * ((ppsum + qqsum) >> 2));

		}
}
