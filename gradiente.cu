#include "defines.hpp"

extern __shared__ unsigned int Hs[];
__global__ void GradientCalculation(int width,int height, const unsigned char *input,
		unsigned int *pixHist_, const int stop, int *d_p, int *d_q, int *d_pq,
		int BINS, int BINSp, int R){//se le debe pasar el offset*

	int row= blockIdx.y*dimOfBlock+threadIdx.y;
	int col= blockIdx.x*dimOfBlock+threadIdx.x;
	int id=row*width+col;
	int temp=0;
	int size=width*height;

	///////////////////
	// Block and thread index
    const int bx = blockIdx.x;
    //const int tx = threadIdx.x;

    const int tx = threadIdx.y*dimOfBlock+threadIdx.x;

	//ty*blockdim+tx

    // Offset to per-block sub-histograms
    const unsigned int off_rep = BINSp * (tx % R);//total 11 bins, el modulo se multiplica por el tamaño del bin

	///////////////////
	for(int pos = tx; pos < BINSp*R; pos += blockDim.x*blockDim.y) Hs[pos] = 0;
	__syncthreads();        // Intra-block synchronization

	__shared__ unsigned char s_input[dimOfBlock][dimOfBlock];
	s_input[threadIdx.y][threadIdx.x]=input[id];

	__syncthreads();
	d_p[id]=0;
	d_q[id]=0;
	d_pq[id]=0;


	int p,q;
	if(id>=width+1 && id<stop  ){


		if(threadIdx.x==0){
			p = int(s_input[threadIdx.y][threadIdx.x+1]) - int(input[id - 1]);
		}
		else if(threadIdx.x==dimOfBlock-1){
			p = int(input[id + 1]) - int(s_input[threadIdx.y][threadIdx.x-1]);
		}
		else
			p = int(s_input[threadIdx.y][threadIdx.x+1]) - int(s_input[threadIdx.y][threadIdx.x-1]);

		//Columnas
		if(threadIdx.y==0){
			q = int(s_input[threadIdx.y+1][threadIdx.x]) - int(input[id - width]);
		}
		else if(threadIdx.y==dimOfBlock-1){
			q = int(input[id + width]) - int(s_input[threadIdx.y-1][threadIdx.x]);
		}
		else
			q = int(s_input[threadIdx.y+1][threadIdx.x]) - int(s_input[threadIdx.y-1][threadIdx.x]);

		d_p[id] = p * p;
		d_q[id] = q * q;
		d_pq[id] = p * q;

		/*temp = (abs(d_pq[id]));

		if(temp<=1023)
                	atomicAdd(&Hs[off_rep + temp], 1);
                else
                	atomicAdd(&Hs[off_rep + 1023], 1);*/
		}
	 /*__syncthreads();      // Intra-block synchronization

        // Merge per-block histograms and write to global memory

	for(int pos = tx; pos < BINS; pos += blockDim.x*blockDim.y){
		unsigned int sum = 0;
		for(int base = 0; base < BINSp*R; base += BINSp)
		        sum += Hs[base + pos];
		// Atomic addition in global memory
		atomicAdd(pixHist_ + pos, sum);
		}*/

}
