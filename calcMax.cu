#include "defines.hpp"


__device__ void warpReduce(volatile int *sdata, unsigned int tid, int blockSize) {

if (blockSize >= 64)
	/*if(sdata[tid]<sdata[tid+32])
		sdata[tid] = sdata[tid + 32];*/
		sdata[tid]=MAX(sdata[tid],sdata[tid+32]);

if (blockSize >= 32)
	/*if(sdata[tid]<sdata[tid+16])
		sdata[tid] = sdata[tid + 16];*/
		sdata[tid]=MAX(sdata[tid],sdata[tid+16]);
if (blockSize >= 16)
	/*if(sdata[tid]<sdata[tid+8])
		sdata[tid] = sdata[tid + 8];*/
		sdata[tid]=MAX(sdata[tid],sdata[tid+8]);
if (blockSize >= 8)
		/* if(sdata[tid]<sdata[tid+4])
			sdata[tid] = sdata[tid + 4];*/
		sdata[tid]=MAX(sdata[tid],sdata[tid+4])	;
if (blockSize >= 4)
	 /*if(sdata[tid]<sdata[tid+2])
		sdata[tid] = sdata[tid + 2];*/
		sdata[tid]=MAX(sdata[tid],sdata[tid+2]);
if (blockSize >= 2)
	/*if(sdata[tid]<sdata[tid+1])
		sdata[tid] = sdata[tid + 1];*/
		sdata[tid]=MAX(sdata[tid],sdata[tid+1]);

}


extern __shared__  int sdata[];
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n, int blockSize) {

	extern __shared__  int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;

	while (i < n) {

		  sdata[tid]=MAX(sdata[tid],g_idata[i]);

		  //if (i + blockSize < n){
		  sdata[tid]=MAX(sdata[tid],g_idata[i+blockSize]);
		  //}
		  i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid]=MAX(sdata[tid],sdata[tid+256]);
		 }
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid]=MAX(sdata[tid],sdata[tid+128]);
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid]=MAX(sdata[tid],sdata[tid+64]);
		}
		__syncthreads();
	 }

	if (tid < 32) warpReduce(sdata, tid,blockSize);

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}





//////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size excceed the upbound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = MIN(maxBlocks, blocks);
    }

}

