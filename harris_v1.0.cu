/*
 ============================================================================
 Name        : harris_v1.0.cu
 Author      : Julio Gonzalez Carranza
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */


#include "defines.hpp"


#define __TIEMPO__KERNELS__ false
#define __TIEMPO__TOTAL__ true
#define __GRADIENTE__ 3
#define __HISTOGRAMA__ CPU
#define __UMBRAL__ CPU
#define __HARRIS__ 3
#define __MAXIMO__GPU__
//#define __MAXIMO__CPU__
//#define __GOODPIXELS__CPU__
#define __GOODPIXELS__GPU__
#define __SORT__ CPU

using namespace std;
void GenFileListSorted(char *dirName, char **fileList, int *count)
{
	struct dirent **namelist;
	int n, fileCnt = 0;

	n = scandir(dirName, &namelist, 0, alphasort);
	if (n < 0)
	{
		//printf(" [SMD][ER]Unable to open directory \n");
		*count = 0;
	}
	else
	{
		int iter = 0;

		while (iter < n)
		{
			if((strcmp(namelist[iter]->d_name, ".") != 0) && (strcmp(namelist[iter]->d_name, "..") != 0))
			{
				// process only .bmp files
				if(strstr(namelist[iter]->d_name, ".bmp"))
				{
					fileList[fileCnt] = (char *)malloc(strlen(dirName) + strlen(namelist[iter]->d_name) + 2);
					if(!fileList[fileCnt])
					{
						//printf(" ERROR: malloc failed ! \n");
						return;
					}
		  			strcpy(fileList[fileCnt], dirName);
		  			strcat(fileList[fileCnt], "/");
		  			strcat(fileList[fileCnt], namelist[iter]->d_name);
					printf(" [SMD][OK]File name: %s \n", fileList[fileCnt]);
					free(namelist[iter]);
					fileCnt++;
				}
			}
			iter++;
		}
		*count = fileCnt;
	}
}

void ReadBMP(char* filename, unsigned char* grayData, int &ancho, int &alto)
{
    int i;
    FILE* f = fopen(filename, "rb");

    if(f == NULL)
        throw "Argument Exception";

    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    /*cout << endl;
    cout << "  Name: " << filename << endl;
    cout << " Width: " << width << endl;
    cout << "Height: " << height << endl;*/
    ancho = width;
    alto = height;
    int row_padded = (width*3 + 3) & (~3);
    unsigned char* data = new unsigned char[row_padded];

    unsigned char tmp;

    for(int i = 0; i < height; i++)
    {
        fread(data, sizeof(unsigned char), row_padded, f);
        for(int j = 0; j < width*3; j += 3)
        {
            // Convert (B, G, R) to (R, G, B)
            tmp = data[j];
            data[j] = data[j+2];
            data[j+2] = tmp;
            //cout << "[" << i*width+j/3 << "]= " << (data[j]+data[j+1]+data[j+2])/3 << endl;

            //funciona
            //grayData[i*width+j/3] = (int)(data[j]+data[j+1]+data[j+2])/3;
            //prueba
            grayData[i*width+j/3]=(data[j]+(data[j+1]<<1)+data[j+2]+2)>>2;
            //funciona
            //grayData[i*width+j/3] = int(data[j]*0.299+data[j+1]*0.587+data[j+2]*0.114);

            //cout << "[" << i*width+j/3 << "]" << endl;
            //cout << "R: "<< (int)data[j] << " G: " << (int)data[j+1]<< " B: " << (int)data[j+2]<< endl;
            //cout << "data["<< i << "," << j/3 << "]= " << (unsigned int)grayData[i*width+j/3] << endl;
            //printf("data: %d\n",grayData[i*width+j/3]);
        }
    }

    fclose(f);

}

int cornerDetector(unsigned char* grayImage, vector<punto> puntosDeInteres, const int ancho, const int alto){

	const unsigned char *input = grayImage;
	unsigned int *pixHist;
	const int nPixels = ancho*alto;
	const int width = ancho;
	const int height = alto;
	const int stop = nPixels - width - 1;
	pixHist=(unsigned int*)malloc(sizeof(unsigned int)*1024);
	for(int i = 0; i < 1024; i++)
			pixHist[i] = 0;

	#if __TIEMPO__KERNELS__
	float milliseconds = 0;
	cudaEvent_t e_start, e_stop;
	cudaEventCreate(&e_start);
	cudaEventCreate(&e_stop);
	#endif
	int *d_p;
	int *d_q;
	int *d_pq;

	unsigned int *d_pixHist;
	cudaMalloc(&d_pixHist,1024*sizeof(unsigned int));


	cudaMemcpy(d_pixHist,pixHist,1024*sizeof(unsigned int),cudaMemcpyHostToDevice);



	/**
	 * Declaracion de los vectores p, q y pq para accesos coalescentes a memoria
	 * de la gpu
	 *
	 */
	int *p = new int[nPixels];
	int *q = new int[nPixels];
	int *pq = new int[nPixels];
	cudaMalloc(&d_p,nPixels*sizeof(int));
	cudaMalloc(&d_q,nPixels*sizeof(int));
	cudaMalloc(&d_pq,nPixels*sizeof(int));

	/**
	 * Declaracion del puntero d_input
	 * este puntero
	 */
	unsigned char *d_input;
	cudaMalloc(&d_input,nPixels*sizeof(unsigned char));
	cudaMemcpy(d_input,input,nPixels*sizeof(unsigned char),cudaMemcpyHostToDevice);

	/**
	 * Declaracion de dimensiones de grid y tamaño de bloques
	 */
	dim3 dimGrid(width/dimOfBlock,height/dimOfBlock);//numero de tiles
   	dim3 dimBlock(dimOfBlock,dimOfBlock);//tamanio de los tiles

	#if __TIEMPO__KERNELS__
   	cudaEventRecord(e_start);
	#endif

	#if __GRADIENTE__ == 3
	GradientCalculation<<<dimGrid,dimBlock,1025*sizeof(unsigned int)*3>>>(width, height, d_input, d_pixHist, stop, d_p, d_q, d_pq, 1024, 1025, 3);
	#endif

	#if __TIEMPO__KERNELS__
	cudaEventRecord(e_stop);
	cudaEventSynchronize(e_stop);
	milliseconds=0;
	cudaEventElapsedTime(&milliseconds, e_start, e_stop);
	printf("calculo gradiente: %f\n",milliseconds);
	#endif

	/*cudaMemcpy(pixHist,d_pixHist,1024*sizeof(unsigned int),cudaMemcpyDeviceToHost);

	#if __TIEMPO__KERNELS__
	cudaEventRecord(e_start);
	#endif

	#if __HISTOGRAMA__ == CPU
	int tempVal;
	tempVal=0;
	for(int i = 1023; i >=0 ; i--)
		{

		tempVal = tempVal + pixHist[i];

		pixHist[i] = tempVal;
	}
	#endif


	int maskThreshold = 1023;
	int processed = 0;
	int workload = 100;
	for(int i = 0; i < 1024; i++)
	{
		if(((pixHist[i] * 100) / nPixels) <= workload)
		{
			maskThreshold = i;
			processed = ((pixHist[i] * 100) / nPixels);
			break;
		}
	}

	cudaFree(d_pixHist);
	#if __TIEMPO__KERNELS__
	cudaEventRecord(e_stop);
	cudaEventSynchronize(e_stop);
	milliseconds=0;
	cudaEventElapsedTime(&milliseconds, e_start, e_stop);
	printf("hist+thres: %f\n",milliseconds);
	#endif
	/*printf("libera pixhist\n");
	cudaEventRecord(e_stop);
	cudaEventSynchronize(e_stop);
	milliseconds=0;
	cudaEventElapsedTime(&milliseconds, e_start, e_stop);
	printf("calculo hist y umbral: %f\n",milliseconds);*/

	const int diff = HARRIS_WINDOW_SIZE + 1;
	int pOutputImage[width*height];
	/*for(int i=0;i<nPixels;i++){
		pOutputImage[i]=0;
	}*/
	int despl = (HARRIS_WINDOW_SIZE / 2) * (width + 1);
	int *R = pOutputImage + (HARRIS_WINDOW_SIZE / 2) * (width + 1);

	int *d_R;
	unsigned char *d_pCovImage;

	cudaMalloc(&d_R,nPixels * sizeof(int));

	//cudaMalloc(&d_pCovImage,nPixels*sizeof(unsigned char));
	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////

	#if __TIEMPO__KERNELS__
	cudaEventRecord(e_start);
	#endif

	#if __HARRIS__ == 3
	harrisResponseFunction<<<dimGrid,dimBlock>>>(diff, width, height, d_R, d_p, d_q, d_pq, 1023,d_pCovImage,despl);
	#endif

	#if __TIEMPO__KERNELS__
	cudaEventRecord(e_stop);
	cudaEventSynchronize(e_stop);
	milliseconds=0;
	cudaEventElapsedTime(&milliseconds, e_start, e_stop);
	printf("calculo harris: %f\n",milliseconds);
	#endif

	//#ifdef __MAXIMO__CPU__
	cudaMemcpy(R,d_R+ (HARRIS_WINDOW_SIZE / 2) * (width + 1),(nPixels - (HARRIS_WINDOW_SIZE / 2) * (width + 1))*sizeof(int),cudaMemcpyDeviceToHost);
	//#endif
	cudaFree(d_p);
	cudaFree(d_q);
	cudaFree(d_pq);
	//cudaFree(d_R);
	cudaFree(d_pCovImage);


	int *data = pOutputImage;
	int *d_data=d_R;
	//cudaMalloc(&d_data,nPixels*sizeof(int));


	//cudaMemcpy(d_data,data,nPixels*sizeof(int),cudaMemcpyHostToDevice);

	#if __TIEMPO__KERNELS__
	cudaEventRecord(e_start);
	#endif
	// determine maximum value
	int max[1];
	int maximum=0;
	#ifdef __MAXIMO__GPU__



	int numBlocks=0;
	int numThreads=0;
	getNumBlocksAndThreads(6, nPixels, 32, 64, numBlocks, numThreads);


	int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(int) : numThreads *sizeof(int);

	int h_odata[numBlocks];

	for (int i=0;i<numBlocks;i++)
		h_odata[i]=0;

	int *d_odata;
	cudaMalloc(&d_odata,numBlocks*sizeof(int));

	cudaMemcpy(d_odata,h_odata,numBlocks*sizeof(int),cudaMemcpyHostToDevice);

	dim3 adimBlock(numThreads, 1);
    	dim3 adimGrid(numBlocks, 1);

	reduce<<< adimGrid, adimBlock, smemSize >>>(d_data, d_odata,(unsigned int)nPixels,numThreads);



	cudaMemcpy(h_odata,d_odata,numBlocks*sizeof(int),cudaMemcpyDeviceToHost);


	for (int i=0; i<numBlocks; i++)
            {
                max[0]=maximum=MAX(h_odata[i],maximum );

            }
	#endif

	#ifdef __MAXIMO__CPU__
	maximum = 0;
	for (int i = 0; i < nPixels; ++i)
	{
		if (data[i] > maximum)
			maximum = data[i];

	}
	#endif

	#if __TIEMPO__KERNELS__
	cudaEventRecord(e_stop);
	cudaEventSynchronize(e_stop);
	milliseconds=0;
	cudaEventElapsedTime(&milliseconds, e_start, e_stop);
	printf("maximo: %f\n",milliseconds);
	#endif

	//int *nCandidates = new int[1];
	max[0]=maximum;
	max[0] = int(max[0] * 0.005f);// + 0.5f
	int *pCandidateOffsets = new int[nPixels];
	int *nCandidates = new int[1];
	int *pCandidateOffsets2 = new int[nPixels];
	int *nCandidates2 = new int[1];
	nCandidates[0]=0;
	nCandidates2[0]=0;
	// only accept good pixels

	#ifdef  __GOODPIXELS__CPU__
		for (int i = 0; i < nPixels; i++)
		{
			if (data[i] >= max[0])
				pCandidateOffsets[nCandidates[0]++] = i;
		}
	#endif

	#ifdef  __GOODPIXELS__GPU__
		int *d_pCandidateOffsets;
		int *d_nCandidates;
		int *d_aux;
		cudaMalloc(&d_pCandidateOffsets,nPixels*sizeof(int));
		cudaMalloc(&d_nCandidates,sizeof(int)*2);
		//cudaMalloc(&d_aux,nPixels*sizeof(int));

		cudaMemcpy(d_nCandidates,nCandidates,sizeof(int)*2,cudaMemcpyHostToDevice);
		#if __TIEMPO__KERNELS__
		cudaEventRecord(e_start);
		#endif
		goodPixels<<<dimGrid,dimBlock>>>(d_data,d_pCandidateOffsets, d_aux,d_nCandidates,width,max[0]);
		#if __TIEMPO__KERNELS__
		cudaEventRecord(e_stop);
		cudaEventSynchronize(e_stop);
		milliseconds=0;
		cudaEventElapsedTime(&milliseconds, e_start, e_stop);
		printf("good pixels: %f\n",milliseconds);
		#endif
		cudaMemcpy(nCandidates,d_nCandidates,sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(pCandidateOffsets,d_pCandidateOffsets,sizeof(int)*nCandidates[0],cudaMemcpyDeviceToHost);
		//cudaMemcpy(aux,d_aux,sizeof(int)*nPixels,cudaMemcpyDeviceToHost);

		cudaFree(d_pCandidateOffsets);
		cudaFree(d_nCandidates);
		//cudaFree(d_aux);
	#endif

/*	if(nCandidates[0] == nCandidates2[0]) cout << "misma cantidad" << endl;
	else cout << "distinta cantidad" << endl;*/

	// sort by quality
	//cout << nCandidates[0] << "-+-+-+"<<endl;
#if __TIEMPO__KERNELS__
		cudaEventRecord(e_start);
		#endif
	#if __SORT__ == CPU
	QuicksortInverse(pCandidateOffsets, data, 0, nCandidates[0] - 1);
	//QuicksortInverse(pCandidateOffsets2, data, 0, nCandidates2[0] - 1);
	/*sort(pCandidateOffsets);
	sort();*/
#if __TIEMPO__KERNELS__
		cudaEventRecord(e_stop);
		cudaEventSynchronize(e_stop);
		milliseconds=0;
		cudaEventElapsedTime(&milliseconds, e_start, e_stop);
		printf("quicksort: %f\n",milliseconds);
		#endif
	#endif
	/*bool fallo;
	fallo=false;
	for (int i=0;i<nCandidates[0];i++){
		if(pCandidateOffsets[i]!=pCandidateOffsets2[i] && fallo==false) {
			fallo=true;
			cout << pCandidateOffsets[i] << "+" << pCandidateOffsets[i+1] <<"+" <<  pCandidateOffsets[i+2] << " | " << pCandidateOffsets2[i] << "+" <<pCandidateOffsets2[i+1] << "+" << pCandidateOffsets2[i+2]<< " | " << i << endl;
			cout << data[pCandidateOffsets[i]] << "+" << data[pCandidateOffsets[i+1]] <<"+" <<  data[pCandidateOffsets[i+2]] << " | " << data[pCandidateOffsets2[i]] << "+" << data[pCandidateOffsets2[i+1]] << "+" << data[pCandidateOffsets2[i+2]]<< " | " << i << endl;
			cout << endl;
		}
	}*/
	cudaFree(d_R);
	cudaFree(d_data);

	//printf(" %f\t",milliseconds);
	float fMinDistance = 5.0f;
	const int nMinDistance = int(fMinDistance );//+ 0.5f
#if __TIEMPO__KERNELS__
		cudaEventRecord(e_start);
		#endif
	unsigned char image[nPixels];
	for (int i=0;i<nPixels;i++) image[i]=0;
	int nInterestPoints = 0;
	const int nMaxPoints=700;
	for (int i = 0; i < nCandidates[0] && nInterestPoints < nMaxPoints; i++)
	{
		const int offset = pCandidateOffsets[i];

		const int x = offset % width;
		const int y = offset / width;

		bool bTake = true;

		const int minx = x - nMinDistance < 0 ? 0 : x - nMinDistance;
		const int miny = y - nMinDistance < 0 ? 0 : y - nMinDistance;
		const int maxx = x + nMinDistance >= width ? width - 1 : x + nMinDistance;
		const int maxy = y + nMinDistance >= height ? height - 1 : y + nMinDistance;
		const int diff = width - (maxx - minx + 1);

		for (int l = miny, offset2 = miny * width + minx; l <= maxy; l++, offset2 += diff)
			for (int k = minx; k <= maxx; k++, offset2++)
				if (image[l * width + k]==1)
				{
					bTake = false;
					break;
				}

		if (bTake)
		{
			// store  point
			puntosDeInteres[nInterestPoints].x = float(x);
			puntosDeInteres[nInterestPoints].y = float(y);
			nInterestPoints++;

			// mark location in grid for distance constraint check
			image[offset] = 1;
		}
	}
#if __TIEMPO__KERNELS__
		cudaEventRecord(e_stop);
		cudaEventSynchronize(e_stop);
		milliseconds=0;
		cudaEventElapsedTime(&milliseconds, e_start, e_stop);
		printf("puntos finales: %f\n",milliseconds);
		#endif
	/*for (int i=0;i<nPixels;i++){
		if(image.pixels[i]==1)
			printf("%d ,%d\n",image.pixels[i], i);
		}*/
	//printf("%d\t",nInterestPoints);
	//cudaFree(raw_pointer_cast(d_pCandidateOffsets.data()));
	free(pCandidateOffsets);
	return nInterestPoints;
}



class harrisCornerDetector{
public:
	harrisCornerDetector(): m_fQualityThreshold(0.005f), m_nMaxPoints(700), maskThreshold(0), workload(40)
	{

	}

	int run(){
		char *fileList[250];
		int imgIndex = 0, count = 0;
		//GenFileListSorted("./InputImages", fileList, &count);
		GenFileListSorted(SRC_IMAGE, fileList, &count);

		int maskThreshold = 0;
		int processed, index = 0;
		cudaEvent_t start, stop;
		int contador=0;
		float milliseconds=0;

		vector<punto> puntosDeInteres(700);
		int ancho, alto;
		while(contador < 3)
			{
				// main loop
				//cout << "count: " << count << endl;
				while(index < count)
				{
					unsigned char *grayImage = new unsigned char[640*480];

					/**
					 * Convierte la imagen fileList[index] a escala de grises y la guarda
					 * en grayimage
					 */
					ReadBMP(fileList[index], grayImage, ancho, alto);
					//cout << ancho << "-" << alto << endl;



					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					#if __TIEMPO__TOTAL__
					cudaEventRecord(start);
					#endif
					const int nPuntos=cornerDetector(grayImage,puntosDeInteres, ancho, alto);
					#if __TIEMPO__TOTAL__
					cudaEventRecord(stop);
					cudaEventSynchronize(stop);
					milliseconds=0;
					cudaEventElapsedTime(&milliseconds, start, stop);
					printf("TIEMPO TOTAL: %f\n",milliseconds);
					#endif
					cout << fileList[index] << " - numero de puntos: " << nPuntos << endl;
					//printf("%d\t",nPuntos);
					//printf("tiempo: %f\n",milliseconds);

					index++;
					free(grayImage);
				}
				index = 0;
				contador++;
			}

		return 0;
	}

private:
	float m_fQualityThreshold;
	int m_nMaxPoints;
	int maskThreshold;
	int workload;
};


int main(void)

{
	harrisCornerDetector demo;
	return demo.run();

}



