#include <stdio.h>
#include <stdlib.h>
#include <iostream>
//#define xlns16_ideal
#define xlns16_alt
#include "xlns16.cpp"
//#include "xlns16d.cu"

#define ITER (36)
//#define LNS_TESTS (1048576)
#define LNS_TESTS (2000)
//#define LNS_THREAD_CNT (32)
//#define LNS_CTA_CNT (256)

struct xlnsParams
{
 xlns16_float *arglnsx;
 xlns16_float *arglnsy;
 xlns16_float *reslns;
 int n;
};


/*
__global__ void xlns_kernel(struct xlnsParams parms)
{
 int i;
 int totalThreads = gridDim.x * blockDim.x;
 int ctaStart = blockDim.x * blockIdx.x;
 for (i = ctaStart + threadIdx.x; i < parms.n; i += totalThreads)  
  {
    parms.reslns[i] = parms.arglnsx[i]-parms.arglnsy[i];
  }
}
*/

/*Mandelbrot set */

xlns16_float manxlnsiter(xlns16_float x, xlns16_float y, int iter)
{
	int count;
	xlns16_float x1,y1,xnew,ynew,two,four,res;
	two = 2.0;
	four = 4.0;
			x1 = x;
			y1 = y;
			for (count=0; count<iter; count++)
			{
				xnew = x*x - y*y + x1;
				ynew = x*y*two + y1;
				res = x*x+y*y;
				x = (res < four) ? xnew : x;
				y = (res < four) ? ynew : y;
			}
	return res;
}


int main (int argc, char *argv[])
{
 int i;
 unsigned short x,y;
 FILE* f;

 //xlns16d_float * lnsRes = 0;
 //xlns16d_float * lnsArgx = 0;
 //xlns16d_float * lnsArgy = 0;
 //xlns16d_float * argx = 0;
 //xlns16d_float * argy = 0;
 //xlns16d_float * res = 0;
 xlns16_float * lnsRes = 0;
 xlns16_float * lnsArgx = 0;
 xlns16_float * lnsArgy = 0;
 xlns16_float * argx = 0;
 xlns16_float * argy = 0;
 xlns16_float * res = 0;
 struct xlnsParams funcParams;
 int ix,iy;
 xlns16_float four,xscale,yscale;

 //cudaMalloc ((void **)&lnsArgx, LNS_TESTS * sizeof(unsigned short));
 //cudaMalloc ((void **)&lnsArgy, LNS_TESTS * sizeof(unsigned short));
 //cudaMalloc ((void **)&lnsRes, LNS_TESTS * sizeof(unsigned short));

 argx = (xlns16_float *) malloc (LNS_TESTS * sizeof(argx[0]));
 argy = (xlns16_float *) malloc (LNS_TESTS * sizeof(argy[0]));
 res = (xlns16_float *) malloc (LNS_TESTS * sizeof(res[0]));

 for (i=0; i<LNS_TESTS; i++)
 {
   argx[i] = 0;
   argy[i] = 0;
 }
 i=0;
 four = 4.0;
 yscale = 12.0;
 xscale = 24.0;
 for (iy = 11; iy >= -11; iy--)
	{
		for (ix=-40; ix <= 38; ix++)
		{
			argy[i] = ((float)iy)/yscale;
			argx[i] = ((float)ix)/xscale;
			i++;
		}
	}
 //cudaMemcpy (lnsArgx, argx, LNS_TESTS * sizeof(argx[0]), cudaMemcpyHostToDevice);
 //cudaMemcpy (lnsArgy, argy, LNS_TESTS * sizeof(argy[0]), cudaMemcpyHostToDevice);

 funcParams.arglnsx = lnsArgx;
 funcParams.arglnsy = lnsArgy;
 funcParams.reslns = lnsRes;
 funcParams.n = LNS_TESTS;
 
 //printf("start CUDA\n");
 //xlns_kernel<<<LNS_CTA_CNT,LNS_THREAD_CNT>>>(funcParams);
 //printf("done CUDA\n");

 //cudaMemcpy (res, lnsRes, LNS_TESTS * sizeof(res[0]), cudaMemcpyDeviceToHost);

 for (i=0; i<LNS_TESTS; i++)
 {
   res[i] = manxlnsiter(argx[i], argy[i], ITER); 
 }

 i = 0;
 for (iy = 11; iy >= -11; iy--)
	{
		for (ix=-40; ix <= 38; ix++)
		{
			if (res[i] >= four)
				printf("*");
			else
				printf(" ");
			i++;
		}
		printf("\n");
	}

/*
 printf("writing mitchjunk3\n");
 f=fopen("mitchjunk3.txt","w");
 for (x=0; x<(1<<(K+F)); x++)
 {
     for (y=0; y<(1<<(K+F)); y++)
     {
	res[(x<<(K+F))+y] = xlns16_sub(argx[(x<<(K+F))+y],argy[(x<<(K+F))+y]);
        fprintf(f,"%04x\n",res[(x<<(K+F))+y]);
     }
 }
fclose(f);
*/
return 0;
}

