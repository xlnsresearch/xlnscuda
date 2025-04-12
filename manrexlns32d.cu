#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#define xlns32_ideal
//#define xlns32_alt 
#include "xlns32.cpp"
#include "xlns32d.cu"

#define ITER (37)
#define LNS_TESTS (1048576)
#define LNS_THREAD_CNT (32)
//#define LNS_CTA_CNT (256)
#define LNS_CTA_CNT (32768)

struct xlnsParams
{
 xlns32d_float *arglnsx;
 xlns32d_float *arglnsy;
 xlns32d_float *reslns;
 int n;
};



/*
 for (i=0; i<LNS_TESTS; i++)
 {
   res[i] = manxlnsiter(argx[i], argy[i], ITER); 
 }
*/


/*Mandelbrot set */

//__device__ xlns32d_float manxlnsiterd(xlns32d_float x, xlns32d_float y, int iter)
__device__ xlns32d_float manxlnsiterd(xlns32d_float x, xlns32d_float y)
{
	int count;
	xlns32d_float x1,y1,xnew,ynew,two,four,res;
	two = 2.0;
	four = 4.0;
	x1 = x;
	y1 = y;
	for (count=0; count<ITER; count++)
	{
		xnew = x*x - y*y + x1;
		ynew = x*y*two + y1;
		res = x*x+y*y;
		x = (res < four) ? xnew : x;
		y = (res < four) ? ynew : y;
	}
	return res;
}

__global__ void xlns_kernel(struct xlnsParams parms)
{
 int i;
 int totalThreads = gridDim.x * blockDim.x;
 int ctaStart = blockDim.x * blockIdx.x;
 for (i = ctaStart + threadIdx.x; i < parms.n; i += totalThreads)  
  {
    parms.reslns[i] = manxlnsiterd(parms.arglnsx[i],parms.arglnsy[i]);  //,ITER);
  }
}



int main (int argc, char *argv[])
{
 int i;
 //FILE* f;

 xlns32d_float * lnsRes = 0;
 xlns32d_float * lnsArgx = 0;
 xlns32d_float * lnsArgy = 0;
 xlns32_float * argx = 0;
 xlns32_float * argy = 0;
 xlns32_float * res = 0;
 xlns32_float zero;

 struct xlnsParams funcParams;
 int ix,iy;
 xlns32_float four,xscale,yscale;

 cudaMalloc ((void **)&lnsArgx, LNS_TESTS * sizeof(xlns32d_float));
 cudaMalloc ((void **)&lnsArgy, LNS_TESTS * sizeof(xlns32d_float));
 cudaMalloc ((void **)&lnsRes, LNS_TESTS * sizeof(xlns32d_float));

 argx = (xlns32_float *) malloc (LNS_TESTS * sizeof(argx[0]));
 argy = (xlns32_float *) malloc (LNS_TESTS * sizeof(argy[0]));
 res = (xlns32_float *) malloc (LNS_TESTS * sizeof(res[0]));
 zero = 0;

 for (i=0; i<LNS_TESTS; i++)
 {
   argx[i] = zero;
   argy[i] = zero;
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
 cudaMemcpy (lnsArgx, argx, LNS_TESTS * sizeof(argx[0]), cudaMemcpyHostToDevice);
 cudaMemcpy (lnsArgy, argy, LNS_TESTS * sizeof(argy[0]), cudaMemcpyHostToDevice);

 funcParams.arglnsx = lnsArgx;
 funcParams.arglnsy = lnsArgy;
 funcParams.reslns = lnsRes;
 funcParams.n = LNS_TESTS;
 
 printf("start CUDA\n");
 xlns_kernel<<<LNS_CTA_CNT,LNS_THREAD_CNT>>>(funcParams);
 printf("done CUDA\n");

 cudaMemcpy (res, lnsRes, LNS_TESTS * sizeof(res[0]), cudaMemcpyDeviceToHost);
 printf("done CUDA memcpy\n");

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


return 0;
}

