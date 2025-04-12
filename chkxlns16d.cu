#include <stdlib.h>
#include <stdio.h>
//#define xlns16_ideal
//#define xlns16_alt
#include "xlns16.cpp"
#include "xlns16d.cu"

#define LNS_TESTS (1048576)
#define LNS_THREAD_CNT (32)
#define LNS_CTA_CNT (32768)
//#define LNS_CTA_CNT (256)
#define F 7
#define K (3)

struct lnsParams
{
 unsigned short *arglnsx;
 unsigned short *arglnsy;
 unsigned short *reslns;
 int n;
 char operand;
};

__global__ void lns_kernel(struct lnsParams parms)
{
 int i;
 int totalThreads = gridDim.x * blockDim.x;
 int ctaStart = blockDim.x * blockIdx.x;
 for (i = ctaStart + threadIdx.x; i < parms.n; i += totalThreads)  
  {
    parms.reslns[i] = 	(parms.operand=='*') ? 	xlns16d_mul(parms.arglnsx[i],parms.arglnsy[i]):
    			(parms.operand=='/') ? 	xlns16d_div(parms.arglnsx[i],parms.arglnsy[i]):
    			(parms.operand=='+') ? 	xlns16d_add(parms.arglnsx[i],parms.arglnsy[i]):
    						xlns16d_sub(parms.arglnsx[i],parms.arglnsy[i]);
  }
}


int main (int argc, char *argv[])
{
 int i;
 unsigned short x,y;
 FILE* f;
 char operand;

 unsigned short* lnsRes = 0;
 unsigned short* lnsArgx = 0;
 unsigned short* lnsArgy = 0;
 unsigned short* argx = 0;
 unsigned short* argy = 0;
 unsigned short* res = 0;
 struct lnsParams funcParams;

 if (argc>1) operand = argv[1][0];
        else operand = '*';

 printf("chkxlns16d %c %d %d %d %d\n",operand,operand=='+',operand=='-',operand=='*',operand=='/');

 cudaMalloc ((void **)&lnsArgx, LNS_TESTS * sizeof(unsigned short));
 cudaMalloc ((void **)&lnsArgy, LNS_TESTS * sizeof(unsigned short));
 cudaMalloc ((void **)&lnsRes, LNS_TESTS * sizeof(unsigned short));

 argx = (unsigned short *) malloc (LNS_TESTS * sizeof(argx[0]));
 argy = (unsigned short *) malloc (LNS_TESTS * sizeof(argy[0]));
 res = (unsigned short *) malloc (LNS_TESTS * sizeof(res[0]));
 for (i=0; i<LNS_TESTS; i++)
 {
   argx[i] = 0x4000|(i >> (K+F));
   argy[i] = 0x4000|(i & ((1 << (K+F))-1));
 }
 cudaMemcpy (lnsArgx, argx, LNS_TESTS * sizeof(argx[0]), cudaMemcpyHostToDevice);
 cudaMemcpy (lnsArgy, argy, LNS_TESTS * sizeof(argy[0]), cudaMemcpyHostToDevice);

 funcParams.arglnsx = lnsArgx;
 funcParams.arglnsy = lnsArgy;
 funcParams.reslns = lnsRes;
 funcParams.n = LNS_TESTS;
 funcParams.operand = operand;

 printf("start CUDA\n");
 lns_kernel<<<LNS_CTA_CNT,LNS_THREAD_CNT>>>(funcParams);
 printf("done CUDA\n");

 cudaMemcpy (res, lnsRes, LNS_TESTS * sizeof(res[0]), cudaMemcpyDeviceToHost);

 printf("writing wrongxlns\n");
 f=fopen("wrongxlns.txt","w");
 int wrong = 0;
 for (x=0; x<(1<<(K+F)); x++)
 {
     for (y=0; y<(1<<(K+F)); y++)
     {
	if (((operand=='*')&&(res[(x<<(K+F))+y] != xlns16_mul(argx[(x<<(K+F))+y],argy[(x<<(K+F))+y]))) ||
	    ((operand=='/')&&(res[(x<<(K+F))+y] != xlns16_div(argx[(x<<(K+F))+y],argy[(x<<(K+F))+y]))) ||
	    ((operand=='+')&&(res[(x<<(K+F))+y] != xlns16_add(argx[(x<<(K+F))+y],argy[(x<<(K+F))+y]))) ||
	    ((operand=='-')&&(res[(x<<(K+F))+y] != xlns16_sub(argx[(x<<(K+F))+y],argy[(x<<(K+F))+y]))))
	{ wrong++;
          fprintf(f,"%04x %04x %04x!=%04x\n", argx[(x<<(K+F))+y],argy[(x<<(K+F))+y],
			  		res[(x<<(K+F))+y],
					(operand=='*')?xlns16_mul(argx[(x<<(K+F))+y],argy[(x<<(K+F))+y]):
					(operand=='/')?xlns16_div(argx[(x<<(K+F))+y],argy[(x<<(K+F))+y]):
					(operand=='+')?xlns16_add(argx[(x<<(K+F))+y],argy[(x<<(K+F))+y]):
					               xlns16_sub(argx[(x<<(K+F))+y],argy[(x<<(K+F))+y]));
	}
     }
 }
 printf("%i wrong\n",wrong); 
 fclose(f);

// for (x=0; x<(1<<(K+F)); x++)
//	 printf("%d ",res[x]-0x4000-x);
return 0;
}

