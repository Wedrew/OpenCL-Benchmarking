#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef __APPLE__
	#include <OpenCL/cl.hpp>
#else
	#include <CL/cl.h>
#endif


int main(void)
{
	const int N_ELEMENTS = 1024;
	int *A = malloc(sizeof(int)* N_ELEMENTS);
	int *B = malloc(sizeof(int)* N_ELEMENTS);
	int *C = malloc(sizeof(int)* N_ELEMENTS);

	for(int i = 0; i < N_ELEMENTS; ++i)
	{
		A[i] = i;
		B[i] = i;
	}






	return 0;
}