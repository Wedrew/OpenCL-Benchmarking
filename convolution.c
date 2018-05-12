#include <stdio.h>
#include "load_kernel.h"

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif


int main(void)
{
	//Host data
	float *A = NULL;

	char *KernelSource;
	char* file = "./multiply.cl";

	const size_t elements = 16;
	size_t globalSize = elements;
	size_t localSize = 64;
	size_t dataSize = sizeof(float)*elements;

	A = malloc(dataSize);

	long lFileSize = LoadOpenCLKernel(file, &KernelSource);
    if( lFileSize < 0L ) 
    {
        perror("File read failed");
        return 0;
    }

	for(int x = 0; x < elements; ++x)
	{
		A[x] = x+1.0;
	}

	//Use this to check the output of each API cal
	cl_int status;

	//Retrieve the number of platforms
	cl_uint numPlatforms = 0;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);

	//Allocate enough space or each platform
	cl_platform_id *platforms = NULL;
	platforms = malloc(numPlatforms * sizeof(cl_platform_id));

	//Fill the platforms
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);

	//Retrieve the number of devices
	cl_uint numDevices = 0;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

	//Allocated enough space for each device
	cl_device_id *devices;
	devices = malloc(numDevices*sizeof(cl_device_id));

	//Fill in the devices
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

	//Create a context and associate it with the devices
	cl_context context;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	//Create a command queue and associate it with the device
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);

	//Create a buffer object that will hold the output data cl_mem buffA
	cl_mem buffA;
	buffA = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize, NULL, &status);

	//Write input array A to the device buffer bufferA
	status = clEnqueueWriteBuffer(cmdQueue, buffA, CL_FALSE, 0, dataSize, A, 0, NULL, NULL);

	//Create a program with source code
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &status);

	//Build (compile) the program for the device
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	//Create the kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, "multiply", &status);

	//Associate the input output buffers with the kernel
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffA);

	//Execute the kernel for execution
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

	//Read the device output buffer to the host output array
	clEnqueueReadBuffer(cmdQueue, buffA, CL_TRUE, 0, dataSize, A, 0, NULL, NULL);

	for(int x = 0; x < elements; ++x)
	{
		printf("Testing: %f\n", A[x]);
	}

	// status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

	// //Read the device output buffer to the host output array
	// clEnqueueReadBuffer(cmdQueue, buffA, CL_TRUE, 0, dataSize, A, 0, NULL, NULL);

	// for(int x = 0; x < elements; ++x)
	// {
	// 	printf("Testing: %f\n", A[x]);
	// }

	clReleaseKernel(kernel); 
	clReleaseProgram(program); 
	clReleaseCommandQueue(cmdQueue); 
	clReleaseMemObject(buffA); 
	clReleaseContext(context);
	// Free host resources 
	free(A);
	free(platforms); 
	free(devices);


	return 0;
}