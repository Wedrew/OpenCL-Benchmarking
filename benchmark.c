#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "load_kernel.h"

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#define VECTOR_SIZE 134217728 //Increase this for more calculations (Must be of the form n^2 i.e. 4096*4096)
#define GPU 1 //Change this to 0 for integrated graphics card
const char* file = "./saxpy.cl";

void DisplayDevices(cl_device_id *deviceList, size_t valueSize, cl_uint maxComputeUnits);

int main(int argc, char *argv[])
{
    unsigned long long i;
    size_t valueSize;
    cl_uint maxComputeUnits;

    //Allocate spave for vectors A, B and C
    float alpha = 2.0;
    float *A = malloc(sizeof(float)*VECTOR_SIZE);
    float *B = malloc(sizeof(float)*VECTOR_SIZE);
    float *C = malloc(sizeof(float)*VECTOR_SIZE);

    size_t globalSize = VECTOR_SIZE; // Process the entire lists
    size_t localSize = 64;           // Process one item at a time
    char *KernelSource;
    long lFileSize;
    cl_ulong gpuStart;
    cl_ulong gpuEnd;
    cl_event event;

    lFileSize = LoadOpenCLKernel(file, &KernelSource);
    if( lFileSize < 0L ) 
    {
        perror("File read failed");
        return 0;
    }

    //Init array values
    for(i = 0; i < VECTOR_SIZE; i++)
    {
        A[i] = i;
        B[i] = VECTOR_SIZE - i;
    }
    
    // Get platform and device information
    cl_platform_id *platforms = NULL;
    cl_uint numPlatforms;

    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = malloc(sizeof(cl_platform_id)*numPlatforms);
    clStatus = clGetPlatformIDs(numPlatforms, platforms, NULL);
    //Get the devices list and choose the device you want to run on
    cl_device_id *deviceList = NULL;
    cl_uint numDevices;

    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    deviceList = malloc(sizeof(cl_device_id)*numDevices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, deviceList, NULL);

    DisplayDevices(deviceList, valueSize, maxComputeUnits);

    // Create one OpenCL context for each device in the platform
    cl_context context;
    context = clCreateContext(NULL, numDevices, deviceList, NULL, NULL, &clStatus);
    // Create a command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceList[GPU], CL_QUEUE_PROFILING_ENABLE, &clStatus);
    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(commandQueue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(commandQueue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL);
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &clStatus);
    // Build the program
    clStatus = clBuildProgram(program, numDevices, deviceList, NULL, NULL, NULL);
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "saxpy", &clStatus);
    // Set the arguments of the kernel
    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem),(void *)&A_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem),(void *)&C_clmem);

    // Execute the OpenCL kernel on the list
    printf("Executing on GPU...\n");
    clStatus = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &event);
    clWaitForEvents(1, &event);
    // Read the cl memory C_clmem on device to the host variable C
    clStatus = clEnqueueReadBuffer(commandQueue, C_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), C, 0, NULL, NULL);
    // Clean up and wait for all the comands to complete.
    clStatus = clFinish(commandQueue);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(gpuStart), &gpuStart, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(gpuEnd), &gpuEnd, NULL);

    double gpuRunTime = (float)(gpuEnd-gpuStart) / 1000000000.0;
    printf("GPU Runtime: %0.6f seconds\n", gpuRunTime);

    // //Display the result to the screen
    // for(i = 0; i < VECTOR_SIZE; ++i)
    // {
    //   printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);
    // }

    //Cpu equivalent of saxpy
    printf("Executing on CPU...\n");
    clock_t cpuBegin = clock();
    for (i = 0; i < VECTOR_SIZE; ++i)
    {
       C[i] = alpha * A[i] + C[i];
    }
    clock_t cpuEnd = clock();

    double cpuRunTime = (double)(cpuEnd - cpuBegin) / CLOCKS_PER_SEC;
    printf("CPU Runtime: %0.6f seconds\n", cpuRunTime);

    if(gpuRunTime > cpuRunTime)
        printf("Your CPU performed %0.2f times better than your GPU\n", gpuRunTime/cpuRunTime);
    else if(cpuRunTime > gpuRunTime)
        printf("Your GPU performed %0.2f times better than your CPU\n", cpuRunTime/gpuRunTime);
    else
        printf("It's a tie!");
    printf("----------------------------\n");

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    clStatus = clReleaseCommandQueue(commandQueue);
    clStatus = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    free(platforms);
    free(deviceList);
    return 0;
}

void DisplayDevices(cl_device_id *deviceList, size_t valueSize, cl_uint maxComputeUnits)
{
    char* value;
    printf("----------------------------\n");
    // print device name
    clGetDeviceInfo(deviceList[GPU], CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(deviceList[GPU], CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device: %s\n", value);

    // print hardware device version
    clGetDeviceInfo(deviceList[GPU], CL_DEVICE_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(deviceList[GPU], CL_DEVICE_VERSION, valueSize, value, NULL);
    printf("Hardware version: %s\n", value);

    // print software driver version
    clGetDeviceInfo(deviceList[GPU], CL_DRIVER_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(deviceList[GPU], CL_DRIVER_VERSION, valueSize, value, NULL);
    printf("Software version: %s\n", value);

    // print c version supported by compiler for device
    clGetDeviceInfo(deviceList[GPU], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(deviceList[GPU], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
    printf("OpenCL C version: %s\n", value);

    // print parallel compute units
    clGetDeviceInfo(deviceList[GPU], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Parallel compute units: %d\n", maxComputeUnits);
    free(value);
}

