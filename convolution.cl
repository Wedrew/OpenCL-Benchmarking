__kernel void convolution(
	__constant int *kernel,
	__global float *output,
	__global float *image,
	__private const height,
	__private const width)
{
	long id = get_global_id(0);


}