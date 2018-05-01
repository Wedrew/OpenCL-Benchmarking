__kernel void saxpy(
	__private const float alpha,
	__global const float *A,
	__global const float *B,
	__global float *C)
{
	//Get the gid of the work-item
	long gid = get_global_id(0);
	C[gid] = alpha * A[gid] + B[gid];
}