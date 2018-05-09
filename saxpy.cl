__kernel void saxpy(
	__private float alpha,
	__global float *A,
	__global float *B,
	__global float *C)
{
	//Get the gid of the work-item
	long gid = get_global_id(0);
	C[gid] = alpha * A[gid] + B[gid];
}