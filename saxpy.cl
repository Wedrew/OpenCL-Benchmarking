__kernel void saxpy(
	__global float *A,
	__global float *B,
	__global float *C)
{
	size_t gid = get_global_id(0);
	C[gid] = mad(A[gid], B[gid], 2);
}