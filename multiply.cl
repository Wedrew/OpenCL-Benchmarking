__kernel void multiply(
	__global float *A)
{
	long id = get_global_id(0);
	A[id] *= 2.0f;
}