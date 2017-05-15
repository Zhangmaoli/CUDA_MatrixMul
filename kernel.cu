

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "time.h"
#include <windows.h>
#include "device_launch_parameters.h"
#include "stdio.h"
// includes, project
// Thread block size
#define BLOCK_SIZE 32

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (3 * BLOCK_SIZE) // Matrix A width
#define HA (5 * BLOCK_SIZE) // Matrix A height
#define WB (8 * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

//sequential code implemented on cpu
void computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
	for (unsigned int i = 0; i < hA; ++i)
		for (unsigned int j = 0; j < wB; ++j)
		{
		double sum = 0;
		for (unsigned int k = 0; k < wA; ++k)
		{
			double a = A[i * wA + k];
			double b = B[k * wB + j];
			sum += a * b;
		}
		C[i * wB + j] = (float)sum;
		}
}

// Initialize a matrix with random float entries.
void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

__device__  float * GetSubMatrix(float *matrix, int m, int index, int width)
{
	return  matrix + width*BLOCK_SIZE*index + BLOCK_SIZE*m;
}

//Kernel code
__global__ void matrixMul(float* C, float* A, float* B, int wA, int wB)
{
	// Declaration of the shared memory array As used to
	// store the sub-matrix of A
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

	// Declaration of the shared memory array Bs used to
	// store the sub-matrix of B
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int m = 0; m<wA / BLOCK_SIZE; m++)
	{
		//get the address of submatrixA
		//float *subA=A+wA*BLOCK_SIZE*by+BLOCK_SIZE*m;
		float *subA = GetSubMatrix(A, m, by, wA);
		//get the address of submatrixB
		//float *subB=B+wB*BLOCK_SIZE*m+BLOCK_SIZE*bx;
		float *subB = GetSubMatrix(B, bx, m, wB);
		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = *(subA + wA * ty + tx);
		Bs[ty][tx] = *(subB + wB * ty + tx);

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Bs[k][tx];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	//float *subC = C+wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	float *subC = GetSubMatrix(C, bx, by, wB);
	*(subC + wB * ty + tx) = Csub;
}


int main(int argc, char **argv)
{
	LARGE_INTEGER start, finish;
	LARGE_INTEGER freq;
	double costtime1;
	double costtime2;
	double speedup;
	// set seed for rand()
	srand((unsigned)time(NULL));

	// allocate host memory for matrices A and B
	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);
	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*)malloc(mem_size_B);

	// initialize host memory
	randomInit(h_A, size_A);
	randomInit(h_B, size_B);

	// allocate device memory
	float* d_A;
	cudaMalloc((void**)&d_A, mem_size_A);
	float* d_B;
	cudaMalloc((void**)&d_B, mem_size_B);

	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	// allocate device memory for result
	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* d_C;
	cudaMalloc((void**)&d_C, mem_size_C);

	// allocate host memory for the result
	float* h_C = (float*)malloc(mem_size_C);

	// create and start timer
	unsigned int timer = 0;


	// setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(WC / threads.x, HC / threads.y);

	// execute the kernel
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	matrixMul << < grid, threads >> >(d_C, d_A, d_B, WA, WB);
	cudaThreadSynchronize();
	QueryPerformanceCounter(&finish);
	// stop and destroy timer
	costtime1 = (double)(finish.QuadPart - start.QuadPart) * 1000 / freq.QuadPart;   //ms
	// copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	// compute reference solution
	float* reference = (float*)malloc(mem_size_C);
	QueryPerformanceCounter(&start);
	computeGold(reference, h_A, h_B, HA, WA, WB);
	QueryPerformanceCounter(&finish);
	costtime2 = (double)(finish.QuadPart - start.QuadPart) * 1000 / freq.QuadPart;   //ms
	speedup = costtime2 / costtime1;
	printf("time1: %f  ms\n", costtime1);
	printf("time2: %f  ms\n", costtime2);
	printf("speedup is %f\n", speedup);

	// check result

	// clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(reference);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	getchar();
}