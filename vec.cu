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
#define BLOCK_SIZE 512

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (3 * BLOCK_SIZE) // Matrix A width
#define HA (5 * BLOCK_SIZE) // Matrix A height
#define WB 1 // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

//sequential code implemented on cpu
void computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA)
{
	for (unsigned int i = 0; i < hA; ++i)
		{
			double sum = 0;
			for (unsigned int k = 0; k < wA; ++k)
			{
				double a = A[i * wA + k];
				double b = B[k ];
				sum += a * b;
			}
			C[i] = (float)sum;
		}
}

// Initialize a matrix with random float entries.
void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}


__global__ void matrixMul(float* C, float* A, float* B,int wA)
{
	// Declaration of the shared memory array As used to
	// store the sub-matrix of A
	// Block index
	int bx = blockIdx.x;
	// Thread index
	int tx = threadIdx.x;
	int Cindex = bx*BLOCK_SIZE + tx;
	int ABase = Cindex*wA;
	int i;
	float sum=0;
	for (i = 0; i < wA; i++)
	{
		sum += B[i] * A[ABase + i];
	}
	C[Cindex] = sum;
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
	dim3 threads(BLOCK_SIZE);
	dim3 grid(HA/BLOCK_SIZE);

	// execute the kernel
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	matrixMul <<< grid,threads >>>(d_C, d_A, d_B,WA);
	cudaThreadSynchronize();
	QueryPerformanceCounter(&finish);
	// stop and destroy timer
	costtime1 = (double)(finish.QuadPart - start.QuadPart) * 1000 / freq.QuadPart;   //ms
																					 // copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	// compute reference solution
	float* reference = (float*)malloc(mem_size_C);
	QueryPerformanceCounter(&start);
	computeGold(reference, h_A, h_B, HA, WA);
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
