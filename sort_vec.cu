/****************************************************************************
 *
 * sort_vec.cu - Sort the array with CUDA
 *
 * Created by Afshin Khodaveisi Afshin.khodaveisi@studio.unibo.it
 *
 * ---------------------------------------------------------------------------
 *
 * Using Merge Sort algorithm
 *
 * Run in various condition including:
 * Sequential Implementation
 *
 ****************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#define N (2048 * 1024)
#define BLKDIM (32)
#define SHOWITEMS (400)


void printArray(int arr[], int n) {
	for(int unused_i=0 ; unused_i< SHOWITEMS ; unused_i++){
		printf("%d ", *(arr+unused_i));
		if (unused_i % 20 == 0 && unused_i != 0)
			printf("\n");
	}
	printf("\n");

}

void showTime(double start, double end , char msg[])
{
	double elapsed = end - start ;
	printf("Elapsed time in %s: %.6f seconds\n",msg, elapsed);
}

double get_time(){
	clock_t	time = clock();
	return (double)(time / CLOCKS_PER_SEC);
}


void initilizeArray(int* arr) {
	for (int unused_i = 0; unused_i < N; unused_i++)
		*(arr + unused_i) = rand();
}

__device__ void _merge(int* arr, int* left, int leftSize, int* right, int rightSize) {
	int i = 0, j = 0, k = 0;

	while (i < leftSize && j < rightSize) {
		if (left[i] <= right[j]) {
			arr[k++] = left[i++];
		}
		else {
			arr[k++] = right[j++];
		}
	}

	while (i < leftSize) {
		arr[k++] = left[i++];
	}

	while (j < rightSize) {
		arr[k++] = right[j++];
	}
}

__global__ void mergeSort(int* arr, int n) {
	int currentSize, leftStart;

	for (currentSize = 1; currentSize < n; currentSize *= 2) {
		for (leftStart = 0; leftStart < n - 1; leftStart += 2 * currentSize) {
			int mid = leftStart + currentSize - 1;
			int rightEnd = (leftStart + 2 * currentSize - 1 < n - 1) ? leftStart + 2 * currentSize - 1 : n - 1;
			int leftSize = mid - leftStart + 1;
			int rightSize = rightEnd - mid;

			int* left = (int*)malloc(leftSize * sizeof(int));
			int* right = (int*)malloc(rightSize * sizeof(int));
			//int left[leftSize], right[rightSize];

			// Copy data from original array to temporary left and right arrays
			for (int i = 0; i < leftSize; i++) {
				left[i] = *(arr + leftStart + i);
			}
			for (int i = 0; i < rightSize; i++) {
				right[i] = *(arr + mid + 1 + i);
			}
			if (currentSize >= 2048 * 32 && leftStart == 0)
				printf("arr[0]: %d , %d , %d , %d\n", arr[0] , arr[1] , arr[2] , arr[3]);
			_merge(arr + leftStart, left, leftSize, right, rightSize);
			if (currentSize >= 2048 * 32 && leftStart == 0)
				printf("mid: %d, rightEnd: %d, leftsize: %d, rightsize: %d \n", mid, rightEnd, leftSize, rightSize);
			free(left);
			free(right);
		}
	}
	printf("last arr[0]: %d , %d , %d , %d", arr[0], arr[1], arr[2], arr[3]);
}

// Helper function checks validation of cuda codes and merge sort as sequential
cudaError_t mergeSortHelper(int* arr, size_t size){
	cudaError_t cudaStatus;
	int* d_arr;
	double tstart, tend;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
	}

	cudaStatus = cudaMalloc((void**)&d_arr, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
	}

	cudaStatus = cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}

	// Host code: demonstrating the array before sorting
	printf("The orginal array is (%d Items): \n", SHOWITEMS);
	printArray(arr, N);

	// Launch merge sort on the GPU with one thread and one block.
	tstart = get_time();
	mergeSort <<<1, 1 >>> (d_arr, N);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mergeSort launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	tend = get_time();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mergeSort!\n", cudaStatus);
		fprintf(stderr, "Error is: %s \n",cudaGetErrorString(cudaStatus));
	}

	showTime(tstart, tend, (char*)"Sequential Implementation");
	cudaStatus = cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}

	cudaFree(d_arr);

	return cudaStatus;

}

int main() {
	int *arr;
	const size_t size = N * sizeof(int);
	cudaError_t cudaStatus;

	// Host Space
	arr = (int*)malloc(size);

	// Sequential Implementation
	initilizeArray(arr);
	cudaStatus = mergeSortHelper(arr, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mergeSortHelper failed!\n");
		//return 1;
	}

	printf("The sorted array is: \n");
	printArray(arr, N);

	free(arr);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

