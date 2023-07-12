/****************************************************************************
 *
 * sort_vec.cu - Sorting a vector using CUDA C
 *
 * Created by Afshin Khodaveisi Afshin.khodaveisi@studio.unibo.it
 *
 * ---------------------------------------------------------------------------
 *
 * Implementing vector sorting algorithms using Cuda C
 *
 * Implemented algorithms including:
 * Merge Sort - Sequential Implementation (1 block , 1 Thread)
 * Extended Merge Sort - Parallel Implementation (Multi Threads)
 ****************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#define N (1024 * 1024)
#define BLKDIM (1024)
#define SHOWITEMS (100)



void printArray(int arr[], int n) {
	for(int unused_i=0 ; unused_i< SHOWITEMS ; unused_i++){
		printf("%d ", *(arr+unused_i));
		if (unused_i % 20 == 0 && unused_i != 0)
			printf("\n");
	}
	printf("...\n");

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
	//printf("_merge : lef[0]:%d , arr[0]:%d , right[0]:%d , arr[1]:%d\n", left[0], arr[0], right[0], arr[1]);
	while (i < leftSize && j < rightSize) {
		if (left[i] <= right[j]) {
			//printf("$$$$$$$$$\n");
			arr[k++] = left[i++];
			//printf("k:%d\n", k);
		}
		else {
			//printf("%%%%%%%%%%%\n");
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

			_merge(arr + leftStart, left, leftSize, right, rightSize);
	
			free(left);
			free(right);
		}
	}
}

__global__ void mergeSortWithThreads(int* arr, int n) {
	int currentSize, leftStart;
	int idx = threadIdx.x;
	int partSize = blockDim.x;
	int nPart = (n + blockDim.x - 1) / blockDim.x;

	for (currentSize = 1; currentSize < n; currentSize *= 2) {
		for(int p = 0 ; p < nPart ; p++){
			int leftStart = (idx + (p * partSize)) * 2 * currentSize;
			//if (currentSize > 512 * 1   ) printf("leftstart: %d , p: %d , n: %d , partSize: %d , nPart: %d ,idx: %d\n", leftStart, p, n , partSize, nPart,idx);
			if (leftStart + currentSize > n) break;
			//if (currentSize > 512 * 1 && leftStart > 2095103) printf("SECOND: leftstart: %d , p: %d , n: %d , partSize: %d , nPart: %d ,idx: %d ,currentsize :%d\n", leftStart, p, n, partSize, nPart, idx,currentSize);
				int mid = leftStart + currentSize - 1;
				int rightEnd = (leftStart + 2 * currentSize - 1 < n - 1) ? leftStart + 2 * currentSize - 1 : n - 1;
				int leftSize = mid - leftStart + 1;
				int rightSize = rightEnd - mid;
				//if (currentSize > 512 * 1 && leftStart > 2095103) printf("leftsize: %d , rightsize: %d", leftSize, rightSize);
				//printf("index:%d , mid:%d , rightEnd:%d , leftsize:%d , rightsize:%d\n", index, mid, rightEnd, leftSize, rightSize);
				int* left = (int*)malloc(leftSize * sizeof(int));
				int* right = (int*)malloc(rightSize * sizeof(int));
				//if (currentSize > 512 * 1 && leftStart > 2095103) printf("here1\n");
				// Copy data from original array to temporary left and right arrays
				for (int i = 0; i < leftSize; i++) {
					//if (currentSize > 512 * 1 && leftStart > 2095103) printf("########\n");
					left[i] = *(arr + leftStart + i);
				}
				//if (currentSize > 512 * 1 && leftStart > 2095103) printf("here2\n");
				for (int i = 0; i < rightSize; i++) {
					right[i] = *(arr + mid + 1 + i);
				}
				//if (currentSize > 512 * 1 && leftStart > 2095103) printf("here3\n");
				//printf("before : arr0:%d , arr1:%d , arr2:%d , arr3:%d\n", arr[0], arr[1], arr[2], arr[3]);
				_merge(arr + leftStart, left, leftSize, right, rightSize);
				//printf("after : arr0:%d , arr1:%d , arr2:%d , arr3:%d\n", arr[0], arr[1], arr[2], arr[3]);
				free(left);
				free(right);
				
				__syncthreads();
			
		}
		

	}
}

__global__ void mergeSortWithSharedMemmory(int* arr, int n) {
	extern __shared__ int sharedArray[];

	int currentSize, leftStart;

	for (currentSize = 1; currentSize < n; currentSize *= 2) {
		leftStart = blockIdx.x * (2 * currentSize);
		int mid = leftStart + currentSize - 1;
		int rightEnd = (leftStart + 2 * currentSize - 1 < n - 1) ? leftStart + 2 * currentSize - 1 : n - 1;
		int leftSize = mid - leftStart + 1;
		int rightSize = rightEnd - mid;

		int* left = &sharedArray[leftStart];
		int* right = &sharedArray[mid + 1];

		// Copy data to shared memory
		if (threadIdx.x < leftSize)
			left[threadIdx.x] = arr[leftStart + threadIdx.x];
		if (threadIdx.x < rightSize)
			right[threadIdx.x] = arr[mid + 1 + threadIdx.x];

		__syncthreads();

		_merge(arr + leftStart, left, leftSize, right, rightSize);

		__syncthreads();
	}
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

// Helper function checks validation of cuda codes and merge sort as sequential
cudaError_t mergeSortWithThreadsHelper(int* arr, size_t size) {
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

	// Launch merge sort on the GPU with one thread and one block.
	tstart = get_time();
	mergeSortWithThreads << <1, BLKDIM >> > (d_arr, N);
	//mergeSortWithSharedMemmory << <1, 1, N * sizeof(int) >> > (d_arr, N);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mergeSortWithThreads launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	tend = get_time();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mergeSortWithThreads!\n", cudaStatus);
		fprintf(stderr, "Error is: %s \n", cudaGetErrorString(cudaStatus));
	}

	showTime(tstart, tend, (char*)"Parallel Implementation with Threads");
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
	printf("Calculation Is Starting ...\n");
	// Sequential Implementation
	initilizeArray(arr);
	// Host code: demonstrating the array before sorting
	printf("The orginal array is (%d Items): \n", SHOWITEMS);
	printArray(arr, N);
	cudaStatus = mergeSortHelper(arr, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mergeSortHelper failed!\n");
		// Todo: another alternative would be exit program after receiving error in any step
		//return 1;
	}
	printf("The sorted array is: \n");
	printArray(arr, N);

	// Parallel Implementation with threads and one block
	initilizeArray(arr);
	/** (arr + 0) = 2;
	*(arr + 1) = 5;
	*(arr + 2) = 9;
	*(arr + 3) = 4;*/
	printf("The orginal array is (%d Items): \n", SHOWITEMS);
	printArray(arr, N);
	cudaStatus = mergeSortWithThreadsHelper(arr, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mergeSortWithThreadsHelper failed!\n");
		//return 1;
	}
	printf("The sorted array is: \n");
	printArray(arr, N);

	printf("Calculation Is Finished!\n");

	free(arr);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

