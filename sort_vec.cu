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
#define N (4 * 4)
#define BLKDIM (2)
#define BLKSIZE (2)
#define SHOWITEMS (16)



void printArray(int arr[], int n) {
	for(int unused_i=0 ; unused_i< SHOWITEMS ; unused_i++){
		printf("%d ", *(arr+unused_i));
		if (unused_i % 10 == 0 && unused_i != 0)
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

__device__ void _merge_shared(int* arr, int* left, int leftStart, int leftSize, int* right, int rightStart, int rightSize) {
	int i = leftSize, j = rightStart, k = 0;
	printf("_merge : lef[0]:%d , left[1]:%d, arr[0]:%d , right[0]:%d ,right[1]:%d, arr[1]:%d\n", left[0],left[1], arr[0], right[0],right[1], arr[1]);
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

__global__ void mergeSortWithThreadsAndBlocks_limited(int* arr, int n) {
	int currentSize, leftStart;
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	int partSize = BLKSIZE * blockDim.x;
	//Warning : partsize * 2 ?????
	int nPart = (n + partSize - 1) / partSize;

	for (currentSize = 1; currentSize < n; currentSize *= 2) {
		for (int p = 0; p < nPart; p++) {
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

__global__ void mergeSortWithThreadsAndBlocks_limited_shared(int* arr, int n) {
	int currentSize, leftStart;
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	int partSize = BLKSIZE * blockDim.x;
	int nPart = (n + partSize - 1) / (partSize * 2);
	int blockSize = blockDim.x;
	__shared__ int* left;
	__shared__ int* right;
	//extern __shared__ int sharedArray[];

	for (currentSize = 1; currentSize < n; currentSize *= 2) {
		const int sharedSize_side =  (blockSize* currentSize * 2 < n) ? blockSize * currentSize : n/2 ;
		printf("sharedsize_side: %d , currentsize: %d \n", sharedSize_side , currentSize);
		left = (int*)malloc(sharedSize_side * sizeof(int));
		right = (int*)malloc(sharedSize_side * sizeof(int));
		//int* left = &sharedArray[sharedSize_side];
		//int* right = &sharedArray[sharedSize_side];

		for (int p = 0; p < nPart; p++) {
			int leftStart = (idx + (p * partSize)) * 2 * currentSize;
			
			if (leftStart + currentSize > n) break;
			printf("leftstart: %d , p: %d , n: %d , partSize: %d , nPart: %d ,idx: %d\n", leftStart, p, n, partSize, nPart, idx);
			//if (currentSize > 512 * 1 && leftStart > 2095103) printf("SECOND: leftstart: %d , p: %d , n: %d , partSize: %d , nPart: %d ,idx: %d ,currentsize :%d\n", leftStart, p, n, partSize, nPart, idx,currentSize);
			int mid = leftStart + currentSize - 1;
			int rightEnd = (leftStart + 2 * currentSize - 1 < n - 1) ? leftStart + 2 * currentSize - 1 : n - 1;
			int leftSize = mid - leftStart + 1;
			int rightSize = rightEnd - mid;
			int rightStart = mid + 1;
			//if (currentSize > 512 * 1 && leftStart > 2095103) printf("leftsize: %d , rightsize: %d", leftSize, rightSize);
			//printf("index:%d , mid:%d , rightEnd:%d , leftsize:%d , rightsize:%d\n", index, mid, rightEnd, leftSize, rightSize);
			//if (currentSize > 512 * 1 && leftStart > 2095103) printf("here1\n");
			// Copy data from original array to temporary left and right arrays
			int _unused_idx = 0;
			for (int i = 0; i < leftSize; i++) {
				//if (currentSize > 512 * 1 && leftStart > 2095103) printf("########\n");
				_unused_idx = i % sharedSize_side;
				left[leftStart % sharedSize_side] = *(arr + leftStart + i);
			}
			//if (currentSize > 512 * 1 && leftStart > 2095103) printf("here2\n");
			for (int i = 0; i < rightSize; i++) {
				_unused_idx = i % sharedSize_side;
				right[i % sharedSize_side] = *(arr + rightStart + i);
			}

			__syncthreads();
			//if (currentSize > 512 * 1 && leftStart > 2095103) printf("here3\n");
			printf("before : arr0:%d , arr1:%d , arr2:%d , arr3:%d\n", arr[0], arr[1], arr[2], arr[3]);
			printf("leftStart: %d , rigfhtStart: %d , sharedSize: %d\n", leftStart, rightStart, sharedSize_side);
			_merge_shared(arr + leftStart, left , (leftStart% sharedSize_side) , leftSize + leftStart, right,(rightStart% sharedSize_side), rightSize + rightStart);
			printf("after : arr0:%d , arr1:%d , arr2:%d , arr3:%d\n", arr[0], arr[1], arr[2], arr[3]);
			__syncthreads();
	
		}

		//free(left);
		//free(right);
		
	}
}

__global__ void mergeSortWithBlocksAndThreads(int* arr, int n) {
	int currentSize = 1, leftStart;
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	int partSize = blockDim.x;
	int nPart = (n + blockDim.x - 1) / blockDim.x;

	while (nPart > 0) {
		
			int leftStart = idx * 2 * currentSize;
			//printf("leftstart: %d , n: %d , partSize: %d , nPart: %d ,idx: %d , currentsize:%d\n", leftStart, n , partSize, nPart,idx, currentSize);
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

			nPart /= 2;
			currentSize *= 2;
		
	}
}

__global__ void mergeSortWithBlocks(int* arr, int n) {
	int currentSize, leftStart;
	int idx = blockIdx.x;
	//int partSize = 16;
	int nPart = n/2 ; 

	
			while (nPart > 0) {
				currentSize = n / (nPart * 2);
				int leftStart = idx * 2 * currentSize;
				if (leftStart + currentSize > n) break;
				//printf("npart: %d , idx: %d , currentsize: %d , leftStart:%d\n", nPart, idx, currentSize, leftStart);
				int mid = leftStart + currentSize - 1;
				int rightEnd = (leftStart + 2 * currentSize - 1 < n - 1) ? leftStart + 2 * currentSize - 1 : n - 1;
				int leftSize = mid - leftStart + 1;
				int rightSize = rightEnd - mid;
				int* left = (int*)malloc(leftSize * sizeof(int));
				int* right = (int*)malloc(rightSize * sizeof(int));

				for (int i = 0; i < leftSize; i++) {
					//if (currentSize > 512 * 1 && leftStart > 2095103) printf("########\n");
					left[i] = *(arr + leftStart + i);
				}
				//if (currentSize > 512 * 1 && leftStart > 2095103) printf("here2\n");
				for (int i = 0; i < rightSize; i++) {
					right[i] = *(arr + mid + 1 + i);
				}

				_merge(arr + leftStart, left, leftSize, right, rightSize);
				//printf("after : arr0:%d , arr1:%d , arr2:%d , arr3:%d\n", arr[0], arr[1], arr[2], arr[3]);
				free(left);
				free(right);

				__syncthreads();

				nPart /= 2;

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
	//mergeSortWithThreads << <1, BLKDIM >> > (d_arr, N);
	//mergeSortWithBlocks << <N, 1 >> > (d_arr, N);
	//mergeSortWithBlocksAndThreads << <(N + BLKDIM -1 )/ (BLKDIM * 2), BLKDIM >> > (d_arr, N);
	//mergeSortWithThreadsAndBlocks_limited << <BLKSIZE , BLKDIM >> > (d_arr, N);
	mergeSortWithThreadsAndBlocks_limited_shared << <BLKSIZE , BLKDIM >> > (d_arr, N);
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
	arr = (int*)malloc(size);

	printf("Calculation Is Starting ...\n");

	// Sequential Implementation
	printf("\n########Sequential Implementation########\n");
	//initilizeArray(arr);
	//// Host code: demonstrating the array before sorting
	//printf("The orginal array is (%d Items): \n", SHOWITEMS);
	//printArray(arr, N);
	//cudaStatus = mergeSortHelper(arr, size);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "mergeSortHelper failed!\n");
	//	// Todo: another alternative would be exit program after receiving error in any step
	//	//return 1;
	//}
	//printf("The sorted array is: \n");
	//printArray(arr, N);

	// Parallel Implementation with threads and one block
	printf("\n########Parallel Impelementation########\n");
	initilizeArray(arr);
	printf("The orginal array is (%d Items): \n", SHOWITEMS);
	printArray(arr, N);
	cudaStatus = mergeSortWithThreadsHelper(arr, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mergeSortWithThreadsHelper failed!\n");
		//return 1;
	}
	printf("The sorted array is: \n");
	printArray(arr, N);

	printf("\nCalculation Is Finished!\n");

	free(arr);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

