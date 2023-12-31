# Vector Sorting Algorithm Implementation

This project provides an implementation of a vector sorting algorithm. The aim of this project is to analyze the performance of different sorting algorithms in parallel using CUDA C. By studying the implemented algorithms, you can gain a comprehensive understanding of their efficiency and effectiveness in sorting arrays.

## The Implemented Algorithms include:
- Merge Sort - Sequential Implementation (1 block , 1 Thread)
- Extended Merge Sort - Parallel Implementation (Multi threads and one block) 
- Extended Merge Sort - Parallel Implementation (Multi blocks and one thread inside each block)
- Extended Merge Sort - Parallel Implementation (Multi threads and multi blocks)
- Extended Merge Sort - Parallel Implementation (Multi-threads and multi-blocks are limited with respect to the total number of threads)

## Get Started
To use this repository, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies (CUDA Toolkit, C compiler, etc.).
3. If you are use Visual Studio, start by adding Cuda C dependency to your project. Then, build and run within Visual Studio.Otherwise, if you are running the project on Slurm Cluster, use follow command: '$sbatch project.sbatch'
4. Experiment with different input sizes and compare the results.

## License
This project is licensed under the MIT License.

