#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for matrix multiplication
template<typename T>
__global__ void matrixMultiplyKernel(T *d_A, T *d_B, T *d_C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate column index

    if (row < m && col < p) {
        double sum = 0;
        for (int k = 0; k < n; k++) {
            sum += d_A[row * n + k] * d_B[k * p + col];
        }
        d_C[row * p + col] = sum; // Store the computed value in output matrix
    }
}

int main() {
    int m, n, p;
    double *h_A, *h_B, *h_C; // Host matrices
    double *d_A, *d_B, *d_C; // Device matrices

    // Open input files for matrices A and B
    FILE* fpA = fopen("matrixA.txt", "r");
    FILE* fpB = fopen("matrixB.txt", "r");
    
    // Read dimensions of matrix A
    fscanf(fpA, "%d %d", &m, &n);
    
    // Read dimensions of matrix B
    int n2, p2;
    fscanf(fpB, "%d %d", &n2, &p);

    // Allocate memory for matrices on host
    size_t sizeA = m * n * sizeof(double);
    size_t sizeB = n * p * sizeof(double);
    size_t sizeC = m * p * sizeof(double);

    h_A = (double*)malloc(sizeA);
    h_B = (double*)malloc(sizeB);
    h_C = (double*)malloc(sizeC);

    // Read matrix A from file
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            fscanf(fpA, "%lf", &h_A[i * n + j]);

    // Read matrix B from file
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            fscanf(fpB, "%lf", &h_B[i * p + j]);

    fclose(fpA);
    fclose(fpB);

    // Allocate memory on device
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int threads = 16;  
    
    // Define grid and block dimensions
    dim3 blockDim(threads, threads);
    dim3 gridDim((p + threads - 1) / threads, (m + threads - 1) / threads);
    
    // Launch CUDA kernel for matrix multiplication
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, p);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Measure execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %.2f ms\n", milliseconds);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Write output matrix to file
    FILE* fp = fopen("matrixC.txt", "w");
    fprintf(fp, "%d %d\n", m, p);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            fprintf(fp, "%.2f ", h_C[i * p + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}