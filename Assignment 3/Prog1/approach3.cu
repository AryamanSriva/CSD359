#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(double *A, double *B, double *C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        double sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

int main() {
    int m, n, p;

    // Read matrices from files
    ifstream inA("matrixA.txt"), inB("matrixB.txt");
    
    // Read matrix A
    inA >> m >> n;
    vector<double> h_A(m * n);
    for (int i = 0; i < m * n; i++) inA >> h_A[i];

    // Read matrix B
    inB >> n >> p;
    vector<double> h_B(n * p);
    for (int i = 0; i < n * p; i++) inB >> h_B[i];

    // Initialize result matrix C
    vector<double> h_C(m * p, 0);

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(double));
    cudaMalloc((void**)&d_B, n * p * sizeof(double));
    cudaMalloc((void**)&d_C, m * p * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), n * p * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block size
    int threads = 16;  // Change this to 2, 4, 8, 16, 32, 64
    dim3 blockSize(threads, threads);
    dim3 gridSize((p + threads - 1) / threads, (m + threads - 1) / threads);


    // Start CUDA timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, p);

    // Stop CUDA timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, m * p * sizeof(double), cudaMemcpyDeviceToHost);

    // Print execution time
    cout << "Execution time: " << milliseconds << " ms\n";

    // Save result to file
    ofstream out("matrixC.txt");
    out << m << " " << p << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            out << h_C[i * p + j] << " ";
        }
        out << endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
