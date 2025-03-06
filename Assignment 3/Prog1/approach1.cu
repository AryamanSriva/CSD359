#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(double* A, double* B, double* C, int m, int n, int p) {
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
    // File names for input matrices
    string fileA = "matrixA.txt", fileB = "matrixB.txt";
    ifstream inA(fileA), inB(fileB);
    
    // Read matrix A dimensions and values
    int m, n, p;
    inA >> m >> n;
    vector<double> A(m * n);
    for(int i = 0; i < m * n; i++)
        inA >> A[i];
    
    // Read matrix B dimensions and values
    inB >> n >> p;
    vector<double> B(n * p);
    for(int i = 0; i < n * p; i++)
        inB >> B[i];
    
    // Resultant matrix C initialized with zeros
    vector<double> C(m * p, 0);
    
    // Device pointers
    double *d_A, *d_B, *d_C;
    size_t sizeA = m * n * sizeof(double);
    size_t sizeB = n * p * sizeof(double);
    size_t sizeC = m * p * sizeof(double);
    
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeB, cudaMemcpyHostToDevice);
    
    int threads = 16; 
    
    // Define grid and block dimensions
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((p + threads - 1) / threads, 
                   (m + threads - 1) / threads);
    
    // CUDA events for measuring execution time
    cudaEvent_t start, stop;
    float elapsedTime;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Launch the matrix multiplication kernel
    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Copy result matrix from device to host
    cudaMemcpy(C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Print execution time
    cout << "Execution time: " << elapsedTime << " ms\n";
    
    // Write output matrix to file
    ofstream out("matrixC.txt");
    out << m << " " << p << endl;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            out << C[i * p + j] << " ";
        }
        out << endl;
    }
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}