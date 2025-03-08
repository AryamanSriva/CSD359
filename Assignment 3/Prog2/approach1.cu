#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace chrono;

// Device function to check if a number is prime
__device__ bool isPrime(long long num) {
    if (num < 2) return false;  // 0 and 1 are not prime numbers
    for (long long i = 2; i * i <= num; i++) { // Check divisibility up to sqrt(num)
        if (num % i == 0) return false;
    }
    return true;
}

// CUDA kernel function to count prime numbers within a given range
__global__ void countPrimes(long long upperLimit, unsigned long long* totalPrimes) {
    // Determine thread ID within the grid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x; // Total number of threads
    
    // Distribute workload among threads
    long long rangePerThread = (upperLimit - 1) / totalThreads + 1;
    long long start = 1 + tid * rangePerThread; // Start number for this thread
    long long end = min(start + rangePerThread - 1, upperLimit); // End number for this thread
    
    unsigned long long localCount = 0; // Local prime count for this thread
    
    // Each thread counts prime numbers in its assigned range
    for (long long i = start; i <= end; i++) {
        if (isPrime(i)) {
            localCount++;
        }
    }
    
    // Atomically update the global count of primes
    atomicAdd(totalPrimes, localCount);
}

int main() {
    int threads = 2;  
    int n = 8; // Exponent for upper limit (10^n)
    long long upperLimit = pow(10, n); // Upper limit of prime search
    
    // Calculate number of blocks needed
    int blocks = (upperLimit + threads - 1) / threads;
    
    // Allocate memory for the total prime count on the device
    unsigned long long *d_totalPrimes, h_totalPrimes = 0;
    cudaMalloc(&d_totalPrimes, sizeof(unsigned long long));
    cudaMemcpy(d_totalPrimes, &h_totalPrimes, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // Start timing the computation
    auto startTime = high_resolution_clock::now();

    // Launch kernel with computed blocks and threads per block
    countPrimes<<<blocks, threads>>>(upperLimit, d_totalPrimes);
    cudaDeviceSynchronize(); // Ensure all threads complete execution

    // Stop timing the computation
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(endTime - startTime);

    // Copy result back to host
    cudaMemcpy(&h_totalPrimes, d_totalPrimes, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Print the result
    cout << "Total number of primes: " << h_totalPrimes << "\nTime taken: " << duration.count() << " ms" << endl;

    // Free allocated device memory
    cudaFree(d_totalPrimes);
    return 0;
}