#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace chrono;

// Device function to check if a number is prime
__device__ bool isPrime(long long num) {
    if (num < 2) return false;
    // Check divisibility up to square root of num
    for (long long i = 2; i * i <= num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

// Each thread takes numbers from a shared counter and checks if they're prime
__global__ void countPrimes(long long upperLimit, unsigned long long* totalPrimes, unsigned long long* counter) {
    // Local counter for each thread to minimize atomic operations
    unsigned long long localCount = 0;
    unsigned long long current;
    
    // Atomically increment counter to get next number to check
    while ((current = atomicAdd(counter, 1)) <= upperLimit) {
        if (isPrime(current)) {
            localCount++;
        }
    }
    
    // Add this thread's count to the global total
    atomicAdd(totalPrimes, localCount);
}

int main() {
    int threads = 2;  
    int n = 8;        // Calculate primes up to 10^n
    long long upperLimit = pow(10, n);
    
    // Calculate number of blocks needed
    // Ensures we have enough total threads to handle the workload
    int blocks = (upperLimit + threads - 1) / threads;
    
    // Host variables for storing results
    unsigned long long *d_totalPrimes, h_totalPrimes = 0;
    unsigned long long *d_counter, h_counter = 1;
    
    // Allocate memory on GPU
    cudaMalloc(&d_totalPrimes, sizeof(unsigned long long));
    cudaMalloc(&d_counter, sizeof(unsigned long long));
    
    // Copy initial values to GPU memory
    cudaMemcpy(d_totalPrimes, &h_totalPrimes, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counter, &h_counter, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // Start timing the computation
    auto startTime = high_resolution_clock::now();

    // Launch kernel with specified number of blocks and threads
    countPrimes<<<blocks, threads>>>(upperLimit, d_totalPrimes, d_counter);
    // Wait for all threads to complete
    cudaDeviceSynchronize();

    // Stop timing and calculate duration
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(endTime - startTime);

    // Copy results back from GPU to host
    cudaMemcpy(&h_totalPrimes, d_totalPrimes, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Output results
    cout << "Total number of primes: " << h_totalPrimes << "\nTime taken: " << duration.count() << " ms" << endl;

    // Free GPU memory
    cudaFree(d_totalPrimes);
    cudaFree(d_counter);
    return 0;
}