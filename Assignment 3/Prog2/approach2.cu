#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace chrono;

// Device function to check if a number is prime
__device__ bool isPrime(long long num) {
    if (num < 2) return false; // Numbers less than 2 are not prime
    for (long long i = 2; i * i <= num; i++) { // Check divisibility up to sqrt(num)
        if (num % i == 0) return false; // If divisible, it's not prime
    }
    return true; // Otherwise, it's prime
}

// CUDA kernel to count prime numbers up to a given upper limit
__global__ void countPrimes(long long upperLimit, unsigned long long* totalPrimes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Compute global thread index
    int stride = gridDim.x * blockDim.x; // Determine stride to distribute work among threads
    unsigned long long localCount = 0; // Local count of primes for this thread
    
    // Each thread processes numbers in steps of "stride" to avoid redundant calculations
    for (long long i = tid + 1; i <= upperLimit; i += stride) {
        if (isPrime(i)) {
            localCount++; // Increment local count if the number is prime
        }
    }
    
    // Use atomic addition to safely update the total prime count in global memory
    atomicAdd(totalPrimes, localCount);
}

int main() {
    int threads = 2;  
    int n = 5; // Define the exponent for the upper limit (10^n)
    long long upperLimit = pow(10, n); // Compute the upper limit
    
    // Calculate the number of blocks needed based on the number of threads
    int blocks = (upperLimit + threads - 1) / threads;

    // Allocate memory for total prime count on device
    unsigned long long *d_totalPrimes, h_totalPrimes = 0;
    cudaMalloc(&d_totalPrimes, sizeof(unsigned long long));
    cudaMemcpy(d_totalPrimes, &h_totalPrimes, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    auto startTime = high_resolution_clock::now(); // Start timing

    // Launch CUDA kernel
    countPrimes<<<blocks, threads>>>(upperLimit, d_totalPrimes);
    cudaDeviceSynchronize(); // Ensure all threads have completed execution

    auto endTime = high_resolution_clock::now(); // End timing
    auto duration = duration_cast<milliseconds>(endTime - startTime); // Calculate duration

    // Copy the result back from device to host
    cudaMemcpy(&h_totalPrimes, d_totalPrimes, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Output the results
    cout << "Total number of primes: " << h_totalPrimes << "\nTime taken: " << duration.count() << " ms" << endl;

    // Free allocated memory on device
    cudaFree(d_totalPrimes);
    return 0;
}
