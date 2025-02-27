#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

// Function to check if a number is prime
bool isPrime(long long num) {
    if (num < 2) return false;
    for (long long i = 2; i * i <= num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

int main() {
    int n = 8; // Set the upper limit
    int threadCount = 64; // Number of threads
    long long upperLimit = pow(10, n); // Calculate the upper limit
    long long totalPrimes = 0; // Variable to store the total number of primes

    auto startTime = high_resolution_clock::now(); // Start measuring time

    long long rangeSize = upperLimit / threadCount; // Divide the range equally among threads

    // Use OpenMP to parallelize the prime counting process
    #pragma omp parallel num_threads(threadCount) reduction(+:totalPrimes)
    {
        int threadID = omp_get_thread_num();
        long long start = threadID * rangeSize + 1;
        long long end = (threadID == threadCount - 1) ? upperLimit : (threadID + 1) * rangeSize;
        long long localPrimeCount = 0;

        for (long long i = start; i <= end; i++) {
            if (isPrime(i)) {
                localPrimeCount++;
            }
        }

        totalPrimes += localPrimeCount;
    }

    auto endTime = high_resolution_clock::now(); // End measuring time
    auto duration = duration_cast<milliseconds>(endTime - startTime); // Calculate time taken

    // Output the result
    cout << "Total number of primes: " << totalPrimes << "\nTime taken: " << duration.count() << " ms" << endl;
    
    return 0;
}