#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

int main() {
    // Variables
    int threadCount = 64;           // Number of threads to use
    const int BATCH_SIZE = 1000;    // Batch size for each thread to process
    int n = 8;                      // Exponent for upper limit (10^n)
    long long upperLimit = pow(10, n); // Calculate upper limit
    long long totalPrimes = 0;      // Counter for total primes found

    // Function to check if a number is prime
    auto isPrime = [](long long num) {
        if (num < 2) return false;
        for (long long i = 2; i * i <= num; i++) {
            if (num % i == 0) return false;
        }
        return true;
    };

    // Set number of threads
    omp_set_num_threads(threadCount);

    // Measure start time
    auto startTime = high_resolution_clock::now();

    // Parallel prime counting using OpenMP
    #pragma omp parallel
    {
        long long localCount = 0;  // Local counter for each thread
        
        #pragma omp for schedule(dynamic, BATCH_SIZE)
        for (long long i = 2; i <= upperLimit; i++) {
            if (isPrime(i)) {
                localCount++;
            }
        }

        // Atomic addition of local count to total
        #pragma omp atomic
        totalPrimes += localCount;
    }

    // Measure end time
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(endTime - startTime);

    // Output results
    cout << "Total number of primes: " << totalPrimes << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;

    return 0;
}