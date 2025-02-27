#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

int main() {
    // Number of threads
    int threadCount = 64;
    int n = 8; // Power of 10 for upperLimit
    long long upperLimit = pow(10, n);

    // Variable to store the total number of primes
    long long totalPrimes = 0;

    // Function to check if a number is prime
    auto isPrime = [](long long num) -> bool {
        if (num < 2) return false;
        for (long long i = 2; i * i <= num; i++) {
            if (num % i == 0) return false;
        }
        return true;
    };

    // Start measuring time
    auto startTime = high_resolution_clock::now();

    // Use OpenMP to parallelize the prime counting process with cyclic distribution
    #pragma omp parallel num_threads(threadCount) reduction(+:totalPrimes)
    {
        int threadID = omp_get_thread_num();
        for (long long i = threadID + 1; i <= upperLimit; i += threadCount) {
            if (isPrime(i)) {
                totalPrimes++;
            }
        }
    }

    // End measuring time
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(endTime - startTime);

    // Output the results
    cout << "Total number of primes: " << totalPrimes << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;

    return 0;
}