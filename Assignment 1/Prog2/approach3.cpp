#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <chrono>
#include <atomic>

using namespace std;
using namespace chrono;

int main() {
    // Variables
    int threadCount = 64;                           // Number of threads to use
    atomic<long long> sharedCounter(1);             // Shared counter for dynamic load balancing
    vector<long long> primeCounts(threadCount, 0);  // Vector to store prime counts per thread
    const int BATCH_SIZE = 1000;                    // Batch size for each thread to process
    int n = 5;                                      // Exponent for upper limit (10^n)

    long long upperLimit = pow(10, n);              // Calculate upper limit
    vector<thread> threads;                         // Vector to hold thread objects

    // Function to check if a number is prime
    auto isPrime = [](long long num) {
        if (num < 2) return false;
        for (long long i = 2; i * i <= num; i++) {
            if (num % i == 0) return false;
        }
        return true;
    };

    // Function for threads to count primes dynamically
    auto countPrimesDynamic = [&](long long* primeCount) {
        *primeCount = 0;

        while (true) {
            long long start = sharedCounter.fetch_add(BATCH_SIZE);  // Atomically fetch and update counter

            if (start > upperLimit) break;  // Exit if all numbers are processed

            long long end = min(start + BATCH_SIZE - 1, upperLimit); // Determine batch end
            for (long long i = start; i <= end; i++) {
                if (isPrime(i)) {
                    (*primeCount)++;
                }
            }
        }
    };

    // Measure start time
    auto startTime = high_resolution_clock::now();

    // Create threads
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(countPrimesDynamic, &primeCounts[i]);
    }

    // Wait for all threads to complete
    for (auto &t : threads) {
        t.join();
    }

    // Measure end time
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(endTime - startTime);

    // Calculate total prime count
    long long totalPrimes = 0;
    for (const auto &count : primeCounts) {
        totalPrimes += count;
    }

    // Output results
    cout << "Total number of primes: " << totalPrimes << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;

    return 0;
}
