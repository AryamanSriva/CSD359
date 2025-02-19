#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <chrono>

using namespace std;
using namespace chrono;

int main() {
    // Define the number of threads and the upper limit for prime counting
    int threadCount = 64;
    int n = 5; // Change this value to adjust the power of 10 for upperLimit
    long long upperLimit = pow(10, n);
    
    // Vector to store the count of primes found by each thread
    vector<long long> primeCounts(threadCount, 0);
    vector<thread> threads;

    // Function to check if a number is prime
    auto isPrime = [](long long num) -> bool {
        if (num < 2) return false;
        for (long long i = 2; i * i <= num; i++) {
            if (num % i == 0) return false;
        }
        return true;
    };

    // Function to count primes using cyclic distribution
    auto countPrimesCyclic = [&](long long start, long long step, int index) {
        primeCounts[index] = 0;
        for (long long i = start; i <= upperLimit; i += step) {
            if (isPrime(i)) {
                primeCounts[index]++;
            }
        }
    };

    // Start measuring time
    auto startTime = high_resolution_clock::now();

    // Launch threads with cyclic distribution strategy
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(countPrimesCyclic, i + 1, threadCount, i);
    }

    // Join all threads
    for (auto &t : threads) {
        t.join();
    }

    // End measuring time
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(endTime - startTime);

    // Sum up the total number of primes found by all threads
    long long totalPrimes = 0;
    for (const auto &count : primeCounts) {
        totalPrimes += count;
    }

    // Output the results
    cout << "Total number of primes: " << totalPrimes << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;

    return 0;
}
