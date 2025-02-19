#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <chrono>

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

// Function to count primes in a given range
void countPrimesInRange(long long start, long long end, long long* primeCount) {
    *primeCount = 0;
    for (long long i = start; i <= end; i++) {
        if (isPrime(i)) {
            (*primeCount)++;
        }
    }
}

int main() {
    int n = 8; // Set the power of 10 for the upper limit
    int threadCount = 4; // Number of threads to use
    long long upperLimit = pow(10, n); // Calculate the upper limit
    vector<thread> threads; // Vector to store threads
    vector<long long> primeCounts(threadCount, 0); // Vector to store prime counts for each thread

    auto startTime = high_resolution_clock::now(); // Start measuring time

    long long rangeSize = upperLimit / threadCount; // Divide the range equally among threads
    
    // Create and start threads
    for (int i = 0; i < threadCount; i++) {
        long long start = i * rangeSize + 1;
        long long end = (i == threadCount - 1) ? upperLimit : (i + 1) * rangeSize;
        threads.emplace_back(countPrimesInRange, start, end, &primeCounts[i]);
    }

    // Wait for all threads to finish
    for (auto &t : threads) {
        t.join();
    }

    auto endTime = high_resolution_clock::now(); // End measuring time
    auto duration = duration_cast<milliseconds>(endTime - startTime); // Calculate time taken

    // Sum up the total prime numbers found by all threads
    long long totalPrimes = 0;
    for (int i = 0; i < threadCount; i++) {
        totalPrimes += primeCounts[i];
    }

    // Output the result
    cout << "Total number of primes: " << totalPrimes << "\nTime taken: " << duration.count() << " ms" << endl;
    
    return 0;
}
