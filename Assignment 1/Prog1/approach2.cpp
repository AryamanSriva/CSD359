#include <iostream>
#include <vector>
#include <fstream>
#include <pthread.h>
#include <sys/time.h>

using namespace std;

// Number of threads to use for parallel computation
int numThreads = 64;

// Structure to store thread data
struct ThreadData {
    const vector<vector<int>> *A;
    const vector<vector<int>> *B;
    vector<vector<int>> *C;
    int threadID;
    int numThreads;
};

// Function executed by each thread for matrix multiplication
void* multiplyRows(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int m = data->A->size();    // Number of rows in A
    int n = data->A->at(0).size(); // Number of columns in A (same as rows in B)
    int p = data->B->at(0).size(); // Number of columns in B
    
    // Each thread processes a subset of rows
    for (int i = data->threadID; i < m; i += data->numThreads) {
        for (int j = 0; j < p; ++j) {
            data->C->at(i)[j] = 0;
            for (int k = 0; k < n; ++k) {
                data->C->at(i)[j] += data->A->at(i)[k] * data->B->at(k)[j];
            }
        }
    }
    return nullptr;
}

int main() {
    // Load matrices from files
    ifstream fileA("matrixA.txt");
    ifstream fileB("matrixB.txt");
    
    int m, n, p;
    fileA >> m >> n;
    fileB >> n >> p;
    
    vector<vector<int>> A(m, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(p));
    vector<vector<int>> C(m, vector<int>(p, 0));
    
    // Read matrix A from file
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            fileA >> A[i][j];
    
    // Read matrix B from file
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            fileB >> B[i][j];
    
    fileA.close();
    fileB.close();
    
    // Initialize thread-related variables
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Create threads for matrix multiplication
    for (int i = 0; i < numThreads; ++i) {
        threadData[i].A = &A;
        threadData[i].B = &B;
        threadData[i].C = &C;
        threadData[i].threadID = i;
        threadData[i].numThreads = numThreads;
        pthread_create(&threads[i], nullptr, multiplyRows, &threadData[i]);
    }
    
    // Join threads to ensure completion
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    
    gettimeofday(&end, NULL);
    double timeTaken = (end.tv_sec - start.tv_sec) * 1e3;
    timeTaken = (timeTaken + (end.tv_usec - start.tv_usec) * 1e-3);
    
    // Save result matrix to file
    ofstream fileC("matrixC.txt");
    for (const auto& row : C) {
        for (const auto& val : row) {
            fileC << val << " ";
        }
        fileC << endl;
    }
    fileC.close();
    
    // Print execution time
    cout << "Time taken: " << timeTaken << " ms" << endl;
    
    return 0;
}
