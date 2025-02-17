#include <iostream>
#include <vector>
#include <fstream>
#include <pthread.h>
#include <sys/time.h>

using namespace std;

int numThreads = 64; // Number of threads
pthread_mutex_t rowMutex; // Mutex for synchronizing row access
int currentRow = 0; // Shared row index to be processed by threads

// Structure to hold matrix multiplication data
struct ThreadData {
    const vector<vector<int>> *A;
    const vector<vector<int>> *B;
    vector<vector<int>> *C;
};

// Thread function for multiplying rows of matrices
void* multiplyRows(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int m = data->A->size();
    int n = data->A->at(0).size();
    int p = data->B->at(0).size();

    while (true) {
        pthread_mutex_lock(&rowMutex);
        int row = currentRow++; // Get the next row to process
        pthread_mutex_unlock(&rowMutex);

        if (row >= m) break; // Stop when all rows are processed

        for (int j = 0; j < p; ++j) {
            data->C->at(row)[j] = 0;
            for (int k = 0; k < n; ++k) {
                data->C->at(row)[j] += data->A->at(row)[k] * data->B->at(k)[j];
            }
        }
    }
    return nullptr;
}

int main() {
    // Load matrices from files
    ifstream fileA("matrixA.txt"), fileB("matrixB.txt");
    int rowsA, colsA, rowsB, colsB;
    fileA >> rowsA >> colsA;
    fileB >> rowsB >> colsB;
    
    vector<vector<int>> A(rowsA, vector<int>(colsA));
    vector<vector<int>> B(rowsB, vector<int>(colsB));
    
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsA; ++j)
            fileA >> A[i][j];
    
    for (int i = 0; i < rowsB; ++i)
        for (int j = 0; j < colsB; ++j)
            fileB >> B[i][j];
    
    fileA.close();
    fileB.close();

    int m = rowsA, n = colsA, p = colsB;
    vector<vector<int>> C(m, vector<int>(p, 0)); // Result matrix

    pthread_t threads[numThreads];
    ThreadData threadData = {&A, &B, &C};
    pthread_mutex_init(&rowMutex, nullptr);

    struct timeval start, end;
    gettimeofday(&start, NULL); // Start timing

    // Create threads for matrix multiplication
    for (int i = 0; i < numThreads; ++i) {
        pthread_create(&threads[i], nullptr, multiplyRows, &threadData);
    }

    // Wait for all threads to finish
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    gettimeofday(&end, NULL); // End timing
    double timeTaken = (end.tv_sec - start.tv_sec) * 1e3;
    timeTaken += (end.tv_usec - start.tv_usec) * 1e-3;

    pthread_mutex_destroy(&rowMutex);

    // Save result matrix to file
    ofstream fileC("matrixC.txt");
    for (const auto& row : C) {
        for (const auto& val : row) {
            fileC << val << " ";
        }
        fileC << endl;
    }
    fileC.close();

    // Output execution time
    cout << "Time taken: " << timeTaken << " ms" << endl;
    return 0;
}
