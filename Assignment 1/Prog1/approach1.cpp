#include <iostream>
#include <vector>
#include <fstream>
#include <pthread.h>
#include <sys/time.h>

using namespace std;

// Structure to store data for each thread
struct ThreadData {
    const vector<vector<int>> *A;
    const vector<vector<int>> *B;
    vector<vector<int>> *C;
    int startRow, endRow;
};

// Function executed by each thread to compute part of the matrix multiplication
void* multiplyRows(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->startRow; i < data->endRow; ++i) {
        for (size_t j = 0; j < data->B->at(0).size(); ++j) {
            data->C->at(i)[j] = 0;
            for (size_t k = 0; k < data->A->at(0).size(); ++k) {
                data->C->at(i)[j] += data->A->at(i)[k] * data->B->at(k)[j];
            }
        }
    }
    return nullptr;
}

int main() {
    int numThreads = 64; 
    
    // Load matrices from files
    ifstream fileA("matrixA.txt"), fileB("matrixB.txt");
    int rowsA, colsA, rowsB, colsB;
    fileA >> rowsA >> colsA;
    fileB >> rowsB >> colsB;
    
    vector<vector<int>> A(rowsA, vector<int>(colsA));
    vector<vector<int>> B(rowsB, vector<int>(colsB));
    vector<vector<int>> C(rowsA, vector<int>(colsB, 0));
    
    // Read matrix A from file
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsA; ++j)
            fileA >> A[i][j];
    
    // Read matrix B from file
    for (int i = 0; i < rowsB; ++i)
        for (int j = 0; j < colsB; ++j)
            fileB >> B[i][j];
    
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    int rowsPerThread = rowsA / numThreads;
    
    struct timeval start, end;
    gettimeofday(&start, NULL); // Start time measurement
    
    // Create threads to perform matrix multiplication
    for (int i = 0; i < numThreads; ++i) {
        threadData[i].A = &A;
        threadData[i].B = &B;
        threadData[i].C = &C;
        threadData[i].startRow = i * rowsPerThread;
        threadData[i].endRow = (i == numThreads - 1) ? rowsA : (i + 1) * rowsPerThread;
        pthread_create(&threads[i], nullptr, multiplyRows, &threadData[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }
    
    gettimeofday(&end, NULL); // End time measurement
    double timeTaken = (end.tv_sec - start.tv_sec) * 1e3;
    timeTaken = (timeTaken + (end.tv_usec - start.tv_usec) * 1e-3);
    
    // Save the resulting matrix to file
    ofstream fileC("matrixC.txt");
    for (const auto& row : C) {
        for (const auto& val : row) {
            fileC << val << " ";
        }
        fileC << endl;
    }
    
    // Print execution time
    cout << "Time taken: " << timeTaken << " ms" << endl;
    return 0;
}
