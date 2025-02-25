#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <chrono>

using namespace std;

class MatrixMultiplier {
private:
    vector<vector<double>> A, B, C; // Matrices A, B, and result matrix C
    int m, n, p;
    const int NUM_THREADS = 64;

public:
    // Constructor to load matrices from files
    MatrixMultiplier(const string& fileA, const string& fileB) {
        loadMatrices(fileA, fileB);
    }

    // Function to read matrices from input files
    void loadMatrices(const string& fileA, const string& fileB) {
        ifstream inA(fileA), inB(fileB);

        // Read matrix A from file
        inA >> m >> n;
        A.resize(m, vector<double>(n));
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                inA >> A[i][j];

        // Read matrix B from file
        inB >> n >> p;
        B.resize(n, vector<double>(p));
        for(int i = 0; i < n; i++)
            for(int j = 0; j < p; j++)
                inB >> B[i][j];

        // Initialize result matrix C with zeros
        C.resize(m, vector<double>(p, 0));
    }

    // Function to perform parallel matrix multiplication
    void multiply() {
        auto start = chrono::high_resolution_clock::now(); // Start timing

        omp_set_num_threads(NUM_THREADS); // Set number of threads for OpenMP
        #pragma omp parallel // Parallel region starts
        {
            #pragma omp for schedule(static) // Distribute iterations among threads
            for(int i = 0; i < m; i++) {
                for(int j = 0; j < p; j++) {
                    double sum = 0;
                    for(int k = 0; k < n; k++) {
                        sum += A[i][k] * B[k][j]; // Compute dot product
                    }
                    C[i][j] = sum; // Store result in matrix C
                }
            }
        }

        auto end = chrono::high_resolution_clock::now(); // End timing
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "Execution time: " << duration.count() << " ms\n"; // Print execution time
    }

    // Function to save
    void saveResult(const string& filename) {
        ofstream out(filename);
        out << m << " " << p << endl; // Write matrix dimensions
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < p; j++) {
                out << C[i][j] << " "; // Write matrix values
            }
            out << endl;
        }
    }
};

int main() {
    // Create a MatrixMultiplier object and load matrices from files
    MatrixMultiplier multiplier("matrixA.txt", "matrixB.txt");
    
    // Perform matrix multiplication
    multiplier.multiply();
    
    // Save the result
    multiplier.saveResult("matrixC.txt");

    return 0;
}
