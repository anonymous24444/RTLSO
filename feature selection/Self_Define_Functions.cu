#include "Self_Define_Functions.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << err__ << " ("                             \
                      << cudaGetErrorString(err__) << ")" << std::endl;        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define KNN_BLOCK_SIZE 256

// [Modified] GPU nearest neighbor selection and voting
__global__ void knnVoteKernel(
    const float* __restrict__ trainData,
    const int*   __restrict__ trainLabels,
    const float* __restrict__ testData,
    int trainSize, int testSize, int numSelected, int KNN_neighbors,
    int classes,
    int* __restrict__ predictions)
{
    extern __shared__ unsigned char smem_raw[];
    float* s_dist  = reinterpret_cast<float*>(smem_raw);
    int*   s_votes = reinterpret_cast<int*>(s_dist + trainSize);

    __shared__ float redMin[KNN_BLOCK_SIZE];
    __shared__ int   redIdx[KNN_BLOCK_SIZE];

    for (int ti = blockIdx.x; ti < testSize; ti += gridDim.x) {
        const float* tvec = testData + (size_t)ti * numSelected;

        // Compute squared Euclidean distances
        for (int tr = threadIdx.x; tr < trainSize; tr += blockDim.x) {
            const float* rvec = trainData + (size_t)tr * numSelected;
            float d = 0.f;
            for (int j = 0; j < numSelected; ++j) {
                float diff = rvec[j] - tvec[j];
                d += diff * diff;
            }
            s_dist[tr] = d;
        }
        __syncthreads();

        for (int c = threadIdx.x; c < classes; c += blockDim.x)
            s_votes[c] = 0;
        __syncthreads();

        // Select neighbors by parallel reduction
        for (int iter = 0; iter < KNN_neighbors; ++iter) {
            float localMin = FLT_MAX;
            int   localIdx = -1;
            for (int tr = threadIdx.x; tr < trainSize; tr += blockDim.x) {
                float dv = s_dist[tr];
                if (dv < localMin) { localMin = dv; localIdx = tr; }
            }
            redMin[threadIdx.x] = localMin;
            redIdx[threadIdx.x] = localIdx;
            __syncthreads();

            for (int s = KNN_BLOCK_SIZE / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    if (redMin[threadIdx.x + s] < redMin[threadIdx.x]) {
                        redMin[threadIdx.x] = redMin[threadIdx.x + s];
                        redIdx[threadIdx.x] = redIdx[threadIdx.x + s];
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                int idx = redIdx[0];
                if (idx >= 0) {
                    int lab = trainLabels[idx];
                    if (lab >= 0 && lab < classes)
                        atomicAdd(&s_votes[lab], 1);
                    s_dist[idx] = FLT_MAX;
                }
            }
            __syncthreads();
        }

        // Vote for the class; ties favor the smaller label
        if (threadIdx.x == 0) {
            int pred = 0;
            for (int c = 1; c < classes; ++c)
                if (s_votes[c] > s_votes[pred]) pred = c;
            predictions[ti] = pred;
        }
        __syncthreads();
    }
}

// Reusable buffers for cross-validation
static cudaStream_t g_streams[2]     = {nullptr, nullptr};
static cudaEvent_t  g_ev[2]          = {nullptr, nullptr};
static void*        g_d_buf[2]       = {nullptr, nullptr};
static size_t       g_d_bufCap[2]    = {0, 0};
static int*         g_d_pred[2]      = {nullptr, nullptr};
static size_t       g_d_predCap[2]   = {0, 0};
static void*        g_h_pinned[2]    = {nullptr, nullptr};
static size_t       g_h_pinnedCap[2] = {0, 0};
static int*         g_h_pred_pool    = nullptr;
static size_t       g_h_pred_poolCap = 0;

// [Modified] Training fitness from cross-validation
double nFoldCrossValidation(double** trainData, int* trainLabels, int trainSize,
                            int NF, int CV_folds, int KNN_neighbors, int* fs,
                            int classes, double omega)
{
    // Check cross-validation parameters
    if (CV_folds < 2 || CV_folds > trainSize)   throw std::invalid_argument("CV_folds must be between 2 and trainSize");
    if (KNN_neighbors <= 0)                    throw std::invalid_argument("KNN_neighbors must be positive");
    if (classes <= 0) throw std::invalid_argument("classes must be positive");
    if (NF <= 0) throw std::invalid_argument("NF must be positive");

    // Select features
    std::vector<int> selectedIndices;
    selectedIndices.reserve(NF);
    for (int i = 0; i < NF; ++i)
        if (fs[i]) selectedIndices.push_back(i);
    const int numSelected = (int)selectedIndices.size();
    if (numSelected == 0) return 1.0;

    // Pack selected features as floats
    std::vector<float> selData((size_t)trainSize * numSelected);
    for (int i = 0; i < trainSize; ++i) {
        const double* src = trainData[i];
        float* dst = selData.data() + (size_t)i * numSelected;
        for (int j = 0; j < numSelected; ++j)
            dst[j] = (float)src[selectedIndices[j]];
    }

    // Shuffle samples
    std::vector<int> indices(trainSize);
    for (int i = 0; i < trainSize; ++i) indices[i] = i;
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }

    // Create validation folds
    std::vector<int> foldSizes(CV_folds);
    std::vector<std::vector<int>> folds(CV_folds);
    int cur = 0;
    for (int f = 0; f < CV_folds; ++f) {
        int foldSize = trainSize / CV_folds + (f < trainSize % CV_folds ? 1 : 0);
        foldSizes[f] = foldSize;
        folds[f].resize(foldSize);
        for (int i = 0; i < foldSize; ++i) folds[f][i] = indices[cur++];
    }

    // Create training folds
    std::vector<std::vector<int>> trainIdx(CV_folds);
    for (int f = 0; f < CV_folds; ++f) {
        int ts = trainSize - foldSizes[f];
        trainIdx[f].reserve(ts);
        for (int g = 0; g < CV_folds; ++g) {
            if (g == f) continue;
            for (int i = 0; i < foldSizes[g]; ++i)
                trainIdx[f].push_back(folds[g][i]);
        }
    }

    // Initialize streams and events
    if (g_streams[0] == nullptr) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaStreamCreate(&g_streams[i]));
            CUDA_CHECK(cudaEventCreateWithFlags(&g_ev[i], cudaEventDisableTiming));
        }
    }

    if (trainSize > (int)g_h_pred_poolCap) {
        if (g_h_pred_pool) cudaFreeHost(g_h_pred_pool);
        CUDA_CHECK(cudaHostAlloc(&g_h_pred_pool,
                                 (size_t)trainSize * sizeof(int),
                                 cudaHostAllocDefault));
        g_h_pred_poolCap = trainSize;
    }

    static bool smemAttrSet = false;
    if (!smemAttrSet) {
        CUDA_CHECK(cudaFuncSetAttribute(
            knnVoteKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            64 * 1024));
        smemAttrSet = true;
    }

    // Process folds with two CUDA streams
    int testBase = 0;
    for (int f = 0; f < CV_folds; ++f) {
        const int s = f & 1;
        const int testSize = foldSizes[f];
        const int trSize   = (int)trainIdx[f].size();

        // Wait before reusing this buffer
        if (f >= 2) CUDA_CHECK(cudaEventSynchronize(g_ev[s]));

        const size_t trainBytes = (size_t)trSize   * numSelected * sizeof(float);
        const size_t labelBytes = (size_t)trSize   * sizeof(int);
        const size_t testBytes  = (size_t)testSize * numSelected * sizeof(float);
        const size_t totalBytes = trainBytes + labelBytes + testBytes;

        if (totalBytes > g_h_pinnedCap[s]) {
            if (g_h_pinned[s]) cudaFreeHost(g_h_pinned[s]);
            CUDA_CHECK(cudaHostAlloc(&g_h_pinned[s], totalBytes, cudaHostAllocDefault));
            g_h_pinnedCap[s] = totalBytes;
        }
        if (totalBytes > g_d_bufCap[s]) {
            if (g_d_buf[s]) cudaFree(g_d_buf[s]);
            CUDA_CHECK(cudaMalloc(&g_d_buf[s], totalBytes));
            g_d_bufCap[s] = totalBytes;
        }
        if ((size_t)testSize > g_d_predCap[s]) {
            if (g_d_pred[s]) cudaFree(g_d_pred[s]);
            CUDA_CHECK(cudaMalloc(&g_d_pred[s], (size_t)testSize * sizeof(int)));
            g_d_predCap[s] = testSize;
        }

        // Split the host buffer
        char* hp = reinterpret_cast<char*>(g_h_pinned[s]);
        float* h_train = reinterpret_cast<float*>(hp);
        int*   h_lbl   = reinterpret_cast<int*>(hp + trainBytes);
        float* h_test  = reinterpret_cast<float*>(hp + trainBytes + labelBytes);

        // Pack training data and labels
        const int* trIds = trainIdx[f].data();
        for (int i = 0; i < trSize; ++i) {
            std::memcpy(h_train + (size_t)i * numSelected,
                        selData.data() + (size_t)trIds[i] * numSelected,
                        (size_t)numSelected * sizeof(float));
            h_lbl[i] = trainLabels[trIds[i]];
        }
        // Pack validation data
        const int* teIds = folds[f].data();
        for (int t = 0; t < testSize; ++t) {
            std::memcpy(h_test + (size_t)t * numSelected,
                        selData.data() + (size_t)teIds[t] * numSelected,
                        (size_t)numSelected * sizeof(float));
        }

        // Copy data to GPU
        CUDA_CHECK(cudaMemcpyAsync(g_d_buf[s], g_h_pinned[s], totalBytes,
                                   cudaMemcpyHostToDevice, g_streams[s]));

        // Split the device buffer
        char* dp = reinterpret_cast<char*>(g_d_buf[s]);
        float* d_train = reinterpret_cast<float*>(dp);
        int*   d_lbl   = reinterpret_cast<int*>(dp + trainBytes);
        float* d_test  = reinterpret_cast<float*>(dp + trainBytes + labelBytes);

        // Predict validation labels
        size_t smemBytes = (size_t)trSize * sizeof(float)
                         + (size_t)classes * sizeof(int);
        int grid = (testSize < 1024) ? testSize : 1024;
        const int neighbors = (KNN_neighbors > trSize) ? trSize : KNN_neighbors;
        knnVoteKernel<<<grid, KNN_BLOCK_SIZE, smemBytes, g_streams[s]>>>(
            d_train, d_lbl, d_test,
            trSize, testSize, numSelected, neighbors,
            classes, g_d_pred[s]);
        CUDA_CHECK(cudaGetLastError());

        // Copy predictions to CPU
        CUDA_CHECK(cudaMemcpyAsync(g_h_pred_pool + testBase, g_d_pred[s],
                                   (size_t)testSize * sizeof(int),
                                   cudaMemcpyDeviceToHost, g_streams[s]));

        // Mark completion of this fold
        CUDA_CHECK(cudaEventRecord(g_ev[s], g_streams[s]));

        testBase += testSize;
    }

    // Wait for all predictions
    CUDA_CHECK(cudaStreamSynchronize(g_streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(g_streams[1]));

    // Count errors in each fold
    std::vector<int> foldErrors(CV_folds, 0);
    int tb = 0;
    for (int f = 0; f < CV_folds; ++f) {
        const int testSize = foldSizes[f];
        const int* teIds = folds[f].data();
        for (int t = 0; t < testSize; ++t) {
            if (g_h_pred_pool[tb + t] != trainLabels[teIds[t]]) ++foldErrors[f];
        }
        tb += testSize;
    }

    // Combine classification error and feature ratio
    double totalErrorRate = 0.0;
    for (int f = 0; f < CV_folds; ++f)
        totalErrorRate += (double)foldErrors[f] / foldSizes[f];

    return omega * totalErrorRate / CV_folds
         + (1.0 - omega) * (double)numSelected / NF;
}

// [Modified] Encode feature selection status as a cache key
void fastKeyGen(int* fs, int NF, std::string& key) {
    const size_t numBytes = (NF + 7) / 8;
    key.resize(numBytes, 0);
    for (size_t byteIdx = 0; byteIdx < numBytes; ++byteIdx) {
        const int startBit = static_cast<int>(byteIdx) * 8;
        const int endBit   = std::min(startBit + 8, NF);
        unsigned char byte = 0;
        if (endBit - startBit == 8) {
            byte |= (fs[startBit + 0] ? 0x80 : 0);
            byte |= (fs[startBit + 1] ? 0x40 : 0);
            byte |= (fs[startBit + 2] ? 0x20 : 0);
            byte |= (fs[startBit + 3] ? 0x10 : 0);
            byte |= (fs[startBit + 4] ? 0x08 : 0);
            byte |= (fs[startBit + 5] ? 0x04 : 0);
            byte |= (fs[startBit + 6] ? 0x02 : 0);
            byte |= (fs[startBit + 7] ? 0x01 : 0);
        } else {
            for (int b = 0; b < endBit - startBit; ++b)
                byte |= (fs[startBit + b] ? (0x80 >> b) : 0);
        }
        key[byteIdx] = byte;
    }
}

// [Modified] GPU test validation with the training kernel
double testValidation(double** trainData, int* trainLabels, int trainSize,
                      double** testData, int* testLabels, int testSize,
                      int NF, int KNN_neighbors, int* final_fs, int classes){
    if (trainSize <= 0 || testSize <= 0 || NF <= 0 || classes <= 0){
        throw std::invalid_argument("Invalid test validation dimensions");
    }
    if (KNN_neighbors <= 0){
        throw std::invalid_argument("KNN_neighbors must be positive");
    }

    // Select features
    std::vector<int> selectedIndices;
    selectedIndices.reserve(NF);
    for (int i = 0; i < NF; ++i){
        if (final_fs[i]){
            selectedIndices.push_back(i);
        }
    }
    const int numSelected = static_cast<int>(selectedIndices.size());
    if (numSelected == 0){
        return 1.0;
    }

    // Pack selected features as floats
    std::vector<float> h_train(static_cast<size_t>(trainSize) * numSelected);
    std::vector<float> h_test(static_cast<size_t>(testSize) * numSelected);
    std::vector<int> predictions(testSize);
    for (int i = 0; i < trainSize; ++i){
        for (int j = 0; j < numSelected; ++j){
            h_train[static_cast<size_t>(i) * numSelected + j]
                = static_cast<float>(trainData[i][selectedIndices[j]]);
        }
    }
    for (int i = 0; i < testSize; ++i){
        for (int j = 0; j < numSelected; ++j){
            h_test[static_cast<size_t>(i) * numSelected + j]
                = static_cast<float>(testData[i][selectedIndices[j]]);
        }
    }

    // Allocate GPU buffers
    float* d_train = nullptr;
    float* d_test = nullptr;
    int* d_labels = nullptr;
    int* d_predictions = nullptr;
    const size_t trainBytes = h_train.size() * sizeof(float);
    const size_t testBytes = h_test.size() * sizeof(float);
    const size_t labelBytes = static_cast<size_t>(trainSize) * sizeof(int);
    const size_t predictionBytes = static_cast<size_t>(testSize) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_train, trainBytes));
    CUDA_CHECK(cudaMalloc(&d_test, testBytes));
    CUDA_CHECK(cudaMalloc(&d_labels, labelBytes));
    CUDA_CHECK(cudaMalloc(&d_predictions, predictionBytes));

    // Copy training and test data to GPU
    CUDA_CHECK(cudaMemcpy(d_train, h_train.data(), trainBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test, h_test.data(), testBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, trainLabels, labelBytes, cudaMemcpyHostToDevice));

    // Predict test labels with the shared kernel
    CUDA_CHECK(cudaFuncSetAttribute(
        knnVoteKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    const size_t sharedBytes = static_cast<size_t>(trainSize) * sizeof(float)
                             + static_cast<size_t>(classes) * sizeof(int);
    const int grid = std::min(testSize, 1024);
    const int neighbors = std::min(KNN_neighbors, trainSize);
    knnVoteKernel<<<grid, KNN_BLOCK_SIZE, sharedBytes>>>(
        d_train, d_labels, d_test, trainSize, testSize,
        numSelected, neighbors, classes, d_predictions);
    CUDA_CHECK(cudaGetLastError());

    // Copy predictions to CPU and count errors
    CUDA_CHECK(cudaMemcpy(predictions.data(), d_predictions,
                          predictionBytes, cudaMemcpyDeviceToHost));
    int misclassified_count = 0;
    for (int i = 0; i < testSize; ++i){
        if (predictions[i] != testLabels[i]){
            ++misclassified_count;
        }
    }

    // Release GPU buffers
    CUDA_CHECK(cudaFree(d_train));
    CUDA_CHECK(cudaFree(d_test));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_predictions));
    return static_cast<double>(misclassified_count) / testSize;
}