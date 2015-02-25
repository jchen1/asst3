/* Copyright 2014 15418 Staff */

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>

#include <random>
#include <vector>
#include <numeric>

#include "parallelSort.h"

using namespace std;

void printArr(const char* arrName, int *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %d %d %d %d\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0,
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0);
  }
#endif
}

void printArr(const char* arrName, float *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %f %f %f %f\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0,
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0);
  }
#endif
}

void randomSample(float *data, size_t dataSize, float *sample, size_t sampleSize) {
  for (size_t i=0; i<sampleSize; i++) {
    sample[i] = data[rand()%dataSize];
  }
}

void parallelSort(float *data, float *&sortedData, int procs, int procId, size_t dataSize, size_t &localSize) {
  // Implement parallel sort algorithm as described in assignment 3
  // handout.
  // Input:
  //  data[]: input arrays of unsorted data, *distributed* across p processors
  //          note that data[] array on each process contains *a fraction* of all data
  //  sortedData[]: output arrays of sorted data, initially unallocated
  //                please update the size of sortedData[] to localSize!
  //  procs: total number of processes
  //  procId: id of this process
  //  dataSize: aggregated size of *all* data[] arrays
  //  localSize: size of data[] array on *this* process (as input)
  //             size of sortedData[] array on *this* process (as output)
  //
  //
  // Step 1: Choosing Pivots to Define Buckets
  // Step 2: Bucketing Elements of the Input Array
  // Step 3: Redistributing Elements
  // Step 5: Final Local Sort
  // ***********************************************************************


  // Step 1:  each process chooses 12 * lg(dataSize) values and sends them to
  //          process 0, which sorts them and sends the pivots back out

  int numValues = 12 * log(dataSize);

  float* samples, *pivots = new float[procs - 1];
  MPI_Request* requests;
  // Process 0 receives all of the samples so needs a bigger array, receives
  // non-blocking so that it can generate random samples as well
  if (procId == 0) {
    samples = new float[procs * numValues];
    requests = new MPI_Request[procs - 1];
    for (int i = 1; i < procs; i++) {
      MPI_Irecv(samples + (i * numValues), numValues, MPI_FLOAT, i, i, MPI_COMM_WORLD, &requests[i - 1]);
    }
  }
  else {
    samples = new float[numValues];
  }

  // Generate random samples using fancy C++11 things
  random_device rd;
  default_random_engine el;
  uniform_int_distribution<size_t> uniform_dist(0, localSize - 1);

  for (int i = 0; i < numValues; i++) {
    samples[i] = data[uniform_dist(el)];
  }

  // Process 0 now waits for the other processors to finish sending data, then
  // sorts and selects pivots, and sends them back out to every other processor
  if (procId == 0) {
    for (int i = 0; i < procs - 1; i++) {
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
    }

    sort(samples, samples + (procs * numValues));
    for (int i = 1; i < procs; i++) {
      pivots[i - 1] = samples[i * numValues];
    }

    delete[] requests;
  }
  else {
    MPI_Send(samples, numValues, MPI_FLOAT, 0, procId, MPI_COMM_WORLD);
  }

  MPI_Bcast(pivots, procs - 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  delete[] samples;

  // Step 2:  each process buckets their local data

  vector<vector<float>> localBuckets(procs);

  for (int i = 0; i < localSize; i++) {
    int bucketIndex = lower_bound(pivots, pivots + procs - 1, data[i]) - pivots;
    localBuckets[bucketIndex].push_back(data[i]);
  }

  // Step 3:  each process determines how big their bucket is, then collects
  //          its bucket from the other processes

  int* localBucketSizes = new int[procs], *localBucketOffsets = new int[procs + 1];
  int* bucketOffsets = new int[procs + 1], *bucketSizes = new int[procs];
  for (int i = 0; i < procs; i++) {
    localBucketSizes[i] = localBuckets[i].size();
  }

  localBucketOffsets[0] = 0;
  partial_sum(localBucketSizes, localBucketSizes + procs, localBucketOffsets + 1);

  bucketOffsets[0] = 0;
  MPI_Alltoall(localBucketSizes, 1, MPI_INT, bucketSizes, 1, MPI_INT, MPI_COMM_WORLD);

  partial_sum(bucketSizes, bucketSizes + procs, bucketOffsets + 1);
  localSize = bucketOffsets[procs];

  sortedData = new float[localSize];

  // Copy the data for the alltoall, since the vectors aren't contiguous
  float* localData = new float[localBucketOffsets[procs]];
  for (int i = 0, idx = 0; i < localBuckets.size(); i++) {
    for (int j = 0; j < localBuckets[i].size(); j++) {
      localData[idx++] = localBuckets[i][j];
    }
  }

  // Send data to the proper buckets
  MPI_Alltoallv(localData, localBucketSizes, localBucketOffsets, MPI_FLOAT,
                sortedData, bucketSizes, bucketOffsets, MPI_FLOAT, MPI_COMM_WORLD);


  // Step 4:  locally sort the data
  sort(sortedData, sortedData + localSize);

  // Output:
  //  sortedData[]: output arrays of sorted data, initially unallocated
  //                please update the size of sortedData[] to localSize!
  //  localSize: size of data[] array on *this* process (as input)
  //             size of sortedData[] array on *this* process (as output)
  return;
}

