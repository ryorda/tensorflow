#include <stdio.h>
#include <cstdlib>

#include "tensorflow/contrib/lite/java/src/main/native/numeric_test.h"

/// logging
#include <android/log.h>
#include <fstream>
#include <time.h>
/// end of logging

#include <vector>
#include <omp.h>

int num_thread;

float PortableClip(float f, float abs_limit) {
  float result = (abs_limit < f) ? abs_limit : f;
  result = (-abs_limit > result) ? -abs_limit : result;
  return result;
}


void PortableVectorVectorCwiseProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      float* result) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v++) {
    *result++ = *vector1++ * *vector2++;
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableVectorVectorCwiseProduct : %d : %d : %f sec\n", num_thread, v_size, delta_time);
}


void PortableVectorVectorCwiseProductAccumulate(const float* vector1,
                                                const float* vector2,
                                                int v_size, float* result) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v++) {
    *result++ += *vector1++ * *vector2++;
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableVectorVectorCwiseProductAccumulate : %d : %d : %f sec\n", num_thread, v_size, delta_time);
    
}

void PortableVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < v_size; v++) {
      *result++ += vector[v] * *batch_vector++;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableVectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", num_thread, v_size, delta_time);

}

void Block2VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=2) {
    float vec[2];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[2];


      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "BlockVectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", 2, v_size, delta_time);

}

void Block4VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=4) {
    float vec[4];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;
    *(vec + 2) = v + 2 < v_size ? *(vector + v + 2) : 0;
    *(vec + 3) = v + 3 < v_size ? *(vector + v + 3) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[4];

      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 
      *(batch_vec + 2) = v + 2 < v_size ? *(batch + 2) : 0; 
      *(batch_vec + 3) = v + 3 < v_size ? *(batch + 3) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);
      if (v + 2 < v_size) *(restemp + 2)  += *(vec + 2) * *(batch_vec + 2);
      if (v + 3 < v_size) *(restemp + 3)  += *(vec + 3) * *(batch_vec + 3);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "BlockVectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", 4, v_size, delta_time);

}

void Block8VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=8) {
    float vec[8];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;
    *(vec + 2) = v + 2 < v_size ? *(vector + v + 2) : 0;
    *(vec + 3) = v + 3 < v_size ? *(vector + v + 3) : 0;
    *(vec + 4) = v + 4 < v_size ? *(vector + v + 4) : 0;
    *(vec + 5) = v + 5 < v_size ? *(vector + v + 5) : 0;
    *(vec + 6) = v + 6 < v_size ? *(vector + v + 6) : 0;
    *(vec + 7) = v + 7 < v_size ? *(vector + v + 7) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[8];

      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 
      *(batch_vec + 2) = v + 2 < v_size ? *(batch + 2) : 0; 
      *(batch_vec + 3) = v + 3 < v_size ? *(batch + 3) : 0; 
      *(batch_vec + 4) = v + 4 < v_size ? *(batch + 4) : 0; 
      *(batch_vec + 5) = v + 5 < v_size ? *(batch + 5) : 0; 
      *(batch_vec + 6) = v + 6 < v_size ? *(batch + 6) : 0; 
      *(batch_vec + 7) = v + 7 < v_size ? *(batch + 7) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);
      if (v + 2 < v_size) *(restemp + 2)  += *(vec + 2) * *(batch_vec + 2);
      if (v + 3 < v_size) *(restemp + 3)  += *(vec + 3) * *(batch_vec + 3);
      if (v + 4 < v_size) *(restemp + 4)  += *(vec + 4) * *(batch_vec + 4);
      if (v + 5 < v_size) *(restemp + 5)  += *(vec + 5) * *(batch_vec + 5);
      if (v + 6 < v_size) *(restemp + 6)  += *(vec + 6) * *(batch_vec + 6);
      if (v + 7 < v_size) *(restemp + 7)  += *(vec + 7) * *(batch_vec + 7);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "BlockVectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", 8, v_size, delta_time);

}

void Block16VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=16) {
    float vec[16];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;
    *(vec + 2) = v + 2 < v_size ? *(vector + v + 2) : 0;
    *(vec + 3) = v + 3 < v_size ? *(vector + v + 3) : 0;
    *(vec + 4) = v + 4 < v_size ? *(vector + v + 4) : 0;
    *(vec + 5) = v + 5 < v_size ? *(vector + v + 5) : 0;
    *(vec + 6) = v + 6 < v_size ? *(vector + v + 6) : 0;
    *(vec + 7) = v + 7 < v_size ? *(vector + v + 7) : 0;
    *(vec + 8) = v + 8 < v_size ? *(vector + v + 8) : 0;
    *(vec + 9) = v + 9 < v_size ? *(vector + v + 9) : 0;
    *(vec + 10) = v + 10 < v_size ? *(vector + v + 10) : 0;
    *(vec + 11) = v + 11 < v_size ? *(vector + v + 11) : 0;
    *(vec + 12) = v + 12 < v_size ? *(vector + v + 12) : 0;
    *(vec + 13) = v + 13 < v_size ? *(vector + v + 13) : 0;
    *(vec + 14) = v + 14 < v_size ? *(vector + v + 14) : 0;
    *(vec + 15) = v + 15 < v_size ? *(vector + v + 15) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[16];

      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 
      *(batch_vec + 2) = v + 2 < v_size ? *(batch + 2) : 0; 
      *(batch_vec + 3) = v + 3 < v_size ? *(batch + 3) : 0; 
      *(batch_vec + 4) = v + 4 < v_size ? *(batch + 4) : 0; 
      *(batch_vec + 5) = v + 5 < v_size ? *(batch + 5) : 0; 
      *(batch_vec + 6) = v + 6 < v_size ? *(batch + 6) : 0; 
      *(batch_vec + 7) = v + 7 < v_size ? *(batch + 7) : 0; 
      *(batch_vec + 8) = v + 8 < v_size ? *(batch + 8) : 0; 
      *(batch_vec + 9) = v + 9 < v_size ? *(batch + 9) : 0; 
      *(batch_vec + 10) = v + 10 < v_size ? *(batch + 10) : 0; 
      *(batch_vec + 11) = v + 11 < v_size ? *(batch + 11) : 0; 
      *(batch_vec + 12) = v + 12 < v_size ? *(batch + 12) : 0; 
      *(batch_vec + 13) = v + 13 < v_size ? *(batch + 13) : 0; 
      *(batch_vec + 14) = v + 14 < v_size ? *(batch + 14) : 0; 
      *(batch_vec + 15) = v + 15 < v_size ? *(batch + 15) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);
      if (v + 2 < v_size) *(restemp + 2)  += *(vec + 2) * *(batch_vec + 2);
      if (v + 3 < v_size) *(restemp + 3)  += *(vec + 3) * *(batch_vec + 3);
      if (v + 4 < v_size) *(restemp + 4)  += *(vec + 4) * *(batch_vec + 4);
      if (v + 5 < v_size) *(restemp + 5)  += *(vec + 5) * *(batch_vec + 5);
      if (v + 6 < v_size) *(restemp + 6)  += *(vec + 6) * *(batch_vec + 6);
      if (v + 7 < v_size) *(restemp + 7)  += *(vec + 7) * *(batch_vec + 7);
      if (v + 8 < v_size) *(restemp + 8)  += *(vec + 8) * *(batch_vec + 8);
      if (v + 9 < v_size) *(restemp + 9)  += *(vec + 9) * *(batch_vec + 9);
      if (v + 10 < v_size) *(restemp + 10)  += *(vec + 10) * *(batch_vec + 10);
      if (v + 11 < v_size) *(restemp + 11)  += *(vec + 11) * *(batch_vec + 11);
      if (v + 12 < v_size) *(restemp + 12)  += *(vec + 12) * *(batch_vec + 12);
      if (v + 13 < v_size) *(restemp + 13)  += *(vec + 13) * *(batch_vec + 13);
      if (v + 14 < v_size) *(restemp + 14)  += *(vec + 14) * *(batch_vec + 14);
      if (v + 15 < v_size) *(restemp + 15)  += *(vec + 15) * *(batch_vec + 15);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "BlockVectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", 16, v_size, delta_time);

}

void Block32VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=16) {
    float vec[16];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;
    *(vec + 2) = v + 2 < v_size ? *(vector + v + 2) : 0;
    *(vec + 3) = v + 3 < v_size ? *(vector + v + 3) : 0;
    *(vec + 4) = v + 4 < v_size ? *(vector + v + 4) : 0;
    *(vec + 5) = v + 5 < v_size ? *(vector + v + 5) : 0;
    *(vec + 6) = v + 6 < v_size ? *(vector + v + 6) : 0;
    *(vec + 7) = v + 7 < v_size ? *(vector + v + 7) : 0;
    *(vec + 8) = v + 8 < v_size ? *(vector + v + 8) : 0;
    *(vec + 9) = v + 9 < v_size ? *(vector + v + 9) : 0;
    *(vec + 10) = v + 10 < v_size ? *(vector + v + 10) : 0;
    *(vec + 11) = v + 11 < v_size ? *(vector + v + 11) : 0;
    *(vec + 12) = v + 12 < v_size ? *(vector + v + 12) : 0;
    *(vec + 13) = v + 13 < v_size ? *(vector + v + 13) : 0;
    *(vec + 14) = v + 14 < v_size ? *(vector + v + 14) : 0;
    *(vec + 15) = v + 15 < v_size ? *(vector + v + 15) : 0;
    *(vec + 16) = v + 16 < v_size ? *(vector + v + 16) : 0;
    *(vec + 17) = v + 17 < v_size ? *(vector + v + 17) : 0;
    *(vec + 18) = v + 18 < v_size ? *(vector + v + 18) : 0;
    *(vec + 19) = v + 19 < v_size ? *(vector + v + 19) : 0;
    *(vec + 20) = v + 20 < v_size ? *(vector + v + 20) : 0;
    *(vec + 21) = v + 21 < v_size ? *(vector + v + 21) : 0;
    *(vec + 22) = v + 22 < v_size ? *(vector + v + 22) : 0;
    *(vec + 23) = v + 23 < v_size ? *(vector + v + 23) : 0;
    *(vec + 24) = v + 24 < v_size ? *(vector + v + 24) : 0;
    *(vec + 25) = v + 25 < v_size ? *(vector + v + 25) : 0;
    *(vec + 26) = v + 26 < v_size ? *(vector + v + 26) : 0;
    *(vec + 27) = v + 27 < v_size ? *(vector + v + 27) : 0;
    *(vec + 28) = v + 28 < v_size ? *(vector + v + 28) : 0;
    *(vec + 29) = v + 29 < v_size ? *(vector + v + 29) : 0;
    *(vec + 30) = v + 30 < v_size ? *(vector + v + 30) : 0;
    *(vec + 31) = v + 31 < v_size ? *(vector + v + 31) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[16];

      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 
      *(batch_vec + 2) = v + 2 < v_size ? *(batch + 2) : 0; 
      *(batch_vec + 3) = v + 3 < v_size ? *(batch + 3) : 0; 
      *(batch_vec + 4) = v + 4 < v_size ? *(batch + 4) : 0; 
      *(batch_vec + 5) = v + 5 < v_size ? *(batch + 5) : 0; 
      *(batch_vec + 6) = v + 6 < v_size ? *(batch + 6) : 0; 
      *(batch_vec + 7) = v + 7 < v_size ? *(batch + 7) : 0; 
      *(batch_vec + 8) = v + 8 < v_size ? *(batch + 8) : 0; 
      *(batch_vec + 9) = v + 9 < v_size ? *(batch + 9) : 0; 
      *(batch_vec + 10) = v + 10 < v_size ? *(batch + 10) : 0; 
      *(batch_vec + 11) = v + 11 < v_size ? *(batch + 11) : 0; 
      *(batch_vec + 12) = v + 12 < v_size ? *(batch + 12) : 0; 
      *(batch_vec + 13) = v + 13 < v_size ? *(batch + 13) : 0; 
      *(batch_vec + 14) = v + 14 < v_size ? *(batch + 14) : 0; 
      *(batch_vec + 15) = v + 15 < v_size ? *(batch + 15) : 0; 
      *(batch_vec + 16) = v + 16 < v_size ? *(batch + 16) : 0; 
      *(batch_vec + 17) = v + 17 < v_size ? *(batch + 17) : 0; 
      *(batch_vec + 18) = v + 18 < v_size ? *(batch + 18) : 0; 
      *(batch_vec + 19) = v + 19 < v_size ? *(batch + 19) : 0; 
      *(batch_vec + 20) = v + 20 < v_size ? *(batch + 20) : 0; 
      *(batch_vec + 21) = v + 21 < v_size ? *(batch + 21) : 0; 
      *(batch_vec + 22) = v + 22 < v_size ? *(batch + 22) : 0; 
      *(batch_vec + 23) = v + 23 < v_size ? *(batch + 23) : 0; 
      *(batch_vec + 24) = v + 24 < v_size ? *(batch + 24) : 0; 
      *(batch_vec + 25) = v + 25 < v_size ? *(batch + 25) : 0; 
      *(batch_vec + 26) = v + 26 < v_size ? *(batch + 26) : 0; 
      *(batch_vec + 27) = v + 27 < v_size ? *(batch + 27) : 0; 
      *(batch_vec + 28) = v + 28 < v_size ? *(batch + 28) : 0; 
      *(batch_vec + 29) = v + 29 < v_size ? *(batch + 29) : 0; 
      *(batch_vec + 30) = v + 30 < v_size ? *(batch + 30) : 0; 
      *(batch_vec + 31) = v + 31 < v_size ? *(batch + 31) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);
      if (v + 2 < v_size) *(restemp + 2)  += *(vec + 2) * *(batch_vec + 2);
      if (v + 3 < v_size) *(restemp + 3)  += *(vec + 3) * *(batch_vec + 3);
      if (v + 4 < v_size) *(restemp + 4)  += *(vec + 4) * *(batch_vec + 4);
      if (v + 5 < v_size) *(restemp + 5)  += *(vec + 5) * *(batch_vec + 5);
      if (v + 6 < v_size) *(restemp + 6)  += *(vec + 6) * *(batch_vec + 6);
      if (v + 7 < v_size) *(restemp + 7)  += *(vec + 7) * *(batch_vec + 7);
      if (v + 8 < v_size) *(restemp + 8)  += *(vec + 8) * *(batch_vec + 8);
      if (v + 9 < v_size) *(restemp + 9)  += *(vec + 9) * *(batch_vec + 9);
      if (v + 10 < v_size) *(restemp + 10)  += *(vec + 10) * *(batch_vec + 10);
      if (v + 11 < v_size) *(restemp + 11)  += *(vec + 11) * *(batch_vec + 11);
      if (v + 12 < v_size) *(restemp + 12)  += *(vec + 12) * *(batch_vec + 12);
      if (v + 13 < v_size) *(restemp + 13)  += *(vec + 13) * *(batch_vec + 13);
      if (v + 14 < v_size) *(restemp + 14)  += *(vec + 14) * *(batch_vec + 14);
      if (v + 15 < v_size) *(restemp + 15)  += *(vec + 15) * *(batch_vec + 15);
      if (v + 16 < v_size) *(restemp + 16)  += *(vec + 16) * *(batch_vec + 16);
      if (v + 17 < v_size) *(restemp + 17)  += *(vec + 17) * *(batch_vec + 17);
      if (v + 18 < v_size) *(restemp + 18)  += *(vec + 18) * *(batch_vec + 18);
      if (v + 19 < v_size) *(restemp + 19)  += *(vec + 19) * *(batch_vec + 19);
      if (v + 20 < v_size) *(restemp + 20)  += *(vec + 20) * *(batch_vec + 20);
      if (v + 21 < v_size) *(restemp + 21)  += *(vec + 21) * *(batch_vec + 21);
      if (v + 22 < v_size) *(restemp + 22)  += *(vec + 22) * *(batch_vec + 22);
      if (v + 23 < v_size) *(restemp + 23)  += *(vec + 23) * *(batch_vec + 23);
      if (v + 24 < v_size) *(restemp + 24)  += *(vec + 24) * *(batch_vec + 24);
      if (v + 25 < v_size) *(restemp + 25)  += *(vec + 25) * *(batch_vec + 25);
      if (v + 26 < v_size) *(restemp + 26)  += *(vec + 26) * *(batch_vec + 26);
      if (v + 27 < v_size) *(restemp + 27)  += *(vec + 27) * *(batch_vec + 27);
      if (v + 28 < v_size) *(restemp + 28)  += *(vec + 28) * *(batch_vec + 28);
      if (v + 29 < v_size) *(restemp + 29)  += *(vec + 29) * *(batch_vec + 29);
      if (v + 30 < v_size) *(restemp + 30)  += *(vec + 30) * *(batch_vec + 30);
      if (v + 31 < v_size) *(restemp + 31)  += *(vec + 31) * *(batch_vec + 31);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "BlockVectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", 32, v_size, delta_time);

}

void PortableVectorBatchVectorAssign(const float* vector, int v_size,
                                     int n_batch, float* batch_vector) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int b = 0; b < n_batch; b++) {
    memcpy(batch_vector + b * v_size, vector, v_size * sizeof(float));
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableVectorBatchVectorAssign : %d : %d : %f sec\n", num_thread, v_size, delta_time);

}


void PortableSub1Vector(const float* vector, int v_size, float* result) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v++) {
    *result++ = 1.0f - *vector++;
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableSub1Vector : %d : %d : %f sec\n", num_thread, v_size, delta_time);
}


void PortableClipVector(const float* vector, int v_size, float abs_limit,
                        float* result) {


  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v++) {
    *result++ = PortableClip(*vector++, abs_limit);
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableClipVector : %d : %d : %f sec\n", num_thread, v_size, delta_time);
}


void PortableReductionSumVector(const float* input_vector, float* output_vector,
                                int output_size, int reduction_size) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  const float* input_vector_ptr = input_vector;
  #pragma omp for schedule(static)
  for (int o = 0; o < output_size; o++) {
    for (int r = 0; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableReductionSumVector : %d : %d : %f sec\n", num_thread, output_size, delta_time);
}


// SEQUENTIAL

void SequentialPortableVectorVectorCwiseProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      float* result) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v++) {
    *result++ = *vector1++ * *vector2++;
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableVectorVectorCwiseProduct : 1 : %d : %f sec\n", v_size, delta_time);
}


void SequentialPortableVectorVectorCwiseProductAccumulate(const float* vector1,
                                                const float* vector2,
                                                int v_size, float* result) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v++) {
    *result++ += *vector1++ * *vector2++;
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableVectorVectorCwiseProductAccumulate : 1 : %d : %f sec\n", v_size, delta_time);
    
}

void SequentialPortableVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < v_size; v++) {
      *result++ += vector[v] * *batch_vector++;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableVectorBatchVectorCwiseProductAccumulate : 1 : %d : %f sec\n", v_size, delta_time);

}

void SequentialPortableVectorBatchVectorAssign(const float* vector, int v_size,
                                     int n_batch, float* batch_vector) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int b = 0; b < n_batch; b++) {
    memcpy(batch_vector + b * v_size, vector, v_size * sizeof(float));
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableVectorBatchVectorAssign : 1 : %d : %f sec\n", v_size, delta_time);

}


void SequentialPortableSub1Vector(const float* vector, int v_size, float* result) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v++) {
    *result++ = 1.0f - *vector++;
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableSub1Vector : 1 : %d : %f sec\n", v_size, delta_time);
}


void SequentialPortableClipVector(const float* vector, int v_size, float abs_limit,
                        float* result) {


  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v++) {
    *result++ = PortableClip(*vector++, abs_limit);
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableClipVector : 1 : %d : %f sec\n", v_size, delta_time);
}


void SequentialPortableReductionSumVector(const float* input_vector, float* output_vector,
                                int output_size, int reduction_size) {

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  const float* input_vector_ptr = input_vector;
  // #pragma omp for schedule(static)
  for (int o = 0; o < output_size; o++) {
    for (int r = 0; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "PortableReductionSumVector : 1 : %d : %f sec\n", output_size, delta_time);
}

void ParallelBlock2VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=2) {
    float vec[2];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[2];


      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "ParallelBlock2VectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", num_thread, v_size, delta_time);

}

void ParallelBlock4VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=4) {
    float vec[4];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;
    *(vec + 2) = v + 2 < v_size ? *(vector + v + 2) : 0;
    *(vec + 3) = v + 3 < v_size ? *(vector + v + 3) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[4];

      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 
      *(batch_vec + 2) = v + 2 < v_size ? *(batch + 2) : 0; 
      *(batch_vec + 3) = v + 3 < v_size ? *(batch + 3) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);
      if (v + 2 < v_size) *(restemp + 2)  += *(vec + 2) * *(batch_vec + 2);
      if (v + 3 < v_size) *(restemp + 3)  += *(vec + 3) * *(batch_vec + 3);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "ParallelBlock4VectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", num_thread, v_size, delta_time);

}

void ParallelBlock8VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=8) {
    float vec[8];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;
    *(vec + 2) = v + 2 < v_size ? *(vector + v + 2) : 0;
    *(vec + 3) = v + 3 < v_size ? *(vector + v + 3) : 0;
    *(vec + 4) = v + 4 < v_size ? *(vector + v + 4) : 0;
    *(vec + 5) = v + 5 < v_size ? *(vector + v + 5) : 0;
    *(vec + 6) = v + 6 < v_size ? *(vector + v + 6) : 0;
    *(vec + 7) = v + 7 < v_size ? *(vector + v + 7) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[8];

      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 
      *(batch_vec + 2) = v + 2 < v_size ? *(batch + 2) : 0; 
      *(batch_vec + 3) = v + 3 < v_size ? *(batch + 3) : 0; 
      *(batch_vec + 4) = v + 4 < v_size ? *(batch + 4) : 0; 
      *(batch_vec + 5) = v + 5 < v_size ? *(batch + 5) : 0; 
      *(batch_vec + 6) = v + 6 < v_size ? *(batch + 6) : 0; 
      *(batch_vec + 7) = v + 7 < v_size ? *(batch + 7) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);
      if (v + 2 < v_size) *(restemp + 2)  += *(vec + 2) * *(batch_vec + 2);
      if (v + 3 < v_size) *(restemp + 3)  += *(vec + 3) * *(batch_vec + 3);
      if (v + 4 < v_size) *(restemp + 4)  += *(vec + 4) * *(batch_vec + 4);
      if (v + 5 < v_size) *(restemp + 5)  += *(vec + 5) * *(batch_vec + 5);
      if (v + 6 < v_size) *(restemp + 6)  += *(vec + 6) * *(batch_vec + 6);
      if (v + 7 < v_size) *(restemp + 7)  += *(vec + 7) * *(batch_vec + 7);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "ParallelBlock8VectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", num_thread, v_size, delta_time);

}

void ParallelBlock16VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=16) {
    float vec[16];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;
    *(vec + 2) = v + 2 < v_size ? *(vector + v + 2) : 0;
    *(vec + 3) = v + 3 < v_size ? *(vector + v + 3) : 0;
    *(vec + 4) = v + 4 < v_size ? *(vector + v + 4) : 0;
    *(vec + 5) = v + 5 < v_size ? *(vector + v + 5) : 0;
    *(vec + 6) = v + 6 < v_size ? *(vector + v + 6) : 0;
    *(vec + 7) = v + 7 < v_size ? *(vector + v + 7) : 0;
    *(vec + 8) = v + 8 < v_size ? *(vector + v + 8) : 0;
    *(vec + 9) = v + 9 < v_size ? *(vector + v + 9) : 0;
    *(vec + 10) = v + 10 < v_size ? *(vector + v + 10) : 0;
    *(vec + 11) = v + 11 < v_size ? *(vector + v + 11) : 0;
    *(vec + 12) = v + 12 < v_size ? *(vector + v + 12) : 0;
    *(vec + 13) = v + 13 < v_size ? *(vector + v + 13) : 0;
    *(vec + 14) = v + 14 < v_size ? *(vector + v + 14) : 0;
    *(vec + 15) = v + 15 < v_size ? *(vector + v + 15) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[16];

      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 
      *(batch_vec + 2) = v + 2 < v_size ? *(batch + 2) : 0; 
      *(batch_vec + 3) = v + 3 < v_size ? *(batch + 3) : 0; 
      *(batch_vec + 4) = v + 4 < v_size ? *(batch + 4) : 0; 
      *(batch_vec + 5) = v + 5 < v_size ? *(batch + 5) : 0; 
      *(batch_vec + 6) = v + 6 < v_size ? *(batch + 6) : 0; 
      *(batch_vec + 7) = v + 7 < v_size ? *(batch + 7) : 0; 
      *(batch_vec + 8) = v + 8 < v_size ? *(batch + 8) : 0; 
      *(batch_vec + 9) = v + 9 < v_size ? *(batch + 9) : 0; 
      *(batch_vec + 10) = v + 10 < v_size ? *(batch + 10) : 0; 
      *(batch_vec + 11) = v + 11 < v_size ? *(batch + 11) : 0; 
      *(batch_vec + 12) = v + 12 < v_size ? *(batch + 12) : 0; 
      *(batch_vec + 13) = v + 13 < v_size ? *(batch + 13) : 0; 
      *(batch_vec + 14) = v + 14 < v_size ? *(batch + 14) : 0; 
      *(batch_vec + 15) = v + 15 < v_size ? *(batch + 15) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);
      if (v + 2 < v_size) *(restemp + 2)  += *(vec + 2) * *(batch_vec + 2);
      if (v + 3 < v_size) *(restemp + 3)  += *(vec + 3) * *(batch_vec + 3);
      if (v + 4 < v_size) *(restemp + 4)  += *(vec + 4) * *(batch_vec + 4);
      if (v + 5 < v_size) *(restemp + 5)  += *(vec + 5) * *(batch_vec + 5);
      if (v + 6 < v_size) *(restemp + 6)  += *(vec + 6) * *(batch_vec + 6);
      if (v + 7 < v_size) *(restemp + 7)  += *(vec + 7) * *(batch_vec + 7);
      if (v + 8 < v_size) *(restemp + 8)  += *(vec + 8) * *(batch_vec + 8);
      if (v + 9 < v_size) *(restemp + 9)  += *(vec + 9) * *(batch_vec + 9);
      if (v + 10 < v_size) *(restemp + 10)  += *(vec + 10) * *(batch_vec + 10);
      if (v + 11 < v_size) *(restemp + 11)  += *(vec + 11) * *(batch_vec + 11);
      if (v + 12 < v_size) *(restemp + 12)  += *(vec + 12) * *(batch_vec + 12);
      if (v + 13 < v_size) *(restemp + 13)  += *(vec + 13) * *(batch_vec + 13);
      if (v + 14 < v_size) *(restemp + 14)  += *(vec + 14) * *(batch_vec + 14);
      if (v + 15 < v_size) *(restemp + 15)  += *(vec + 15) * *(batch_vec + 15);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "ParallelBlock16VectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", num_thread, v_size, delta_time);

}

void ParallelBlock32VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  #pragma omp for schedule(static)
  for (int v = 0; v < v_size; v+=32) {
    float vec[32];
    *vec = v < v_size ? *(vector + v) : 0;
    *(vec + 1) = v + 1 < v_size ? *(vector + v + 1) : 0;
    *(vec + 2) = v + 2 < v_size ? *(vector + v + 2) : 0;
    *(vec + 3) = v + 3 < v_size ? *(vector + v + 3) : 0;
    *(vec + 4) = v + 4 < v_size ? *(vector + v + 4) : 0;
    *(vec + 5) = v + 5 < v_size ? *(vector + v + 5) : 0;
    *(vec + 6) = v + 6 < v_size ? *(vector + v + 6) : 0;
    *(vec + 7) = v + 7 < v_size ? *(vector + v + 7) : 0;
    *(vec + 8) = v + 8 < v_size ? *(vector + v + 8) : 0;
    *(vec + 9) = v + 9 < v_size ? *(vector + v + 9) : 0;
    *(vec + 10) = v + 10 < v_size ? *(vector + v + 10) : 0;
    *(vec + 11) = v + 11 < v_size ? *(vector + v + 11) : 0;
    *(vec + 12) = v + 12 < v_size ? *(vector + v + 12) : 0;
    *(vec + 13) = v + 13 < v_size ? *(vector + v + 13) : 0;
    *(vec + 14) = v + 14 < v_size ? *(vector + v + 14) : 0;
    *(vec + 15) = v + 15 < v_size ? *(vector + v + 15) : 0;
    *(vec + 16) = v + 16 < v_size ? *(vector + v + 16) : 0;
    *(vec + 17) = v + 17 < v_size ? *(vector + v + 17) : 0;
    *(vec + 18) = v + 18 < v_size ? *(vector + v + 18) : 0;
    *(vec + 19) = v + 19 < v_size ? *(vector + v + 19) : 0;
    *(vec + 20) = v + 20 < v_size ? *(vector + v + 20) : 0;
    *(vec + 21) = v + 21 < v_size ? *(vector + v + 21) : 0;
    *(vec + 22) = v + 22 < v_size ? *(vector + v + 22) : 0;
    *(vec + 23) = v + 23 < v_size ? *(vector + v + 23) : 0;
    *(vec + 24) = v + 24 < v_size ? *(vector + v + 24) : 0;
    *(vec + 25) = v + 25 < v_size ? *(vector + v + 25) : 0;
    *(vec + 26) = v + 26 < v_size ? *(vector + v + 26) : 0;
    *(vec + 27) = v + 27 < v_size ? *(vector + v + 27) : 0;
    *(vec + 28) = v + 28 < v_size ? *(vector + v + 28) : 0;
    *(vec + 29) = v + 29 < v_size ? *(vector + v + 29) : 0;
    *(vec + 30) = v + 30 < v_size ? *(vector + v + 30) : 0;
    *(vec + 31) = v + 31 < v_size ? *(vector + v + 31) : 0;

    const float *batch = batch_vector + v;
    float* restemp = result + v;
    
    for (int b = 0; b < n_batch; b++) {
      float batch_vec[16];

      *batch_vec = v < v_size ? *batch : 0; 
      *(batch_vec + 1) = v + 1 < v_size ? *(batch + 1) : 0; 
      *(batch_vec + 2) = v + 2 < v_size ? *(batch + 2) : 0; 
      *(batch_vec + 3) = v + 3 < v_size ? *(batch + 3) : 0; 
      *(batch_vec + 4) = v + 4 < v_size ? *(batch + 4) : 0; 
      *(batch_vec + 5) = v + 5 < v_size ? *(batch + 5) : 0; 
      *(batch_vec + 6) = v + 6 < v_size ? *(batch + 6) : 0; 
      *(batch_vec + 7) = v + 7 < v_size ? *(batch + 7) : 0; 
      *(batch_vec + 8) = v + 8 < v_size ? *(batch + 8) : 0; 
      *(batch_vec + 9) = v + 9 < v_size ? *(batch + 9) : 0; 
      *(batch_vec + 10) = v + 10 < v_size ? *(batch + 10) : 0; 
      *(batch_vec + 11) = v + 11 < v_size ? *(batch + 11) : 0; 
      *(batch_vec + 12) = v + 12 < v_size ? *(batch + 12) : 0; 
      *(batch_vec + 13) = v + 13 < v_size ? *(batch + 13) : 0; 
      *(batch_vec + 14) = v + 14 < v_size ? *(batch + 14) : 0; 
      *(batch_vec + 15) = v + 15 < v_size ? *(batch + 15) : 0; 
      *(batch_vec + 16) = v + 16 < v_size ? *(batch + 16) : 0; 
      *(batch_vec + 17) = v + 17 < v_size ? *(batch + 17) : 0; 
      *(batch_vec + 18) = v + 18 < v_size ? *(batch + 18) : 0; 
      *(batch_vec + 19) = v + 19 < v_size ? *(batch + 19) : 0; 
      *(batch_vec + 20) = v + 20 < v_size ? *(batch + 20) : 0; 
      *(batch_vec + 21) = v + 21 < v_size ? *(batch + 21) : 0; 
      *(batch_vec + 22) = v + 22 < v_size ? *(batch + 22) : 0; 
      *(batch_vec + 23) = v + 23 < v_size ? *(batch + 23) : 0; 
      *(batch_vec + 24) = v + 24 < v_size ? *(batch + 24) : 0; 
      *(batch_vec + 25) = v + 25 < v_size ? *(batch + 25) : 0; 
      *(batch_vec + 26) = v + 26 < v_size ? *(batch + 26) : 0; 
      *(batch_vec + 27) = v + 27 < v_size ? *(batch + 27) : 0; 
      *(batch_vec + 28) = v + 28 < v_size ? *(batch + 28) : 0; 
      *(batch_vec + 29) = v + 29 < v_size ? *(batch + 29) : 0; 
      *(batch_vec + 30) = v + 30 < v_size ? *(batch + 30) : 0; 
      *(batch_vec + 31) = v + 31 < v_size ? *(batch + 31) : 0; 

      if (v < v_size) *restemp  += *vec * *batch_vec;
      if (v + 1 < v_size) *(restemp + 1)  += *(vec + 1) * *(batch_vec + 1);
      if (v + 2 < v_size) *(restemp + 2)  += *(vec + 2) * *(batch_vec + 2);
      if (v + 3 < v_size) *(restemp + 3)  += *(vec + 3) * *(batch_vec + 3);
      if (v + 4 < v_size) *(restemp + 4)  += *(vec + 4) * *(batch_vec + 4);
      if (v + 5 < v_size) *(restemp + 5)  += *(vec + 5) * *(batch_vec + 5);
      if (v + 6 < v_size) *(restemp + 6)  += *(vec + 6) * *(batch_vec + 6);
      if (v + 7 < v_size) *(restemp + 7)  += *(vec + 7) * *(batch_vec + 7);
      if (v + 8 < v_size) *(restemp + 8)  += *(vec + 8) * *(batch_vec + 8);
      if (v + 9 < v_size) *(restemp + 9)  += *(vec + 9) * *(batch_vec + 9);
      if (v + 10 < v_size) *(restemp + 10)  += *(vec + 10) * *(batch_vec + 10);
      if (v + 11 < v_size) *(restemp + 11)  += *(vec + 11) * *(batch_vec + 11);
      if (v + 12 < v_size) *(restemp + 12)  += *(vec + 12) * *(batch_vec + 12);
      if (v + 13 < v_size) *(restemp + 13)  += *(vec + 13) * *(batch_vec + 13);
      if (v + 14 < v_size) *(restemp + 14)  += *(vec + 14) * *(batch_vec + 14);
      if (v + 15 < v_size) *(restemp + 15)  += *(vec + 15) * *(batch_vec + 15);
      if (v + 16 < v_size) *(restemp + 16)  += *(vec + 16) * *(batch_vec + 16);
      if (v + 17 < v_size) *(restemp + 17)  += *(vec + 17) * *(batch_vec + 17);
      if (v + 18 < v_size) *(restemp + 18)  += *(vec + 18) * *(batch_vec + 18);
      if (v + 19 < v_size) *(restemp + 19)  += *(vec + 19) * *(batch_vec + 19);
      if (v + 20 < v_size) *(restemp + 20)  += *(vec + 20) * *(batch_vec + 20);
      if (v + 21 < v_size) *(restemp + 21)  += *(vec + 21) * *(batch_vec + 21);
      if (v + 22 < v_size) *(restemp + 22)  += *(vec + 22) * *(batch_vec + 22);
      if (v + 23 < v_size) *(restemp + 23)  += *(vec + 23) * *(batch_vec + 23);
      if (v + 24 < v_size) *(restemp + 24)  += *(vec + 24) * *(batch_vec + 24);
      if (v + 25 < v_size) *(restemp + 25)  += *(vec + 25) * *(batch_vec + 25);
      if (v + 26 < v_size) *(restemp + 26)  += *(vec + 26) * *(batch_vec + 26);
      if (v + 27 < v_size) *(restemp + 27)  += *(vec + 27) * *(batch_vec + 27);
      if (v + 28 < v_size) *(restemp + 28)  += *(vec + 28) * *(batch_vec + 28);
      if (v + 29 < v_size) *(restemp + 29)  += *(vec + 29) * *(batch_vec + 29);
      if (v + 30 < v_size) *(restemp + 30)  += *(vec + 30) * *(batch_vec + 30);
      if (v + 31 < v_size) *(restemp + 31)  += *(vec + 31) * *(batch_vec + 31);

      restemp += v_size;
      batch += v_size;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "ParallelBlock32VectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", num_thread, v_size, delta_time);

}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_runNumericTest(JNIEnv* env, jclass /*clazz*/) {
  omp_set_dynamic(0);     // Explicitly disable dynamic teams

  for (int n_thread : {1, 2, 4, 8}){
    num_thread = n_thread;

    omp_set_num_threads(num_thread);
   
    for (int v_size = 4; v_size <= 1024; v_size *= 4){

      float *vector1, *vector2, *result;
      float *batch_vector, *batch_result;

      vector1 = (float *) malloc(v_size * sizeof(float *));
      vector2 = (float *) malloc(v_size * sizeof(float *));
      result = (float *) malloc(v_size * sizeof(float *));
      batch_vector = (float *) malloc(v_size * v_size * sizeof(float *));
      batch_result = (float *) malloc(v_size * v_size * sizeof(float *));

      for (int i = 0; i < v_size; i++){
        vector1[i] = 1;
        vector2[i] = 1;
        for (int j = 0; j < v_size; j++){
          batch_vector[i * v_size + j] = 1;
        }
      }

      if (num_thread > 1){

        PortableVectorVectorCwiseProduct(vector1, vector2, v_size, result);
        PortableVectorVectorCwiseProductAccumulate(vector1, vector2, v_size, result);
        PortableVectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        PortableVectorBatchVectorAssign(vector1, v_size, v_size, batch_result);
        PortableSub1Vector(vector1, v_size, result);
        PortableClipVector(vector1, v_size, 120, result);
        PortableReductionSumVector(batch_vector, result, v_size, v_size);

        ParallelBlock2VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        ParallelBlock4VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        ParallelBlock8VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        ParallelBlock16VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        ParallelBlock32VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);

      }
      else {

        SequentialPortableVectorVectorCwiseProduct(vector1, vector2, v_size, result);
        SequentialPortableVectorVectorCwiseProductAccumulate(vector1, vector2, v_size, result);
        SequentialPortableVectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        SequentialPortableVectorBatchVectorAssign(vector1, v_size, v_size, batch_result);
        SequentialPortableSub1Vector(vector1, v_size, result);
        SequentialPortableClipVector(vector1, v_size, 120, result);
        SequentialPortableReductionSumVector(batch_vector, result, v_size, v_size);

        Block2VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        Block4VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        Block8VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        Block16VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);
        Block32VectorBatchVectorCwiseProductAccumulate(vector1, v_size, batch_vector, v_size, batch_result);

      }
    } 
  }
}