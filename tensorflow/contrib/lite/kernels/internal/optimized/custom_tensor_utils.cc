/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <string.h>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

#include "tensorflow/contrib/lite/kernels/internal/optimized/tensor_utils_impl.h"

#include "tensorflow/contrib/android_renderscript_ops/jni/RScommon.h"
#include "tensorflow/contrib/android_renderscript_ops/jni/rsMatmul.h"

/// logging
#include <android/log.h>
#include <fstream>
#include <time.h>
/// end of logging


#include <omp.h>

namespace tflite {
namespace tensor_utils {

void CustomMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {
  

  // timespec start, finish;
  // clock_gettime(CLOCK_MONOTONIC, &start);

  omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(8);

    float* result_in_batch = result;
      for (int b = 0; b < n_batch; b++) {
          const float* matrix_ptr = matrix;
          #pragma omp for schedule(static)
          for (int r = 0; r < m_rows; r++) {
            for (int c = 0; c < m_cols; c++) {
              *result_in_batch += *matrix_ptr++ * *(vector + b * m_cols + c);
            }
            result_in_batch += result_stride;
          }
      }

    omp_set_num_threads(8);
    omp_set_dynamic(1);     // Explicitly disable dynamic teams
    // renderscript
      

  // if (result_stride == 1){
  //   androidrs::matmul::rsMatmul_sgemm(static_cast<void*>(const_cast<float*>(matrix)), 0,
  //     static_cast<void*>(const_cast<float*>(vector)), 0, 
  //     static_cast<void*>(const_cast<float*>(result)), m_rows, n_batch, m_cols, 1, 0);
  // }
  // else {
  //   PortableMatrixBatchVectorMultiplyAccumulate(matrix,
  //                                                m_rows, m_cols,
  //                                                vector,
  //                                                n_batch, result,
  //                                                result_stride);
  // }

  // clock_gettime(CLOCK_MONOTONIC, &finish);
  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
  
  // __android_log_print(ANDROID_LOG_INFO, "LOG_OPS", " CustomMatrixBatchVector %d x %d x %d , consume time : %f sec", m_rows, m_cols, n_batch, delta_time );
  
}

void CustomVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  // timespec start, finish;
  // clock_gettime(CLOCK_MONOTONIC, &start);

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

  // clock_gettime(CLOCK_MONOTONIC, &finish);
  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

  // __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", "BlockVectorBatchVectorCwiseProductAccumulate : %d : %d : %f sec\n", 16, v_size, delta_time);

}

void CustomMatrixMatrixMultiply(const float* matrix,
                                   int m_rows, int m_cols,
                                   const float* matrix2,
                                   int m2_cols, float* result) {

  // timespec start, finish;
  // clock_gettime(CLOCK_MONOTONIC, &start);


  for (int offset_i = 0; offset_i < m_rows; offset_i += 2 ){
    #pragma omp for schedule(static)
    for (int offset_j = 0; offset_j < m2_cols; offset_j += 2 ){

      float mat1[4], mat2[4];
      float restemp[4] = {0};
      
      for (int k = 0; k < m_cols; k+=2){

        for (int ii = 0; ii < 2; ii++){
          for (int jj = 0; jj < 2; jj++){
            int iii = offset_i + ii;
            int jjj = k + jj;
            mat1[ii * 2 + jj] = (iii < m_rows && jjj < m_cols) ? *(matrix + iii * m_cols + jjj) : 0;
          }
        }

        for (int ii = 0; ii < 2; ii++){
          for (int jj = 0; jj < 2; jj++){
            int iii = k + ii;
            int jjj = offset_j + jj;
            mat2[ii * 2 + jj] = (iii < m_cols && jjj < m2_cols) ? *(matrix2 + iii * m2_cols + jjj) : 0;
          }
        }


        *restemp += *mat1 * *mat2
                + *(mat1 + 1) * *(mat2 + 1 * 2);
        *(restemp + 1) += *(mat1) * *(mat2 + 1)
                + *(mat1 + 1) * *(mat2 + 1 * 2 + 1);
        *(restemp + 2) += *(mat1 + 1 * 2) * *mat2
                + *(mat1 + 1 * 2 + 1) * *(mat2 + 1 * 2);
        *(restemp + 3) += *(mat1 + 1 * 2) * *(mat2 + 1)
                + *(mat1 + 1 * 2 + 1) * *(mat2 + 1 * 2 + 1);

      }
      

      for (int ii = 0; ii < 2; ii++){
          for (int jj = 0; jj < 2; jj++){
            int iii = offset_i + ii;
            int jjj = offset_j + jj;
            if (iii < m_rows && jjj < m2_cols) 
              *(result + iii * m2_cols + jjj) = *(restemp + ii * 2 + jj);
          }
        }

    }
  }

  // clock_gettime(CLOCK_MONOTONIC, &finish);
  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
  
  // printf(" BlockMatrixMatrix  : 2 : %d x %d x %d : %f sec\n", m_rows, m_cols, m2_cols, delta_time );
}



}  // namespace tensor_utils
}  // namespace tflite
