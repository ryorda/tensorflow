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

/// logging
#include <android/log.h>
#include <fstream>
#include <time.h>
/// end of logging


#include <omp.h>

namespace tflite {
namespace tensor_utils {

void RenderScriptMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {
  

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(4);

  float* result_in_batch = result;

  float vec[2];
  float vec16[4];
  int row = m_rows;
  int col = m_cols;
  
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    const float* vector_in_batch = vector + b * m_cols;


    #pragma omp for schedule(static)
    for (int i = 0; i < row; i+= 2){
      float res_total[2] = {0};
      
      for (int j = 0; j < col; j+= 2){

        const float* start_l0, *start_l1;
        const float* start_r = vector_in_batch + j;

        *vec = (j < col) ? *(start_r) : 0;
        *(vec + 1) = (j + 1 < col) ? *(start_r + 1) : 0;

        if ( i + 1 < row && j + 1 < col ){
          start_l0 = matrix_ptr + j + i * col;
          start_l1 = matrix_ptr + j + (i + 1) * col;
        }
        else {

          for (int ii = 0; ii < 2; ii++){
            int iii = ii + i;
            if (iii < row){
              *(vec16 + ii * 2 + 0) = (j < col) ? *( matrix_ptr + 0 + (iii * col)) : 0;
              *(vec16 + ii * 2 + 1) = (j + 1 < col) ? *( matrix_ptr + 1 + (iii * col)) : 0;
            }
            else {
              *(vec16 + ii * 2 + 0) = 0;
              *(vec16 + ii * 2 + 1) = 0;
            }
          }

          start_l0 = vec16;
          start_l1 = vec16 + 4;
        }

        *res_total += *start_l0 * *vec 
                    + *(start_l0 + 1) * *(vec + 1);

        *(res_total + 1) += *start_l1 * *vec 
                    + *(start_l1 + 1) * *(vec + 1);

      }

      const float *pSource = res_total;

      if (i < row)
        *(result_in_batch) = *pSource;
      if (i + 1 < row)
        *(result_in_batch + 1 * result_stride) = *(pSource + 1);

      result_in_batch += 2 * result_stride;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
  
  __android_log_print(ANDROID_LOG_INFO, "LOG_OPS", " RenderScriptMatrixBatchVector %d x %d x %d , consume time : %f sec", m_rows, m_cols, n_batch, delta_time );
  
  omp_set_num_threads(8);
  omp_set_dynamic(1);     // Explicitly disable dynamic teams
}


}  // namespace tensor_utils
}  // namespace tflite
