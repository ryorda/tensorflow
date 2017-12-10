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

// opengl support
#include <GLES2/gl2.h>
#include <glm/glm.hpp>
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4x4
#include <glm/gtc/type_ptr.hpp>
// opengl support


namespace tflite {
namespace tensor_utils {

void GLMMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {
  

  timespec start, finish;
  clock_gettime(CLOCK_MONOTONIC, &start);

  float* result_in_batch = result;
  glm::mat4x4 lsh;
  glm::vec4 rsh;
  glm::vec4 res;
  glm::vec4 res_total;

  float vec[4];
  float vec16[16];
  int row = m_rows;
  int col = m_cols;
  
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    const float* vector_in_batch = vector + b * m_cols;

    for (int i = 0; i < row; i+= 4){
      res_total = glm::vec4(0, 0, 0, 0);
      
      for (int j = 0; j < col; j+=4){

        const float* start_l0, *start_l1, *start_l2, *start_l3;
        const float* start_r = vector_in_batch + j;

        *vec = (j < col) ? *(start_r) : 0;
        *(vec + 1) = (j + 1 < col) ? *(start_r + 1) : 0;
        *(vec + 2) = (j + 2 < col) ? *(start_r + 2) : 0;
        *(vec + 3) = (j + 3 < col) ? *(start_r + 3) : 0;

        rsh = glm::vec4( *(vec), *(vec+1), *(vec+2), *(vec+3)); 

        if ( i + 3 < row && j + 3 < col ){
          start_l0 = matrix_ptr + j + i * col;
          start_l1 = matrix_ptr + j + (i + 1) * col;
          start_l2 = matrix_ptr + j + (i + 2) * col;
          start_l3 = matrix_ptr + j + (i + 3) * col;
        }
        else {

          for (int ii = 0; ii < 4; ii++)
            for (int jj = 0; jj < 4; jj++){
              int iii = i + ii;
              int jjj = j + jj;
              if (iii < row && jjj < col)
                *(vec16 + ii * 4 + jj) = *( matrix_ptr + jjj + (iii * col));
              else 
                *(vec16 + ii * 4 + jj) = 0;
            }


          start_l0 = vec16;
          start_l1 = vec16 + 4;
          start_l2 = vec16 + 8;
          start_l3 = vec16 + 12;
        }


        lsh = glm::mat4x4(
            *(start_l0), *(start_l1), *(start_l2), *(start_l3), 
            *(start_l0+1), *(start_l1+1), *(start_l2+1), *(start_l3+1), 
            *(start_l0+2), *(start_l1+2), *(start_l2+2), *(start_l3+2), 
            *(start_l0+3), *(start_l1+3), *(start_l2+3), *(start_l3+3)
          );

        /**
        * WARNING: glm multiply matrix and vector as follow
        * c = A' * b
        * Don't forget to transpose if you need to multiply as usual
        */
        res = lsh*rsh;
        res_total = res_total + res;
      }

      const float *pSource = (const float*)glm::value_ptr(res_total);

      if (i < row)
        *(result_in_batch) = *pSource;
      if (i + 1 < row)
        *(result_in_batch + 1 * result_stride) = *(pSource + 1);
      if (i + 2 < row)
        *(result_in_batch + 2 * result_stride) = *(pSource + 2);
      if (i + 3 < row)
        *(result_in_batch + 3 * result_stride) = *(pSource + 3);

      result_in_batch += 4 * result_stride;
    }

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
  
  __android_log_print(ANDROID_LOG_INFO, "LOG_OPS", " GLMMatrixBatchVector %d x %d x %d , consume time : %f sec", m_rows, m_cols, n_batch, delta_time );
  
}


}  // namespace tensor_utils
}  // namespace tflite
