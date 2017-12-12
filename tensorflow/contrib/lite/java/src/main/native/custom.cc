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

#include <stdio.h>

#include "tensorflow/contrib/lite/java/src/main/native/custom.h"

/// logging
#include <android/log.h>
#include <fstream>
#include <time.h>
/// end of logging

#include "tensorflow/contrib/android_renderscript_ops/jni/RScommon.h"
#include "tensorflow/contrib/android_renderscript_ops/jni/rsMatmul.h"

#include <vector>

static sp<RS> mRS = new RS();

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_copyVectorTest(JNIEnv* env, jclass /*clazz*/) {
	
	if(!mRS->getContext()){
        mRS->init(kCachePath);
    }

	for (auto size : {2, 4, 8, 16, 32, 64, 128, 256, 512,1024, 2048, 4096}){

		std::vector<float> vec;

		for (float i = 0; i < size; i++)
			vec.push_back(255);

		// allocation time
		// timespec start, finish;
		// clock_gettime(CLOCK_MONOTONIC, &start);
		
		sp<const Element> e = Element::F32(mRS);

        sp<const Type> t = Type::create(mRS, e, size, 1, 0);

        sp<Allocation> alloc = Allocation::createTyped(mRS, t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);


		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyVector %d : allocation , consume time : %f sec", size, delta_time );


		// copy time

		// clock_gettime(CLOCK_MONOTONIC, &start);

		alloc->copy1DRangeFrom(0, size, &vec[0]);

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyVector %d : copy , consume time : %f sec", size, delta_time );
	}
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_copyMatrixTest(JNIEnv* env, jclass /*clazz*/) {
	

	if(!mRS->getContext()){
        mRS->init(kCachePath);
    }

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}){

		float matrix[size][size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i][j] = 255;
		
		// allocation time
		// timespec start, finish;
		// clock_gettime(CLOCK_MONOTONIC, &start);
		
		sp<const Element> e = Element::F32(mRS);

        sp<const Type> t = Type::create(mRS, e, size, size, 0);

        sp<Allocation> alloc = Allocation::createTyped(mRS, t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);


		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyMatrix 2D %d : allocation , consume time : %f sec", size, delta_time );


		// copy time

		// clock_gettime(CLOCK_MONOTONIC, &start);

		alloc->copy2DRangeFrom(0, 0, size, size, matrix);

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyMatrix 2D %d : copy , consume time : %f sec", size, delta_time );

	}  

	// flattened matrix
	if(!mRS->getContext()){
        mRS->init(kCachePath);
    }

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}){

		float matrix[size * size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255;
		
		// allocation time
		// timespec start, finish;
		// clock_gettime(CLOCK_MONOTONIC, &start);
		
		sp<const Element> e = Element::F32(mRS);

        sp<const Type> t = Type::create(mRS, e, size * size, 1, 0);

        sp<Allocation> alloc = Allocation::createTyped(mRS, t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);


		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyMatrix 1D %d : allocation , consume time : %f sec", size, delta_time );


		// copy time

		// clock_gettime(CLOCK_MONOTONIC, &start);

		alloc->copy1DRangeFrom(0, size * size, matrix);

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyMatrix 1D %d : copy , consume time : %f sec", size, delta_time );

	} 
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_multiplicationMatrixTest(JNIEnv* env, jclass /*clazz*/) {
	

	if(!mRS->getContext()){
        mRS->init(kCachePath);
    }

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512}){

		float matrix[size * size], matrix2[size * size], result[size * size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255, matrix2[i * size + j] = 255;
		
		// allocation time
		// timespec start, finish;
		// clock_gettime(CLOCK_MONOTONIC, &start);

		androidrs::matmul::rsMatmul_sgemm(static_cast<void*>(const_cast<float*>(matrix)), 0,
			static_cast<void*>(const_cast<float*>(matrix2)), 0, 
			static_cast<void*>(const_cast<float*>(result)), size, size, size, 1, 0);
	

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " multiplicationMatrix 2D %d : allocation , consume time : %f sec", size, delta_time );

	}  

}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorTest(JNIEnv* env, jclass /*clazz*/) {
	

	if(!mRS->getContext()){
        mRS->init(kCachePath);
    }

	for (int size : {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}){

		float matrix[size * size], matrix2[size], result[size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255, matrix2[i] = 255;
		
		// allocation time
		// timespec start, finish;
		// clock_gettime(CLOCK_MONOTONIC, &start);



		androidrs::matmul::rsMatmul_sgemv(
			static_cast<void*>(const_cast<float*>(matrix2)), 1, 
			static_cast<void*>(const_cast<float*>(matrix)), 
			static_cast<void*>(const_cast<float*>(result)), 1, 
            size, size, 1, 0, 0
            );

		// androidrs::matmul::rsMatmul_sgemm(static_cast<void*>(const_cast<float*>(matrix)), 0,
		// 	static_cast<void*>(const_cast<float*>(matrix2)), 0, 
		// 	static_cast<void*>(const_cast<float*>(result)), size, 1, size, 1, 0);


		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " RSMatrixVector 2D %d , consume time : %f sec", size, delta_time );

	}  

	// test cache
	for (int i = 0; i < 4	; i++){

		int size = 1024;


		float matrix[size * size], matrix2[size], result[size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255, matrix2[i] = 255;
		
		// allocation time
		// timespec start, finish;
		// clock_gettime(CLOCK_MONOTONIC, &start);


		androidrs::matmul::rsMatmul_sgemv(
			static_cast<void*>(const_cast<float*>(matrix2)), 1, 
			static_cast<void*>(const_cast<float*>(matrix)), 
			static_cast<void*>(const_cast<float*>(result)), 1, 
            size, size, 1, 0, 1
            );

		// androidrs::matmul::rsMatmul_sgemm(static_cast<void*>(const_cast<float*>(matrix)), 0,
		// 	static_cast<void*>(const_cast<float*>(matrix2)), 0, 
		// 	static_cast<void*>(const_cast<float*>(result)), size, 1, size, 1, 0);


		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CachedRSMatrixVector 2D %d , consume time : %f sec", size, delta_time );

	}  

}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorNaiveTest(JNIEnv* env, jclass /*clazz*/) {
	
	for (int size : {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}){

		// float matrix[size * size], matrix2[size], result[size];

		float *matrix = (float *)malloc(size * size * sizeof(float *));
		float *matrix2 = (float *)malloc(size * sizeof(float *));
		float *result = (float *)malloc(size * sizeof(float *));

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 1, matrix2[i] = 1;


		int result_stride = 1;
		int n_batch = 1;
		int m_cols = size;
		int m_rows = size;
		// allocation time
		// timespec start, finish;
		// clock_gettime(CLOCK_MONOTONIC, &start);


		float* result_in_batch = result;
		  for (int b = 0; b < n_batch; b++) {
		    const float* matrix_ptr = matrix;
		    for (int r = 0; r < m_rows; r++) {
		      const float* vector_in_batch = matrix2 + b * m_cols;
		      for (int c = 0; c < m_cols; c++) {
		        *result_in_batch += *matrix_ptr++ * *vector_in_batch++;
		      }
		      result_in_batch += result_stride;
		    }
		  }

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		bool isRight = true;
		for (int i = 0; i < size; i++){
			isRight = isRight && result[i] == size;
		}
		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " NaiveMatrixVector 1D %d : %s , consume time : %f sec", size, (isRight ? "true" : "false"), delta_time );

		free(matrix);
		free(matrix2);
		free(result);
	}


	return;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorBlock2Test(JNIEnv* env, jclass /*clazz*/) {
	
	for (int size : {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}){

		// float matrix[size * size], vector[size], result[size];

		float *matrix = (float *)malloc(size * size * sizeof(float *));
		float *vector = (float *)malloc(size * sizeof(float *));
		float *result = (float *)malloc(size * sizeof(float *));

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 1, vector[i] = 1;


		int result_stride = 1;
		int n_batch = 1;
		int m_cols = size;
		int m_rows = size;
		
		// allocation time
		
		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

		  float* result_in_batch = result;

		  float vec[2];
		  float vec16[4];
		  int row = m_rows;
		  int col = m_cols;
		  
		  for (int b = 0; b < n_batch; b++) {
		    const float* matrix_ptr = matrix;
		    const float* vector_in_batch = vector + b * m_cols;

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

		  // clock_gettime(CLOCK_MONOTONIC, &finish);
		  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
		  
		  

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " BlockMatrixVector 2 : %d , consume time : %f sec", size, delta_time );

		free(matrix);
		free(vector);
		free(result);
	}


	return;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorBlock4Test(JNIEnv* env, jclass /*clazz*/) {
	
	for (int size : {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}){

		// float matrix[size * size], vector[size], result[size];

		float *matrix = (float *)malloc(size * size * sizeof(float *));
		float *vector = (float *)malloc(size * sizeof(float *));
		float *result = (float *)malloc(size * sizeof(float *));

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 1, vector[i] = 1;


		int result_stride = 1;
		int n_batch = 1;
		int m_cols = size;
		int m_rows = size;
		
		// allocation time
		
		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

		  float* result_in_batch = result;

		  float vec[4];
		  float vec16[16];
		  int row = m_rows;
		  int col = m_cols;
		  
		  for (int b = 0; b < n_batch; b++) {
		    const float* matrix_ptr = matrix;
		    const float* vector_in_batch = vector + b * m_cols;

		    for (int i = 0; i < row; i+= 4){
		      float res_total[4] = {0};
		      
		      for (int j = 0; j < col; j+=4){

		        const float* start_l0, *start_l1, *start_l2, *start_l3;
		        const float* start_r = vector_in_batch + j;

		        *vec = (j < col) ? *(start_r) : 0;
		        *(vec + 1) = (j + 1 < col) ? *(start_r + 1) : 0;
		        *(vec + 2) = (j + 2 < col) ? *(start_r + 2) : 0;
		        *(vec + 3) = (j + 3 < col) ? *(start_r + 3) : 0;

		        if ( i + 3 < row && j + 3 < col ){
		          start_l0 = matrix_ptr + j + i * col;
		          start_l1 = matrix_ptr + j + (i + 1) * col;
		          start_l2 = matrix_ptr + j + (i + 2) * col;
		          start_l3 = matrix_ptr + j + (i + 3) * col;
		        }
		        else {

		          for (int ii = 0; ii < 4; ii++){
		            int iii = ii + i;
		            if (iii < row){
		              *(vec16 + ii * 4 + 0) = (j < col) ? *( matrix_ptr + 0 + (iii * col)) : 0;
		              *(vec16 + ii * 4 + 1) = (j + 1 < col) ? *( matrix_ptr + 1 + (iii * col)) : 0;
		              *(vec16 + ii * 4 + 2) = (j + 2 < col) ? *( matrix_ptr + 2 + (iii * col)) : 0;
		              *(vec16 + ii * 4 + 3) = (j + 3 < col) ? *( matrix_ptr + 3 + (iii * col)) : 0;
		            }
		            else {
		              *(vec16 + ii * 4 + 0) = *( matrix_ptr + j + (iii * col));
		              *(vec16 + ii * 4 + 1) = *( matrix_ptr + j + 1 + (iii * col));
		              *(vec16 + ii * 4 + 2) = *( matrix_ptr + j + 2 + (iii * col));
		              *(vec16 + ii * 4 + 3) = *( matrix_ptr + j + 3 + (iii * col));
		            }
		          }

		          start_l0 = vec16;
		          start_l1 = vec16 + 4;
		          start_l2 = vec16 + 8;
		          start_l3 = vec16 + 12;
		        }


		        *res_total += *start_l0 * *vec 
		                    + *(start_l0 + 1) * *(vec + 1)
		                    + *(start_l0 + 2) * *(vec + 2)
		                    + *(start_l0 + 3) * *(vec + 3);

		        *(res_total + 1) += *start_l1 * *vec 
		                    + *(start_l1 + 1) * *(vec + 1)
		                    + *(start_l1 + 2) * *(vec + 2)
		                    + *(start_l1 + 3) * *(vec + 3);

		        *(res_total + 2) += *start_l2 * *vec 
		                    + *(start_l2 + 1) * *(vec + 1)
		                    + *(start_l2 + 2) * *(vec + 2)
		                    + *(start_l2 + 3) * *(vec + 3);

		        *(res_total + 3) += *start_l3 * *vec 
		                    + *(start_l3 + 1) * *(vec + 1)
		                    + *(start_l3 + 2) * *(vec + 2)
		                    + *(start_l3 + 3) * *(vec + 3);
		      }

		      const float *pSource = res_total;

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

		  // clock_gettime(CLOCK_MONOTONIC, &finish);
		  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
		  

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " BlockMatrixVector 4 : %d , consume time : %f sec", size, delta_time );

		free(matrix);
		free(vector);
		free(result);
	}


	return;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorBlock8Test(JNIEnv* env, jclass /*clazz*/) {
	
	for (int size : {8, 16, 32, 64, 128, 256, 512, 1024, 2048}){

		// float matrix[size * size], vector[size], result[size];

		float *matrix = (float *)malloc(size * size * sizeof(float *));
		float *vector = (float *)malloc(size * sizeof(float *));
		float *result = (float *)malloc(size * sizeof(float *));

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 1, vector[i] = 1;


		int result_stride = 1;
		int n_batch = 1;
		int m_cols = size;
		int m_rows = size;
		
		// allocation time
		
		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

		  float* result_in_batch = result;


		      float vec[8];
		      float vec16[8*8];
		      int row = m_rows;
		      int col = m_cols;
		      
		      for (int b = 0; b < n_batch; b++) {
		        const float* matrix_ptr = matrix;
		        const float* vector_in_batch = vector + b * m_cols;

		        for (int i = 0; i < row; i+= 8){
		          float res_total[8] = {0};
		          
		          for (int j = 0; j < col; j+= 8){

		            const float* start_l0, *start_l1, *start_l2, *start_l3,
		              *start_l4, *start_l5, *start_l6, *start_l7;
		            const float* start_r = vector_in_batch + j;

		            *vec = (j < col) ? *(start_r) : 0;
		            *(vec + 1) = (j + 1 < col) ? *(start_r + 1) : 0;
		            *(vec + 2) = (j + 2 < col) ? *(start_r + 2) : 0;
		            *(vec + 3) = (j + 3 < col) ? *(start_r + 3) : 0;
		            *(vec + 4) = (j + 4 < col) ? *(start_r + 4) : 0;
		            *(vec + 5) = (j + 5 < col) ? *(start_r + 5) : 0;
		            *(vec + 6) = (j + 6 < col) ? *(start_r + 6) : 0;
		            *(vec + 7) = (j + 7 < col) ? *(start_r + 7) : 0;

		            if ( i + 7 < row && j + 7 < col ){
		              start_l0 = matrix_ptr + j + i * col;
		              start_l1 = matrix_ptr + j + (i + 1) * col;
		              start_l2 = matrix_ptr + j + (i + 2) * col;
		              start_l3 = matrix_ptr + j + (i + 3) * col;
		              start_l4 = matrix_ptr + j + (i + 4) * col;
		              start_l5 = matrix_ptr + j + (i + 5) * col;
		              start_l6 = matrix_ptr + j + (i + 6) * col;
		              start_l7 = matrix_ptr + j + (i + 7) * col;
		            }
		            else {

		              for (int ii = 0; ii < 8; ii++){
		                int iii = ii + i;
		                if (iii < row){
		                  *(vec16 + ii * 4 + 0) = (j < col) ? *( matrix_ptr + 0 + (iii * col)) : 0;
		                  *(vec16 + ii * 4 + 1) = (j + 1 < col) ? *( matrix_ptr + 1 + (iii * col)) : 0;
		                  *(vec16 + ii * 4 + 2) = (j + 2 < col) ? *( matrix_ptr + 2 + (iii * col)) : 0;
		                  *(vec16 + ii * 4 + 3) = (j + 3 < col) ? *( matrix_ptr + 3 + (iii * col)) : 0;
		                  *(vec16 + ii * 4 + 4) = (j + 4 < col) ? *( matrix_ptr + 4 + (iii * col)) : 0;
		                  *(vec16 + ii * 4 + 5) = (j + 5 < col) ? *( matrix_ptr + 5 + (iii * col)) : 0;
		                  *(vec16 + ii * 4 + 6) = (j + 6 < col) ? *( matrix_ptr + 6 + (iii * col)) : 0;
		                  *(vec16 + ii * 4 + 7) = (j + 7 < col) ? *( matrix_ptr + 7 + (iii * col)) : 0;
		                }
		                else {
		                  *(vec16 + ii * 4 + 0) = *( matrix_ptr + j + (iii * col));
		                  *(vec16 + ii * 4 + 1) = *( matrix_ptr + j + 1 + (iii * col));
		                  *(vec16 + ii * 4 + 2) = *( matrix_ptr + j + 2 + (iii * col));
		                  *(vec16 + ii * 4 + 3) = *( matrix_ptr + j + 3 + (iii * col));
		                  *(vec16 + ii * 4 + 4) = *( matrix_ptr + j + 4 + (iii * col));
		                  *(vec16 + ii * 4 + 5) = *( matrix_ptr + j + 5 + (iii * col));
		                  *(vec16 + ii * 4 + 6) = *( matrix_ptr + j + 6 + (iii * col));
		                  *(vec16 + ii * 4 + 7) = *( matrix_ptr + j + 7 + (iii * col));
		                }
		              }

		              start_l0 = vec16;
		              start_l1 = vec16 + 8;
		              start_l2 = vec16 + 16;
		              start_l3 = vec16 + 24;
		              start_l4 = vec16 + 32;
		              start_l5 = vec16 + 40;
		              start_l6 = vec16 + 48;
		              start_l7 = vec16 + 56;
		            }


		            *res_total += *start_l0 * *vec 
		                        + *(start_l0 + 1) * *(vec + 1)
		                        + *(start_l0 + 2) * *(vec + 2)
		                        + *(start_l0 + 3) * *(vec + 3)
		                        + *(start_l0 + 4) * *(vec + 4)
		                        + *(start_l0 + 5) * *(vec + 5)
		                        + *(start_l0 + 6) * *(vec + 6)
		                        + *(start_l0 + 7) * *(vec + 7);

		            *(res_total + 1) += *start_l1 * *vec 
		                        + *(start_l1 + 1) * *(vec + 1)
		                        + *(start_l1 + 2) * *(vec + 2)
		                        + *(start_l1 + 3) * *(vec + 3)
		                        + *(start_l1 + 4) * *(vec + 4)
		                        + *(start_l1 + 5) * *(vec + 5)
		                        + *(start_l1 + 6) * *(vec + 6)
		                        + *(start_l1 + 7) * *(vec + 7);

		            *(res_total + 2) += *start_l2 * *vec 
		                        + *(start_l2 + 1) * *(vec + 1)
		                        + *(start_l2 + 2) * *(vec + 2)
		                        + *(start_l2 + 3) * *(vec + 3)
		                        + *(start_l2 + 4) * *(vec + 4)
		                        + *(start_l2 + 5) * *(vec + 5)
		                        + *(start_l2 + 6) * *(vec + 6)
		                        + *(start_l2 + 7) * *(vec + 7);

		            *(res_total + 3) += *start_l3 * *vec 
		                        + *(start_l3 + 1) * *(vec + 1)
		                        + *(start_l3 + 2) * *(vec + 2)
		                        + *(start_l3 + 3) * *(vec + 3)
		                        + *(start_l3 + 4) * *(vec + 4)
		                        + *(start_l3 + 5) * *(vec + 5)
		                        + *(start_l3 + 6) * *(vec + 6)
		                        + *(start_l3 + 7) * *(vec + 7);

		            *(res_total + 4) += *start_l4 * *vec 
		                        + *(start_l4 + 1) * *(vec + 1)
		                        + *(start_l4 + 2) * *(vec + 2)
		                        + *(start_l4 + 3) * *(vec + 3)
		                        + *(start_l4 + 4) * *(vec + 4)
		                        + *(start_l4 + 5) * *(vec + 5)
		                        + *(start_l4 + 6) * *(vec + 6)
		                        + *(start_l4 + 7) * *(vec + 7);

		            *(res_total + 5) += *start_l5 * *vec 
		                        + *(start_l5 + 1) * *(vec + 1)
		                        + *(start_l5 + 2) * *(vec + 2)
		                        + *(start_l5 + 3) * *(vec + 3)
		                        + *(start_l5 + 4) * *(vec + 4)
		                        + *(start_l5 + 5) * *(vec + 5)
		                        + *(start_l5 + 6) * *(vec + 6)
		                        + *(start_l5 + 7) * *(vec + 7);

		            *(res_total + 6) += *start_l6 * *vec 
		                        + *(start_l6 + 1) * *(vec + 1)
		                        + *(start_l6 + 2) * *(vec + 2)
		                        + *(start_l6 + 3) * *(vec + 3)
		                        + *(start_l6 + 4) * *(vec + 4)
		                        + *(start_l6 + 5) * *(vec + 5)
		                        + *(start_l6 + 6) * *(vec + 6)
		                        + *(start_l6 + 7) * *(vec + 7);

		            *(res_total + 7) += *start_l7 * *vec 
		                        + *(start_l7 + 1) * *(vec + 1)
		                        + *(start_l7 + 2) * *(vec + 2)
		                        + *(start_l7 + 3) * *(vec + 3)
		                        + *(start_l7 + 4) * *(vec + 4)
		                        + *(start_l7 + 5) * *(vec + 5)
		                        + *(start_l7 + 6) * *(vec + 6)
		                        + *(start_l7 + 7) * *(vec + 7);
		          }

		          const float *pSource = res_total;

		          if (i < row)
		            *(result_in_batch) = *pSource;
		          if (i + 1 < row)
		            *(result_in_batch + 1 * result_stride) = *(pSource + 1);
		          if (i + 2 < row)
		            *(result_in_batch + 2 * result_stride) = *(pSource + 2);
		          if (i + 3 < row)
		            *(result_in_batch + 3 * result_stride) = *(pSource + 3);
		          if (i + 4 < row)
		            *(result_in_batch + 4 * result_stride) = *(pSource + 4);
		          if (i + 5 < row)
		            *(result_in_batch + 5 * result_stride) = *(pSource + 5);
		          if (i + 6 < row)
		            *(result_in_batch + 6 * result_stride) = *(pSource + 6);
		          if (i + 7 < row)
		            *(result_in_batch + 7 * result_stride) = *(pSource + 7);

		          result_in_batch += 8 * result_stride;
		        }

		      }

		  // clock_gettime(CLOCK_MONOTONIC, &finish);
		  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
		  
		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " BlockMatrixVector 8 : %d , consume time : %f sec", size, delta_time );

		free(matrix);
		free(vector);
		free(result);
	}


	return;
}


JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_naiveDotVectorTest(JNIEnv* env, jclass /*clazz*/) {
	
	for (int size : {8, 16, 32, 64, 128, 256}){

		float *vec = (float *)malloc(size * size * sizeof(float *));
		float *vec2 = (float *)malloc(size * size * sizeof(float *));

		for (int i = 0; i < size * size; i++)
			for (int j = 0; j < size * size; j++)
				vec[i * size + j] = 1, vec2[i * size + j] = 1;


		int v_size = size * size;
		float* vector1 = vec;
		float* vector2 = vec2;
		
		// allocation time
		
		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

		  float result = 0.0;
		  for (int v = 0; v < v_size; v++) {
		    result += *vector1++ * *vector2++;
		  }
		  
		  // clock_gettime(CLOCK_MONOTONIC, &finish);
		  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
		  
		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " NaiveDotVector  %d , consume time : %f sec", size * size , delta_time );

		free(vec);
		free(vec2);
	}


	return;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_block8DotVectorTest(JNIEnv* env, jclass /*clazz*/) {
	
	for (int size : {8, 16, 32, 64, 128, 256}){

		float *vect = (float *)malloc(size * size * sizeof(float *));
		float *vect2 = (float *)malloc(size * size * sizeof(float *));

		for (int i = 0; i < size * size; i++)
			for (int j = 0; j < size * size; j++)
				vect[i * size + j] = 1, vect2[i * size + j] = 1;


		int v_size = size * size;
		float* vector1 = vect;
		float* vector2 = vect2;
		
		// allocation time
		
		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

		  float result = 0.0;
		  float vec[8], vec2[8];

		  for (int v = 0; v < v_size; v+= 8) {
		    const float* vec_ptr = vector1 + v;
		    const float* vec_ptr2 = vector2 + v;

		    *vec = (v < v_size) ? *(vec_ptr) : 0;
		    *(vec + 1) = (v + 1 < v_size) ? *(vec_ptr + 1) : 0;
		    *(vec + 2) = (v + 2 < v_size) ? *(vec_ptr + 2) : 0;
		    *(vec + 3) = (v + 3 < v_size) ? *(vec_ptr + 3) : 0;
		    *(vec + 4) = (v + 4 < v_size) ? *(vec_ptr + 4) : 0;
		    *(vec + 5) = (v + 5 < v_size) ? *(vec_ptr + 5) : 0;
		    *(vec + 6) = (v + 6 < v_size) ? *(vec_ptr + 6) : 0;
		    *(vec + 7) = (v + 7 < v_size) ? *(vec_ptr + 7) : 0;

		    *vec2 = (v < v_size) ? *(vec_ptr2) : 0;
		    *(vec2 + 1) = (v + 1 < v_size) ? *(vec_ptr2 + 1) : 0;
		    *(vec2 + 2) = (v + 2 < v_size) ? *(vec_ptr2 + 2) : 0;
		    *(vec2 + 3) = (v + 3 < v_size) ? *(vec_ptr2 + 3) : 0;
		    *(vec2 + 4) = (v + 4 < v_size) ? *(vec_ptr2 + 4) : 0;
		    *(vec2 + 5) = (v + 5 < v_size) ? *(vec_ptr2 + 5) : 0;
		    *(vec2 + 6) = (v + 6 < v_size) ? *(vec_ptr2 + 6) : 0;
		    *(vec2 + 7) = (v + 7 < v_size) ? *(vec_ptr2 + 7) : 0;

		    float* vectemp = vec;
		    float* vec2temp = vec2;

		    for (int i = 0; i < 8; i++)
		      result += *vectemp++ * *vec2temp++;

		    vec_ptr += 8;
		    vec_ptr2 += 8;
		  }


		  // clock_gettime(CLOCK_MONOTONIC, &finish);
		  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
  

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " BlockDotVector 8 :  %d , consume time : %f sec", size * size , delta_time );


		free(vect);
		free(vect2);
	}


	return;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_block16DotVectorTest(JNIEnv* env, jclass /*clazz*/) {
	
	for (int size : {8, 16, 32, 64, 128, 256}){

		float *vect = (float *)malloc(size * size * sizeof(float *));
		float *vect2 = (float *)malloc(size * size * sizeof(float *));

		for (int i = 0; i < size * size; i++)
			for (int j = 0; j < size * size; j++)
				vect[i * size + j] = 1, vect2[i * size + j] = 1;


		int v_size = size * size;
		float* vector1 = vect;
		float* vector2 = vect2;
		
		// allocation time
		
		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

		  float result = 0.0;
		  float vec[16], vec2[16];

		  for (int v = 0; v < v_size; v+= 16) {
		    const float* vec_ptr = vector1 + v;
		    const float* vec_ptr2 = vector2 + v;

		    *vec = (v < v_size) ? *(vec_ptr) : 0;
		    *(vec + 1) = (v + 1 < v_size) ? *(vec_ptr + 1) : 0;
		    *(vec + 2) = (v + 2 < v_size) ? *(vec_ptr + 2) : 0;
		    *(vec + 3) = (v + 3 < v_size) ? *(vec_ptr + 3) : 0;
		    *(vec + 4) = (v + 4 < v_size) ? *(vec_ptr + 4) : 0;
		    *(vec + 5) = (v + 5 < v_size) ? *(vec_ptr + 5) : 0;
		    *(vec + 6) = (v + 6 < v_size) ? *(vec_ptr + 6) : 0;
		    *(vec + 7) = (v + 7 < v_size) ? *(vec_ptr + 7) : 0;

		    *(vec + 8) = (v + 8 < v_size) ? *(vec_ptr + 8) : 0;
		    *(vec + 9) = (v + 9 < v_size) ? *(vec_ptr + 9) : 0;
		    *(vec + 10) = (v + 10 < v_size) ? *(vec_ptr + 10) : 0;
		    *(vec + 11) = (v + 11 < v_size) ? *(vec_ptr + 11) : 0;
		    *(vec + 12) = (v + 12 < v_size) ? *(vec_ptr + 12) : 0;
		    *(vec + 13) = (v + 13 < v_size) ? *(vec_ptr + 13) : 0;
		    *(vec + 14) = (v + 14 < v_size) ? *(vec_ptr + 14) : 0;
		    *(vec + 15) = (v + 15 < v_size) ? *(vec_ptr + 15) : 0;


		    *vec2 = (v < v_size) ? *(vec_ptr2) : 0;
		    *(vec2 + 1) = (v + 1 < v_size) ? *(vec_ptr2 + 1) : 0;
		    *(vec2 + 2) = (v + 2 < v_size) ? *(vec_ptr2 + 2) : 0;
		    *(vec2 + 3) = (v + 3 < v_size) ? *(vec_ptr2 + 3) : 0;
		    *(vec2 + 4) = (v + 4 < v_size) ? *(vec_ptr2 + 4) : 0;
		    *(vec2 + 5) = (v + 5 < v_size) ? *(vec_ptr2 + 5) : 0;
		    *(vec2 + 6) = (v + 6 < v_size) ? *(vec_ptr2 + 6) : 0;
		    *(vec2 + 7) = (v + 7 < v_size) ? *(vec_ptr2 + 7) : 0;

		    *(vec2 + 8) = (v + 8 < v_size) ? *(vec_ptr2 + 8) : 0;
		    *(vec2 + 9) = (v + 9 < v_size) ? *(vec_ptr2 + 9) : 0;
		    *(vec2 + 10) = (v + 10 < v_size) ? *(vec_ptr2 + 10) : 0;
		    *(vec2 + 11) = (v + 11 < v_size) ? *(vec_ptr2 + 11) : 0;
		    *(vec2 + 12) = (v + 12 < v_size) ? *(vec_ptr2 + 12) : 0;
		    *(vec2 + 13) = (v + 13 < v_size) ? *(vec_ptr2 + 13) : 0;
		    *(vec2 + 14) = (v + 14 < v_size) ? *(vec_ptr2 + 14) : 0;
		    *(vec2 + 15) = (v + 15 < v_size) ? *(vec_ptr2 + 15) : 0;

		    float* vectemp = vec;
		    float* vec2temp = vec2;

		    for (int i = 0; i < 16; i++)
		      result += *vectemp++ * *vec2temp++;

		    vec_ptr += 16;
		    vec_ptr2 += 16;
		  }


		  // clock_gettime(CLOCK_MONOTONIC, &finish);
		  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
  

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " BlockDotVector 16 : %d , consume time : %f sec", size * size , delta_time );

		free(vect);
		free(vect2);
	}


	return;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_block32DotVectorTest(JNIEnv* env, jclass /*clazz*/) {
	
	for (int size : {8, 16, 32, 64, 128, 256}){

		float *vect = (float *)malloc(size * size * sizeof(float *));
		float *vect2 = (float *)malloc(size * size * sizeof(float *));

		for (int i = 0; i < size * size; i++)
			for (int j = 0; j < size * size; j++)
				vect[i * size + j] = 1, vect2[i * size + j] = 1;


		int v_size = size * size;
		float* vector1 = vect;
		float* vector2 = vect2;
		
		// allocation time
		
		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

		  float result = 0.0;
		  float vec[32], vec2[32];

		  for (int v = 0; v < v_size; v+= 32) {
		    const float* vec_ptr = vector1 + v;
		    const float* vec_ptr2 = vector2 + v;

		    *vec = (v < v_size) ? *(vec_ptr) : 0;
		    *(vec + 1) = (v + 1 < v_size) ? *(vec_ptr + 1) : 0;
		    *(vec + 2) = (v + 2 < v_size) ? *(vec_ptr + 2) : 0;
		    *(vec + 3) = (v + 3 < v_size) ? *(vec_ptr + 3) : 0;
		    *(vec + 4) = (v + 4 < v_size) ? *(vec_ptr + 4) : 0;
		    *(vec + 5) = (v + 5 < v_size) ? *(vec_ptr + 5) : 0;
		    *(vec + 6) = (v + 6 < v_size) ? *(vec_ptr + 6) : 0;
		    *(vec + 7) = (v + 7 < v_size) ? *(vec_ptr + 7) : 0;

		    *(vec + 8) = (v + 8 < v_size) ? *(vec_ptr + 8) : 0;
		    *(vec + 9) = (v + 9 < v_size) ? *(vec_ptr + 9) : 0;
		    *(vec + 10) = (v + 10 < v_size) ? *(vec_ptr + 10) : 0;
		    *(vec + 11) = (v + 11 < v_size) ? *(vec_ptr + 11) : 0;
		    *(vec + 12) = (v + 12 < v_size) ? *(vec_ptr + 12) : 0;
		    *(vec + 13) = (v + 13 < v_size) ? *(vec_ptr + 13) : 0;
		    *(vec + 14) = (v + 14 < v_size) ? *(vec_ptr + 14) : 0;
		    *(vec + 15) = (v + 15 < v_size) ? *(vec_ptr + 15) : 0;

		    *(vec + 16) = (v + 16 < v_size) ? *(vec_ptr + 16) : 0;
		    *(vec + 17) = (v + 17 < v_size) ? *(vec_ptr + 17) : 0;
		    *(vec + 18) = (v + 18 < v_size) ? *(vec_ptr + 18) : 0;
		    *(vec + 19) = (v + 19 < v_size) ? *(vec_ptr + 19) : 0;
		    *(vec + 20) = (v + 20 < v_size) ? *(vec_ptr + 20) : 0;
		    *(vec + 21) = (v + 21 < v_size) ? *(vec_ptr + 21) : 0;
		    *(vec + 22) = (v + 22 < v_size) ? *(vec_ptr + 22) : 0;
		    *(vec + 23) = (v + 23 < v_size) ? *(vec_ptr + 23) : 0;

		    *(vec + 24) = (v + 24 < v_size) ? *(vec_ptr + 24) : 0;
		    *(vec + 25) = (v + 25 < v_size) ? *(vec_ptr + 25) : 0;
		    *(vec + 26) = (v + 26 < v_size) ? *(vec_ptr + 26) : 0;
		    *(vec + 27) = (v + 27 < v_size) ? *(vec_ptr + 27) : 0;
		    *(vec + 28) = (v + 28 < v_size) ? *(vec_ptr + 28) : 0;
		    *(vec + 29) = (v + 29 < v_size) ? *(vec_ptr + 29) : 0;
		    *(vec + 30) = (v + 30 < v_size) ? *(vec_ptr + 30) : 0;
		    *(vec + 31) = (v + 31 < v_size) ? *(vec_ptr + 31) : 0;



		    *vec2 = (v < v_size) ? *(vec_ptr2) : 0;
		    *(vec2 + 1) = (v + 1 < v_size) ? *(vec_ptr2 + 1) : 0;
		    *(vec2 + 2) = (v + 2 < v_size) ? *(vec_ptr2 + 2) : 0;
		    *(vec2 + 3) = (v + 3 < v_size) ? *(vec_ptr2 + 3) : 0;
		    *(vec2 + 4) = (v + 4 < v_size) ? *(vec_ptr2 + 4) : 0;
		    *(vec2 + 5) = (v + 5 < v_size) ? *(vec_ptr2 + 5) : 0;
		    *(vec2 + 6) = (v + 6 < v_size) ? *(vec_ptr2 + 6) : 0;
		    *(vec2 + 7) = (v + 7 < v_size) ? *(vec_ptr2 + 7) : 0;

		    *(vec2 + 8) = (v + 8 < v_size) ? *(vec_ptr2 + 8) : 0;
		    *(vec2 + 9) = (v + 9 < v_size) ? *(vec_ptr2 + 9) : 0;
		    *(vec2 + 10) = (v + 10 < v_size) ? *(vec_ptr2 + 10) : 0;
		    *(vec2 + 11) = (v + 11 < v_size) ? *(vec_ptr2 + 11) : 0;
		    *(vec2 + 12) = (v + 12 < v_size) ? *(vec_ptr2 + 12) : 0;
		    *(vec2 + 13) = (v + 13 < v_size) ? *(vec_ptr2 + 13) : 0;
		    *(vec2 + 14) = (v + 14 < v_size) ? *(vec_ptr2 + 14) : 0;
		    *(vec2 + 15) = (v + 15 < v_size) ? *(vec_ptr2 + 15) : 0;

		    *(vec2 + 16) = (v + 16 < v_size) ? *(vec_ptr2 + 16) : 0;
		    *(vec2 + 17) = (v + 17 < v_size) ? *(vec_ptr2 + 17) : 0;
		    *(vec2 + 18) = (v + 18 < v_size) ? *(vec_ptr2 + 18) : 0;
		    *(vec2 + 19) = (v + 19 < v_size) ? *(vec_ptr2 + 19) : 0;
		    *(vec2 + 20) = (v + 20 < v_size) ? *(vec_ptr2 + 20) : 0;
		    *(vec2 + 21) = (v + 21 < v_size) ? *(vec_ptr2 + 21) : 0;
		    *(vec2 + 22) = (v + 22 < v_size) ? *(vec_ptr2 + 22) : 0;
		    *(vec2 + 23) = (v + 23 < v_size) ? *(vec_ptr2 + 23) : 0;

		    *(vec2 + 24) = (v + 24 < v_size) ? *(vec_ptr2 + 24) : 0;
		    *(vec2 + 25) = (v + 25 < v_size) ? *(vec_ptr2 + 25) : 0;
		    *(vec2 + 26) = (v + 26 < v_size) ? *(vec_ptr2 + 26) : 0;
		    *(vec2 + 27) = (v + 27 < v_size) ? *(vec_ptr2 + 27) : 0;
		    *(vec2 + 28) = (v + 28 < v_size) ? *(vec_ptr2 + 28) : 0;
		    *(vec2 + 29) = (v + 29 < v_size) ? *(vec_ptr2 + 29) : 0;
		    *(vec2 + 30) = (v + 30 < v_size) ? *(vec_ptr2 + 30) : 0;
		    *(vec2 + 31) = (v + 31 < v_size) ? *(vec_ptr2 + 31) : 0;

		    float* vectemp = vec;
		    float* vec2temp = vec2;

		    for (int i = 0; i < 32; i++)
		      result += *vectemp++ * *vec2temp++;

		    vec_ptr += 32;
		    vec_ptr2 += 32;
		  }


		  // clock_gettime(CLOCK_MONOTONIC, &finish);
		  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
		  
		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " BlockDotVector 32 :  %d , consume time : %f sec", size * size , delta_time );

		free(vect);
		free(vect2);
	}


	return;
}