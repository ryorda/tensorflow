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
#include <omp.h>

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

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyVector %d : allocation : %f sec", size, delta_time );


		// copy time

		// clock_gettime(CLOCK_MONOTONIC, &start);

		alloc->copy1DRangeFrom(0, size, &vec[0]);

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyVector %d : copy : %f sec", size, delta_time );
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

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyMatrix 2D %d : allocation : %f sec", size, delta_time );


		// copy time

		// clock_gettime(CLOCK_MONOTONIC, &start);

		alloc->copy2DRangeFrom(0, 0, size, size, matrix);

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyMatrix 2D %d : copy : %f sec", size, delta_time );

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

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyMatrix 1D %d : allocation : %f sec", size, delta_time );


		// copy time

		// clock_gettime(CLOCK_MONOTONIC, &start);

		alloc->copy1DRangeFrom(0, size * size, matrix);

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " CopyMatrix 1D %d : copy : %f sec", size, delta_time );

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

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " RSMatrixMatrixMultiplication : %d : %f sec", size, delta_time );

	}  

}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorTest(JNIEnv* env, jclass /*clazz*/) {
	

	if(!mRS->getContext()){
        mRS->init(kCachePath);
    }

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}){

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

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " RSMatrixVector : %d : %f sec", size, delta_time );

	}  

	// test cache
	// for (int i = 0; i < 4	; i++){

	// 	int size = 1024;


	// 	float matrix[size * size], matrix2[size], result[size];

	// 	for (int i = 0; i < size; i++)
	// 		for (int j = 0; j < size; j++)
	// 			matrix[i * size + j] = 255, matrix2[i] = 255;
		
	// 	// allocation time
	// 	// timespec start, finish;
	// 	// clock_gettime(CLOCK_MONOTONIC, &start);


	// 	androidrs::matmul::rsMatmul_sgemv(
	// 		static_cast<void*>(const_cast<float*>(matrix2)), 1, 
	// 		static_cast<void*>(const_cast<float*>(matrix)), 
	// 		static_cast<void*>(const_cast<float*>(result)), 1, 
 //            size, size, 1, 0, 1
 //            );

	// 	// androidrs::matmul::rsMatmul_sgemm(static_cast<void*>(const_cast<float*>(matrix)), 0,
	// 	// 	static_cast<void*>(const_cast<float*>(matrix2)), 0, 
	// 	// 	static_cast<void*>(const_cast<float*>(result)), size, 1, size, 1, 0);


	// 	// clock_gettime(CLOCK_MONOTONIC, &finish);
	// 	// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

	// 	// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " Cache RSMatrixVector : %d : %f sec", size, delta_time );

	// }  

}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorNaiveTest(JNIEnv* env, jclass /*clazz*/) {
	

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}){

		float matrix[size * size], matrix2[size], result[size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255, matrix2[i] = 255;
		
		int result_stride = 1;
		int n_batch = 1;
		int m_rows = size;
		int m_cols = size;
		float* vector = matrix2;

		// allocation time
		// timespec start, finish;
		// clock_gettime(CLOCK_MONOTONIC, &start);

		float* result_in_batch = result;
		for (int b = 0; b < 1; b++) {
			const float* matrix_ptr = matrix;
			for (int r = 0; r < size; r++) {
			  const float* vector_in_batch = matrix2 + b * size;
			  for (int c = 0; c < size; c++) {
			    *result_in_batch += *matrix_ptr++ * *vector_in_batch++;
			  }
			  result_in_batch += result_stride;
			}
		}

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " NaiveMatrixVector : %d : %f sec", size, delta_time );

	}  

}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorOpenMP2Test(JNIEnv* env, jclass /*clazz*/) {

	omp_set_dynamic(0);     // Explicitly disable dynamic teams
  	omp_set_num_threads(2);

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}){

		float matrix[size * size], matrix2[size], result[size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255, matrix2[i] = 255;
		
		int result_stride = 1;
		int n_batch = 1;
		int m_rows = size;
		int m_cols = size;
		float* vector = matrix2;

		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

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

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " OpenMPMatrixVector 2 : %d : %f sec", size, delta_time );
	}
}


JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorOpenMP4Test(JNIEnv* env, jclass /*clazz*/) {

	omp_set_dynamic(0);     // Explicitly disable dynamic teams
  	omp_set_num_threads(4);

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}){

		float matrix[size * size], matrix2[size], result[size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255, matrix2[i] = 255;
		
		int result_stride = 1;
		int n_batch = 1;
		int m_rows = size;
		int m_cols = size;
		float* vector = matrix2;

		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

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

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " OpenMPMatrixVector 4 : %d : %f sec", size, delta_time );
	}
}


JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorOpenMP8Test(JNIEnv* env, jclass /*clazz*/) {

	omp_set_dynamic(0);     // Explicitly disable dynamic teams
  	omp_set_num_threads(8);

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}){

		float matrix[size * size], matrix2[size], result[size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255, matrix2[i] = 255;
		
		int result_stride = 1;
		int n_batch = 1;
		int m_rows = size;
		int m_cols = size;
		float* vector = matrix2;

		// timespec start, finish;
		  // clock_gettime(CLOCK_MONOTONIC, &start);

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

		// clock_gettime(CLOCK_MONOTONIC, &finish);
		// float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " OpenMPMatrixVector 8 : %d : %f sec", size, delta_time );
	}
}



JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorBlock2Test(JNIEnv* env, jclass /*clazz*/) {
	

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}){

		float matrix[size * size], matrix2[size], result[size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255, matrix2[i] = 255;
		
		int result_stride = 1;
		int n_batch = 1;
		int m_rows = size;
		int m_cols = size;
		float* vector = matrix2;
		
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

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " BlockMatrixVector 2 : %d : %f sec", size, delta_time );

	}  

}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Custom_matrixVectorOpenMPBlock2Test(JNIEnv* env, jclass /*clazz*/) {
	
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
  	omp_set_num_threads(4);

	for (int size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}){

		float matrix[size * size], matrix2[size], result[size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				matrix[i * size + j] = 255, matrix2[i] = 255;
		
		int result_stride = 1;
		int n_batch = 1;
		int m_rows = size;
		int m_cols = size;
		float* vector = matrix2;

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

		  // clock_gettime(CLOCK_MONOTONIC, &finish);
		  // float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

		// __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " OpenMPBlockMatrixVector 8 2 : %d : %f sec", size, delta_time );

	}  


  	omp_set_num_threads(8);
	omp_set_dynamic(1);     // Explicitly disable dynamic teams
}