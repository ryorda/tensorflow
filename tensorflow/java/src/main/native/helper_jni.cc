/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/java/src/main/native/helper_jni.h"
#include "tensorflow/c/c_api.h"

#ifdef TENSORFLOW_USE_SYCL
#define SYCL_AVAILABLE 1
#else
#define TENSORFLOW_USE_SYCL 0
#define SYCL_AVAILABLE 0
#endif // TENSORFLOW_USE_SYCL

#ifdef GOOGLE_CUDA
#define CUDA_AVAILABLE 1
#else
#define GOOGLE_CUDA 0
#define CUDA_AVAILABLE 0
#endif // GOOGLE_CUDA

#include <limits>
#include <string.h>
#include <memory>
#include <cstdio>
#include <string>
#include <vector>

#include "tensorflow/java/src/main/native/exception_jni.h"

TF_Session* requireHandle(JNIEnv* env, jlong handle) {
  static_assert(sizeof(jlong) >= sizeof(TF_Session*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(env, kNullPointerException,
                   "close() has been called on the Session");
    return nullptr;
  }
  return reinterpret_cast<TF_Session*>(handle);
}

JNIEXPORT jboolean JNICALL Java_org_tensorflow_Helper_isCudaAvailable(JNIEnv* env,
                                                                 jclass clazz) {
  return CUDA_AVAILABLE;
}

JNIEXPORT jboolean JNICALL Java_org_tensorflow_Helper_isSyclAvailable(JNIEnv* env,
                                                                 jclass clazz) {
  return SYCL_AVAILABLE;
}

JNIEXPORT jboolean JNICALL Java_org_tensorflow_Helper_isCudaEnabled(JNIEnv* env,
                                                                 jclass clazz) {
  return GOOGLE_CUDA;
}

JNIEXPORT jboolean JNICALL Java_org_tensorflow_Helper_isSyclEnabled(JNIEnv* env,
                                                                 jclass clazz) {
  return TENSORFLOW_USE_SYCL;
}

JNIEXPORT jstring JNICALL Java_org_tensorflow_Helper_listDevices(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  

  TF_Session* session = requireHandle(env, handle);
  TF_Status* status = TF_NewStatus();

  TF_DeviceList* devices = TF_SessionListDevices(session, status);

  int size = TF_DeviceListCount(devices);
  int i = 0;
  std::string ret = "";

  for (i = 0; i < size; i++){
    std::string s( TF_DeviceListName(devices, i, status) );
    if (i > 0) ret += ",";
    ret += s;
  }

  return env->NewStringUTF(ret.c_str());
}

