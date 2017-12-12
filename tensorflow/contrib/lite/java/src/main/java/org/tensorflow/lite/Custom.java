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

package org.tensorflow.lite;

public class Custom {

  public Custom() {
    try {
      System.loadLibrary("tensorflowlite_jni");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("TensorFlowLite: failed to load native library: " + e.getMessage());
    }
  }

  public native void copyVectorTest();
  public native void copyMatrixTest();
  public native void multiplicationMatrixTest();
  public native void matrixVectorTest();
  
  public native void matrixVectorNaiveTest();
  public native void matrixVectorBlock2Test();
  public native void matrixVectorBlock4Test();
  public native void matrixVectorBlock8Test();

  public native void naiveDotVectorTest();
  public native void block8DotVectorTest();
  public native void block16DotVectorTest();
  public native void block32DotVectorTest();

}
