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

package org.tensorflow;

/** Unchecked exception thrown when executing TensorFlow Graphs. */
public final class Helper {
  
  long handle;

  public Helper(long handle){
  	this.handle = handle;
  }

  public String[] getAvailableDevices(){
 		 String s = listDevices(handle);
 		 
 		 return s.split(",");
  }

  public String[] getEnabledDevices(){
 		 String s = "";
 		 
 		 if (isCudaEnabled()) s += "CUDA ";	
 		 if (isSyclEnabled()) s += "SYCL ";

 		 return s.split(" ");
  }

  private static native boolean isCudaEnabled();

  private static native boolean isSyclEnabled();

  private static native boolean isCudaAvailable();

  private static native boolean isSyclAvailable();

  private static native String listDevices(long handle);
}
