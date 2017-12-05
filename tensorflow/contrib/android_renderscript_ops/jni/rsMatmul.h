//
// Created by WangYingnan on 3/12/17.
//

#ifndef RSKERNELSTEST_RSMATMUL_H
#define RSKERNELSTEST_RSMATMUL_H

#include "tensorflow/contrib/android_renderscript_ops/jni/RScommon.h"
#include <vector>

namespace androidrs {

namespace matmul {

static sp<RS> mRS = new RS();
static sp<ScriptIntrinsicBLAS> sc = nullptr; //ScriptIntrinsicBLAS::create(androidrs::matmul::mRS);
//static const char* cachePath = "/data/user/0/org.tensorflow.demo/cache";
static int tot_matmul_cnt = 2;
static int count = 0;
static sp<Allocation> last_gemv_alloc[8];
static bool last_gemv_alloc_visit[8];

sp<ScriptIntrinsicBLAS>& initSC()
{
    
    if (sc == nullptr) {
    mRS->init(kCachePath);
    sc = ScriptIntrinsicBLAS::create(androidrs::matmul::mRS);   
    }
    return sc;
}

// float
void rsMatmul_sgemm(void* a_ptr, bool a_trans, void* b_ptr, bool b_trans, void* c_ptr,
                    int m, int n, int k, float alpha, float beta)
{
    int idx = count%tot_matmul_cnt;

    if(!androidrs::matmul::mRS->getContext()){
        androidrs::matmul::mRS->init(kCachePath);
    }

    //if(count<tot_matmul_cnt){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);

        sp<const Type> a_t = Type::create(androidrs::matmul::mRS, e, k, m, 0);
        sp<const Type> b_t = Type::create(androidrs::matmul::mRS, e, n, k, 0);
        sp<const Type> c_t = Type::create(androidrs::matmul::mRS, e, n, m, 0);

        sp<Allocation> a_alloc = Allocation::createTyped(androidrs::matmul::mRS, a_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        sp<Allocation> b_alloc = Allocation::createTyped(androidrs::matmul::mRS, b_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        sp<Allocation> c_alloc = Allocation::createTyped(androidrs::matmul::mRS, c_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);

        //a_alloc_vec.push_back(a_alloc);
        //b_alloc_vec.push_back(b_alloc);
        //c_alloc_vec.push_back(c_alloc);
    // }

    //a_alloc_vec[idx]->copy2DRangeFrom(0, 0, k, m, a_ptr);
    //b_alloc_vec[idx]->copy2DRangeFrom(0, 0, n, k, b_ptr);
    a_alloc->copy2DRangeFrom(0, 0, k, m, a_ptr);
    b_alloc->copy2DRangeFrom(0, 0, n, k, b_ptr);

    RsBlasTranspose a_transpose = a_trans ? RsBlasTranspose::RsBlasTrans : RsBlasTranspose::RsBlasNoTrans;
    RsBlasTranspose b_transpose = b_trans ? RsBlasTranspose::RsBlasTrans : RsBlasTranspose::RsBlasNoTrans;

    sp<ScriptIntrinsicBLAS> script = initSC();

    timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // script->SGEMM(a_transpose, b_transpose, alpha, a_alloc_vec[idx], b_alloc_vec[idx], beta, c_alloc_vec[idx]);
    script->SGEMM(a_transpose, b_transpose, alpha, a_alloc, b_alloc, beta, c_alloc);

    // c_alloc_vec[idx]->copy2DRangeTo(0, 0, n, m, c_ptr);
    c_alloc->copy2DRangeTo(0, 0, n, m, c_ptr);
    //count++;

    clock_gettime(CLOCK_MONOTONIC, &finish);
    float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);

    __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " RS GEMM %dx%dx%d , consume time : %f sec", m, k, n, delta_time );
    
};

// float
void rsMatmul_sgemv(void* x_ptr, int inc_x, void* a_ptr, void* y_ptr, 
                    int inc_y, int m, int k, float alpha, float beta, int code)
{
    int idx = count%tot_matmul_cnt;

    if(!androidrs::matmul::mRS->getContext()){
        androidrs::matmul::mRS->init(kCachePath);
    }

    //if(count<tot_matmul_cnt){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);

        sp<const Type> x_t = Type::create(androidrs::matmul::mRS, e, k, 1, 0);
        sp<const Type> a_t = Type::create(androidrs::matmul::mRS, e, m, k, 0);
        sp<const Type> y_t = Type::create(androidrs::matmul::mRS, e, m, 1, 0);

        sp<Allocation> a_alloc = Allocation::createTyped(androidrs::matmul::mRS, a_t, RS_ALLOCATION_USAGE_SCRIPT);
        sp<Allocation> x_alloc = Allocation::createTyped(androidrs::matmul::mRS, x_t, RS_ALLOCATION_USAGE_SCRIPT);
        sp<Allocation> y_alloc = Allocation::createTyped(androidrs::matmul::mRS, y_t, RS_ALLOCATION_USAGE_SCRIPT);

        //a_alloc_vec.push_back(a_alloc);
        //b_alloc_vec.push_back(b_alloc);
        //c_alloc_vec.push_back(c_alloc);
    // }

    //a_alloc_vec[idx]->copy2DRangeFrom(0, 0, k, m, a_ptr);
    //b_alloc_vec[idx]->copy2DRangeFrom(0, 0, n, k, b_ptr);

    timespec start, finish, finish2, finish3;
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (code > 0 && code < 8 && last_gemv_alloc_visit[code])
        a_alloc = last_gemv_alloc[code];
    else {
        a_alloc->copy2DRangeFrom(0, 0, k, m, a_ptr);
        last_gemv_alloc[code] = a_alloc;
        last_gemv_alloc_visit[code] = true;
    }
    

    x_alloc->copy1DRangeFrom(0, k, x_ptr);

    clock_gettime(CLOCK_MONOTONIC, &finish);


    sp<ScriptIntrinsicBLAS> script = initSC();
    // script->SGEMM(a_transpose, b_transpose, alpha, a_alloc, b_alloc, beta, c_alloc);
    script->SGEMM(RsBlasTranspose::RsBlasNoTrans, RsBlasTranspose::RsBlasNoTrans, alpha, x_alloc, a_alloc, beta, y_alloc);
    // script->SGEMV(a_transpose, alpha, a_alloc, x_alloc, inc_x, beta, y_alloc, inc_y);

    clock_gettime(CLOCK_MONOTONIC, &finish2);

    // c_alloc_vec[idx]->copy2DRangeTo(0, 0, n, m, c_ptr);
    y_alloc->copy1DRangeTo(0, m, y_ptr);
    //count++;

    clock_gettime(CLOCK_MONOTONIC, &finish3);

    float delta_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
    float delta_time2 = (finish2.tv_sec - finish.tv_sec) + ((float)(finish2.tv_nsec - finish.tv_nsec)/1000000000.0f);
    float delta_time3 = (finish3.tv_sec - start.tv_sec) + ((float)(finish3.tv_nsec - start.tv_nsec)/1000000000.0f);

    __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " RS GEMV %dx%dx1 : copyTo , consume time : %f sec", m, k, delta_time );
    __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " RS GEMV %dx%dx1 : compute, consume time : %f sec", m, k, delta_time2 );
    __android_log_print(ANDROID_LOG_INFO, "LOG_TEST", " RS GEMV %dx%dx1 : overall , consume time : %f sec", m, k, delta_time3 );
};


}
}

#endif //RSKERNELSTEST_RSMATMUL_H
