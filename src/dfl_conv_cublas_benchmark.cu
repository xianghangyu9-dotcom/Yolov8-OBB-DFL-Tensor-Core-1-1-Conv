#include "dfl_conv_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <iomanip>

using namespace std;

#define INPUT  "../input_case_v3"
#define OUTPUT "../output_case_v3"

#define BIAS_FILE "p3_bias_fp32.bin"

#define CUDACHECK(expr)                 \          
    do{                                 \
        cudaError_t error = (expr);     \
        if(error != cudaSuccess)        \
        {                               \
            cerr<<"error is "           \
            <<cudaGetErrorString(error) \
            <<" on "                    \
            <<__FILE__                  \
            <<" : "                     \
            <<__LINE__                  \
            <<endl;                     \
            return 1;                   \
        }                               \
    }while(0)     

const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CUBLAS_CHECK(expr)                                           \
    do {                                                             \
        cublasStatus_t status = (expr);                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                       \
            cerr << "cuBLAS error: "                                 \
                    << cublas_status_string(status)                  \
                    << " at " << __FILE__ << ":" << __LINE__         \
                    << endl;                                         \
            return 1;                                                \
        }                                                            \
    } while (0)

__global__
void bias_add_kernel(
    float* matrix_c,
    const float* bias,
    int m,
    int n
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < m * n) {
        int row = index / n;
        matrix_c[index] += bias[row];
    }
}

cublasStatus_t run_cublas_gemm(
    cublasHandle_t handle,
    const __half* device_a,
    const __half* device_w,
    float* device_c,
    int m,
    int n,
    int k
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    return cublasGemmEx(
        handle,

        CUBLAS_OP_N,
        CUBLAS_OP_N,

        n, 
        m, 
        k,

        &alpha,

        device_a,
        CUDA_R_16F,
        n,      

        device_w,
        CUDA_R_16F,
        k,      

        &beta,

        device_c,
        CUDA_R_32F,
        n,

        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

int main() {
    constexpr int M         = 64;
    constexpr int K         = 64;
    constexpr int height    = 80;
    constexpr int width     = 80;
    constexpr int N         = width * height;
    
    constexpr int w_cap     = M * K;
    constexpr int a_cap     = K * N;
    constexpr int bias_cap  = M;
    constexpr int c_cap     = M * N;

    vector<__half> host_w(w_cap);
    vector<__half> host_a(a_cap);
    vector<float> host_bias(bias_cap);

    if(!input_load(INPUT "/p3_w_fp16.bin",host_w)){
        return 1;
    };
    if(!input_load(INPUT "/p3_a_fp16.bin",host_a)){
        return 1;
    };
    if(!input_load(INPUT "/p3_bias_fp32.bin",host_bias)){
        return 1;
    };

    __half* device_w   = nullptr;
    __half* device_a   = nullptr;
    float* device_bias = nullptr;
    float* device_c    = nullptr;

    CUDACHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_w),
        w_cap * sizeof(__half))
    );

    CUDACHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_a),
        a_cap * sizeof(__half))
    );

    CUDACHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_bias),
        bias_cap * sizeof(float))
    );

    CUDACHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_c),
        c_cap * sizeof(float))
    );

    CUDACHECK(
        cudaMemcpy(
            device_w,
            host_w.data(),
            w_cap * sizeof(__half),
            cudaMemcpyHostToDevice
        )
    );

    CUDACHECK(
        cudaMemcpy(
            device_a,
            host_a.data(),
            a_cap * sizeof(__half),
            cudaMemcpyHostToDevice
        )
    );

    CUDACHECK(
        cudaMemcpy(
            device_bias,
            host_bias.data(),
            bias_cap * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    constexpr int WARMUP = 100;
    constexpr int ITERS = 1000;

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, 0));

    constexpr int BIAS_THREADS = 256;
    constexpr int BIAS_BLOCKS = (c_cap + BIAS_THREADS - 1) / BIAS_THREADS;

    for (int i = 0; i < WARMUP; i++) {
        CUBLAS_CHECK(run_cublas_gemm(
            handle,
            device_a,
            device_w,
            device_c,
            M,
            N,
            K
        ));

        bias_add_kernel<<<BIAS_BLOCKS, BIAS_THREADS>>>(
            device_c,
            device_bias,
            M,
            N
        );
    }

    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERS; i++) {
        CUBLAS_CHECK(run_cublas_gemm(
            handle,
            device_a,
            device_w,
            device_c,
            M,
            N,
            K
        ));
    }

    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float gemm_only_total_ms = 0.0f;

    CUDACHECK(cudaEventElapsedTime(
        &gemm_only_total_ms,
        start,
        stop
    ));

    const float gemm_only_avg_us =
        gemm_only_total_ms * 1000.0f / ITERS;

    CUDACHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERS; i++) {
        CUBLAS_CHECK(run_cublas_gemm(
            handle,
            device_a,
            device_w,
            device_c,
            M,
            N,
            K
        ));

        bias_add_kernel<<<BIAS_BLOCKS, BIAS_THREADS>>>(
            device_c,
            device_bias,
            M,
            N
        );
    }

    CUDACHECK(cudaGetLastError());

    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float gemm_bias_total_ms = 0.0f;

    CUDACHECK(cudaEventElapsedTime(
        &gemm_bias_total_ms,
        start,
        stop
    ));

    const float gemm_bias_avg_us =
        gemm_bias_total_ms * 1000.0f / ITERS;

    cout << fixed << setprecision(4);

    cout << "cuBLAS GEMM-only average latency: "
              << gemm_only_avg_us
              << " us"
              << endl;

    cout << "cuBLAS GEMM + bias average latency: "
              << gemm_bias_avg_us
              << " us"
              << endl;

    vector<float> host_c(c_cap);

    CUDACHECK(cudaMemcpy(
        host_c.data(),
        device_c,
        c_cap * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    if (!output_write(OUTPUT "/output_dfl_p3.bin", host_c)) {
        return 1;
    }

    cout << "cublas is successful!" << endl;

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDACHECK(cudaFree(device_a));
    CUDACHECK(cudaFree(device_w));
    CUDACHECK(cudaFree(device_bias));
    CUDACHECK(cudaFree(device_c));

    return 0;
}