#include"dfl_conv_kernels.cuh"

#include<cuda_runtime.h>
#include <cuda_fp16.h>
#include<mma.h>

using namespace std;
using namespace nvcuda::wmma;

#define INPUT "../input_case_v3"
#define OUTPUT "../output_case_v3"

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


__global__
void kernel_martix(
    const __half *matrix_a1,
    const __half *matrix_w1,
    const float *bias,
    int m,
    int n,
    int k,
    float *matrix_c1
){
    constexpr int BM      = 64;
    constexpr int BN      = 128;
    constexpr int BK      = 32;
    
    constexpr int WMMA_M  = 16;
    constexpr int WMMA_K  = 16;
    constexpr int WMMA_N  = 16;

    constexpr int warp_size_m = 2;
    constexpr int warp_size_n = 4;

    constexpr int frag_size_m = BM / (warp_size_m * WMMA_M);
    constexpr int frag_size_k = BK / WMMA_K;
    constexpr int frag_size_n = BN / (warp_size_n * WMMA_N);

    int tid      = threadIdx.x;
    int warp_id  = tid >> 5;
    int warp_y   = warp_id / warp_size_n;
    int warp_x   = warp_id % warp_size_n;
    
    int offset_y = BM * blockIdx.y;
    int offset_x = BN * blockIdx.x;

    __shared__ __align__ (32) __half s_mem_w[BM][BK];
    __shared__ __align__ (32) __half s_mem_a[BK][BN];
    __shared__ __align__(32) float   s_mem_c[BM][BN];

    fragment<
        matrix_a,
        WMMA_M, WMMA_N, WMMA_K,
        __half,
        row_major
    > w_frag[frag_size_m][frag_size_k];

    fragment<
        matrix_b,
        WMMA_M, WMMA_N, WMMA_K,
        __half,
        row_major
    > a_frag[frag_size_k][frag_size_n];

    fragment<
        accumulator,
        WMMA_M,WMMA_N,WMMA_K,
        float
    > c_frag[frag_size_m][frag_size_n];

    #pragma unroll
    for(int w_m = 0; w_m < frag_size_m; w_m++)
    {
        #pragma unroll
        for(int a_n = 0; a_n < frag_size_n; a_n++)
        {
            fill_fragment(c_frag[w_m][a_n], 0.0f);
        }
    }

    int k_size_w = BK >> 3;
    int m_size_w = 256 / k_size_w;

    int n_size_a = BN >> 3;
    int k_size_a = 256 / n_size_a;

    for(int k0 = 0; k0 < k; k0 += BK)
    {
        //global to share
        if(k0 < k)
        {
            #pragma unroll
            for(int i = 0; i < BM; i += m_size_w)
            {
                int local_m = tid / k_size_w + i;
                int local_k = tid % k_size_w * 8; 
                const float4 w8 = *reinterpret_cast<const float4*>(matrix_w1 + (offset_y + local_m) * k + k0 + local_k);
                *reinterpret_cast<float4*>(&s_mem_w[local_m][local_k]) = w8;
            }

            #pragma unroll
            for(int i = 0; i < BK; i += k_size_a)
            {
                int local_k = tid / n_size_a + i;
                int local_n = tid % n_size_a * 8;
                const float4 a8 = *reinterpret_cast<const float4 *>(matrix_a1 + (local_k + k0) * n + offset_x + local_n);
                *reinterpret_cast<float4*>(&s_mem_a[local_k][local_n]) = a8;
            }
        }

        __syncthreads();
      
        //加载frag_w,frag_a
        #pragma unroll
        for(int i = 0; i < frag_size_m; i++)
        {
            #pragma unroll
            for(int j = 0; j < frag_size_k; j++)
            {
                load_matrix_sync(
                    w_frag[i][j],
                    &s_mem_w[(warp_y * frag_size_m + i) * WMMA_M][j * WMMA_K],
                    BK
                );
            }
        }
        #pragma unroll
        for(int i = 0; i < frag_size_k; i++)
        {
            #pragma unroll
            for(int j = 0; j < frag_size_n; j++)
            {
                load_matrix_sync(
                    a_frag[i][j],
                    &s_mem_a[i * WMMA_K][(warp_x * frag_size_n + j) * WMMA_N],
                    BN
                );
            }
        }

        //计算c_frag
        #pragma unroll
        for(int i = 0; i < frag_size_m; i++)
        {
            #pragma unroll
            for(int j = 0; j < frag_size_n; j++)
            {
                #pragma unroll
                for(int index = 0; index < frag_size_k; index++)
                {
                    mma_sync(
                        c_frag[i][j],
                        w_frag[i][index],
                        a_frag[index][j],
                        c_frag[i][j]
                    );
                }
            }
        }
        __syncthreads();
    }
    //先写回share中加上bias后写回matrix_c1
    int share_m = warp_y * frag_size_m * WMMA_M;
    int share_n = warp_x * frag_size_n * WMMA_N;
    #pragma unroll
    for(int i = 0; i < frag_size_m; i++)
    {
        #pragma unroll
        for(int j = 0;j < frag_size_n; j++)
        {
            store_matrix_sync(&s_mem_c[share_m + i * WMMA_M][share_n + j * WMMA_N], c_frag[i][j],BN,mem_row_major);
        }
    }

    __syncthreads();

    constexpr int block_n = 16;
    constexpr int block_m = 256 / block_n;

    int lane_y = tid / block_n;
    int lane_x = tid % block_n;

    #pragma unroll
    for(int i = 0; i < BM; i += block_m)
    {
        int global_m = offset_y + lane_y + i;
        #pragma unroll
        for(int j = 0;j < BN; j += block_n)
        {
            int global_n = offset_x + lane_x + j;
            matrix_c1[global_m * n + global_n] = s_mem_c[lane_y + i][lane_x + j] + bias[global_m];
        }
    }
}

int main(){
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

    constexpr int BM    =  64;
    constexpr int BN    =  128;
    constexpr int grid_x  = (N + BN-1) / BN;
    constexpr int grid_y  = (M + BM-1) / BM;

    const dim3 block(
        256,
        1,
        1
    );

    const dim3 grid(
        grid_x,
        grid_y,
        1
    );

    constexpr int WARMUP = 100;
    constexpr int ITERS = 1000;

    //预热
    for (int i = 0; i < WARMUP; ++i) {
        kernel_martix<<<grid, block>>>(
            device_a,
            device_w,
            device_bias,
            M,
            N,
            K,
            device_c
        );
    }

    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    //计时
    cudaEvent_t start,stop;

    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaEventRecord(start));
    for(int i = 0; i<ITERS; i++){
        kernel_martix<<<
            grid,
            block
        >>>(
            device_a,
            device_w,
            device_bias,
            M,
            N,
            K,
            device_c
        );
    }

    CUDACHECK(cudaGetLastError());

    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDACHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );
    
    float avg_us = total_ms * 1000.0f / ITERS;

    cout<<"v3 average latency is "<<avg_us<<"us"<<endl;

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    vector<float> host_c(c_cap);

    CUDACHECK(
        cudaMemcpy(
            host_c.data(),
            device_c,
            c_cap * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    if(!output_write(OUTPUT "/output_dfl_p3.bin",host_c)){
        return 1;
    }

    cout<<"v3 is successful!"<<endl;

    return 0;
}