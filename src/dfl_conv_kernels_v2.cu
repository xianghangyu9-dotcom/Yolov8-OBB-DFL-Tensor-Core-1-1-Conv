#include"dfl_conv_kernels.cuh"

#include<cuda_runtime.h>
#include <cuda_fp16.h>

using namespace std;

#define INPUT "../input_case"
#define OUTPUT "../output_case"

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
    }while(0)                           \


__global__
void kernel_martix(
    const float *martix_a,
    const float *martix_w,
    const float *bias,
    int m,
    int col_num,
    int k,
    float *martix_c
){
    constexpr int BM      = 64;
    constexpr int BN      = 64;
    constexpr int BK      = 16;
    constexpr int block_m = 16;
    constexpr int block_n = 16;
    
    int offset_y = BM * blockIdx.y;
    int offset_x = BN * blockIdx.x;
    
    int thread_perblock_y = threadIdx.y;
    int thread_perblock_x = threadIdx.x;

    int thread_id = thread_perblock_y * block_m + thread_perblock_x;

    __shared__ float s_mem_w[2][BK][BM + 1];
    __shared__ float s_mem_a[2][BK][BN + 1];

    float c_temp[4][4];
    for(int i = 0; i < 4; i++)
    {
        int m = offset_y + i * block_m + thread_perblock_y;
        for(int j = 0; j < 4; j++)
        {
            c_temp[i][j] = bias[m];
        }
    }

    constexpr int lane_w_x = BK / 4;
    constexpr int lane_w_y = block_m * block_n /lane_w_x;
    constexpr int lane_a_x = BN / 4;
    constexpr int lane_a_y = block_m * block_n /lane_a_x;

    //global-share 缓冲区预加载 
    for(int i = 0; i < BM; i += lane_w_y)
    {
        int w_m = thread_id / lane_w_x;
        int w_n = thread_id % lane_w_x * 4;
        const float4 w4 = *reinterpret_cast<const float4*>( martix_w + (offset_y + w_m + i) * k + w_n); 
        
        s_mem_w[0][w_n + 0][w_m] = w4.x;
        s_mem_w[0][w_n + 1][w_m] = w4.y;
        s_mem_w[0][w_n + 2][w_m] = w4.z;
        s_mem_w[0][w_n + 3][w_m] = w4.w;
    }

    for(int i = 0; i < BK; i += lane_a_y)
    {
        int a_m = thread_id / lane_a_x;
        int a_n = thread_id % lane_a_x * 4;
        const float4 a4 = *reinterpret_cast<const float4*>( martix_a + (a_m + i) * col_num + offset_x + a_n); 
        
        s_mem_a[0][a_m][a_n + 0] = a4.x;
        s_mem_a[0][a_m][a_n + 1] = a4.y;
        s_mem_a[0][a_m][a_n + 2] = a4.z;
        s_mem_a[0][a_m][a_n + 3] = a4.w;
    }

    __syncthreads();

    //K-LOOP
    for(int k0 = BK; k0 < k + BK; k0 += BK)
    {
        int buffer = ((k0 / BK) & 1);
        if( k0 < k)
        {
            //global-share 
            for(int i = 0; i < BM; i += lane_w_y)
            {
                int w_m = thread_id / lane_w_x;
                int w_n = thread_id % lane_w_x * 4;
                const float4 w4 = *reinterpret_cast<const float4*>( martix_w + (offset_y + w_m + i) * k + w_n + k0); 
                
                s_mem_w[buffer][w_n + 0][w_m] = w4.x;
                s_mem_w[buffer][w_n + 1][w_m] = w4.y;
                s_mem_w[buffer][w_n + 2][w_m] = w4.z;
                s_mem_w[buffer][w_n + 3][w_m] = w4.w;
            }

            for(int i = 0; i < BK; i += lane_a_y)
            {
                int a_m = thread_id / lane_a_x;
                int a_n = thread_id % lane_a_x * 4;
                const float4 a4 = *reinterpret_cast<const float4*>( martix_a + (a_m + i + k0) * col_num + offset_x + a_n); 
                
                s_mem_a[buffer][a_m][a_n + 0] = a4.x;
                s_mem_a[buffer][a_m][a_n + 1] = a4.y;
                s_mem_a[buffer][a_m][a_n + 2] = a4.z;
                s_mem_a[buffer][a_m][a_n + 3] = a4.w;
            }
        }
        buffer ^= 1;

        //tile计算
        constexpr int circle_m = BM / block_m;
        constexpr int circle_n = BN / block_n;
        float Tm[2][circle_m];
        float Tn[2][circle_n];

        //Tm,Tn预加载
        for (int j = 0; j < circle_m; ++j) 
        {
            int temp_w = j * block_m + thread_perblock_y;
            int temp_a = j * block_n + thread_perblock_x;

            Tm[0][j] = s_mem_w[buffer][0][temp_w];
            Tn[0][j] = s_mem_a[buffer][0][temp_a];
        }

        for(int i = 1; i < BK + 1; i++)
        {
            //Tm,Tn缓冲
            int buffer_t = i & 1;
            if( i < BK)
            {
               for (int j = 0; j < circle_m; ++j) 
                {
                    int temp_w = j * block_m + thread_perblock_y;
                    int temp_a = j * block_n + thread_perblock_x;

                    Tm[buffer_t][j] = s_mem_w[buffer][i][temp_w];
                    Tn[buffer_t][j] = s_mem_a[buffer][i][temp_a];
                }
            }
            buffer_t ^= 1;

            //计算
            for(int c_m = 0; c_m < circle_m; c_m++)
            {
                for(int c_n = 0; c_n < circle_n; c_n++)
                {
                    c_temp[c_m][c_n] += Tm[buffer_t][c_m] * Tn[buffer_t][c_n];
                }
            }
        }
        __syncthreads();
    }

    //一次性放回c中
    for(int i_c = 0; i_c < 4; i_c++)
    {
        int m_c = i_c * block_m + offset_y + thread_perblock_y;
        for(int j_c = 0; j_c < 4;j_c++)
        {
            int n_c = j_c * block_n + offset_x + thread_perblock_x;
            if(m_c < m && n_c < col_num)
            {
                martix_c[m_c * col_num + n_c] = c_temp[i_c][j_c];
            }
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

    vector<float> host_w(w_cap);
    vector<float> host_a(a_cap);
    vector<float> host_bias(bias_cap);

    if(!input_load(INPUT "/p3_w_dfl_raw.bin",host_w)){
        return 1;
    };
    if(!input_load(INPUT "/p3_a_dfl_raw.bin",host_a)){
        return 1;
    };
    if(!input_load(INPUT "/p3_bias_dfl_raw.bin",host_bias)){
        return 1;
    };

    float* device_w   = nullptr;
    float* device_a   = nullptr;
    float* device_bias = nullptr;
    float* device_c    = nullptr;

    CUDACHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_w),
        w_cap * sizeof(float))
    );

    CUDACHECK(
        cudaMalloc(reinterpret_cast<void**>(&device_a),
        a_cap * sizeof(float))
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
            w_cap * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDACHECK(
        cudaMemcpy(
            device_a,
            host_a.data(),
            a_cap * sizeof(float),
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

    constexpr int tile_m    =  64;
    constexpr int tile_n    =  64;
    constexpr int grid_x  = (N + tile_n-1) / tile_n;
    constexpr int grid_y  = (M + tile_m-1) / tile_m;

    const dim3 block(
        16,
        16,
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

    cout<<"v2 average latency is "<<avg_us<<"us"<<endl;

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

    cout<<"v2 is successful!"<<endl;

    return 0;
}