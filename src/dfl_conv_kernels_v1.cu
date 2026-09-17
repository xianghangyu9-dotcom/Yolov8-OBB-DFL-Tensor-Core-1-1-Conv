#include"dfl_conv_kernels.cuh"

#include<cuda_runtime.h>

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
    constexpr int TILE_M  = 64;
    constexpr int TILE_N  = 64;
    constexpr int BK      = 16;
    constexpr int block_m = 16;
    constexpr int block_n = 16;

    int offset_y = TILE_M * blockIdx.y;
    int offset_x = TILE_N * blockIdx.x;
    
    int thread_perblock_y = threadIdx.y;
    int thread_perblock_x = threadIdx.x;

    __shared__ float s_mem_w[BK][TILE_M + 1];
    __shared__ float s_mem_a[BK][TILE_N + 1];

    float c_temp[4][4];
    for(int i = 0; i < 4; i++)
    {
        int m = i * block_m + thread_perblock_y;
        for(int j = 0; j < 4; j++)
        {
            c_temp[i][j] = bias[m];
        }
    }

    //K-LOOP
    for(int k0 = 0; k0 < k; k0 += BK)
    {
        //global-share 
        for(int i = 0; i < TILE_M / block_m; i++)
        {
            int w_base_m = thread_perblock_y + i * block_m;
            int a_base_n = thread_perblock_x + i * block_n;
            for(int j = 0; j < BK / block_n; j++)
            {
                int w_base_n = thread_perblock_x + j * block_n;
                int a_base_m = thread_perblock_y + j * block_m;
                s_mem_w[w_base_n][w_base_m] = martix_w[(offset_y + w_base_m) * k + w_base_n + k0];
                s_mem_a[a_base_m][a_base_n] = martix_a[offset_x + a_base_n + (a_base_m + k0) * col_num];
            }
        }
        __syncthreads();

        //tile计算
        for(int i = 0; i < BK; i++)
        {
            for(int j = 0; j < TILE_M / block_m; j++)
            {
                for(int k = 0; k < TILE_N / block_n; k++)
                {
                    int temp_w = j * block_m + thread_perblock_y;
                    int temp_a = k * block_n + thread_perblock_x; 
                    c_temp[j][k] += s_mem_w[i][temp_w] * s_mem_a[i][temp_a];
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

    float* device_w    = nullptr;
    float* device_a    = nullptr;
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

    cout<<"v1 average latency is "<<avg_us<<"us"<<endl;

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

    cout<<"v0 is successful!"<<endl;

    return 0;
}