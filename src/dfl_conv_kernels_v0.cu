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
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;

    int offset_y = TILE_M * blockIdx.y + threadIdx.y;
    int offset_x = TILE_N * blockIdx.x + threadIdx.x;
    
    float c_temp[4][4];

    for(int i = 0; i < 4; i++)
    {
        int m_base = offset_y + i * 16;
        for(int j = 0; j < 4; j++)
        {
            int n_base = offset_x + j * 16;
            float sum = 0.0f;
            if(m_base < m && n_base < col_num)
            {
                sum = bias[m_base];

                for(int index = 0; index < k; index++)
                {
                    float tile_w = martix_w[m_base * k + index]; 
                    float tile_a = martix_a[col_num * index + n_base];
                    sum += tile_w * tile_a;
                }
                c_temp[i][j] = sum;
            }
        }
    }

    for(int i = 0; i < 4; i++)
    {
        int m_c = i * 16 + offset_y;
        for(int j = 0; j < 4;j++)
        {
            int n_c = j * 16 + offset_x;
            if(m_c < m && n_c < col_num)
            {
                martix_c[m_c * col_num + n_c] = c_temp[i][j];
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

    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

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