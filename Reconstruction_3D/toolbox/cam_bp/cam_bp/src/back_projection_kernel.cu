#include <THC.h>
#include <stdbool.h>
#include <stdio.h>
#include <cuda.h>
#include "back_projection_kernel.h"

const int CUDA_NUM_THREADS = 1024;

// 计算出在给定数据量和线程数的情况下，需要启动多少个CUDA块来处理数据
inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// 在不支持Compute Capability 6.0及以上版本的CUDA设备上提供一个替代方案，以实现原子加法操作
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                            __longlong_as_double(assumed)));
  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif


#define EPS 1e-5
// for rounding to the nearest integer


// 将一个浮点数取整为最接近的小于或等于它的整数（向下取整）
#define FLOOR_I(a) \
  ( (a) < 0 ? (int)(a) - 1:(int) (a) )

// 将一个浮点数取整为最接近的小于或等于它的整数，并将结果转换为浮点数类型
#define FLOOR_F(a)                                              \
  (float)(FLOOR_I(a))

// 将一个浮点数进行四舍五入，并返回最接近的整数值
#define ROUND_I(a) \
  ( (a)-FLOOR_F(a) > FLOOR_F(a) + 1.0 - (a) ? FLOOR_I(a)+1:FLOOR_I(a) )

// 将一个浮点数进行四舍五入，并返回最接近的浮点数值
#define ROUND_F(a)                                              \
  ( (a)-FLOOR_F(a) > FLOOR_F(a) + 1.0 - (a) ? FLOOR_F(a)+1:FLOOR_F(a) )

// 判断给定的三个浮点数x1、x2、x3是否在给定的三个范围s1、s2、s3内
#define WITHIN_BOUNDS(x1, x2, x3, s1, s2, s3) ( (x1 >= 0.0f) && (x1 < s1) && (x2 >= 0.0f) && (x2 < s2) && (x3 >= 0.0f) && (x3 < s3))

// 根据给定的索引和步长从四维数据数组中获取对应的元素值
#define GET_DIRECT_4d(data, x0, x1, x2, x3, sd0, sd1, sd2, sd3)         \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3)])

// 在四维数据数组中的特定位置进行原子加法操作
#define ADD_ATOMIC_4d(data, x0, x1, x2, x3, sd0, sd1, sd2, sd3, v)        \
  atomicAdd( data + (x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3), v )
// 在五维数据数组中的特定位置进行原子加法操作
#define ADD_ATOMIC_5d(data, x0, x1, x2, x3, x4, sd0, sd1, sd2, sd3, sd4, v) \
  atomicAdd( data + (x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3) + (x4)*(sd4), v )
// 在四维数据数组中的特定位置设置一个给定的值
#define SET_DIRECT_4d(data, x0, x1, x2, x3, sd0, sd1, sd2, sd3, v)        \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3)]) = v

// 根据给定的索引和步长从三维数据数组中获取对应的元素值
#define GET_DIRECT_3d(data, x0, x1, x2, sd0, sd1, sd2) \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2)])


// 在三维数据数组中的特定位置设置一个给定的值
#define SET_DIRECT_3d(data, x0, x1, x2, sd0, sd1, sd2, v)        \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) ]) = v


// 根据给定的索引和步长从五维数据数组中获取对应的元素值
#define GET_DIRECT_5d(data, x0, x1, x2, x3, x4, stride0, stride1, stride2, stride3, stride4) \
  ((data)[(x0)*(stride0)+(x1)*(stride1)+(x2)*(stride2)+(x3)*(stride3)+(x4)*(stride4)])


// 在五维数据数组中的特定位置设置一个给定的值
#define SET_DIRECT_5d(data, x0, x1, x2, x3, x4, stride0, stride1, stride2, stride3, stride4, value) \
  ((data)[(x0)*(stride0)+(x1)*(stride1)+(x2)*(stride2)+(x3)*(stride3)+(x4)*(stride4)] = (value))


// 将全局坐标下的体素索引转换为体素坐标，将全局坐标(glob_a)加上0.5后乘以体素分辨率(res_a)，然后取整为最接近的小于或等于的整数，从而得到对应的体素坐标
#define VOXIND_TO_VOXC(glob_a, res_a)           \
  ( FLOOR_I( (glob_a+0.5f) * (float)res_a ) )

// 返回两个值中的较大值
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )

// 返回三个值中的最大值
#define MAX3(a,b,c) MAX( MAX((a),(b)), (c) )

// 生成一个用于并行计算的循环结构，在CUDA内核函数中使用
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)


// 进行参数检查和错误处理，用于在条件不满足时抛出异常
#define THCUNN_argCheck(STATE, COND, ARG, T, FORMAT) \
  if (!(COND)) { \
    THCDescBuff s1 = THCudaTensor_sizeDesc(state, T); \
    THArgCheck(COND, ARG, FORMAT, s1.str);           \
  }

// 进行维度和大小的检查，并在条件不满足时抛出异常
#define THCUNN_check_dim_size(STATE, T, DIM, DIM_SIZE, SIZE) \
  if (THCudaTensor_nDimension(STATE, T) != DIM ||             \
      THCudaTensor_size(STATE, T, DIM_SIZE) != SIZE) {        \
      THCDescBuff s1 = THCudaTensor_sizeDesc(state, T);       \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d" \
              " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

// 确保多个张量在同一个GPU上
#define THCUNN_assertSameGPU(...) THAssertMsg(THCudaTensor_checkGPU(__VA_ARGS__), \
  "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")




// 进行输入张量的形状检查
static inline void cambp_shapecheck(THCState* state,
                                    THCudaTensor* depth,
                                    THCudaTensor* camdist,
                                    THCudaTensor* fl,
                                    THCudaTensor* voxel,
                                    THCudaTensor* cnt){
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, depth) == 4, 2, depth,
      "4D input tensor expected but got: %s");
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, camdist) == 2, 2, camdist,
      "3D input tensor expected but got: %s");
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, fl) == 2, 2, fl,
      "3D input tensor expected but got: %s");

  THCUNN_argCheck(state, THCudaTensor_nDimension(state, cnt) == 5, 2, cnt,
      "5D input tensor expected but got: %s");

  THCUNN_argCheck(state, THCudaTensor_nDimension(state, voxel) == 5, 2, voxel,
      "5D input tensor expected but got: %s");

  int nbatch = THCudaTensor_size(state, depth, 0);
  int nc = THCudaTensor_size(state, depth, 1);
  int vszx = THCudaTensor_size(state, voxel, 2);
  int vszy = THCudaTensor_size(state, voxel, 3);
  int vszz = THCudaTensor_size(state, voxel, 4);

  //fprintf(stderr, "argcheck + size pass\n");
  THCUNN_check_dim_size(state, camdist, 2, 0, nbatch);
  THCUNN_check_dim_size(state, camdist, 2, 1, nc);
  //fprintf(stderr, "camdist  pass\n");
  THCUNN_check_dim_size(state, fl, 2, 0, nbatch);
  THCUNN_check_dim_size(state, fl, 2, 1, nc);
  THCUNN_check_dim_size(state, voxel, 5, 0, nbatch);
  THCUNN_check_dim_size(state, voxel, 5, 1, nc);
  THCUNN_check_dim_size(state, cnt, 5, 0, nbatch);
  THCUNN_check_dim_size(state, cnt, 5, 1, nc);
  THCUNN_check_dim_size(state, cnt, 5, 2, vszx);
  THCUNN_check_dim_size(state, cnt, 5, 3, vszy);
  THCUNN_check_dim_size(state, cnt, 5, 4, vszz);

}

// 检查输入张量的维度和大小
static inline void cambp_shapecheck(THCState* state,
                                    THCudaTensor* depth,
                                    THCudaTensor* grid,
                                    THCudaTensor* voxel,
                                    THCudaTensor* cnt){
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, depth) == 4, 2, depth,
      "4D input tensor expected but got: %s");
  THCUNN_argCheck(state, THCudaTensor_nDimension(state, grid) == 5, 2, grid,
      "5D input tensor expected but got: %s");

  THCUNN_argCheck(state, THCudaTensor_nDimension(state, cnt) == 5, 2, cnt,
      "5D input tensor expected but got: %s");

  THCUNN_argCheck(state, THCudaTensor_nDimension(state, voxel) == 5, 2, voxel,
      "5D input tensor expected but got: %s");

  int nbatch = THCudaTensor_size(state, depth, 0);
  int nc = THCudaTensor_size(state, depth, 1);
  int nh = THCudaTensor_size(state, depth, 2);
  int nw = THCudaTensor_size(state, depth, 3);
  int vszx = THCudaTensor_size(state, voxel, 2);
  int vszy = THCudaTensor_size(state, voxel, 3);
  int vszz = THCudaTensor_size(state, voxel, 4);

  //fprintf(stderr, "argcheck + size pass\n");
  THCUNN_check_dim_size(state, grid, 5, 0, nbatch);
  THCUNN_check_dim_size(state, grid, 5, 1, nc);
  THCUNN_check_dim_size(state, grid, 5, 2, nh);
  THCUNN_check_dim_size(state, grid, 5, 3, nw);
  THCUNN_check_dim_size(state, grid, 5, 4, 3);
  //fprintf(stderr, "camdist  pass\n");
  THCUNN_check_dim_size(state, voxel, 5, 0, nbatch);
  THCUNN_check_dim_size(state, voxel, 5, 1, nc);
  THCUNN_check_dim_size(state, cnt, 5, 0, nbatch);
  THCUNN_check_dim_size(state, cnt, 5, 1, nc);
  THCUNN_check_dim_size(state, cnt, 5, 2, vszx);
  THCUNN_check_dim_size(state, cnt, 5, 3, vszy);
  THCUNN_check_dim_size(state, cnt, 5, 4, vszz);
}



// 计算给定值的平方
#define square(a) \
  ((a)*(a))

//inline float sqrt(float a){
//  return sqrtf(a);
//}


// 计算三维向量的欧几里德范数（模长）
#define vec3d_norm( x1, x2, x3) \
  sqrtf(square(x1) + square(x2) + square(x3))


// 用于设置CUDA内核函数的启动配置
__launch_bounds__(CUDA_NUM_THREADS)
// 用于执行后向投影的前向操作
__global__ void back_projection_forward_kernel(float* depth,
                                               int N, int NC,
                                               int dszh, int dszw,
                                               int dsdn, int dsdc, int dsdh, int dsdw,
                                               float* cam_dist_in,
                                               int cdisdn, int cdisdc,
                                               float* fl_in,
                                               int fisdn,int fisdc,
                                               float* voxel,
                                               int vszx, int vszy, int vszz,
                                               int vsdn, int vsdc, int vsdx, int vsdy, int vsdz,
                                               float* cnt,
                                               int csdn, int csdc, int csdx, int csdy, int csdz,
                                               int nthreads){
  // int index;
  CUDA_KERNEL_LOOP(index, nthreads){
    const int n = index % N; // 当前线程所处理的 批次 索引
    const int ind_c = (index / N) % NC; // 根据索引index、批次大小N和通道数NC计算当前线程所处理的 通道 索引
    const int ind_w = (index / (N * NC)) % dszh; // 根据索引index、批次大小N和通道数NC以及深度张量高度dszh计算当前线程所处理的 高度 索引
    const int ind_h = (index / (N * NC * dszh)) % dszw; // 当前线程所处理的 宽度 索引

    // 获取深度张量在给定位置(n, ind_c, ind_h, ind_w)处的值，并将其存储在变量dep_at_pix中
    float dep_at_pix = GET_DIRECT_4d(depth, n, ind_c, ind_h, ind_w, dsdn, dsdc, dsdh, dsdw);
    

    // skip if not in foreground.
    if(dep_at_pix < 0.0f){
      continue;
    }

    // 相机距离张量cam_dist_in中，数据是按照一定顺序存储的。通过索引计算可以确定要访问的特定位置
    float cam_dist = cam_dist_in[n*cdisdn + ind_c*cdisdc]; // 获取 相机距离 张量在给定位置(n, ind_c)处的值
    float fl = fl_in[n*fisdn + ind_c*fisdc]; // 获取 焦距张量 在给定位置(n, ind_c)处的值

    // "高度索引的中心化值"是指将索引值转换为相对于张量高度中心的偏移量或位置，可以将像素或张量的位置相对于张量中心进行表示
    float imind_h = (float)ind_h - ((float)dszh-1.0f)/2.0f; // 计算像素高度索引ind_h的中心化值
    float imind_w = (float)ind_w - ((float)dszw-1.0f)/2.0f; // 计算像素宽度索引ind_w的中心化值
    
    // convert ray depth to plane depth
    float cos_theta = fl / vec3d_norm(imind_h, imind_w, fl);
    
    dep_at_pix = dep_at_pix * cos_theta;

    // find global coord
    // 全局坐标的原点是相机
    // 使用三角形相似的原理计算全局坐标
    float glob_y = -dep_at_pix*imind_w/fl; // 计算全局坐标中的y值
    float glob_z = -dep_at_pix*imind_h/fl; // 计算全局坐标中的z值
    float glob_x = dep_at_pix - cam_dist; // 计算全局坐标中的x值

    
    // find voxel index
    // 索引值表示了在体素数组中相应位置的体素
    int vox_ind_x = VOXIND_TO_VOXC(glob_x, vszx); // 将全局坐标glob_x映射到体素张量中的索引vox_ind_x
    int vox_ind_y = VOXIND_TO_VOXC(glob_y, vszy); // 将全局坐标glob_y映射到体素张量中的索引vox_ind_y
    int vox_ind_z = VOXIND_TO_VOXC(glob_z, vszz); // 将全局坐标glob_z映射到体素张量中的索引vox_ind_z

    
    // skip if out of bounds
    if(!WITHIN_BOUNDS(vox_ind_x, vox_ind_y, vox_ind_z, vszx, vszy, vszz)){
      continue;
    }
    

    // find voxel center
    // 首先需要将体素的索引值转换为相对于体素尺寸的比例值，将体素索引值加上0.5，并除以体素尺寸，这样可以将索引值映射到范围为 [0, 1] 的比例值
    // 接下来将范围为 [0, 1] 的比例值转换为范围为 [-0.5, 0.5] 的中心化值，从比例值中减去0.5，得到体素在 x 方向上的中心化值
    float vox_center_x = (((float)vox_ind_x+0.5f) / (float)vszx) - 0.5f; // 计算体素中心的x坐标
    float vox_center_y = (((float)vox_ind_y+0.5f) / (float)vszy) - 0.5f; // 计算体素中心的y坐标
    float vox_center_z = (((float)vox_ind_z+0.5f) / (float)vszz) - 0.5f; // 计算体素中心的z坐标

    
    //printf("%d %d %d \n", vox_ind_x, vox_ind_y, vox_ind_z);

    // calculate distance
    float dist = vec3d_norm(glob_x - vox_center_x, glob_y - vox_center_y, glob_z - vox_center_z);
    // 根据全局坐标和体素中心坐标，使用宏vec3d_norm计算距离dist

    /*if(vox_ind_x == 1 && vox_ind_y ==1 && vox_ind_z ==1) {
      printf("%f \n", dist - 1.0f/(float)(MAX3(vszx, vszy, vszz)));
      }*/

    // Assuming a tdf threshold of max cell length
    // 使用宏ADD_ATOMIC_5d对计数张量中给定索引处的值进行原子加法操作，将计数值增加
    ADD_ATOMIC_5d(voxel, n, ind_c, vox_ind_x, vox_ind_y, vox_ind_z, vsdn, vsdc, vsdx, vsdy, vsdz, dist);
    ADD_ATOMIC_5d(cnt, n, ind_c, vox_ind_x, vox_ind_y, vox_ind_z, csdn, csdc, csdx, csdy, csdz, 1.0f);
  }
}




__launch_bounds__(CUDA_NUM_THREADS)
__global__ void inplace_safe_divide( float* voxel,
                                     int N, int NC,
                                     int vszx, int vszy, int vszz,
                                     int vsdn, int vsdc, int vsdx, int vsdy, int vsdz,
                                     float* cnt,
                                     int csdn, int csdc,int csdx, int csdy, int csdz,
                                     float dist_bias,
                                     int nthreads){
  //int index;
  CUDA_KERNEL_LOOP(index, nthreads){
    //printf("hi!!!");
    const int n = index % N;
    const int ind_c = (index/ (N) ) % NC;
    const int ind_x = (index / (N * NC) ) % vszx;
    const int ind_y = (index / (N * NC * vszx) ) % vszy;
    const int ind_z = (index / (N * NC * vszx * vszy) ) % vszz;
    float ptnum = GET_DIRECT_5d(cnt, n, ind_c, ind_x, ind_y, ind_z, vsdn, vsdc, vsdx, vsdy, vsdz);
    if(ptnum < EPS){
      // ignore small values
      continue;
    }
    float dist = GET_DIRECT_5d(voxel, n, ind_c, ind_x, ind_y, ind_z, vsdn, vsdc, vsdx, vsdy, vsdz);
    SET_DIRECT_5d(voxel, n, ind_c, ind_x, ind_y, ind_z, csdn, csdc, csdx, csdy, csdz, (dist - dist_bias/(float)(MAX3(vszx, vszy, vszz)))/ptnum);
  }
}


__launch_bounds__(CUDA_NUM_THREADS)
__global__ void get_surface_mask_kernel(float* depth,
                                        int N, int NC,
                                        int dszh, int dszw,
                                        int dsdn, int dsdc, int dsdh, int dsdw,
                                        float* cam_dist_in,
                                        int cdisdn, int cdisdc,
                                        float* fl_in,
                                        int fisdn, int fisdc,
                                        float* cnt,
                                        int vszx, int vszy, int vszz,
                                        int csdn, int csdc, int csdx, int csdy, int csdz,
                                        float* mask,
                                        int msdn, int msdc, int msdx, int msdy, int msdz,
                                        int nthreads){
  CUDA_KERNEL_LOOP(index, nthreads){
    const int n = index % N;
    const int ind_c = (index / N) % NC;
    const int ind_x = (index / (N * NC)) % vszx;
    const int ind_y = (index / (N * NC * vszx) ) % vszy;
    const int ind_z = (index / (N * NC * vszx * vszy) ) % vszz;
    float fl = fl_in[n*fisdn+ind_c*fisdc];
    float cam_dist = cam_dist_in[n*cdisdn+ind_c*cdisdc];
    float ptnum = GET_DIRECT_5d(cnt, n, ind_c, ind_x, ind_y, ind_z, csdn, csdc, csdx, csdy, csdz);
    if(ptnum>EPS){
      continue;
    }
    float vox_center_x = (((float)(ind_x)+0.5)/(float)vszx) - 0.5;
    float vox_center_y = (((float)(ind_y)+0.5)/(float)vszy) - 0.5;
    float vox_center_z = (((float)(ind_z)+0.5)/(float)vszz) - 0.5;
    float im_h = -vox_center_z * fl/(vox_center_x + cam_dist);
    float im_w = -vox_center_y * fl/(vox_center_x + cam_dist);
    int im_idh = ROUND_I( 0.5*((float)dszh-1.0) + im_h );
    int im_idw = ROUND_I( 0.5*((float)dszw-1.0) + im_w );
    if(im_idh<0 || im_idh>=dszh){
      continue;
    }
    if(im_idw<0 || im_idw>=dszw){
      continue;
    }
    float dep_at_pix = GET_DIRECT_4d(depth, n, ind_c, im_idh, im_idw, dsdn, dsdc, dsdh, dsdw);
    if(dep_at_pix<0){
      continue;
    }
    float ray_depth = vec3d_norm(vox_center_x+cam_dist,vox_center_y,vox_center_z);
    if(dep_at_pix<ray_depth){
      SET_DIRECT_5d(mask, n, ind_c, ind_x, ind_y, ind_z, msdn, msdc, msdx, msdy, msdz, 0.0);
    }
  }
}




 

__launch_bounds__(CUDA_NUM_THREADS)
__global__ void back_projection_backward_kernel(float* depth,
                                                int N, int NC,
                                                int dszh, int dszw,
                                                int dsdn, int dsdc, int dsdh, int dsdw,
                                                float* fl_in,
                                                int fisdn, int fisdc,
                                                float* camdist_in,
                                                int cidn, int cidc,
                                                float* cnt,
                                                int cszx, int cszy, int cszz,
                                                int csdn, int csdc, int csdx, int csdy, int csdz,
                                                float* grad_in,
                                                int giszx, int giszy, int giszz,
                                                int gisdn, int gisdc, int gisdx, int gisdy, int gisdz,
                                                float* grad_depth,
                                                int gdsdn, int gdsdc, int gdsdh, int gdsdw,
                                                float* grad_camdist,
                                                int gcsdn, int gcsdc,
                                                float* grad_fl,
                                                int gfsdn, int gfsdc,
                                                int nthreads){
  CUDA_KERNEL_LOOP(index, nthreads){
    const int n = index % N;
    const int ind_c = (index / N) % NC;
    const int ind_h = (index / (N * NC)) % dszh;
    const int ind_w = (index / (N * NC * dszh) ) % dszw;
    // printf("%d %d %d %d\n", n, ind_c, ind_h, ind_w);
    float dep_at_pix_i = GET_DIRECT_4d(depth, n, ind_c, ind_h, ind_w, dsdn, dsdc, dsdh, dsdw);

    // skip if not in foreground.
    if(dep_at_pix_i < 0.0f){
      continue;
    }

    float fl = fl_in[n*fisdn+ind_c*fisdc];
    float cam_dist = camdist_in[n*csdn + ind_c*csdc];
    float imind_h = float(ind_h) - float(dszh-1)/2.0f;
    float imind_w = float(ind_w) - float(dszw-1)/2.0f;
    
    // convert ray depth to plane depth
    float cos_theta = fl / vec3d_norm(imind_h, imind_w, fl);
    float dep_at_pix = dep_at_pix_i * cos_theta;

    // find global coord
    float glob_y = -dep_at_pix*imind_w/fl;
    float glob_z = -dep_at_pix*imind_h/fl;
    float glob_x = dep_at_pix - cam_dist;

    // find voxel index
    int vox_ind_x = VOXIND_TO_VOXC(glob_x, cszx);
    int vox_ind_y = VOXIND_TO_VOXC(glob_y, cszy);
    int vox_ind_z = VOXIND_TO_VOXC(glob_z, cszz);

    // skip if out of bounds
    if(!WITHIN_BOUNDS(vox_ind_x, vox_ind_y, vox_ind_z, cszx, cszy, cszz)){
        continue;
    }
    

    

    // find voxel center
    float vox_center_x = ((float(vox_ind_x)+0.5) / float(cszx)) - 0.5;
    float vox_center_y = ((float(vox_ind_y)+0.5) / float(cszy)) - 0.5;
    float vox_center_z = ((float(vox_ind_z)+0.5) / float(cszz)) - 0.5;

    float pt_vec_len = vec3d_norm(imind_h, imind_w, fl);
    if(pt_vec_len < 1e-5){
      pt_vec_len = 1e-5;
    }
    float pt_dirvec_x = - fl /  pt_vec_len;
    float pt_dirvec_y = imind_w /  pt_vec_len;
    float pt_dirvec_z = imind_h /  pt_vec_len;

    float pt_vc_vec_len = vec3d_norm(glob_x - vox_center_x, glob_y - vox_center_y, glob_z - vox_center_z);
    if(pt_vc_vec_len < 1e-5){
      pt_vc_vec_len = 1e-5;
    }
    float pt_vc_dirvec_x = (glob_x - vox_center_x) /  pt_vc_vec_len;
    float pt_vc_dirvec_y = (glob_y - vox_center_y) /  pt_vc_vec_len;
    float pt_vc_dirvec_z = (glob_z - vox_center_z) /  pt_vc_vec_len;

    float cos_theta_cc = (pt_dirvec_x * pt_vc_dirvec_x) + (pt_dirvec_y * pt_vc_dirvec_y) + (pt_dirvec_z * pt_vc_dirvec_z);
    float ptnum = GET_DIRECT_5d(cnt, n, ind_c, vox_ind_x, vox_ind_y, vox_ind_z, csdn, csdc, csdx, csdy, csdz);
    if(ptnum < 1){
      ptnum = 1;
    }

    float gd_i = GET_DIRECT_5d(grad_in, n, ind_c, vox_ind_x, vox_ind_y, vox_ind_z, gisdn, gisdc, gisdx, gisdy, gisdz);
    SET_DIRECT_4d(grad_depth, n, ind_c, ind_h, ind_w, gdsdn, gdsdc, gdsdh, gdsdw, -gd_i*cos_theta_cc/ptnum);

    // grad_fl

    float grad_fl_x = ((glob_x-vox_center_x)/pt_vc_vec_len) * (square(imind_w) + square(imind_h)) / (pt_vec_len*pt_vec_len*pt_vec_len) ;
    float grad_fl_y = ((glob_y-vox_center_y)/pt_vc_vec_len) * (imind_w * fl) / (pt_vec_len*pt_vec_len*pt_vec_len) ;
    float grad_fl_z = ((glob_z-vox_center_z)/pt_vc_vec_len) * (imind_h * fl) / (pt_vec_len*pt_vec_len*pt_vec_len) ;
    float grad_fl_i = (grad_fl_x + grad_fl_y + grad_fl_z) * gd_i * dep_at_pix_i / ptnum;

    atomicAdd(grad_fl+gfsdn*n + gfsdc*ind_c, grad_fl_i);


    // grad_cam_dist

    atomicAdd(grad_camdist+gcsdn*n+gcsdc*ind_c, -pt_vc_dirvec_x*gd_i/ptnum);
  }
}


__launch_bounds__(CUDA_NUM_THREADS)
__global__ void spherical_back_projection_forward_kernel(float* depth,
                                               int N, int NC,
                                               int dszh, int dszw,
                                               int dsdn, int dsdc, int dsdh, int dsdw,
                                               float* grid_in,
                                               int gisdn, int gisdc,int gisdh, int gisdw,int gisddim,
                                               float* voxel,
                                               int vszx, int vszy, int vszz,
                                               int vsdn, int vsdc, int vsdx, int vsdy, int vsdz,
                                               float* cnt,
                                               int csdn, int csdc, int csdx, int csdy, int csdz,
                                               int nthreads){
  // int index;
  CUDA_KERNEL_LOOP(index, nthreads){
    const int n = index % N;
    const int ind_c = (index / N) % NC;
    const int ind_w = (index / (N * NC)) % dszh;
    const int ind_h = (index / (N * NC * dszh)) % dszw;
    
    float dep_at_pix = GET_DIRECT_4d(depth, n, ind_c, ind_h, ind_w, dsdn, dsdc, dsdh, dsdw);
    float grid_x = GET_DIRECT_5d(grid_in, n, ind_c, ind_h, ind_w, 0, gisdn, gisdc, gisdh, gisdw,gisddim);
    float grid_y = GET_DIRECT_5d(grid_in, n, ind_c, ind_h, ind_w, 1, gisdn, gisdc, gisdh, gisdw,gisddim);
    float grid_z = GET_DIRECT_5d(grid_in, n, ind_c, ind_h, ind_w, 2, gisdn, gisdc, gisdh, gisdw,gisddim);
    

    // skip if not in foreground.
    if(dep_at_pix < 0.0f){
      continue;
    }


    float glob_x = grid_x * dep_at_pix;
    float glob_y = grid_y * dep_at_pix;
    float glob_z = grid_z * dep_at_pix;

    
    // find voxel index
    int vox_ind_x = VOXIND_TO_VOXC(glob_x, vszx);
    int vox_ind_y = VOXIND_TO_VOXC(glob_y, vszy);
    int vox_ind_z = VOXIND_TO_VOXC(glob_z, vszz);

    
    // skip if out of bounds
    if(!WITHIN_BOUNDS(vox_ind_x, vox_ind_y, vox_ind_z, vszx, vszy, vszz)){
      continue;
    }
    

    // find voxel center
    float vox_center_x = (((float)vox_ind_x+0.5f) / (float)vszx) - 0.5f;
    float vox_center_y = (((float)vox_ind_y+0.5f) / (float)vszy) - 0.5f;
    float vox_center_z = (((float)vox_ind_z+0.5f) / (float)vszz) - 0.5f;

    
    //printf("%d %d %d \n", vox_ind_x, vox_ind_y, vox_ind_z);

    // calculate distance
    float dist = vec3d_norm(glob_x - vox_center_x, glob_y - vox_center_y, glob_z - vox_center_z);

    /*if(vox_ind_x == 1 && vox_ind_y ==1 && vox_ind_z ==1) {
      printf("%f \n", dist - 1.0f/(float)(MAX3(vszx, vszy, vszz)));
      }*/

    // Assuming a tdf threshold of max cell length
    ADD_ATOMIC_5d(voxel, n, ind_c, vox_ind_x, vox_ind_y, vox_ind_z, vsdn, vsdc, vsdx, vsdy, vsdz, dist);
    ADD_ATOMIC_5d(cnt, n, ind_c, vox_ind_x, vox_ind_y, vox_ind_z, csdn, csdc, csdx, csdy, csdz, 1.0f);
  }
}

__launch_bounds__(CUDA_NUM_THREADS)
__global__ void spherical_back_projection_backward_kernel(float* depth,
                                                int N, int NC,
                                                int dszh, int dszw,
                                                int dsdn, int dsdc, int dsdh, int dsdw,
                                                float* grid_in,
                                                int gisdn, int gisdc,int gisdh, int gisdw,int gisddim,
                                                float* cnt,
                                                int cszx, int cszy, int cszz,
                                                int csdn, int csdc, int csdx, int csdy, int csdz,
                                                float* grad_in,
                                                int giszx, int giszy, int giszz,
                                                int gradsdn, int gradsdc, int gradsdx, int gradsdy, int gradsdz,
                                                float* grad_depth,
                                                int gdsdn, int gdsdc, int gdsdh, int gdsdw,
                                                int nthreads){
  CUDA_KERNEL_LOOP(index, nthreads){

    const int n = index % N;
    const int ind_c = (index / N) % NC;
    const int ind_h = (index / (N * NC)) % dszh;
    const int ind_w = (index / (N * NC * dszh) ) % dszw;
    //printf("test");
    float dep_at_pix_i = GET_DIRECT_4d(depth, n, ind_c, ind_h, ind_w, dsdn, dsdc, dsdh, dsdw);
    float grid_x = GET_DIRECT_5d(grid_in, n, ind_c, ind_h, ind_w, 0, gisdn, gisdc, gisdh, gisdw,gisddim);
    float grid_y = GET_DIRECT_5d(grid_in, n, ind_c, ind_h, ind_w, 1, gisdn, gisdc, gisdh, gisdw,gisddim);
    float grid_z = GET_DIRECT_5d(grid_in, n, ind_c, ind_h, ind_w, 2, gisdn, gisdc, gisdh, gisdw,gisddim);
    
    

    // skip if not in foreground.
    if(dep_at_pix_i < 0.0f){
      continue;
    }
    float glob_x = grid_x * dep_at_pix_i;
    float glob_y = grid_y * dep_at_pix_i;
    float glob_z = grid_z * dep_at_pix_i;

    // find voxel index
    int vox_ind_x = VOXIND_TO_VOXC(glob_x, cszx);
    int vox_ind_y = VOXIND_TO_VOXC(glob_y, cszy);
    int vox_ind_z = VOXIND_TO_VOXC(glob_z, cszz);
    
    // skip if out of bounds
    if(!WITHIN_BOUNDS(vox_ind_x, vox_ind_y, vox_ind_z, cszx, cszy, cszz)){
        continue;
    }
    

    

    // find voxel center
    float vox_center_x = ((float(vox_ind_x)+0.5) / float(cszx)) - 0.5;
    float vox_center_y = ((float(vox_ind_y)+0.5) / float(cszy)) - 0.5;
    float vox_center_z = ((float(vox_ind_z)+0.5) / float(cszz)) - 0.5;

    float pt_vec_len = vec3d_norm(glob_x, glob_y, glob_z);
    if(pt_vec_len < 1e-5){
      pt_vec_len = 1e-5;
    }
    float pt_dirvec_x = glob_x /  pt_vec_len;
    float pt_dirvec_y = glob_y /  pt_vec_len;
    float pt_dirvec_z = glob_z /  pt_vec_len;

    float cos_theta_cc = (pt_dirvec_x * vox_center_x) + (pt_dirvec_y * vox_center_y) + (pt_dirvec_z * vox_center_z);
    float dist = vec3d_norm(glob_x - vox_center_x, glob_y - vox_center_y, glob_z - vox_center_z);

    float ptnum = GET_DIRECT_5d(cnt, n, ind_c, vox_ind_x, vox_ind_y, vox_ind_z, csdn, csdc, csdx, csdy, csdz);
    // pts may be in diffrent voxel if its really near some places.
    //if( ptnum > 1-1e-5){
    if(ptnum < 1){
      ptnum = 1;
    }
    if(dist < 1e-5){
      dist=1e-5;
    }
    float gd_i = GET_DIRECT_5d(grad_in, n, ind_c, vox_ind_x, vox_ind_y, vox_ind_z, gradsdn, gradsdc, gradsdx, gradsdy, gradsdz);
    SET_DIRECT_4d(grad_depth, n, ind_c, ind_h, ind_w, gdsdn, gdsdc, gdsdh, gdsdw, gd_i*(dep_at_pix_i - cos_theta_cc)/(ptnum*dist));
      //}
      //else{
      //SET_DIRECT_4d(grad_depth, n, ind_c, ind_h, ind_w, gdsdn, gdsdc, gdsdh, gdsdw, 0);
      //}
  }
}

int spherical_back_proj_forward_wrap(THCState* state, THCudaTensor* depth, THCudaTensor* grid_in, THCudaTensor* voxel, THCudaTensor* cnt){
  
  THCUNN_assertSameGPU(state, 4, depth, grid_in, voxel, cnt);
  cambp_shapecheck(state, depth,grid_in, voxel, cnt);
  int N = THCudaTensor_size(state, depth, 0);
  int NC = THCudaTensor_size(state, depth, 1);
  int dszh = THCudaTensor_size(state, depth, 2);
  int dszw = THCudaTensor_size(state, depth, 3);
  int vszx = THCudaTensor_size(state, voxel, 2);
  int vszy = THCudaTensor_size(state, voxel, 3);
  int vszz = THCudaTensor_size(state, voxel, 4);
  THCudaTensor_resize5d(state, voxel, N, NC, vszx, vszy, vszz);
  THCudaTensor_resize5d(state, cnt, N, NC, vszx, vszy, vszz);
  THCudaTensor_zero(state, cnt);
  int count_im = (N * NC* dszh * dszw);
  int count_vox = (N*NC*vszx*vszy*vszz);
  
  spherical_back_projection_forward_kernel
    <<<GET_BLOCKS(count_im), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, depth),
    N,NC, dszh, dszw,
    THCudaTensor_stride(state, depth, 0),
    THCudaTensor_stride(state, depth, 1),
    THCudaTensor_stride(state, depth, 2),
    THCudaTensor_stride(state, depth, 3),
    THCudaTensor_data(state,grid_in),
    THCudaTensor_stride(state, grid_in, 0),
    THCudaTensor_stride(state, grid_in, 1),
    THCudaTensor_stride(state, grid_in, 2),
    THCudaTensor_stride(state, grid_in, 3),
    THCudaTensor_stride(state, grid_in, 4),
    THCudaTensor_data(state,voxel),
    vszx, vszy, vszz,
    THCudaTensor_stride(state, voxel, 0),
    THCudaTensor_stride(state, voxel, 1),
    THCudaTensor_stride(state, voxel, 2),
    THCudaTensor_stride(state, voxel, 3),
    THCudaTensor_stride(state, voxel, 4),
    THCudaTensor_data(state,cnt),
    THCudaTensor_stride(state, cnt, 0),
    THCudaTensor_stride(state, cnt, 1),
    THCudaTensor_stride(state, cnt, 2),
    THCudaTensor_stride(state, cnt, 3),
    THCudaTensor_stride(state, cnt, 4),
    count_im);
  cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess) {
    printf("error in projection foward: %s\n", cudaGetErrorString(err));
    return 0;
  }
 //fprintf(stderr,"calling divide\n");
  inplace_safe_divide
    <<<GET_BLOCKS(count_vox), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, voxel),
    N, NC, vszx, vszy, vszz,
    THCudaTensor_stride(state, voxel, 0),
    THCudaTensor_stride(state, voxel, 1),
    THCudaTensor_stride(state, voxel, 2),
    THCudaTensor_stride(state, voxel, 3),
    THCudaTensor_stride(state, voxel, 4),
    THCudaTensor_data(state, cnt),
    THCudaTensor_stride(state, cnt, 0),
    THCudaTensor_stride(state, cnt, 1),
    THCudaTensor_stride(state, cnt, 2),
    THCudaTensor_stride(state, cnt, 3),
    THCudaTensor_stride(state, cnt, 4),
    0.0f,
    count_vox);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in inplace safe divide: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}
int spherical_back_proj_backward_wrap(THCState* state, THCudaTensor* depth, THCudaTensor* grid_in, THCudaTensor* cnt, THCudaTensor* grad_in, THCudaTensor* grad_depth){
  THCUNN_assertSameGPU(state, 5, depth, grid_in, cnt, grad_in, grad_depth);
  
  int N = THCudaTensor_size(state, depth, 0);
  int NC = THCudaTensor_size(state, depth, 1);
  int dszh = THCudaTensor_size(state, depth, 2);
  int dszw = THCudaTensor_size(state, depth, 3);
  int vszx = THCudaTensor_size(state, cnt, 2);
  int vszy = THCudaTensor_size(state, cnt, 3);
  int vszz = THCudaTensor_size(state, cnt, 4);
  int count_im = (N * NC* dszh * dszw);
  
  spherical_back_projection_backward_kernel
    <<<GET_BLOCKS(count_im), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, depth),
    N,NC, dszh, dszw,
    THCudaTensor_stride(state, depth, 0),
    THCudaTensor_stride(state, depth, 1),
    THCudaTensor_stride(state, depth, 2),
    THCudaTensor_stride(state, depth, 3),
    THCudaTensor_data(state,grid_in),
    THCudaTensor_stride(state, grid_in, 0),
    THCudaTensor_stride(state, grid_in, 1),
    THCudaTensor_stride(state, grid_in, 2),
    THCudaTensor_stride(state, grid_in, 3),
    THCudaTensor_stride(state, grid_in, 4),
    THCudaTensor_data(state,cnt),
    vszx, vszy, vszz,
    THCudaTensor_stride(state, cnt, 0),
    THCudaTensor_stride(state, cnt, 1),
    THCudaTensor_stride(state, cnt, 2),
    THCudaTensor_stride(state, cnt, 3),
    THCudaTensor_stride(state, cnt, 4),
    THCudaTensor_data(state, grad_in),
    vszx, vszy, vszz,
    THCudaTensor_stride(state, grad_in, 0),
    THCudaTensor_stride(state, grad_in, 1),
    THCudaTensor_stride(state, grad_in, 2),
    THCudaTensor_stride(state, grad_in, 3),
    THCudaTensor_stride(state, grad_in, 4),
    THCudaTensor_data(state, grad_depth),
    THCudaTensor_stride(state, grad_depth, 0),
    THCudaTensor_stride(state, grad_depth, 1),
    THCudaTensor_stride(state, grad_depth, 2),
    THCudaTensor_stride(state, grad_depth, 3),
    count_im);
  
  cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess) {
    printf("error in projection foward: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}


int back_projection_forward_wrap(THCState* state, THCudaTensor* depth, THCudaTensor* camdist, THCudaTensor* fl, THCudaTensor* voxel, THCudaTensor* cnt){
  //fprintf(stderr,"calling cuda!!\n");
  THCUNN_assertSameGPU(state, 5, depth, camdist, fl, voxel, cnt);
  cambp_shapecheck(state, depth, camdist, fl, voxel, cnt);
  int N = THCudaTensor_size(state, depth, 0);
  int NC = THCudaTensor_size(state, depth, 1);
  int dszh = THCudaTensor_size(state, depth, 2);
  int dszw = THCudaTensor_size(state, depth, 3);
  int vszx = THCudaTensor_size(state, voxel, 2);
  int vszy = THCudaTensor_size(state, voxel, 3);
  int vszz = THCudaTensor_size(state, voxel, 4);
  THCudaTensor_resize5d(state, voxel, N, NC, vszx, vszy, vszz);
  THCudaTensor_resize5d(state, cnt, N, NC, vszx, vszy, vszz);
  THCudaTensor_zero(state, cnt);
  int count_im = (N * NC* dszh * dszw);
  int count_vox = (N*NC*vszx*vszy*vszz);
  //fprintf(stderr,"calling forawrd\n");

 back_projection_forward_kernel
    <<<GET_BLOCKS(count_im), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, depth),
    N,NC, dszh, dszw,
    THCudaTensor_stride(state, depth, 0),
    THCudaTensor_stride(state, depth, 1),
    THCudaTensor_stride(state, depth, 2),
    THCudaTensor_stride(state, depth, 3),
    THCudaTensor_data(state, camdist),
    THCudaTensor_stride(state,camdist, 0),
    THCudaTensor_stride(state,camdist, 1),
    THCudaTensor_data(state,fl),
    THCudaTensor_stride(state, fl, 0),
    THCudaTensor_stride(state, fl, 1),
    THCudaTensor_data(state,voxel),
    vszx, vszy, vszz,
    THCudaTensor_stride(state, voxel, 0),
    THCudaTensor_stride(state, voxel, 1),
    THCudaTensor_stride(state, voxel, 2),
    THCudaTensor_stride(state, voxel, 3),
    THCudaTensor_stride(state, voxel, 4),
    THCudaTensor_data(state,cnt),
    THCudaTensor_stride(state, cnt, 0),
    THCudaTensor_stride(state, cnt, 1),
    THCudaTensor_stride(state, cnt, 2),
    THCudaTensor_stride(state, cnt, 3),
    THCudaTensor_stride(state, cnt, 4),
    count_im);
 cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess) {
    printf("error in projection foward: %s\n", cudaGetErrorString(err));
    return 0;
  }

 //fprintf(stderr,"calling divide\n");

  inplace_safe_divide
    <<<GET_BLOCKS(count_vox), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, voxel),
    N, NC, vszx, vszy, vszz,
    THCudaTensor_stride(state, voxel, 0),
    THCudaTensor_stride(state, voxel, 1),
    THCudaTensor_stride(state, voxel, 2),
    THCudaTensor_stride(state, voxel, 3),
    THCudaTensor_stride(state, voxel, 4),
    THCudaTensor_data(state, cnt),
    THCudaTensor_stride(state, cnt, 0),
    THCudaTensor_stride(state, cnt, 1),
    THCudaTensor_stride(state, cnt, 2),
    THCudaTensor_stride(state, cnt, 3),
    THCudaTensor_stride(state, cnt, 4),
    1.0f,
    count_vox);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in inplace safe divide: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

int get_surface_mask_wrap(THCState* state, THCudaTensor* depth, THCudaTensor* camdist, THCudaTensor* fl, THCudaTensor* cnt, THCudaTensor* mask){
  
  THCUNN_assertSameGPU(state, 5, depth, camdist, fl, mask, cnt);
  cambp_shapecheck(state, depth, camdist, fl, mask, cnt);
  int N = THCudaTensor_size(state, depth, 0);
  int NC = THCudaTensor_size(state, depth, 1);
  int dszh = THCudaTensor_size(state, depth, 2);
  int dszw = THCudaTensor_size(state, depth, 3);
  int vszx = THCudaTensor_size(state, mask, 2);
  int vszy = THCudaTensor_size(state, mask, 3);
  int vszz = THCudaTensor_size(state, mask, 4);
  THCudaTensor_resize5d(state, mask, N, NC, vszx, vszy, vszz);
  THCudaTensor_resize5d(state, cnt, N, NC, vszx, vszy, vszz);
  THCudaTensor_fill(state, mask, 1.0);
  
  int count_vox = (N*NC*vszx*vszy*vszz);
  //fprintf(stderr,"calling forawrd\n");
 get_surface_mask_kernel
    <<<GET_BLOCKS(count_vox), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, depth),
    N, NC, dszh, dszw,
    THCudaTensor_stride(state, depth, 0),
    THCudaTensor_stride(state, depth, 1),
    THCudaTensor_stride(state, depth, 2),
    THCudaTensor_stride(state, depth, 3),
    THCudaTensor_data(state, camdist),
    THCudaTensor_stride(state,camdist, 0),
    THCudaTensor_stride(state,camdist, 1),
    THCudaTensor_data(state,fl),
    THCudaTensor_stride(state, fl, 0),
    THCudaTensor_stride(state, fl, 1),
    THCudaTensor_data(state,cnt),
    vszx, vszy, vszz,
    THCudaTensor_stride(state,cnt, 0),
    THCudaTensor_stride(state, cnt, 1),
    THCudaTensor_stride(state, cnt, 2),
    THCudaTensor_stride(state, cnt, 3),
    THCudaTensor_stride(state, cnt, 4),
    THCudaTensor_data(state,mask),
    THCudaTensor_stride(state, mask, 0),
    THCudaTensor_stride(state, mask, 1),
    THCudaTensor_stride(state, mask, 2),
    THCudaTensor_stride(state, mask, 3),
    THCudaTensor_stride(state, mask, 4),
    count_vox);
 cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess) {
    printf("error in projection foward: %s\n", cudaGetErrorString(err));
    return 0;
  }
 return 1;
}



//backward

int back_projection_backward_wrap (THCState* state, THCudaTensor* depth, THCudaTensor* fl, THCudaTensor* camdist, THCudaTensor* cnt, THCudaTensor* grad_in, THCudaTensor* grad_depth, THCudaTensor* grad_camdist, THCudaTensor* grad_fl ){
  THCUNN_assertSameGPU(state, 7, depth, fl, cnt, grad_in, grad_depth, grad_camdist, grad_fl);
  int N = THCudaTensor_size(state, depth, 0);
  int NC = THCudaTensor_size(state, depth, 1);
  int dszh = THCudaTensor_size(state, depth, 2);
  int dszw = THCudaTensor_size(state, depth, 3);
  int cszx = THCudaTensor_size(state, cnt, 2);
  int cszy = THCudaTensor_size(state, cnt, 3);
  int cszz = THCudaTensor_size(state, cnt, 4);
  THCudaTensor_resize4d(state, grad_depth, N, NC, dszh, dszw);
  THCudaTensor_resize2d(state, grad_camdist, N, NC);
  THCudaTensor_resize2d(state, grad_fl, N, NC);
  THCudaTensor_zero(state, grad_depth);
  THCudaTensor_zero(state, grad_camdist);
  THCudaTensor_zero(state, grad_fl);
  int count_im = (N * NC* dszh * dszw);

  back_projection_backward_kernel
    <<<GET_BLOCKS(count_im), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
                                                                                   THCudaTensor_data(state, depth),
                                                                                   N, NC,
                                                                                   dszh, dszw,
                                                                                   THCudaTensor_stride(state, depth, 0),
                                                                                   THCudaTensor_stride(state, depth, 1),
                                                                                   THCudaTensor_stride(state, depth, 2),
                                                                                   THCudaTensor_stride(state, depth, 3),
                                                                                   THCudaTensor_data(state, fl),
                                                                                   THCudaTensor_stride(state, fl, 0),
                                                                                   THCudaTensor_stride(state, fl, 1),
                                                                                   THCudaTensor_data(state, camdist),
                                                                                   THCudaTensor_stride(state, camdist, 0),
                                                                                   THCudaTensor_stride(state, camdist, 1),
                                                                                   THCudaTensor_data(state, cnt),
                                                                                   cszx, cszy, cszz,
                                                                                   THCudaTensor_stride(state, cnt, 0),
                                                                                   THCudaTensor_stride(state, cnt, 1),
                                                                                   THCudaTensor_stride(state, cnt, 2),
                                                                                   THCudaTensor_stride(state, cnt, 3),
                                                                                   THCudaTensor_stride(state, cnt, 4),
                                                                                   THCudaTensor_data(state, grad_in),
                                                                                   cszx,cszy,cszz,
                                                                                   THCudaTensor_stride(state, grad_in, 0),
                                                                                   THCudaTensor_stride(state, grad_in, 1),
                                                                                   THCudaTensor_stride(state, grad_in, 2),
                                                                                   THCudaTensor_stride(state, grad_in, 3),
                                                                                   THCudaTensor_stride(state, grad_in, 4),
                                                                                   THCudaTensor_data(state, grad_depth),
                                                                                   THCudaTensor_stride(state, grad_depth, 0),
                                                                                   THCudaTensor_stride(state, grad_depth, 1),
                                                                                   THCudaTensor_stride(state, grad_depth, 2),
                                                                                   THCudaTensor_stride(state, grad_depth, 3),
                                                                                   THCudaTensor_data(state, grad_camdist),
                                                                                   THCudaTensor_stride(state,grad_camdist, 0),
                                                                                   THCudaTensor_stride(state,grad_camdist, 1),
                                                                                   THCudaTensor_data(state, grad_fl),
                                                                                   THCudaTensor_stride(state,grad_fl, 0),
                                                                                   THCudaTensor_stride(state,grad_fl, 1),
                                                                                   count_im);

                    
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler3D update gradInput: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
 }
 
