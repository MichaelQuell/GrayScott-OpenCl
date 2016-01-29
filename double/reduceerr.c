    #if defined(cl_amd_fp64) || defined(cl_khr_fp64)

     

    #if defined(cl_amd_fp64)

    #pragma OPENCL EXTENSION cl_amd_fp64 : enable

    #elif defined(cl_khr_fp64)

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    #endif

     

    // function declarations/definitions using double precision doubleing-point arithmetic

     

    #endif
//doesnt work yet
__kernel
void reduce(
            __global float* buffer,
            __local float* scratch,
            __const int length,
            __global float* result) {

  int global_index = get_global_id(0);
  int local_index = get_local_id(0);
  // Load data into local memory
  if (global_index < length) {
    scratch[local_index] = buffer[global_index];
  } else {
    // Infinity is the identity element for the min operation
    scratch[local_index] = 0.0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = 1;
      offset < get_local_size(0);
      offset <<= 1) {
    int mask = (offset << 1) - 1;
    if ((local_index & mask) == 0) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = (mine < other) other ?  : mine;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}
