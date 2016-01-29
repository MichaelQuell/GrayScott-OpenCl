    #if defined(cl_amd_fp64) || defined(cl_khr_fp64)

     

    #if defined(cl_amd_fp64)

    #pragma OPENCL EXTENSION cl_amd_fp64 : enable

    #elif defined(cl_khr_fp64)

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    #endif

     

    // function declarations/definitions using double precision doubleing-point arithmetic

     

    #endif
__kernel
void reduceerr(__global double* buffer,
            __local double* scratch,
            __const int length,
            __global double* result) {

  int global_index = get_global_id(0);
  double accumulator = 10.0;
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    double element = buffer[global_index];
    accumulator = (accumulator < element) ? accumulator : element;
    global_index += get_global_size(0);
  }

  // Perform parallel reduction
  int local_index = get_local_id(0);
  scratch[local_index] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      double other = scratch[local_index + offset];
      double mine = scratch[local_index];
      scratch[local_index] = (mine < other) ? mine : other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}
