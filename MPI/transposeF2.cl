#if defined(cl_amd_fp64) || defined(cl_khr_fp64)
    #if defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #elif defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
    // function declarations/definitions using double precision doubleing-point arithmetic
#endif
__kernel void transposeF2 ( __global const double2* u,__global double2* v, const int xdiff, const int ydiff, const int Ny)
{	
   	const int ind = get_global_id(0);
  	int n=floor((double)ind/((double)xdiff*ydiff));
	int j=floor((double)(ind-n*xdiff*ydiff)/(double)xdiff);
	int i=ind-n*xdiff*ydiff-j*xdiff;
	v[i*Ny+j+n*ydiff].x=u[ind].x;
	v[i*Ny+j+n*ydiff].y=u[ind].y;
}
