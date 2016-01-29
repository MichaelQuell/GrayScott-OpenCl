__kernel void grid ( __global float* x, const float Lx, const int Nx)
{
  const int ind = get_global_id(0);
	x[ind]=(-1.0 + ((float) 2.0*ind/(float)Nx))*M_PI*Lx;   	
	
}
