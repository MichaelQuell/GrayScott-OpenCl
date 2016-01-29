__kernel void frequencies ( __global float* kx, const float Lx, const int Nx)
{
   	const int ind = get_global_id(0);
	if ( ind < Nx/2) kx[ind] = -1.0*((float)((ind))/Lx)*((float)( ind)/Lx);
	if ( ind ==Nx/2) kx[ ind]=0.0;
	if ( ind > Nx/2) kx[ ind]=-1.0*(float)(Nx-ind)/Lx*(float)(Nx-ind)/Lx;
}
