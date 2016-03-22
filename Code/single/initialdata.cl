__kernel void initialdata ( __global float* ure, __global float* vre, __global float* uim, __global float* vim, __global const float* x ,__global const float* y ,__global const float* z , const int Nx, const int Ny, const int Nz)
{
   const int ind = get_global_id(0);

int i,j,k;
k=floor((float)ind/(float)(Ny*Nx));
j=floor((float)(ind-k*(Ny*Nx))/(float)Nx);
i=ind-k*(Ny*Nx)-j*Nx;
ure[ind]=0.5+exp(-1.0*(x[i]*x[i]+y[j]*y[j] )-1.0);//+z[k]*z[k] 	
vre[ind]=0.1+exp(-1.0*(x[i]*x[i]+y[j]*y[j] )-1.0);
uim[ind]=0.0;
vim[ind]=0.0;
}
