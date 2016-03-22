#if defined(cl_amd_fp64) || defined(cl_khr_fp64)
    #if defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #elif defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
    // function declarations/definitions using double precision doubleing-point arithmetic
#endif
__kernel void linearpart ( __global double2* uhat, __global double2* vhat,__global const double* Kx,__global const double* Ky, const double dt, const double Du,const double Dv, const double A,const double B,const double bhighre, const int Nx, const int Ny,const int myid)
{
   const int ind = get_global_id(0);

int i,j;
i=floor((double)(ind)/(double)Ny);
j=ind-i*Ny;
double uexp;
uexp=dt*bhighre*(-1.0*A+Du*(Kx[i]+Ky[j]));
double vexp;
vexp=dt*bhighre*(-1.0*B+Dv*(Kx[i]+Ky[j]));
if(ind==0&&myid==0){
double N=(double)Nx*Ny;

uhat[ind].x=exp(uexp)*(uhat[ind].x-N)+N;
uhat[ind].y=exp(uexp)*uhat[ind].y;
}

else{
uhat[ind].x=exp(uexp)*uhat[ind].x;
uhat[ind].y=exp(uexp)*uhat[ind].y;
}
vhat[ind].x=exp(vexp)*vhat[ind].x;
vhat[ind].y=exp(vexp)*vhat[ind].y;
}
