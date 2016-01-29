#if defined(cl_amd_fp64) || defined(cl_khr_fp64)
    #if defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #elif defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
    // function declarations/definitions using double precision doubleing-point arithmetic
#endif
__kernel void nonlinearpart ( __global double2* ure,__global double2* vre, const double dt, const double are)
{
   const int ind = get_global_id(0);
	const double tol=pown(0.1,12);//0.000000000001;
double chg=1;
const double uoldre=ure[ind].x;

const double voldre=vre[ind].x;

double utempre,vtempre;
double uMre,vMre;
while(chg>tol){
utempre=ure[ind].x;

vtempre=vre[ind].x;


uMre=0.5*(ure[ind].x+uoldre);
vMre=0.5*(vre[ind].x+voldre);
							
ure[ind].x=uoldre-dt*are*uMre*vMre*vMre;

vre[ind].x=voldre+dt*are*uMre*vMre*vMre;
							
chg=sqrt((ure[ind].x-utempre)*(ure[ind].x-utempre))+
    sqrt((vre[ind].x-vtempre)*(vre[ind].x-vtempre));
} 
ure[ind].y=0.0;
vre[ind].y=0.0;  	
}
