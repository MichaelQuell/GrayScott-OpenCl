__kernel void nonlinearpart ( __global float* ure,__global float* uim,__global float* vre,__global float* vim, const float dt, const float are,const float aim)
{
   const int ind = get_global_id(0);
	const float tol=pown(0.1,6);//0.000000000001;
float chg=1;
const float uoldre=ure[ind];
const float uoldim=uim[ind];
const float voldre=vre[ind];
const float voldim=vim[ind];
float utempre,utempim,vtempre,vtempim;
float uMre,uMim,vMre,vMim;
while(chg>tol){
utempre=ure[ind];
utempim=uim[ind];
vtempre=vre[ind];
vtempim=vim[ind];	

uMre=0.5*(ure[ind]+uoldre);
uMim=0.5*(uim[ind]+uoldim);
vMre=0.5*(vre[ind]+voldre);
vMim=0.5*(vim[ind]+voldim);								
							
ure[ind]=uoldre-dt*are*uMre*vMre*vMre+dt*are*uMre*vMim*vMim+2.0*dt*are*uMim*vMre*vMim+2.0*dt*aim*uMre*vMre*vMim+dt*aim*uMim*vMre*vMre-dt*aim*uMim*vMim*vMim;

uim[ind]=uoldim-dt*aim*uMre*vMre*vMre+dt*are*uMim*vMim*vMim+2.0*dt*aim*uMim*vMre*vMim-2.0*dt*are*uMre*vMre*vMim-dt*are*uMim*vMre*vMre+dt*aim*uMre*vMim*vMim;

vre[ind]=voldre+dt*are*uMre*vMre*vMre-dt*are*uMre*vMim*vMim-2.0*dt*are*uMim*vMre*vMim-2.0*dt*aim*uMre*vMre*vMim-dt*aim*uMim*vMre*vMre+dt*aim*uMim*vMim*vMim;

vim[ind]=voldim+dt*aim*uMre*vMre*vMre-dt*are*uMim*vMim*vMim-2.0*dt*aim*uMim*vMre*vMim+2.0*dt*are*uMre*vMre*vMim+dt*are*uMim*vMre*vMre-dt*aim*uMre*vMim*vMim;
							
chg=sqrt((ure[ind]-utempre)*(ure[ind]-utempre)+(uim[ind]-utempim)*(uim[ind]-utempim))+
    sqrt((vre[ind]-vtempre)*(vre[ind]-vtempre)+(vim[ind]-vtempim)*(vim[ind]-vtempim));
}   	
}
