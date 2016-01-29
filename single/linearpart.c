__kernel void linearpart ( __global float* uhatre,__global float* uhatim,__global float* vhatre,__global float* vhatim,__global const float* Kx,__global const float* Ky,__global const float* Kz, const float dt, const float Du,const float Dv, const float A,const float B,const float bhighre,const float bhighim , const int Nx, const int Ny, const int Nz)
{
   const int ind = get_global_id(0);

int i,j,k;
k=floor((float)ind/(float)(Ny*Nx));
j=floor((float)(ind-k*(Ny*Nx))/(float)Nx);
i=ind-k*(Ny*Nx)-j*Nx;
float uexp[2];
uexp[0]=dt*bhighre*(-1.0*A+Du*(Kx[i]+Ky[j]+Kz[k]));
uexp[1]=dt*bhighim*(-1.0*A+Du*(Kx[i]+Ky[j]+Kz[k]));
float vexp[2];
vexp[0]=dt*bhighre*(-1.0*B+Dv*(Kx[i]+Ky[j]+Kz[k]));
vexp[1]=dt*bhighim*(-1.0*B+Dv*(Kx[i]+Ky[j]+Kz[k]));
if(ind==0){
float N=(float)Nx*Ny*Nz;

uhatre[ind]=exp(uexp[0])*(((uhatre[ind]-N)*cos(uexp[1]))-(uhatim[ind]*sin(uexp[1])))+N;
uhatim[ind]=exp(uexp[0])*(((uhatre[ind]-N)*sin(uexp[1]))+(uhatim[ind]*cos(uexp[1])));
}

else{
uhatre[ind]=exp(uexp[0])*((uhatre[ind]*cos(uexp[1]))-(uhatim[ind]*sin(uexp[1])));
uhatim[ind]=exp(uexp[0])*((uhatre[ind]*sin(uexp[1]))+(uhatim[ind]*cos(uexp[1])));
}
vhatre[ind]=exp(vexp[0])*((vhatre[ind]*cos(vexp[1]))-(vhatim[ind]*sin(vexp[1])));
vhatim[ind]=exp(vexp[0])*((vhatre[ind]*sin(vexp[1]))+(vhatim[ind]*cos(vexp[1])));
}
