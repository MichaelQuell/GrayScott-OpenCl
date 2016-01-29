
//#include <CL/cl.h>
//#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "gsh.c"

#if defined(cl_amd_fp64) || defined(cl_khr_fp64)
    #if defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #elif defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
    // function declarations/definitions using double precision floating-point arithmetic
#endif

int main(argc,argv) 
	int argc;
	char *argv[];{
//mpi
    int myid, numprocs;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
//store memorylayout
	int xdiff,ydiff,xoff,yoff;
//time meassuring
  	struct timeval tvs;
  	struct timeval tve;
    	double elapsedTime;
//for loop indices
	int  n=0,i=0,j=0;
//Parameter
	int	  Nx,Ny;
	int	  Tmax=0;
	int 	  plottime=0,plotnum=0,plotgap=0;
	double	  Lx,Ly;
	double	  dt=0.0;
	double	  Du=0.0, Dv=0.0;	
	double	  A=0.0,B=0.0;
//Splitting coeffizienten
	double	  a=1.0, b=0.5;
//arrays on cpu 
	double*	  T1,*T2;
//Read infutfile
	parainit(&Nx,&Ny,&Tmax,&plotgap,&Lx,&Ly,&dt,&Du,&Dv,&A,&B);
//openCL variables
    	cl_platform_id *platform_id = NULL;
    	cl_kernel frequencies = NULL, initialdata = NULL, linearpart=NULL, nonlinearpart=NULL, transposeF1=NULL, transposeF2=NULL;
    	cl_int ret;
    	cl_uint num_platforms;
// Detect how many platforms there are.
	ret = clGetPlatformIDs(0, NULL, &num_platforms);
// Allocate enough space for the number of platforms.
	platform_id = (cl_platform_id*) malloc(num_platforms*sizeof(cl_platform_id));
// Store the platforms
	ret = clGetPlatformIDs(num_platforms, platform_id, NULL);
	printf("Proc %d found %d platform(s)!\n",myid,num_platforms);
    	cl_uint *num_devices;
	num_devices=(cl_uint*) malloc(num_platforms*sizeof(cl_uint));
    	cl_device_id **device_id = NULL;
	device_id =(cl_device_id**) malloc(num_platforms*sizeof(cl_device_id*));
// Detect number of devices in the platforms
	for(i=0;i<num_platforms;i++){
		char buf[65536];
		size_t size;
		ret = clGetPlatformInfo(platform_id[i],CL_PLATFORM_VERSION,sizeof(buf),buf,&size);
		printf("%s\n",buf);
		ret = clGetDeviceIDs(platform_id[i],CL_DEVICE_TYPE_ALL,0,NULL,num_devices);
		printf("Proc %d found %d device(s) on platform %d!\n",myid, num_devices[i],i);
		ret = clGetPlatformInfo(platform_id[i],CL_PLATFORM_NAME,sizeof(buf),buf,&size);
		printf("%s ",buf);
// Store numDevices from platform
		device_id[i]=(cl_device_id*) malloc(num_devices[i]*sizeof(device_id));
		ret = clGetDeviceIDs(platform_id[i],CL_DEVICE_TYPE_ALL,num_devices[i],device_id[i],NULL);
		for(j=0;j<num_devices[i];j++){
			ret = clGetDeviceInfo(device_id[i][j],CL_DEVICE_NAME,sizeof(buf),buf,&size);
			printf("%s (%d,%d)\n",buf,i,j);
	}}
//create context and command_queue
    	cl_context context = NULL;
   	cl_command_queue command_queue = NULL;
//Which platform and device do i choose?
	int	chooseplatform=0;
	int	choosedevice=0;	  
	printf("Proc %d choose platform %d and device %d!\n",myid,chooseplatform,choosedevice);
	context = clCreateContext( NULL, num_devices[chooseplatform], device_id[chooseplatform], NULL, NULL, &ret);
	if(ret!=CL_SUCCESS){printf("createContext ret:%d,%d\n",ret,myid); exit(1); }
	command_queue = clCreateCommandQueue(context, device_id[chooseplatform][choosedevice], 0, &ret);
	if(ret!=CL_SUCCESS){printf("createCommandQueue ret:%d,%d\n",ret,myid); exit(1); }
//OpenCL arrays
    cl_mem cl_u = NULL,cl_v = NULL;
   	cl_mem cl_uhat = NULL, cl_vhat = NULL;
    cl_mem cl_kx = NULL, cl_ky = NULL;
	cl_mem cl_T1 = NULL, cl_T2 = NULL;
//Find array widths on each MPI process
	ydiff=Ny/numprocs;
	xdiff=Nx/numprocs;
	xoff=myid*xdiff;
	yoff=myid*ydiff;
	if(myid==0){printf("xdiff=%d, ydiff=%d\n",xdiff,ydiff);}
//FFT
	clfftPlanHandle planHandleX;
	clfftPlanHandle planHandleY;
    cl_mem tmpBufferX = NULL;
    cl_mem tmpBufferY = NULL;
	fftinit(&planHandleX, &planHandleY, &context, &command_queue, &tmpBufferX, &tmpBufferY, Nx, Ny,xdiff,ydiff);

//allocate the memory
	T1=(double*) malloc(2*Nx*ydiff*sizeof(double));
	T2=(double*) malloc(2*Nx*ydiff*sizeof(double));
//allocate gpu memory/
	cl_u=clCreateBuffer(context, CL_MEM_READ_WRITE, 2*Nx* ydiff* sizeof(double), NULL, &ret);
	cl_v=clCreateBuffer(context, CL_MEM_READ_WRITE, 2*Nx* ydiff* sizeof(double), NULL, &ret);
	cl_uhat=clCreateBuffer(context, CL_MEM_READ_WRITE, 2*Nx * ydiff* sizeof(double), NULL, &ret);
	cl_vhat=clCreateBuffer(context, CL_MEM_READ_WRITE, 2*Nx * ydiff* sizeof(double), NULL, &ret);
	cl_T1=clCreateBuffer(context, CL_MEM_READ_WRITE, 2*Nx * ydiff* sizeof(double), NULL, &ret);
	cl_T2=clCreateBuffer(context, CL_MEM_READ_WRITE, 2*Nx * ydiff* sizeof(double), NULL, &ret);
	cl_kx = clCreateBuffer(context, CL_MEM_READ_WRITE, xdiff * sizeof(double), NULL, &ret);
	cl_ky = clCreateBuffer(context, CL_MEM_READ_WRITE, Ny * sizeof(double), NULL, &ret);

	if(myid==0){printf("allocated space\n");}
//load the kernels
	loadKernel(&frequencies,&context,&device_id[chooseplatform][choosedevice],"frequencies");
	loadKernel(&initialdata,&context,&device_id[chooseplatform][choosedevice],"initialdata"); 
	loadKernel(&linearpart,&context,&device_id[chooseplatform][choosedevice],"linearpart"); 
	loadKernel(&nonlinearpart,&context,&device_id[chooseplatform][choosedevice],"nonlinearpart"); 
	loadKernel(&transposeF1,&context,&device_id[chooseplatform][choosedevice],"transposeF1"); 
	loadKernel(&transposeF2,&context,&device_id[chooseplatform][choosedevice],"transposeF2"); 

//inintial data
	ret = clFinish(command_queue);
    ret = clSetKernelArg(initialdata, 0, sizeof(cl_mem),(void *)&cl_u);
	ret = clSetKernelArg(initialdata, 1, sizeof(cl_mem),(void* )&cl_v);
	ret = clSetKernelArg(initialdata, 2, sizeof(int),(void* )&Nx);
	ret = clSetKernelArg(initialdata, 3, sizeof(int),(void* )&Ny);
	ret = clSetKernelArg(initialdata, 4, sizeof(int),(void* )&yoff);
	ret = clSetKernelArg(initialdata, 5, sizeof(double),(void* )&Lx);
	ret = clSetKernelArg(initialdata, 6, sizeof(double),(void* )&Ly);
	size_t global_work_size[1] = {Nx*ydiff};
        ret = clEnqueueNDRangeKernel(command_queue, initialdata, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	ret = clFinish(command_queue);
//make output
//        writedata(&cl_u, &cl_v, &command_queue,T1,T2,Nx,ydiff,plotnum);
    writeimage(&cl_v, &command_queue,T1,Nx,ydiff,plotnum);
	plottime=plotgap;
	if(myid==0){printf("Got initial data\n");}
//get frequencies x note after fft we are transposed	
	const size_t global_work_size_x[1] ={xdiff};
	cl_kx = clCreateBuffer(context, CL_MEM_READ_WRITE, xdiff * sizeof(double), NULL, &ret);
    ret = clSetKernelArg(frequencies, 0, sizeof(cl_mem), (void *)&cl_kx);
	ret = clSetKernelArg(frequencies, 1, sizeof(double),(void*)&Lx);
	ret = clSetKernelArg(frequencies, 2, sizeof(int),(void*)&Nx);
	ret = clSetKernelArg(frequencies, 3, sizeof(int),(void*)&xoff);
        ret = clEnqueueNDRangeKernel(command_queue, frequencies, 1, NULL, global_work_size_x, NULL, 0, NULL, NULL);
	ret = clFinish(command_queue);
//get frequencies y note after fft we are transposed
	int Null=0;
	const size_t global_work_size_y[1] ={Ny};
	cl_ky = clCreateBuffer(context, CL_MEM_READ_WRITE, Ny * sizeof(double), NULL, &ret);	
	ret = clSetKernelArg(frequencies, 0, sizeof(cl_mem), (void *)&cl_ky);
	ret = clSetKernelArg(frequencies, 1, sizeof(double),(void*)&Ly);
	ret = clSetKernelArg(frequencies, 2, sizeof(int),(void*)&Ny);
	ret = clSetKernelArg(frequencies, 3, sizeof(int),(void*)&Null);
	ret = clEnqueueNDRangeKernel(command_queue, frequencies, 1, NULL, global_work_size_y, NULL, 0, NULL, NULL);
	ret = clFinish(command_queue);

	if(myid==0){printf("Setup fourier frequencies\n");}
    	MPI_Barrier(MPI_COMM_WORLD);
 	if(myid==0){printf("starting timestepping!\n");}
  	gettimeofday(&tvs, NULL); 
	for(n=0;n<=Tmax;n++){
//linear
fft2dfor(&cl_u, &cl_uhat, &planHandleX, &planHandleY, &transposeF1, &transposeF2, &context, &command_queue, &tmpBufferX, &tmpBufferY, &cl_T1,&cl_T2, T1, T2, Nx, Ny, xdiff, ydiff);
fft2dfor(&cl_v, &cl_vhat, &planHandleX, &planHandleY, &transposeF1, &transposeF2, &context, &command_queue, &tmpBufferX, &tmpBufferY, &cl_T1,&cl_T2, T1, T2, Nx, Ny, xdiff, ydiff);

    ret = clSetKernelArg(linearpart, 0, sizeof(cl_mem),(void *)&cl_uhat);
    ret = clSetKernelArg(linearpart, 1, sizeof(cl_mem),(void *)&cl_vhat);
	ret = clSetKernelArg(linearpart, 2, sizeof(cl_mem),(void* )&cl_kx);
	ret = clSetKernelArg(linearpart, 3, sizeof(cl_mem),(void* )&cl_ky);
	ret = clSetKernelArg(linearpart, 4, sizeof(double),(void* )&dt);
	ret = clSetKernelArg(linearpart, 5, sizeof(double),(void* )&Du);
	ret = clSetKernelArg(linearpart, 6, sizeof(double),(void* )&Dv);
	ret = clSetKernelArg(linearpart, 7, sizeof(double),(void* )&A);
	ret = clSetKernelArg(linearpart, 8, sizeof(double),(void* )&B);
	ret = clSetKernelArg(linearpart, 9, sizeof(double),(void* )&b);
	ret = clSetKernelArg(linearpart, 10, sizeof(int),(void* )&Nx);
	ret = clSetKernelArg(linearpart, 11, sizeof(int),(void* )&Ny);
	ret = clSetKernelArg(linearpart, 12, sizeof(int),(void* )&myid);
    ret = clEnqueueNDRangeKernel(command_queue, linearpart, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	ret = clFinish(command_queue);

fft2dback(&cl_u, &cl_uhat, &planHandleX, &planHandleY, &transposeF1, &transposeF2, &context, &command_queue, &tmpBufferX, &tmpBufferY, &cl_T1,&cl_T2, T1, T2, Nx, Ny, xdiff, ydiff);
fft2dback(&cl_v, &cl_vhat, &planHandleX, &planHandleY, &transposeF1, &transposeF2, &context, &command_queue, &tmpBufferX, &tmpBufferY, &cl_T1,&cl_T2, T1, T2, Nx, Ny, xdiff, ydiff);
//nonlinearpart

    ret = clSetKernelArg(nonlinearpart, 0, sizeof(cl_mem),(void *)&cl_u);
	ret = clSetKernelArg(nonlinearpart, 1, sizeof(cl_mem),(void* )&cl_v);
	ret = clSetKernelArg(nonlinearpart, 2, sizeof(double),(void* )&dt);
	ret = clSetKernelArg(nonlinearpart, 3, sizeof(double),(void* )&a);
    ret = clEnqueueNDRangeKernel(command_queue, nonlinearpart, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	ret = clFinish(command_queue);	

//linear
fft2dfor(&cl_u, &cl_uhat, &planHandleX, &planHandleY, &transposeF1, &transposeF2, &context, &command_queue, &tmpBufferX, &tmpBufferY, &cl_T1,&cl_T2, T1, T2, Nx, Ny, xdiff, ydiff);
fft2dfor(&cl_v, &cl_vhat, &planHandleX, &planHandleY, &transposeF1, &transposeF2, &context, &command_queue, &tmpBufferX, &tmpBufferY, &cl_T1,&cl_T2, T1, T2, Nx, Ny, xdiff, ydiff);

    ret = clSetKernelArg(linearpart, 0, sizeof(cl_mem),(void *)&cl_uhat);
    ret = clSetKernelArg(linearpart, 1, sizeof(cl_mem),(void *)&cl_vhat);
	ret = clSetKernelArg(linearpart, 2, sizeof(cl_mem),(void* )&cl_kx);
	ret = clSetKernelArg(linearpart, 3, sizeof(cl_mem),(void* )&cl_ky);
	ret = clSetKernelArg(linearpart, 4, sizeof(double),(void* )&dt);
	ret = clSetKernelArg(linearpart, 5, sizeof(double),(void* )&Du);
	ret = clSetKernelArg(linearpart, 6, sizeof(double),(void* )&Dv);
	ret = clSetKernelArg(linearpart, 7, sizeof(double),(void* )&A);
	ret = clSetKernelArg(linearpart, 8, sizeof(double),(void* )&B);
	ret = clSetKernelArg(linearpart, 9, sizeof(double),(void* )&b);
	ret = clSetKernelArg(linearpart, 10, sizeof(int),(void* )&Nx);
	ret = clSetKernelArg(linearpart, 11, sizeof(int),(void* )&Ny);
	ret = clSetKernelArg(linearpart, 12, sizeof(int),(void* )&myid);
    ret = clEnqueueNDRangeKernel(command_queue, linearpart, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	ret = clFinish(command_queue);

fft2dback(&cl_u, &cl_uhat, &planHandleX, &planHandleY, &transposeF1, &transposeF2, &context, &command_queue, &tmpBufferX, &tmpBufferY, &cl_T1,&cl_T2, T1, T2, Nx, Ny, xdiff, ydiff);
fft2dback(&cl_v, &cl_vhat, &planHandleX, &planHandleY, &transposeF1, &transposeF2, &context, &command_queue, &tmpBufferX, &tmpBufferY, &cl_T1,&cl_T2, T1, T2, Nx, Ny, xdiff, ydiff);

// done
	if(n==plottime){
		if(myid==0){printf("time:%lf, step:%d,%d\n",n*dt,n,plotnum);}
		plottime=plottime+plotgap;
		plotnum++;

        //writedata(&cl_u, &cl_v, &command_queue,T1,T2,Nx,ydiff,plotnum);
        writeimage(&cl_v, &command_queue,T1,Nx,ydiff,plotnum);
	}
}

 	gettimeofday(&tve, NULL); 
	if(myid==0){printf("Finished time stepping\n");}
 	elapsedTime = (tve.tv_sec - tvs.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (tve.tv_usec - tvs.tv_usec) / 1000.0;   // us to ms
   	if(myid==0){printf("Program took %lfms for execution\n",elapsedTime);}
//Release all the allocated memory
	free(T1);
	free(T2);

	clReleaseMemObject(cl_u);
	clReleaseMemObject(cl_v);
	clReleaseMemObject(cl_uhat);
	clReleaseMemObject(cl_vhat);
	clReleaseMemObject(cl_kx);
	clReleaseMemObject(cl_ky);
	clReleaseMemObject(cl_T1);
	clReleaseMemObject(cl_T2);
//clean up
	fftdestroy(&planHandleX, &planHandleY, &tmpBufferX, &tmpBufferY);
    ret = clReleaseKernel(frequencies); 
    ret = clReleaseKernel(linearpart); 
    ret = clReleaseKernel(nonlinearpart); 
    ret = clReleaseKernel(transposeF1); 
    ret = clReleaseKernel(transposeF2); 

	ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
	for(i=0;i<num_platforms;i++){free(device_id[i]);}
	free(device_id);
	free(platform_id);
	free(num_devices);	
	
    MPI_Finalize();	
	if(myid==0){printf("Program execution complete\n");}

	return 0;
}
