//
//
//
//This file contains only functions for grayscottCLs.c
//
//
#include "clFFT.h"
#include <mpi.h>

//Read the INPUTFILE
void parainit(int * Nx, int * Ny, int * Tmax, int * plotgap, double * Lx, double * Ly, double * dt, double * Du, double * Dv, double * A, double * B){

	int intcomm[4];
	double dpcomm[7];
	int myid;
   	 MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	if(myid == 0){
		char	InputFileName[]="./INPUTFILE";
		FILE*fp;
		fp=fopen(InputFileName,"r");
   		 if(!fp) {fprintf(stderr, "Failed to load IPUTFILE.\n");exit(1);}	 
		int ierr=fscanf(fp, "%d %d %d %d %lf %lf %lf %lf %lf %lf %lf", &intcomm[0],&intcomm[1],&intcomm[2],&intcomm[3],&dpcomm[0],&dpcomm[1],&dpcomm[2],&dpcomm[3],&dpcomm[4],&dpcomm[5],&dpcomm[6]);
		if(ierr!=11){fprintf(stderr, "INPUTFILE corrupted.\n");exit(1);}	
		fclose(fp);
		printf("NX %d\n",intcomm[0]); 
		printf("NY %d\n",intcomm[1]); 
		printf("Tmax %d\n",intcomm[2]);
		printf("plotgap %d\n",intcomm[3]);
		printf("Lx %lf\n",dpcomm[0]);
		printf("Ly %lf\n",dpcomm[1]);
		printf("dt %lf\n",dpcomm[2]);		
		printf("Du %lf\n",dpcomm[3]);
		printf("Dv %lf\n",dpcomm[4]);
		printf("F %lf\n",dpcomm[5]);
		printf("k %lf\n",dpcomm[6]);
	}
        MPI_Bcast(intcomm,4,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(dpcomm,7,MPI_DOUBLE,0,MPI_COMM_WORLD);
	*Nx=intcomm[0];
	*Ny=intcomm[1];
	*Tmax=intcomm[2];
	*plotgap=intcomm[3];
	*Lx=dpcomm[0];
	*Ly=dpcomm[1];
	*dt=dpcomm[2];
	*Du=dpcomm[3];
	*Dv=dpcomm[4];
	*A=dpcomm[5];
	*B=dpcomm[6];
	*B=*A+*B;
	if(myid==0){printf("Read Inputfile\n");}
};

//make plans for FFT
void fftinit(clfftPlanHandle *planHandleX,clfftPlanHandle *planHandleY, cl_context* context, cl_command_queue* command_queue,	cl_mem* tmpBufferX, cl_mem* tmpBufferY, int Nx,int Ny,int xdiff,int ydiff){
	clfftDim dim = CLFFT_1D;
	size_t clLengthX = Nx;
	size_t clLenghtY = Ny;
	cl_int ret=0;
	int myid;
    	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	// Setup clFFT. 
	clfftSetupData fftSetupX;
	clfftSetupData fftSetupY;
	ret = clfftInitSetupData(&fftSetupX);
	if(ret!=CL_SUCCESS){printf("clFFT initX ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftInitSetupData(&fftSetupY);
	if(ret!=CL_SUCCESS){printf("clFFT initY ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftSetup(&fftSetupX);
	if(ret!=CL_SUCCESS){printf("clFFT SetupX ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftSetup(&fftSetupY);
	if(ret!=CL_SUCCESS){printf("clFFT SetupY ret:%d,%d\n",ret,myid);exit(1); }
	// Create a default plan for a complex FFT. 
	ret = clfftCreateDefaultPlan(&*planHandleX, *context, dim, &clLengthX);
	if(ret!=CL_SUCCESS){printf("clFFT PlanX ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftCreateDefaultPlan(&*planHandleY, *context, dim, &clLenghtY);
	if(ret!=CL_SUCCESS){printf("clFFT PlanY ret:%d,%d\n",ret,myid);exit(1); }
	// Set plan parameters. 
	ret = clfftSetPlanPrecision(*planHandleX, CLFFT_DOUBLE);
	if(ret!=CL_SUCCESS){printf("clFFT PrecisionX ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftSetPlanPrecision(*planHandleY, CLFFT_DOUBLE);
	if(ret!=CL_SUCCESS){printf("clFFT PrecisionY ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftSetPlanBatchSize(*planHandleX, (size_t) ydiff );
	if(ret!=CL_SUCCESS){printf("clFFT BatchX ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftSetPlanBatchSize(*planHandleY, (size_t) xdiff );
	if(ret!=CL_SUCCESS){printf("clFFT BatchY ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftSetLayout(*planHandleX, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	if(ret!=CL_SUCCESS){printf("clFFT LayoutX ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftSetLayout(*planHandleY, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	if(ret!=CL_SUCCESS){printf("clFFT LayoutY ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftSetResultLocation(*planHandleX, CLFFT_OUTOFPLACE);
	if(ret!=CL_SUCCESS){printf("clFFT PlaceX ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftSetResultLocation(*planHandleY, CLFFT_OUTOFPLACE);
	if(ret!=CL_SUCCESS){printf("clFFT PlaceY ret:%d,%d\n",ret,myid);exit(1); }
	// Bake the plan. 
	ret = clfftBakePlan(*planHandleX, 1, &*command_queue, NULL, NULL);
	if(ret!=CL_SUCCESS){printf("clFFT BakeX ret:%d,%d\n",ret,myid);exit(1); }
	// Create temporary buffer. 
	// Size of temp buffer. 
	size_t tmpBufferSize = 0;
	ret = clfftGetTmpBufSize(*planHandleX, &tmpBufferSize);
	if ((ret == CL_SUCCESS) && (tmpBufferSize > 0)) {
		*tmpBufferX = clCreateBuffer(*context, CL_MEM_READ_WRITE, tmpBufferSize, NULL, &ret);
		if (ret != CL_SUCCESS){printf("Error with tmpBuffer clCreateBufferX\n");exit(1);}
	}
	ret = clfftBakePlan(*planHandleY, 1, &*command_queue, NULL, NULL);
	if(ret!=CL_SUCCESS){printf("clFFT BakeY ret:%d,%d\n",ret,myid);exit(1); }
	ret = clfftGetTmpBufSize(*planHandleY, &tmpBufferSize);
	if ((ret == CL_SUCCESS) && (tmpBufferSize > 0)) {
		*tmpBufferY = clCreateBuffer(*context, CL_MEM_READ_WRITE, tmpBufferSize, NULL, &ret);
		if (ret != CL_SUCCESS){printf("Error with tmpBuffer clCreateBufferY\n");exit(1);}
	}
};

//destroy plans
void fftdestroy(clfftPlanHandle *planHandleX,clfftPlanHandle *planHandleY,cl_mem* tmpBufferX, cl_mem* tmpBufferY){
	cl_int ret=0;	
	clReleaseMemObject(*tmpBufferX);
	clReleaseMemObject(*tmpBufferY);
	ret = clfftDestroyPlan(&*planHandleX);
	if(ret!=0){printf("Error while destroying fftX");exit(1);}
	ret = clfftDestroyPlan(&*planHandleY);
	if(ret!=0){printf("Error while destroying fftY");exit(1);}
	clfftTeardown();
};

//fft2dfoward
void fft2dfor(cl_mem *cl_u, cl_mem *cl_uhat, clfftPlanHandle *planHandleX, clfftPlanHandle *planHandleY, cl_kernel *transposeF1, cl_kernel *transposeF2, cl_context* context, cl_command_queue* command_queue, cl_mem* tmpBufferX, cl_mem* tmpBufferY, cl_mem *cl_T1,cl_mem *cl_T2, double*T1,double*T2, int Nx,int Ny, int xdiff,int ydiff){
	int ret=0,numprocs;
    	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	ret = clfftEnqueueTransform(*planHandleX, CLFFT_FORWARD, 1, command_queue, 0, NULL, NULL,&*cl_u, &*cl_uhat, *tmpBufferX);
	ret = clFinish(*command_queue);

	const size_t global_work_size[1] ={Nx*ydiff};
    ret = clSetKernelArg(*transposeF1, 0, sizeof(cl_mem), (void *)&*cl_uhat);
    ret = clSetKernelArg(*transposeF1, 1, sizeof(cl_mem), (void *)&*cl_T1);
	ret = clSetKernelArg(*transposeF1, 2, sizeof(int),(void*)&xdiff);
	ret = clSetKernelArg(*transposeF1, 3, sizeof(int),(void*)&ydiff);
	ret = clSetKernelArg(*transposeF1, 4, sizeof(int),(void*)&Nx);
        ret = clEnqueueNDRangeKernel(*command_queue, *transposeF1, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	ret = clFinish(*command_queue);	
  	ret = clEnqueueReadBuffer(*command_queue, *cl_T1, CL_TRUE, 0, 2*Nx *ydiff* sizeof(double), T1, 0, NULL, NULL);
	ret = clFinish(*command_queue);	

    	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Alltoall(T1,ydiff*2*xdiff,MPI_DOUBLE,T2,ydiff*2*xdiff,MPI_DOUBLE,MPI_COMM_WORLD);
    	MPI_Barrier(MPI_COMM_WORLD);

  	ret = clEnqueueWriteBuffer(*command_queue, *cl_T2, CL_TRUE, 0, 2*Ny *xdiff* sizeof(double), T2, 0, NULL, NULL);
	ret = clFinish(*command_queue);	
        ret = clSetKernelArg(*transposeF2, 0, sizeof(cl_mem), (void *)&*cl_T2);
        ret = clSetKernelArg(*transposeF2, 1, sizeof(cl_mem), (void *)&*cl_u);
	ret = clSetKernelArg(*transposeF2, 2, sizeof(int),(void*)&xdiff);
	ret = clSetKernelArg(*transposeF2, 3, sizeof(int),(void*)&ydiff);
	ret = clSetKernelArg(*transposeF2, 4, sizeof(int),(void*)&Ny);
        ret = clEnqueueNDRangeKernel(*command_queue, *transposeF2, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	ret = clFinish(*command_queue);

	ret = clfftEnqueueTransform(*planHandleY, CLFFT_FORWARD, 1, &*command_queue, 0, NULL, NULL,&*cl_u, &*cl_uhat, *tmpBufferY);
	ret = clFinish(*command_queue);
	MPI_Barrier(MPI_COMM_WORLD);
if (ret != CL_SUCCESS){printf("FFT failed%d",ret);}
};

//fft2dback
void fft2dback(cl_mem *cl_u, cl_mem *cl_uhat, clfftPlanHandle *planHandleX, clfftPlanHandle *planHandleY, cl_kernel *transposeF1, cl_kernel *transposeF2, cl_context* context, cl_command_queue* command_queue, cl_mem* tmpBufferX, cl_mem* tmpBufferY, cl_mem *cl_T1,cl_mem *cl_T2, double*T1,double*T2, int Nx,int Ny, int xdiff,int ydiff){
	int ret=0,numprocs;
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	ret = clfftEnqueueTransform(*planHandleY, CLFFT_BACKWARD, 1, command_queue, 0, NULL, NULL,&*cl_uhat, &*cl_u, *tmpBufferY);
	ret = clFinish(*command_queue);	

	const size_t global_work_size[1] ={Ny*xdiff};
    ret = clSetKernelArg(*transposeF1, 0, sizeof(cl_mem), (void *)&*cl_u);
    ret = clSetKernelArg(*transposeF1, 1, sizeof(cl_mem), (void *)&*cl_T1);
	ret = clSetKernelArg(*transposeF1, 2, sizeof(int),(void*)&ydiff);
	ret = clSetKernelArg(*transposeF1, 3, sizeof(int),(void*)&xdiff);
	ret = clSetKernelArg(*transposeF1, 4, sizeof(int),(void*)&Ny);
        ret = clEnqueueNDRangeKernel(*command_queue, *transposeF1, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	ret = clFinish(*command_queue);	
  	ret = clEnqueueReadBuffer(*command_queue, *cl_T1, CL_TRUE, 0, 2*Nx *ydiff* sizeof(double), T1, 0, NULL, NULL);
	ret = clFinish(*command_queue);

    MPI_Barrier(MPI_COMM_WORLD);
	MPI_Alltoall(T1,ydiff*2*xdiff,MPI_DOUBLE,T2,ydiff*2*xdiff,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

  	ret = clEnqueueWriteBuffer(*command_queue, *cl_T2, CL_TRUE, 0, 2*Ny *xdiff* sizeof(double), T2, 0, NULL, NULL);
	ret = clFinish(*command_queue);	
    ret = clSetKernelArg(*transposeF2, 0, sizeof(cl_mem), (void *)&*cl_T2);
    ret = clSetKernelArg(*transposeF2, 1, sizeof(cl_mem), (void *)&*cl_uhat);
	ret = clSetKernelArg(*transposeF2, 2, sizeof(int),(void*)&ydiff);
	ret = clSetKernelArg(*transposeF2, 3, sizeof(int),(void*)&xdiff);
	ret = clSetKernelArg(*transposeF2, 4, sizeof(int),(void*)&Nx);
    ret = clEnqueueNDRangeKernel(*command_queue, *transposeF2, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
	ret = clFinish(*command_queue);
	ret = clfftEnqueueTransform(*planHandleX, CLFFT_BACKWARD, 1, &*command_queue, 0, NULL, NULL,&*cl_uhat, &*cl_u, *tmpBufferX);
	ret = clFinish(*command_queue);
	MPI_Barrier(MPI_COMM_WORLD);
	if (ret != CL_SUCCESS){printf("FFT failed%d",ret);}
};

//loadKernel                                                                                                                                                                                                 
#define MAX_SOURCE_SIZE 8192
void loadKernel(cl_kernel *kernel,cl_context *context, cl_device_id *device_id, char*name){
        cl_program p_kernel;
        cl_int ret=0;
        int myid,i;
        MPI_Comm_rank(MPI_COMM_WORLD,&myid);
        size_t source_size;
        char *source_str;
        char nameconfig[100];
    MPI_Barrier(MPI_COMM_WORLD);
        source_str = (char *)malloc(MAX_SOURCE_SIZE*sizeof(char));
        for(i=0;i<MAX_SOURCE_SIZE;i++){source_str[i]='\0';}
    MPI_Barrier(MPI_COMM_WORLD);
        if(myid == 0){
                FILE* fp;
                strcpy(nameconfig,"./");
                strcat(nameconfig,name);
                strcat(nameconfig,".cl");
                fp = fopen(nameconfig, "r");
                if (!fp) {fprintf(stderr, "Failed to load kernel.\n"); exit(1); }
                source_size = fread( source_str, sizeof(char), MAX_SOURCE_SIZE, fp );
                fclose( fp );
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&source_size,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(source_str,source_size,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
        p_kernel = clCreateProgramWithSource(*context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	if(ret!=CL_SUCCESS){printf("createProgram ret:%d,%d\n",ret,myid);exit(1); }
        ret = clBuildProgram(p_kernel, 1, &*device_id, NULL, NULL, NULL);
        if(ret!=CL_SUCCESS){printf("buildProgram ret:%d,%d\n",ret,myid); exit(1); }
        *kernel = clCreateKernel(p_kernel, name, &ret);
        if(ret!=CL_SUCCESS){printf("createKernel ret:%d,%d\n",ret,myid);exit(1); }
        ret = clReleaseProgram(p_kernel);
        if(ret!=CL_SUCCESS){printf("releaseProgram ret:%d,%d\n",ret,myid);exit(1); }
        if(myid==0){printf("got kernel %s\n",name);}
    MPI_Barrier(MPI_COMM_WORLD);
        free(source_str);
    MPI_Barrier(MPI_COMM_WORLD);
};

//write data
void writedata(cl_mem* cl_u, cl_mem *cl_v, cl_command_queue *command_queue,double *T1, double *T2, int Nx,int ydiff, int plotnum){ 
	int i=0,myid;
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	int ret=0;
	ret = clEnqueueReadBuffer(*command_queue, *cl_u, CL_TRUE, 0, 2*Nx*ydiff * sizeof(double), T1, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(*command_queue, *cl_v, CL_TRUE, 0, 2*Nx*ydiff * sizeof(double), T2, 0, NULL, NULL);
	ret = clFinish(*command_queue);
	if(ret!=0){printf("Error hahah");}
//output of data U

	for(i=0;i<Nx*ydiff;i++){
		T1[i]=T1[2*i];
		T2[i]=T2[2*i];}
	MPI_File fp;

	char tmp_str[10];
	char nameconfig[100];
	strcpy(nameconfig,"./data/u");
	sprintf(tmp_str,"%d",10000000+plotnum);
	strcat(nameconfig,tmp_str);
	strcat(nameconfig,".datbin");
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_File_open(MPI_COMM_WORLD, nameconfig,  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_Barrier(MPI_COMM_WORLD);
  /* Set the file view */
  	MPI_File_set_view(fp, myid*Nx*ydiff*sizeof(double), MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
	MPI_Barrier(MPI_COMM_WORLD);
  /* Write buf to the file */
  	MPI_File_write(fp, T1, Nx*ydiff, MPI_DOUBLE, MPI_STATUS_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
  /* Close the file */
  	MPI_File_close(&fp);
	MPI_Barrier(MPI_COMM_WORLD);
    	//if (!fp) {fprintf(stderr, "Failed to write u-data.\n"); exit(1); }
	MPI_Barrier(MPI_COMM_WORLD);
//V
	strcpy(nameconfig,"./data/v");
	sprintf(tmp_str,"%d",10000000+plotnum);
	strcat(nameconfig,tmp_str);
	strcat(nameconfig,".datbin");
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_File_open(MPI_COMM_WORLD, nameconfig,  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_Barrier(MPI_COMM_WORLD);
  /* Set the file view */
  	MPI_File_set_view(fp, myid*Nx*ydiff*sizeof(double), MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
	MPI_Barrier(MPI_COMM_WORLD);
  /* Write buf to the file */
  	MPI_File_write(fp, T2, Nx*ydiff, MPI_DOUBLE, MPI_STATUS_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
  /* Close the file */
  	MPI_File_close(&fp);
	MPI_Barrier(MPI_COMM_WORLD);
    	//if (!fp) {fprintf(stderr, "Failed to write v-data.\n"); exit(1); }
    	MPI_Barrier(MPI_COMM_WORLD);

};

//write image
void writeimage(cl_mem* cl_u, cl_command_queue *command_queue,double *T1, int Nx,int ydiff, int plotnum){ 
        int i=0,myid,numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
        MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
        int ret=0;
        int header=54;
        ret = clEnqueueReadBuffer(*command_queue, *cl_u, CL_TRUE, 0, 2*Nx*ydiff * sizeof(double), T1, 0, NULL, NULL);
        ret = clFinish(*command_queue);
        if(ret!=0){printf("Error hahah");}
//output of data U
        double localmax=0.0;
        double globalmax=0.0;
        for(i=0;i<Nx*ydiff;i++){
                T1[i]=T1[2*i];
                if(T1[i]>localmax){localmax=T1[i];}
        }

   MPI_Allreduce(&localmax,&globalmax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

        if(myid==0){header=54;}else{header=0;}
        unsigned char*picture=(unsigned char*)malloc((3*Nx*ydiff+header)*sizeof(unsigned char));

        for(i=0;i<Nx*ydiff;i++){
                picture[3*i+header+0]=(unsigned char)(255*T1[i]/globalmax);
                picture[3*i+header+1]=(unsigned char)(255*T1[i]/globalmax);
                picture[3*i+header+2]=(unsigned char)(255*T1[i]/globalmax);
        }
	if(myid==0){
		//header for bmp file
		int w=Nx;
		int h=ydiff*numprocs;
		int filesize=54 + 3*h*w;
		picture[ 0]='B';
		picture[ 1]='M';
		picture[ 2] = (unsigned char)(filesize    );
		picture[ 3] = (unsigned char)(filesize>> 8);
		picture[ 4] = (unsigned char)(filesize>>16);
		picture[ 5] = (unsigned char)(filesize>>24);
		picture[ 6] = 0;
		picture[ 7] = 0;
		picture[ 8] = 0;
		picture[ 9] = 0;
		picture[10] = 45;
		picture[11] = 0;
		picture[12] = 0;
		picture[13] = 0;
		picture[14] = 40;
		picture[15] = 0;
		picture[16] = 0;
		picture[17] = 0;//3
		picture[18] = (unsigned char)(       w    );
		picture[19] = (unsigned char)(       w>> 8);
		picture[20] = (unsigned char)(       w>>16);
		picture[21] = (unsigned char)(       w>>24);
		picture[22] = (unsigned char)(       h    );
		picture[23] = (unsigned char)(       h>> 8);
		picture[24] = (unsigned char)(       h>>16);
		picture[25] = (unsigned char)(       h>>24);
		picture[26] = 1;
		picture[27] = 0;
		picture[28] = 24;
		picture[29] = 0;
		for(i=30;i<54;i++){
			picture[i]=0;
		}
	}
	MPI_File fp;
//file name
	char tmp_str[10];
	char nameconfig[100];
	strcpy(nameconfig,"./data/u");
	sprintf(tmp_str,"%d",10000000+plotnum);
	strcat(nameconfig,tmp_str);
	strcat(nameconfig,".bmp");
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_File_open(MPI_COMM_WORLD, nameconfig,  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_Barrier(MPI_COMM_WORLD);
  /* Set the file view */
	if(myid==0){header=0;}else{header=54;}
	MPI_File_set_view(fp, header+myid*3*Nx*ydiff*sizeof(unsigned char), MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
	MPI_Barrier(MPI_COMM_WORLD);
  /* Write buf to the file */
	if(myid==0){header=54;}else{header=0;}
	MPI_File_write(fp, picture, header+3*Nx*ydiff, MPI_CHAR, MPI_STATUS_IGNORE);

	MPI_Barrier(MPI_COMM_WORLD);
  /* Close the file */
  	MPI_File_close(&fp);
	MPI_Barrier(MPI_COMM_WORLD);
    	//if (!fp) {fprintf(stderr, "Failed to write u-data.\n"); exit(1); }
	MPI_Barrier(MPI_COMM_WORLD);

};

