####### Compiler, tools and options

CC            = mpicc
CFLAGS        = -m64 -pipe -O2 -Wno-unused-but-set-parameter -Wall
DEL_FILE      = rm -f

LIBS = -lOpenCL -I../../clFFT/src/package/include/ -L../../clFFT/src/package/lib64 -lclFFT -lm
 
#

####### Build rules 

CL: grayscottOpenCLs.c
	$(DEL_FILE) grayscottOpenCLs
	$(DEL_FILE) *.o
	$(CC) $(CFLAGS) -o grayscottOpenCLs grayscottOpenCLs.c $(LIBS)
	export LD_LIBRARY_PATH=../../clFFT/src/package/lib64:$$LD_LIBRARY_PATH
	export DISPLAY=:0

out: xdmfcreate.f90
	gfortran -o out xdmfcreate.f90

clean: 
	$(DEL_FILE) grayscottOpenCLs
	$(DEL_FILE) infocl
	$(DEL_FILE) out
	$(DEL_FILE) *.xmf
	$(DEL_FILE) *.o
