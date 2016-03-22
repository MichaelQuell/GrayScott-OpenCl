####### Compiler, tools and options

CC            = gcc
CFLAGS        = -m64 -pipe -O2 -Wno-unused-but-set-parameter -Wall
DEL_FILE      = rm -f

LIBS = -lOpenCL -I../../include/ -L../../lib64 -lclFFT -lm


####### Build rules 

CL: grayscottOpenCLs.c
	$(DEL_FILE) grayscottOpenCLs
	$(DEL_FILE) *.o
	$(CC) $(CFLAGS) -o grayscottOpenCLs grayscottOpenCLs.c $(LIBS)


out: xdmfcreate.f90
	gfortran -o out xdmfcreate.f90

clean: 
	$(DEL_FILE) grayscottOpenCLs
	$(DEL_FILE) infocl
	$(DEL_FILE) out
	$(DEL_FILE) *.xmf
	$(DEL_FILE) *.o
