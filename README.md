# Gray-Scott-OpenCl
This is a repsitory to host some code and posters solving the Gray-Scott-euqation using OpenCl. You will need clFFT, which can be found at https://github.com/clMathLibraries/clFFT.
* Code/single contains a basic 3D code to solve the Gray-Scott-euqation in single precision on a single OpenCl device.
* Code/double contains a basic 3D code to solve the Gray-Scott-euqation in double precision on a single OpenCl device.
* Code/MPI contains a basic 2D code to solve the Gray-Scott-euqation in double precision on a cluster. Only power of 2 gridsizes allowed! Can give you directly black-wite images of the solution.
* Poster/grayscott_iwocl contains a poster comparing the code from above on different devices.
