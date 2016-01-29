#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {

	int	  Tmax=1000000000;
 	clock_t begin, end;
int n=0;
	printf("Got initial data, starting timestepping\n");
  	begin = clock();
	for(n=0;n<=Tmax;n++){
double a=n;
double b=2*n+3;
double c=a/b;
	}
	end = clock();
	printf("Programm took %lf for execution\n",(double)(end-begin)/CLOCKS_PER_SEC);

	return 0;
}
