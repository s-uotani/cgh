all : root root-omp fresnel fresnel-omp

root : root.c
	gcc root.c -lm -O3 -o root.out
root-omp : root-omp.c
	gcc root-omp.c -lm -O3 -fopenmp -o root-omp.out

fresnel : fresnel.c
	gcc fresnel.c -lm -O3 -o fresnel.out
fresnel-omp : fresnel-omp.c
	gcc fresnel-omp.c -lm -O3 -fopenmp -o fresnel-omp.out

clean:
	rm root.out root-omp.out fresnel.out fresnel-omp.out root.bmp root-omp.bmp fresnel.bmp fresnel-omp.bmp
