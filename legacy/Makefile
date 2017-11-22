all : root root-k root_g root_omp fresnel fresnel-k fresnel_g fresnel_omp

root : root.c
	gcc root.c -lm -O3 -o root.out
root-k : root-k.c
	gcc root-k.c -lm -O3 -o root-k.out
root_g : root_g.c
	gcc root_g.c -lm -O3 -o root_g.out
root_omp : root_omp.c
	gcc root_omp.c -lm -O3 -fopenmp -o root_omp.out

fresnel : fresnel.c
	gcc fresnel.c -lm -O3 -o fresnel.out
fresnel-k : fresnel-k.c
	gcc fresnel-k.c -lm -O3 -o fresnel-k.out
fresnel_g : fresnel_g.c
	gcc fresnel_g.c -lm -O3 -o fresnel_g.out
fresnel_omp : fresnel_omp.c
	gcc fresnel_omp.c -lm -O3 -fopenmp -o fresnel_omp.out

clean:
	rm root.out root-k.out root_g.out root_omp.out fresnel.out fresnel-k.out fresnel_g.out fresnel_omp.out
