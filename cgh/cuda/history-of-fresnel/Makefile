all : 0 1 2 3 4 5 6 7

0: 0.cu
	nvcc 0.cu -o 0.out -lcuda -lcudart

1: 1.cu
	nvcc 1.cu -o 1.out -lcuda -lcudart

2: 2.cu
	nvcc 2.cu -o 2.out -lcuda -lcudart

3: 3.cu
	nvcc 3.cu -o 3.out -lcuda -lcudart

4: 4.cu
	nvcc 4.cu -o 4.out -lcuda -lcudart

5: 5.cu
	nvcc 5.cu -o 5.out -lcuda -lcudart

6: 6.cu
	nvcc 6.cu -o 6.out -lcuda -lcudart

7: 7.cu
	nvcc 7.cu -o 7.out -lcuda -lcudart

clean:
	rm 1.out 2.out 3.out 4.out 5.out 6.out 7.out fresnel-gpu.bmp
