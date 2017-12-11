all: root fresnel fresnel-2gpu

root: root-gpu.cu
	nvcc root-gpu.cu -o root-gpu.out -lcuda -lcudart
fresnel: fresnel-gpu.cu
	nvcc fresnel-gpu.cu -o fresnel-gpu.out -lcuda -lcudart
fresnel-2gpu: fresnel-2gpu.cu
	nvcc -lcuda -lcudart -Xcompiler -fopenmp fresnel-2gpu.cu -o fresnel-2gpu.out

clean:
	rm root-gpu.out fresnel-gpu.out fresnel-2gpu.out
	rm root-gpu.bmp fresnel-gpu.bmp fresnel-2gpu.bmp
