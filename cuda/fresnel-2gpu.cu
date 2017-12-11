#include <stdio.h>
#include <math.h>
#include <stdint.h>		//uint32_tは符号なしintで4バイトに指定
#include <stdlib.h> 	//記憶域管理を使うため
#include <cuda.h>
#include <omp.h>

//記号定数として横幅と縦幅を定義
#define width 1024
#define heigth 1024
#define pixel width*heigth


/*--------------------bmpの構造体--------------------*/
#pragma pack(push,1)
typedef struct tagBITMAPFILEHEADER{	//構造体BITMAPFILEHEADERはファイルの先頭に来るもので，サイズは14 byte
	unsigned short	bfType;			//bfTypeは，bmp形式であることを示すため，"BM"が入る
	uint32_t 		bfSize;			//bfsizeは，ファイル全体のバイト数
	unsigned short	bfReserved1;	//bfReserved1と2は予約領域で，0になる
	unsigned short	bfReserved2;
	uint32_t		bf0ffBits;		//bf0ffBitsは先頭から画素データまでのバイト数
}BITMAPFILEHEADER;

#pragma pack(pop)
typedef struct tagBITMAPINFOHEADER{		//BITMAPINFOHEADERはbmpファイルの画像の情報の構造体で，サイズは40 byte
	uint32_t		biSize;				//画像のサイズ
	uint32_t		biWidth;			//横の画素数
	uint32_t		biHeight;			//縦の画素数
	unsigned short	biPlanes;			//1
	unsigned short	biBitCount;			//一画素あたりの色の数のbit数．今回は8
	uint32_t		biCompression;		//圧縮タイプを表す．bmpは非圧縮なので0
	uint32_t		biSizeImage;		//bmp配列のサイズを表す．biCompression=0なら基本的に0
	uint32_t		biXPelsPerMeter;	//biXPelsPerMeterとbiYPelsPerMeterは基本的に0
	uint32_t		biYPelsPerMeter;
	uint32_t		biCirUsed;			//0
	uint32_t		biCirImportant;		//0
}BITMAPINFOHEADER;

typedef struct tagRGBQUAD{
	unsigned char	rgbBlue;
	unsigned char	rgbGreen;
	unsigned char	rgbRed;
	unsigned char	rgbReserved;
}RGBQUAD;
/*--------------------------------------------------*/


/*--------------------フレネル近似のカーネル関数--------------------*/
//上半分を計算すｓるカーネル
__global__ void fresnel_gpu_0(int *x_d, int *y_d, float *z_d, float *lumi_intensity_d){
    int i, j, k;
	int adr;
	float xx, yy;

    j = blockDim.x*blockIdx.x+threadIdx.x;	//widthのループの置き換え
	i = blockDim.y*blockIdx.y+threadIdx.y;	//heigthのループの置き換え
	adr = i*width+j;

	float wave_len = 0.633F;		//光波長
	float wave_num = M_PI/wave_len;	//波数の2分の1

	for (k=0; k<284; k++) {
		xx = ((float)j-x_d[k])*((float)j-x_d[k]);
		yy = ((float)i-y_d[k])*((float)i-y_d[k]);
		lumi_intensity_d[adr] = lumi_intensity_d[adr]+__cosf(wave_num*(xx+yy)*z_d[k]);
	}
}
//下半分を計算すｓるカーネル
__global__ void fresnel_gpu_1(int *x_d, int *y_d, float *z_d, float *lumi_intensity_d){
    int i, j, k;
	int adr;
	float xx, yy;

    j = blockDim.x*blockIdx.x+threadIdx.x;				//widthのループの置き換え
	i = blockDim.y*blockIdx.y+threadIdx.y+heigth*0.5;	//heigthのループの置き換え
	adr = (i-heigth*0.5)*width+j;

	float wave_len = 0.633F;		//光波長
	float wave_num = M_PI/wave_len;	//波数の2分の1

	for (k=0; k<284; k++) {
		xx = ((float)j-x_d[k])*((float)j-x_d[k]);
		yy = ((float)i-y_d[k])*((float)i-y_d[k]);
		lumi_intensity_d[adr] = lumi_intensity_d[adr]+__cosf(wave_num*(xx+yy)*z_d[k]);
	}
}
/*--------------------------------------------------*/


//画像生成用の配列
float lumi_intensity[pixel];	//光強度用の配列
float img_tmp0[pixel/2];
float img_tmp1[pixel/2];
unsigned char img[pixel];		//bmp用の配列


/*--------------------main関数--------------------*/
int main(){
	BITMAPFILEHEADER bmpFh;
	BITMAPINFOHEADER bmpIh;
	RGBQUAD rgbQ[256];

	//ホスト側の変数
	int i;
    int points;	//物体点
	float min = 0.0F, max = 0.0F, mid;	//2値化に用いる
	FILE *fp;

	//3Dファイルの読み込み
	fp = fopen("cube284.3d","rb");	//バイナリで読み込み
	if (!fp) {
		printf("3D file not found!\n");
		exit(1);
	}
	fread(&points, sizeof(int), 1, fp);	//データのアドレス，サイズ，個数，ファイルポインタを指定
	printf("the number of points is %d\n", points);

    //取り出した物体点を入れる配列
	int x[points];				//~~データを読み込むことで初めてこの配列が定義できる~~
	int y[points];
	float z[points];
	int x_buf, y_buf, z_buf;	//データを一時的に溜めておくための変数

	//各バッファに物体点座標を取り込み，ホログラム面と物体点の位置を考慮したデータを各配列に入れる
	for (i=0; i<points; i++) {
		fread(&x_buf, sizeof(int), 1, fp);
		fread(&y_buf, sizeof(int), 1, fp);
		fread(&z_buf, sizeof(int), 1, fp);

		x[i] = x_buf*40+width*0.5;	//物体点を離すために物体点座標に40を掛け，中心の座標を足す
		y[i] = y_buf*40+heigth*0.5;
		z[i] = 1.0F/(((float)z_buf)*40+10000.0F);
	}
	fclose(fp);


/*--------------------GPUによるCGH計算--------------------*/
	dim3 block(32,32/2,1);	//ブロックサイズ(スレッド数)の配置
    dim3 grid(ceil(width/block.x),ceil(heigth/(block.y*2)),1);	//グリッドサイズ(ブロック数)の配置
//	dim3 grid((width+block.x-1)/block.x,(heigth+block.y-1)/block.y,1);

	//デバイス側の変数
	int *x_d;
	int *y_d;
	float *z_d;
	float *lumi_intensity_d;

	omp_set_num_threads(2);
	#pragma omp parallel sections
	{	//"{"は次の行に書かないと大量にエラーが吐き出される！！
		#pragma omp section	//GPU0
		{
			//使用するデバイスを指定
			cudaSetDevice(0);
			//デバイス側のメモリ確保
			cudaMalloc((void**)&x_d, points*sizeof(int));
			cudaMalloc((void**)&y_d, points*sizeof(int));
			cudaMalloc((void**)&z_d, points*sizeof(float));
			cudaMalloc((void**)&lumi_intensity_d, pixel/2*sizeof(float));
			//ホスト側からデバイス側にデータ転送
			cudaMemcpy(x_d, x, points*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(y_d, y, points*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(z_d, z, points*sizeof(float), cudaMemcpyHostToDevice);
			//カーネル関数の起動
			fresnel_gpu_0<<< grid, block >>>(x_d, y_d, z_d, lumi_intensity_d);
			cudaDeviceSynchronize();	//同期
			//デバイス側からホスト側にデータ転送
			cudaMemcpy(img_tmp0, lumi_intensity_d, pixel/2*sizeof(float), cudaMemcpyDeviceToHost);
			//デバイスのメモリ解放
			cudaFree(x_d);
			cudaFree(y_d);
			cudaFree(z_d);
			cudaFree(lumi_intensity_d);
		}
		#pragma omp section	//GPU2
		{
			//使用するデバイスを指定
			cudaSetDevice(2);
			//デバイス側のメモリ確保
			cudaMalloc((void**)&x_d, points*sizeof(int));
			cudaMalloc((void**)&y_d, points*sizeof(int));
			cudaMalloc((void**)&z_d, points*sizeof(float));
			cudaMalloc((void**)&lumi_intensity_d, pixel/2*sizeof(float));
			//ホスト側からデバイス側にデータ転送
			cudaMemcpy(x_d, x, points*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(y_d, y, points*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(z_d, z, points*sizeof(float), cudaMemcpyHostToDevice);
			//カーネル関数の起動
			fresnel_gpu_1<<< grid, block >>>(x_d, y_d, z_d, lumi_intensity_d);
			cudaDeviceSynchronize();
			//デバイス側からホスト側にデータ転送
			cudaMemcpy(img_tmp1, lumi_intensity_d, pixel/2*sizeof(float), cudaMemcpyDeviceToHost);
			//デバイスのメモリ解放
			cudaFree(x_d);
			cudaFree(y_d);
			cudaFree(z_d);
			cudaFree(lumi_intensity_d);
		}
	}
/*--------------------------------------------------*/

	//1つの配列に統合
	for (i=0; i<pixel/2; i++) {
		lumi_intensity[i] = img_tmp0[i];
		lumi_intensity[i+pixel/2] = img_tmp1[i];
	}

	//最大値，最小値を求める
	for (i=0; i<pixel; i++) {
		if (min>lumi_intensity[i]) {
			min = lumi_intensity[i];
		}
		if (max<lumi_intensity[i]) {
			max = lumi_intensity[i];
		}
	}
	mid = (min+max)/2;	//中間値（閾値）を求める

	//各々の光強度配列の値を中間値と比較し，2値化する
	for (i=0; i<pixel; i++) {
		if (lumi_intensity[i]<mid) {
			img[i] = 0;
		}
		else{
			img[i] = 255;
		}
	}


/*--------------------BMP関連--------------------*/
		//BITMAPFILEHEADERの構造体
		bmpFh.bfType		= 19778;	//'B'=0x42,'M'=0x4d,'BM'=0x4d42=19778
		bmpFh.bfSize		= 14+40+1024+(pixel);	//1024はカラーパレットのサイズ．256階調で4 byte一組
		bmpFh.bfReserved1	= 0;
		bmpFh.bfReserved2	= 0;
		bmpFh.bf0ffBits		= 14+40+1024;
		//BITMAPINFOHEADERの構造体
		bmpIh.biSize			= 40;
		bmpIh.biWidth			= width;
		bmpIh.biHeight			= heigth;
		bmpIh.biPlanes			= 1;
		bmpIh.biBitCount		= 8;
		bmpIh.biCompression		= 0;
		bmpIh.biSizeImage		= 0;
		bmpIh.biXPelsPerMeter	= 0;
		bmpIh.biYPelsPerMeter	= 0;
		bmpIh.biCirUsed			= 0;
		bmpIh.biCirImportant	= 0;
		//RGBQUADの構造体
		for (i=0; i<256; i++) {
			rgbQ[i].rgbBlue		= i;
			rgbQ[i].rgbGreen	= i;
			rgbQ[i].rgbRed		= i;
			rgbQ[i].rgbReserved	= 0;
		}
/*--------------------------------------------------*/


	fp = fopen("fresnel-2gpu.bmp","wb");	//宣言したfpと使用するファイル名，その読み書きモードを設定．バイナリ(b)で書き込み(w)
	fwrite(&bmpFh, sizeof(bmpFh), 1, fp);	//書き込むデータのアドレス，データのサイズ，データの個数，ファイルのポインタを指定
	fwrite(&bmpIh, sizeof(bmpIh), 1, fp);	//(&bmpFh.bfType, sizeof(bmpFh.bfType), 1, fp);というように個別に書くことも可能
	fwrite(&rgbQ[0], sizeof(rgbQ[0]), 256, fp);
	fwrite(img, sizeof(unsigned char), pixel, fp);	//bmpに書き込み
	printf("'fresnel-2gpu.bmp' was saved.\n\n");
	fclose(fp);

	return 0;
}
/*--------------------main関数--------------------*/
