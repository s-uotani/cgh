#include <stdio.h>
#include <math.h>
#include <stdint.h>		//uint32_tは符号なしintで4バイトに指定
#include <sys/time.h>	//時間計測に使う
#include <sys/resource.h>
#include <omp.h>


/*記号定数として横幅と縦幅を定義*/
#define width 1920
#define heigth 1080


/*bmpの構造体*/
#pragma pack(push,1)	//1byte毎にパッキング（何も設定しないと4byte毎になってしまう）
typedef struct tagBITMAPFILEHEADER{	//構造体BITMAPFILEHEADERはファイルの先頭に来るもので，サイズは14 byte
	unsigned short	bfType;			//bfTypeは，bmp形式であることを示すため，"BM"が入る
	uint32_t 		bfSize;			//bfsizeは，ファイル全体のバイト数
	unsigned short	bfReserved1;	//bfReserved1と2は予約領域で，0になる
	unsigned short	bfReserved2;
	uint32_t		bf0ffBits;		//bf0ffBitsは先頭から画素データまでのバイト数
}BITMAPFILEHEADER;

#pragma pack(pop)
typedef struct tagBITMAPINFOHEADER{	//BITMAPINFOHEADERはbmpファイルの画像の情報の構造体で，サイズは40 byte
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


/*時間を計測する関数*/
double gettimeofday_sec(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}


float lumi_intensity[width*heigth];	//光強度用の配列
unsigned char img[width*heigth];	//bmp用の配列


/*main関数*/
int main(){
	BITMAPFILEHEADER bmpFh;
	BITMAPINFOHEADER bmpIh;
	RGBQUAD rgbQ[256];

	double starttime, endtime;	//時間計測用の変数
	int i, j, k;
	int points;			//物体点
	FILE *fp, *fp1;

/*BITMAPFILEHEADERの構造体*/
	bmpFh.bfType		=19778;	//'B'=0x42,'M'=0x4d,'BM'=0x4d42=19778
	bmpFh.bfSize		=14+40+1024+(width*heigth);	//1024はカラーパレットのサイズ．256階調で4 byte一組
	bmpFh.bfReserved1	=0;
	bmpFh.bfReserved2	=0;
	bmpFh.bf0ffBits		=14+40+1024;

/*BITMAPINFOHEADERの構造体*/
	bmpIh.biSize			=40;
	bmpIh.biWidth			=width;
	bmpIh.biHeight			=heigth;
	bmpIh.biPlanes			=1;
	bmpIh.biBitCount		=8;
	bmpIh.biCompression		=0;
	bmpIh.biSizeImage		=0;
	bmpIh.biXPelsPerMeter	=0;
	bmpIh.biYPelsPerMeter	=0;
	bmpIh.biCirUsed			=0;
	bmpIh.biCirImportant	=0;

/*RGBQUADの構造体*/
	for(i=0; i<256; i++){
		rgbQ[i].rgbBlue		=i;
		rgbQ[i].rgbGreen	=i;
		rgbQ[i].rgbRed		=i;
		rgbQ[i].rgbReserved	=0;
	}


/*3Dファイルの読み込み*/
	fp=fopen("cube284.3d","rb");	//バイナリで読み込み
	fread(&points, sizeof(int), 1, fp);	//読み込むデータのアドレス，データのサイズ，データの個数，格納するファイルポインタを指定
	printf("the number of points is %d\n", points);


/*取り出した物体点を入れる配列*/
	int x[points];
	int y[points];
	float z[points];

	int x_buf, y_buf, z_buf;	//データを一時的に溜めておくための変数

/*各バッファに物体点座標を取り込み，ホログラム面と物体点の位置を考慮したデータを各配列に入れる*/
	for(i=0; i<points; i++){
		fread(&x_buf, sizeof(int), 1, fp);
		fread(&y_buf, sizeof(int), 1, fp);
		fread(&z_buf, sizeof(int), 1, fp);

		x[i]=x_buf*40+width*0.5;	//物体点を離すために物体点座標に40を掛け，中心の座標を足す
		y[i]=y_buf*40+heigth*0.5;
		z[i]=((float)z_buf)*40+100000.0F;
	}
	fclose(fp);


/*計算に必要な変数の定義*/
	float pixel_int=10.5F;	//画素間隔
	float wave_len=0.633F;	//光波長
	float wave_num=2.0*M_PI/wave_len;	//波数
	float cnst=pixel_int*wave_num;

/*CGH作成の式を用いて計算*/
	starttime=gettimeofday_sec();	//時間測定開始
	#pragma omp parallel for	//openMPでマルチスレッド化
	for(i=0; i<heigth; i++){
		for(j=0; j<width; j++){
			for(k=0; k<284; k++){
				lumi_intensity[i*width+j]=lumi_intensity[i*width+j]+cosf(cnst*sqrt((j-x[k])*(j-x[k])+(i-y[k])*(i-y[k])+z[k]*z[k]));
			}
		}
	}
	endtime=gettimeofday_sec();	//時間計測終了


/*最小・最大・中間値の変数を定義*/
	float min, max, mid;
	min=lumi_intensity[0];	//最大・最小値用の変数を比較できるようにとりあえずlumi~[0]を入れる
	max=lumi_intensity[0];

/*最大値，最小値を求める*/
	for(i=0; i<heigth; i++){
		for(j=0; j<width; j++){
			if(min>lumi_intensity[i*width+j]){
				min=lumi_intensity[i*width+j];
			}
			if(max<lumi_intensity[i*width+j]){
				max=lumi_intensity[i*width+j];
			}
		}
	}
	mid=(min+max)*0.5F;	//中間値（閾値）を求める
	printf("min=%lf, max=%lf, mid=%lf\n", min, max, mid);

/*各々の光強度配列の値を中間値と比較し，2値化する*/
	for(i=0; i<width*heigth; i++){
		if(lumi_intensity[i]<mid){
			img[i]=0;
		}
		if(lumi_intensity[i]>mid){
			img[i]=255;
		}
	}


/*宣言したfpと使用するファイル名，その読み書きモードを設定．バイナリ(b)で書き込み(w)*/
	fp1=fopen("root-omp.bmp","wb");

/*書き込むデータのアドレス，データのサイズ，データの個数，ファイルのポインタを指定*/
	fwrite(&bmpFh, sizeof(bmpFh), 1, fp1);	//(&bmpFh.bfType, sizeof(bmpFh.bfType), 1, fp);というように個別に書くことも可能
	fwrite(&bmpIh, sizeof(bmpIh), 1, fp1);
	fwrite(&rgbQ[0], sizeof(rgbQ[0]), 256, fp1);
	fwrite(img, sizeof(unsigned char), width*heigth, fp1);	//bmpに書き込み。配列名は式の中では先頭のアドレスになるので&をつけない

	printf("'root-omp.bmp' was saved.\n");
	printf("Calculation time is %lf\n",endtime-starttime);	//計測した時間を出力
//	printf("Calculation time is %lf\n",end_omp-start_omp);	//計測した時間を出力

	fclose(fp1);
	return 0;
}
