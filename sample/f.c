#include <stdio.h>
#include <math.h>
#include <stdint.h>		//uint32_tは符号なしintで4バイトに指定
#include <stdlib.h> 	//記憶域管理を使うため
#include <sys/time.h>	//時間計測に使う
#include <sys/resource.h>

#define N_x 1920
#define N_y 1080

/*時間を計測する関数*/
double getrusage_sec(){
    struct rusage t;
    struct timeval tv;

    getrusage(RUSAGE_SELF,&t);
    tv = t.ru_utime;

    return tv.tv_sec + (double)tv.tv_usec*1e-6;
}

unsigned char img[N_x*N_y];
float img_tmp[N_x*N_y];
int x[512], y[512];
float z[512];

int main(){
    float min_tmp, max_tmp, mid_tmp;
    float lambda, dtp, k, pi;
    float starttime, endtime;
    float dx, dy, tmp;
    FILE *fp, *fp1;
    int N_ten, x_tmp, y_tmp, z_tmp;
    int i, j, ii;
    lambda = 0.633F;  dtp = 10.5F;  pi = 3.14159265F;
    k = 2.0F * pi * dtp / lambda;

    fp1 = fopen("cube284.3d", "rb");
    fread(&N_ten, sizeof(int), 1, fp1);
    printf("Number of points = %d\n", N_ten);

    for (i = 0; i<N_ten; i++){
        fread(&x_tmp, sizeof(int), 1, fp1);
        fread(&y_tmp, sizeof(int), 1, fp1);
        fread(&z_tmp, sizeof(int), 1, fp1);
        x[i] = 40 * x_tmp + 700;
        y[i] = 40 * y_tmp + 500;
        z[i] = 40 * ((float)z_tmp) + 100000.0F;
    }
    fclose(fp1);

    for (i = 0; i<N_y*N_x; i++){
        img_tmp[i] = 0.0;
    }

    starttime = getrusage_sec();

    for (i = 0; i<N_y; i++){
        for (j = 0; j<N_x; j++){
            tmp = 0.0F;
            for (ii = 0; ii<N_ten; ii++){
                dx = (float)(x[ii] - j);
                dy = (float)(y[ii] - i);
//tmp = tmp + cos(k * (z[ii] +0.5F*(dx*dx +dy*dy)/z[ii]));
                tmp = tmp + cos(k * 0.5F*(dx*dx + dy*dy) / z[ii]);
//tmp = tmp + cos(k * sqrt(dx*dx +dy*dy +z[ii]*z[ii]));
            }
            img_tmp[i*N_x + j] = tmp;
        }
    }
    endtime = getrusage_sec();
    printf("%lf\n", endtime - starttime);
    return 0;
}
