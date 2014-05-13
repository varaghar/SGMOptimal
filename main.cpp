#include <crtdbg.h>
#include <opencv/highgui.h>
#include <libcv/ScaborDllAcquisition.h>
#include <libcv/SmartAcquisition.h>
#include <libcv/OpenCVUI.h>
#include <libcv/Camera.h>
#include <libcv/variant.h>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <ctime>
//ID_OF_CENSUSNUCLEU	NR_OF_BITS_RETURNED		SGM_DATA_TYPE
//		10						2				unsigned long
//		10						1				unsigned int
//		 9						2				unsigned int
//		 9						1				unsigned short
//		 7						2				unsigned int
//		 7						1				unsigned short
#define COST_DATA_TYPE unsigned long
#define DATA_TYPE int
#define ID_OF_CENSUSNUCLEU 9		//number of ID of the censusNucleu
#define NR_OF_BITS_RETURNED 2		//number of bits returned by census iner Function
#define NR_OF_DISPARITIES 60 //number of disparities on witch SGM will be calculated
using namespace std;
using namespace cv;
using namespace libcv;
using namespace libcv::opencv;
using namespace libcv::scabor;
/////////////////////////////////CENSUS NUCLEU /////////////////////////////////////////
////////////struct///////////////
struct censusNucleu
{
	int nrOfOnes;
	bool data [100];
	int rows;
	int cols;
};
////////////////////////////////
censusNucleu cN9x9 = 
{
	16,
	{
		1,   0,   0,   0,   1,   0,   0,   0,   1,
			0,   0,   0,   0,   0,   0,   0,   0,   0,
			0,   0,   1,   0,   0,   0,   1,   0,   0,
			0,   0,   0,   0,   1,   0,   0,   0,   0,
			1,   0,   0,   1,   0,   1,   0,   0,   1,
			0,   0,   0,   0,   1,   0,   0,   0,   0,
			0,   0,   1,   0,   0,   0,   1,   0,   0,
			0,   0,   0,   0,   0,   0,   0,   0,   0,
			1,   0,   0,   0,   1,   0,   0,   0,   1
	},
	9,
	9
};
censusNucleu* censusNucleuAll[11]={0,0,0,0,0,0,0,0,&cN9x9,0,0};
#define getCensusNucleu(a) *(censusNucleuAll[a-1])
/////////////////////////////////CENSUS NUCLEU /////////////////////////////////////////
////////////////////////////////////CENSUS FUNCTION /////////////////////////////////////////
//////////bitFunctionRet///////////////
#define bitFunctionRet2Bit(a,b) (a<b) ? 1: ((a>b) ? 2 : 0);//include define-ul de mai sus
///////////////////////////////////////
template <typename dataStorage>
vector<Mat> censusFunc(Mat ImgML,Mat ImgMR,censusNucleu cN,char censusbitReturnFunction)
{
	Mat imgTempL(ImgML.rows,ImgML.cols,sizeof(dataStorage), cv::Scalar::all(0) );
	Mat imgTempR(ImgMR.rows,ImgMR.cols,sizeof(dataStorage), cv::Scalar::all(0) );
	dataStorage *imgTempDataL = (dataStorage *)imgTempL.data;
	dataStorage *imgTempDataR = (dataStorage *)imgTempR.data;
	dataStorage valCL = 0, valCR=0;
	int n = cN.cols;
	int m = cN.rows;//window size
	int i, j, x, y;
	uchar * imgDataR = ImgML.data; 
	uchar * imgDataL = ImgMR.data; 
	size_t imgStep=ImgML.step;
	int imgSH = ImgML.size().height - m/2;
	int imgSW = ImgML.size().width - n/2;
	char MASCA_BIT_RET=(censusbitReturnFunction & 0x02)+0x01;//se calculeaza o masca in functie de biti care trebuie intorsi
	for (x = m / 2; x < imgSH ; x++)
	{
		for (y = n / 2; y < imgSW ; y++)
		{
			valCL = 0, valCR=0;
			int tempiL=x - m / 2;
			int tempjL=y - n / 2;
			int tempiH=x + m / 2;
			int tempjH=y + n / 2;
			for (i = tempiL; i <=tempiH; i++)
			{
				for (j =tempjL; j <= tempjH; j++)
				{
					if (cN.data[(i-tempiL)*n+(j-tempjL)])//skip the 0 pixel
					{
						valCL <<= censusbitReturnFunction;
						valCR <<= censusbitReturnFunction;
						valCL += MASCA_BIT_RET & bitFunctionRet2Bit(imgDataR[i*imgStep + j],imgDataR[x*imgStep + y]);//compare pixel values in the neighborhood
						valCR += MASCA_BIT_RET & bitFunctionRet2Bit(imgDataL[i*imgStep + j],imgDataL[x*imgStep + y]);//compare pixel values in the neighborhood
					}
				}
			}
			imgTempDataL[x*imgStep + y]=valCL;
			imgTempDataR[x*imgStep + y]=valCR;
		}
	}
	vector<Mat> vM;
	vM.push_back(imgTempL);
	vM.push_back(imgTempR);
	return vM;
}
////////////////////////////////////CENSUS FUNCTION /////////////////////////////////////////
////////////////////////////////////SemiGlobalMatching /////////////////////////////////////////
////////////////////////////////////////dHamming //////////////////////////////////////////////
//110[1]1[1]01 and 110[0]1[0]01 is 2.
#define XOR(a,b) (a)^(b)
#define distantaHamming(a,b) (float)__popcnt(XOR(a,b))
////////////////////////////////////////dHamming //////////////////////////////////////////////
//////////////////////////////////////matVecini///////////////////////////////////////////////
//////////////////////////////////////gasireAdrese///////////////////////////////////////////////
#define matadr(curentRow,currentCol,currentDensity) ((((curentRow)*nrCols+(currentCol))*nrDisparitati)+(currentDensity))
//////////////////////////////////////gasireAdrese///////////////////////////////////////////////
int P1 = 15;
int P2 = 100;
DATA_TYPE * calculateTopDown(DATA_TYPE* matSGM,int rowsNr,int colsNr,int nrDisparitati,int numberOfElements,DATA_TYPE * sum){
	DATA_TYPE * L=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
	////kod
	////innovation
	int nrRows=rowsNr+1;
	int nrCols=colsNr+1;
	POINT dir ;
	dir.x = 0;
	dir.y = -1;
	DATA_TYPE * C = matSGM;
	for(int j = 0; j < colsNr;j++){
		int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
		for(int d = 0; d <finish;d++){
			int index = matadr(0,j,d);
			L[index] = C[index];
		}
	}
#pragma omp parallel for
	for(int i = 1; i < rowsNr; i++){
		for(int j = 0; j <colsNr; j++){
			DATA_TYPE minimum2 = 0;
			DATA_TYPE m1d;
			int start=0;
			int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
			int startPrev=0;
			int finishPrev=(j+dir.x+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j+dir.x;
			m1d =  L[matadr(i+dir.y,j+dir.x,0)];
			for(int di = startPrev+1; di < finishPrev;di++){
				int index = matadr(i+dir.y,j+dir.x,di);
				if( L[index] < m1d){
					m1d =  L[index];
				}
			}
			///calc minimum 2
			minimum2 = m1d;
			for(int d = start; d <finish;d++){
				DATA_TYPE minimum1 = 0;
				///calc minimum 1
				minimum1 = L[matadr(i+dir.y,j+dir.x,d)];
				DATA_TYPE m1b;
				if(d != 0){
					m1b = L[matadr(i+dir.y,j+dir.x,d-1)] + P1;
					if(minimum1 > m1b){
						minimum1 = m1b;
					}
				}
				DATA_TYPE m1c;
				if(d < finishPrev -1){
					m1c = L[matadr(i+dir.y,j+dir.x,d+1)] + P1;
					if(minimum1 > m1c){
						minimum1 = m1c;
					}
				}
				if(minimum1 > (m1d+P2)){
					minimum1 = m1d+P2;
				}
				///minimum 1 
				///summing
				int index =matadr(i,j,d);
				L[index] = C[index] + minimum1 - minimum2;
			}
		}
	}
	return L;
}
DATA_TYPE * calculateLeftRight(DATA_TYPE* matSGM,int rowsNr,int colsNr,int nrDisparitati,int numberOfElements,DATA_TYPE * sum){
	DATA_TYPE * L=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
	////kod
	////innovation
	int nrRows=rowsNr+1;
	int nrCols=colsNr+1;
	POINT dir ;
	dir.x = -1;
	dir.y = 0;
	DATA_TYPE * C = matSGM;
	for(int i = 0; i < rowsNr;i++){
		for(int d = 0; d <nrDisparitati;d++){
			int index = matadr(i,0,d);
			L[index] = C[index];
		}
	}
#pragma omp parallel for
	for(int i = 0; i < rowsNr; i++){
		for(int j = 1; j <colsNr; j++){
			DATA_TYPE minimum2 = 0;
			DATA_TYPE m1d;
			int start=0;
			int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
			int startPrev=0;
			int finishPrev=(j+dir.x+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j+dir.x;
			m1d =  L[matadr(i+dir.y,j+dir.x,0)];
			for(int di = startPrev+1; di < finishPrev;di++){
				int index= matadr(i+dir.y,j+dir.x,di);
				if( L[index] < m1d){
					m1d =  L[index];
				}
			}
			///calc minimum 2
			minimum2 = m1d;
			for(int d = start; d <finish;d++){
				DATA_TYPE minimum1 = 0;
				///calc minimum 1
				minimum1 = L[matadr(i+dir.y,j+dir.x,d)];
				DATA_TYPE m1b;
				if(d != 0){
					m1b = L[matadr(i+dir.y,j+dir.x,d-1)] + P1;
					if(minimum1 > m1b){
						minimum1 = m1b;
					}
				}
				DATA_TYPE m1c;
				if(d < finishPrev -1){
					m1c = L[matadr(i+dir.y,j+dir.x,d+1)] + P1;
					if(minimum1 > m1c){
						minimum1 = m1c;
					}
				}
				if(minimum1 > (m1d+P2)){
					minimum1 = m1d+P2;
				}
				///minimum 1 
				///summing
				int index = matadr(i,j,d) ;
				L[index] = C[index] + minimum1 - minimum2;
			}
		}
	}
	return L;
}
DATA_TYPE * calculateDownTop(DATA_TYPE* matSGM,int rowsNr,int colsNr,int nrDisparitati,int numberOfElements,DATA_TYPE * sum){
	DATA_TYPE * L=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
	////kod
	////innovation
	int nrRows=rowsNr+1;
	int nrCols=colsNr+1;
	POINT dir ;
	dir.x = 0;
	dir.y = +1;
	DATA_TYPE * C = matSGM;
	for(int j = 0; j < colsNr;j++){	
		int index = matadr(rowsNr-1,j,0);
		L[index] = C[index];
	}
#pragma omp parallel for
	for(int i = rowsNr-2; i >= 0; i--){
		for(int j = 0; j <colsNr; j++){
			DATA_TYPE minimum2 = 0;
			DATA_TYPE m1d;
			int start=0;
			int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
			int startPrev=0;
			int finishPrev=(j+dir.x+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j+dir.x;
			m1d =  L[matadr(i+dir.y,j+dir.x,0)];
			for(int di = startPrev+1; di < finishPrev;di++){
				int index = matadr(i+dir.y,j+dir.x,di);
				if( L[index] < m1d){
					m1d =  L[index];
				}
			}
			///calc minimum 2
			minimum2 = m1d;
			for(int d = start; d <finish;d++){
				DATA_TYPE minimum1 = 0;
				///calc minimum 1
				minimum1 = L[matadr(i+dir.y,j+dir.x,d)];
				DATA_TYPE m1b;
				if(d != 0){
					m1b = L[matadr(i+dir.y,j+dir.x,d-1)] + P1;
					if(minimum1 > m1b){
						minimum1 = m1b;
					}
				}
				DATA_TYPE m1c;
				if(d < finishPrev -1){
					m1c = L[matadr(i+dir.y,j+dir.x,d+1)] + P1;
					if(minimum1 > m1c){
						minimum1 = m1c;
					}
				}
				if(minimum1 > (m1d+P2)){
					minimum1 = m1d+P2;
				}
				///minimum 1 
				///summing
				int index =matadr(i,j,d);
				L[index] = C[index] + minimum1 - minimum2;
			}
		}
	}
	return L;
}
DATA_TYPE * calculateRightLeft(DATA_TYPE* matSGM,int rowsNr,int colsNr,int nrDisparitati,int numberOfElements,DATA_TYPE * sum){
	DATA_TYPE * L=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
	////kod
	////innovation
	int nrRows=rowsNr+1;
	int nrCols=colsNr+1;
	POINT dir ;
	dir.x = 1;
	dir.y = 0;
	DATA_TYPE * C = matSGM;
	for(int i = 0; i < rowsNr;i++){
		for(int d = 0; d <nrDisparitati;d++){
			int index = matadr(i, colsNr-1,d);
			L[index] = C[index];
		}
	}
#pragma omp parallel for
	for(int i = 0; i < rowsNr; i++){
		for(int j = colsNr -1 ; j >= 0; j--){
			DATA_TYPE minimum2 = 0;
			DATA_TYPE m1d;
			int start=0;
			int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
			int startPrev=0;
			int finishPrev=(j+dir.x+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j+dir.x;
			m1d =  L[matadr(i+dir.y,j+dir.x,0)];
			for(int di = startPrev+1; di < finishPrev;di++){
				int index = matadr(i+dir.y,j+dir.x,di);
				if( L[index] < m1d){
					m1d =  L[index];
				}
			}
			///calc minimum 2
			minimum2 = m1d;
			for(int d = start; d <finish;d++){
				DATA_TYPE minimum1 = 0;
				///calc minimum 1
				minimum1 = L[matadr(i+dir.y,j+dir.x,d)];
				DATA_TYPE m1b;
				if(d != 0){
					m1b = L[matadr(i+dir.y,j+dir.x,d-1)] + P1;
					if(minimum1 > m1b){
						minimum1 = m1b;
					}
				}
				DATA_TYPE m1c;
				if(d < finishPrev -1){
					m1c = L[matadr(i+dir.y,j+dir.x,d+1)] + P1;
					if(minimum1 > m1c){
						minimum1 = m1c;
					}
				}
				if(minimum1 > (m1d+P2)){
					minimum1 = m1d+P2;
				}
				///minimum 1 
				///summing
				int index = matadr(i,j,d);
				L[index] = C[index] + minimum1 - minimum2;
			}
		}
	}
	return L;
}
/*
DATA_TYPE * calculateTopLeft(DATA_TYPE* matSGM,int rowsNr,int colsNr,int nrDisparitati,int numberOfElements,DATA_TYPE * sum){
DATA_TYPE * L=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
////kod
////innovation
int nrRows=rowsNr+1;
int nrCols=colsNr+1;
POINT dir ;
dir.x = 1;
dir.y = -1;
DATA_TYPE * C = matSGM;
for(int i = 0; i < rowsNr;i++){
for(int j = 0; j <colsNr;j++){
int start=0;
int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
for(int d = start; d <finish;d++){
if(i == 0 || j == colsNr-1 || d == 0 || d == (finish -1)){
L[matadr(i,j,d)] = C[matadr(i,j,d)];
}else{
//L[matadr(i,j,d)] = 0;
}
}
}
}
for(int i = 1; i < rowsNr; i++){
for(int j = colsNr - 1 ; j >= 0; j--){
DATA_TYPE minimum2 = 0;
DATA_TYPE m1d;
int start=0;
int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
int startPrev=0;
int finishPrev=(j+dir.x+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j+dir.x;
m1d =  L[matadr(i+dir.y,j+dir.x,0)];
for(int di = startPrev+1; di < finishPrev;di++){
if( L[matadr(i+dir.y,j+dir.x,di)] < m1d){
m1d =  L[matadr(i+dir.y,j+dir.x,di)];
}
}
///calc minimum 2
minimum2 = m1d;
for(int d = start; d <finish;d++){
DATA_TYPE minimum1 = 0;
///calc minimum 1
minimum1 = L[matadr(i+dir.y,j+dir.x,d)];
DATA_TYPE m1b;
if(d != 0){
m1b = L[matadr(i+dir.y,j+dir.x,d-1)] + P1;
if(minimum1 > m1b){
minimum1 = m1b;
}
}
DATA_TYPE m1c;
if(d < finishPrev -1){
m1c = L[matadr(i+dir.y,j+dir.x,d+1)] + P1;
if(minimum1 > m1c){
minimum1 = m1c;
}
}
if(minimum1 > (m1d+P2)){
minimum1 = m1d+P2;
}
///minimum 1 
///summing
L[matadr(i,j,d)] = C[matadr(i,j,d)] + minimum1 - minimum2;
sum[matadr(i,j,d)] += 	L[matadr(i,j,d)];
}
}
}
return L;
}
DATA_TYPE * calculateTopRight(DATA_TYPE* matSGM,int rowsNr,int colsNr,int nrDisparitati,int numberOfElements,DATA_TYPE * sum){
DATA_TYPE * L=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
////kod
////innovation
int nrRows=rowsNr+1;
int nrCols=colsNr+1;
POINT dir ;
dir.x = -1;
dir.y = -1;
DATA_TYPE * C = matSGM;
for(int i = 0; i < rowsNr;i++){
for(int j = 0; j <colsNr;j++){
int start=0;
int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
for(int d = start; d <finish;d++){
if(i==0 || j == 0 ||  d == 0  || d == (finish -1)){
L[matadr(i,j,d)] = C[matadr(i,j,d)];
}else{
//	L[matadr(i,j,d)] = 0;
}
}
}
}
for(int i = 1; i < rowsNr; i++){
for(int j = 1; j <colsNr; j++){
DATA_TYPE minimum2 = 0;
DATA_TYPE m1d;
int start=0;
int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
int startPrev=0;
int finishPrev=(j+dir.x+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j+dir.x;
m1d =  L[matadr(i+dir.y,j+dir.x,0)];
for(int di = startPrev+1; di < finishPrev;di++){
if( L[matadr(i+dir.y,j+dir.x,di)] < m1d){
m1d =  L[matadr(i+dir.y,j+dir.x,di)];
}
}
///calc minimum 2
minimum2 = m1d;
for(int d = start; d <finish;d++){
DATA_TYPE minimum1 = 0;
///calc minimum 1
minimum1 = L[matadr(i+dir.y,j+dir.x,d)];
DATA_TYPE m1b;
if(d != 0){
m1b = L[matadr(i+dir.y,j+dir.x,d-1)] + P1;
if(minimum1 > m1b){
minimum1 = m1b;
}
}
DATA_TYPE m1c;
if(d < finishPrev -1){
m1c = L[matadr(i+dir.y,j+dir.x,d+1)] + P1;
if(minimum1 > m1c){
minimum1 = m1c;
}
}
if(minimum1 > (m1d+P2)){
minimum1 = m1d+P2;
}
///minimum 1 
///summing
L[matadr(i,j,d)] = C[matadr(i,j,d)] + minimum1 - minimum2;
sum[matadr(i,j,d)] += 	L[matadr(i,j,d)];
}
}
}
return L;
}
DATA_TYPE * calculateDownLeft(DATA_TYPE* matSGM,int rowsNr,int colsNr,int nrDisparitati,int numberOfElements,DATA_TYPE * sum){
DATA_TYPE * L=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
////kod
////innovation
int nrRows=rowsNr+1;
int nrCols=colsNr+1;
POINT dir ;
dir.x = 1;
dir.y = 1;
DATA_TYPE * C = matSGM;
for(int i = 0; i < rowsNr;i++){
for(int j = 0; j <colsNr;j++){
int start=0;
int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
for(int d = start; d <finish;d++){
if(i == (rowsNr-1) || j== (colsNr-1) ||  d == 0  || d == (finish -1)){
L[matadr(i,j,d)] = C[matadr(i,j,d)];
}else{
//	L[matadr(i,j,d)] = 0;
}
}
}
}
#pragma omp parallel for
for(int i = rowsNr-1; i >= 0; i--){
for(int j = colsNr; j >= 0; j--){
DATA_TYPE minimum2 = 0;
DATA_TYPE m1d;
int start=0;
int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
int startPrev=0;
int finishPrev=(j+dir.x+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j+dir.x;
m1d =  L[matadr(i+dir.y,j+dir.x,0)];
for(int di = startPrev+1; di < finishPrev;di++){
if( L[matadr(i+dir.y,j+dir.x,di)] < m1d){
m1d =  L[matadr(i+dir.y,j+dir.x,di)];
}
}
///calc minimum 2
minimum2 = m1d;
for(int d = start; d <finish;d++){
DATA_TYPE minimum1 = 0;
///calc minimum 1
minimum1 = L[matadr(i+dir.y,j+dir.x,d)];
DATA_TYPE m1b;
if(d != 0){
m1b = L[matadr(i+dir.y,j+dir.x,d-1)] + P1;
if(minimum1 > m1b){
minimum1 = m1b;
}
}
DATA_TYPE m1c;
if(d < finishPrev -1){
m1c = L[matadr(i+dir.y,j+dir.x,d+1)] + P1;
if(minimum1 > m1c){
minimum1 = m1c;
}
}
if(minimum1 > (m1d+P2)){
minimum1 = m1d+P2;
}
///minimum 1 
///summing
L[matadr(i,j,d)] = C[matadr(i,j,d)] + minimum1 - minimum2;
sum[matadr(i,j,d)] += 	L[matadr(i,j,d)];
}
}
}
return L;
}
DATA_TYPE * calculateDownRight(DATA_TYPE* matSGM,int rowsNr,int colsNr,int nrDisparitati,int numberOfElements,DATA_TYPE * sum){
DATA_TYPE * L=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
////kod
////innovation
int nrRows=rowsNr+1;
int nrCols=colsNr+1;
POINT dir ;
dir.x = -1;
dir.y = 1;
DATA_TYPE * C = matSGM;
for(int i = 0; i < rowsNr;i++){
for(int j = 0; j <colsNr;j++){
int start=0;
int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
for(int d = 0; d <finish;d++){
if( i == (rowsNr -1) ||j ==  0|| d == 0 || d == (finish -1)){
L[matadr(i,j,d)] = C[matadr(i,j,d)];
}else{
//	L[matadr(i,j,d)] = 0;
}
}
}
}
for(int i = rowsNr -1 ; i >= 0; i--){
for(int j = 1 ; j< colsNr; j++){
DATA_TYPE minimum2 = 0;
DATA_TYPE m1d;
int start=0;
int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
int startPrev=0;
int finishPrev=(j+dir.x+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j+dir.x;
m1d =  L[matadr(i+dir.y,j+dir.x,0)];
for(int di = startPrev+1; di < finishPrev;di++){
if( L[matadr(i+dir.y,j+dir.x,di)] < m1d){
m1d =  L[matadr(i+dir.y,j+dir.x,di)];
}
}
///calc minimum 2
minimum2 = m1d;
for(int d = start; d <finish;d++){
DATA_TYPE minimum1 = 0;
///calc minimum 1
minimum1 = L[matadr(i+dir.y,j+dir.x,d)];
DATA_TYPE m1b;
if(d != 0){
m1b = L[matadr(i+dir.y,j+dir.x,d-1)] + P1;
if(minimum1 > m1b){
minimum1 = m1b;
}
}
DATA_TYPE m1c;
if(d < finishPrev -1){
m1c = L[matadr(i+dir.y,j+dir.x,d+1)] + P1;
if(minimum1 > m1c){
minimum1 = m1c;
}
}
if(minimum1 > (m1d+P2)){
minimum1 = m1d+P2;
}
///minimum 1 
///summing
L[matadr(i,j,d)] = C[matadr(i,j,d)] + minimum1 - minimum2;
sum[matadr(i,j,d)] += 	L[matadr(i,j,d)];
}
}
}
return L;
}
/**/
Mat saltandpepper(int colsNr,int rowsNr,int nrCols,int nrRows,Mat * lpSrc);
template <typename dataStorage>
Mat getCost(Mat ImgL,Mat ImgR,int censusNucNumber,int censusBitFuncReturnSize,int nrDisparitati)
{
	//Aplicarea transformatei Census pe imagini
	DATA_TYPE *matSGM;
	vector<Mat> imgCensus=censusFunc<dataStorage>(ImgL,ImgR,getCensusNucleu(censusNucNumber),censusBitFuncReturnSize);
	Mat CensusLeft = imgCensus[0];
	Mat CensusRight = imgCensus[1];
	///////vizualizare imagini dupa census///////
	//imshow("CensusL"+to_string(censusBitFuncReturnSize), imgCensus[0]);
	//imshow("CensusR"+to_string(censusBitFuncReturnSize), imgCensus[1]);
	/////////////////////////////////////////////
	dataStorage * imgDataR= (dataStorage *)imgCensus[1].data; 
	dataStorage * imgDataL= (dataStorage *)imgCensus[0].data; 
	size_t imgRightstep  = imgCensus[1].step/sizeof(dataStorage);
	size_t imgLeftstep = imgCensus[0].step/sizeof(dataStorage);
	int nrRows=imgCensus[0].rows;
	int nrCols=imgCensus[0].cols;
	int rowsNr=nrRows-1;
	int colsNr=nrCols-1;
	int maxDisparity=nrDisparitati;
	int numberOfElements=nrRows*nrCols*nrDisparitati;
	//alocarea zonei de memorie pentru calculul SGM
	matSGM=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
	//initializerea valorilor cu 512
	for(int j=0;j<numberOfElements;j++)
	{
		matSGM[j]=512.0;
	}
	//declaratie variabile
	DATA_TYPE cost;
	for(int i=0;i<rowsNr;i++)
	{
		for(int j=0;j<colsNr;j++)
		{
			//Calculul intervalului de disparitati pentru fiecare punct
			int startInterval=0;
			int finishInterval=(j+maxDisparity)<colsNr ? (+maxDisparity):colsNr-j;
			for(int disparitateCurenta=startInterval;disparitateCurenta<finishInterval;disparitateCurenta++)
			{
				cost=distantaHamming(imgDataR[i*imgRightstep +j],imgDataL[i*imgRightstep + j+disparitateCurenta]);
				//setare valoare
				matSGM[ matadr(i,j,disparitateCurenta) ] = cost;
			}
		}
	}
	//namedWindow("myWindow", CV_WINDOW_AUTOSIZE);   // create a window to display you image 
	//imshow("myWindow", dataMatrix2);   // draw the image 
	//waitKey(-1);     // Wait forever a key is pressed 
	DATA_TYPE * S=(DATA_TYPE *)malloc(sizeof(DATA_TYPE)*numberOfElements);
	////kod
	////innovatio
	for(int i = 0; i < rowsNr;i++){
		for(int j = 0; j <colsNr;j++){
			for(int d = 0; d <nrDisparitati;d++){
				S[matadr(i,j,d)] = 0;
			}
		}
	}
	/**/
	DATA_TYPE * L1 = calculateTopDown(matSGM,rowsNr,colsNr,nrDisparitati,numberOfElements,S);
	DATA_TYPE * L2 =calculateDownTop(matSGM,rowsNr,colsNr,nrDisparitati,numberOfElements,S);
	DATA_TYPE * L3 =calculateLeftRight(matSGM,rowsNr,colsNr,nrDisparitati,numberOfElements,S);
	DATA_TYPE * L4 =calculateRightLeft(matSGM,rowsNr,colsNr,nrDisparitati,numberOfElements,S);/**/
	/*DATA_TYPE * L5 =calculateTopLeft(matSGM,rowsNr,colsNr,nrDisparitati,numberOfElements,S);
	DATA_TYPE * L6 =calculateTopRight(matSGM,rowsNr,colsNr,nrDisparitati,numberOfElements,S);
	DATA_TYPE * L7 =calculateDownRight(matSGM,rowsNr,colsNr,nrDisparitati,numberOfElements,S);
	DATA_TYPE * L8 =calculateDownLeft(matSGM,rowsNr,colsNr,nrDisparitati,numberOfElements,S);*/
	for(int i = 0; i < rowsNr; i++){
		for(int j = colsNr -1 ; j >= 0; j--){
			int start=0;
			int finish=(j+nrDisparitati)<colsNr ? (+nrDisparitati):colsNr-j;
			for(int  d = start ; d < finish;d++){
				int index = matadr(i,j,d);
				S[index] = 	L1[index]+L2[index]+L3[index]+L4[index];//+L5[index]+L6[index]+L7[index]+L8[index];
			}
		}
	}
	free(L1);
	free(L2);
	free(L3);
	free(L4);
	/*free(L5);
	free(L6);
	free(L7);
	free(L8);*/
	DATA_TYPE * C = matSGM;
	printf("rows %d,cols =%d",nrRows,nrCols);
	DATA_TYPE * L = S;
	cv::Mat dataMatrix(nrRows,nrCols,CV_8UC1);
	for(int i=0;i<rowsNr;i++) 
	{
		for(int j=0;j<colsNr;j++)
		{
			int minid=0;
			DATA_TYPE min = L[matadr(i,j,0)];
			int start=0;
			int finish=(j+maxDisparity)<colsNr ? (+maxDisparity):colsNr-j;
			for(int d=start+1;d<finish;d++){
				int index = matadr(i,j,d);
				if(min >= L[ index]) {
					min =L[ index];
					minid=d;
				}
			}
			dataMatrix.data[i*nrCols+j] = 4* minid;
		}
	}
	/*cv::Mat dataMatrixC(nrRows,nrCols,CV_8UC1);
	for(int i=0;i<rowsNr;i++) 
	{
	for(int j=0;j<colsNr;j++)
	{
	int minid=0;
	DATA_TYPE min = C[matadr(i,j,0)];
	int start=0;
	int finish=(j+maxDisparity)<colsNr ? (+maxDisparity):colsNr-j;
	for(int d=start+1;d<finish;d++)
	if(min >= C[ matadr(i,j,d)]) {
	min =C[ matadr(i,j,d)];
	minid=d;
	}
	dataMatrixC.data[i*nrCols+j] = 4* minid;
	}
	}*/
	// Wait forever a key is pressed
	/*namedWindow("ds", CV_WINDOW_AUTOSIZE);   // create a window to display you image 
	imshow("ds", dataMatrixC);   // draw the image 
	waitKey(-1);     // Wait forever a key is pressed*/
	free(matSGM);
	free(S);
	return saltandpepper(colsNr,rowsNr,nrCols,nrRows,&dataMatrix);
}
Mat saltandpepper(int colsNr,int rowsNr,int nrCols,int nrRows,Mat * lpSrc){
	int v;
	int dx[9]={0,1,0,-1,0,1,1,-1,-1};
	int dy[9]={0,0,-1,0,1,1,-1,-1,1};
	int vals[9]={0};
	cv::Mat lpDst(nrRows,nrCols,CV_8UC1);
		int a[9];
		for (int i=1; i<nrRows; i++) {
			for (int j=1; j<nrCols; j++)
			{	
				int k = 0;
				for (int m = i; m < i + 3; m++){
					for (int n = j; n < j + 3; n++){
						a[k] = lpSrc->data[m*colsNr+n];
						k++;
					}
				}
				//rendez
				for (int p=0; p < k - 1; p++){
					for (int q = p; q < k; q++){
						if (a[p] > a[q]){
							int aux = a[p];
							a[p] = a[q];
							a[q] = aux;
						}
					}
				}
				lpDst.data[i*colsNr + j] = a[k/2];
			}
		}
	return lpDst;
}
void afterosszeadtuk(float * L,int nrDisparitati){
}
void main() {
	String sTemp;
	DWORD cwdsz = GetCurrentDirectory(0,0); // determine size needed
	char *cwd = (char*)malloc(cwdsz);
	if ( GetCurrentDirectory(cwdsz, cwd) == 0 ) 
	{ 
		exit(0);
	}
	else 
	{
		sTemp = string( cwd );
		for(int i=0;i<3;i++)
		{
			string::size_type pos = sTemp.find_last_of( "\\/" );
			sTemp = sTemp.substr( 0, pos);
		}
		sTemp = sTemp.append("\\Images");
	}
	free((void*)cwd);
	Mat right=imread(sTemp + "\\im6.png",CV_LOAD_IMAGE_GRAYSCALE);
	Mat left=imread(sTemp + "\\im2.png",CV_LOAD_IMAGE_GRAYSCALE);
	//	Mat right=imread(sTemp + "\\im2.png",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat left=imread(sTemp + "\\im6.png",CV_LOAD_IMAGE_GRAYSCALE);
	Mat referinta=imread(sTemp + "\\disp5.png",CV_LOAD_IMAGE_GRAYSCALE);
	//imshow("left", left);
	//waitKey(0);
	//imshow("right", right);
	//waitKey(0);
/*	imshow("referinta55", referinta);
	waitKey(0);*/
	//imshow("jucus", matSGM);
	std::clock_t start;
	double duration;
	start = std::clock();
	Mat dest;
	/* Your algorithm here */
	for(int i = 0; i < 10;i++)
		dest = getCost<COST_DATA_TYPE>(left,right,ID_OF_CENSUSNUCLEU,NR_OF_BITS_RETURNED,NR_OF_DISPARITIES);
	/*innovation */
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"\n time: "<< duration /10<<'\n';
	imwrite( sTemp + "\\tedies2.png", dest );
	namedWindow("ds", CV_WINDOW_AUTOSIZE);   // create a window to display you image 
	imshow("ds", dest);   // draw the image 
	waitKey(-1);   
	//imshow("result", dest);
	//waitKey(0);
	cin>>sTemp;
}