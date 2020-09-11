#ifndef conv3x3
#define conv3x3
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
//#include <ap_cint.h>
#define dataw 8

#define FLOAT 0

#if FLOAT==1
	typedef float data_type;
#else
	typedef ap_fixed<16,3> data_type;
#endif


const int M1 = 96, N1 = 3, C1 = 32, H1=42, S1 = 1, K1 = 11;
const int M2 = 256, N2 = 48, C2 = 16, S2 = 1, K2 = 5;
const int H2 = C2*S2 + K2-1;
const int M3 = 384, N3 = 256, C3 = 8, S3 = 1, K3 = 3;
const int H3 = C3*S3 + K3-1;
const int M4 = 48, N4 = 4, C4 = 224, S4 = 1, K4 = 5, H4=228;
//const int M4 = 384, N4 = 192, C4 = 8, S4 = 1, K4 = 3;
//const int H4 = C4 - S4 + K4;
const int M5 = 256, N5 = 192, C5 = 8, S5 = 1, K5 = 3;
const int H5 = C5 - S5 + K5;


struct data_pack{
	data_type data0;
	data_type data1;
	data_type data2;
	data_type data3;
};

struct dma_data{
	data_pack data;
};


//
//void pool(data_type* src,data_type*dst, int factor);
#endif
