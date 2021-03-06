#include "conv3x3.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;
const int TrBuff1 = 4, TcBuff1=4, Tr1=8, Tc1=8, TmBuff1=4, TnBuff1=4, Tm1=4, Tn1=4, Tk1=5, Tri1=8, Tci1=14;
const int TrBuff2 = 4, TcBuff2=4, Tr2=4, Tc2=4, TmBuff2=4, TnBuff2=4, Tm2=4, Tn2=4, Tk2=5, Tri2=8, Tci2=8;
const int TrBuff3 = 4, TcBuff3=4, Tr3=8, Tc3=8, TmBuff3=4, TnBuff3=4, Tm3=4, Tn3=4, Tk3=7, Tri3=14, Tci3=14;
const int TrBuff4 = 4, TcBuff4=4, Tr4=8, Tc4=8, TmBuff4=4, TnBuff4=4, Tm4=4, Tn4=4, Tk4=7, Tri4=14, Tci4=14;
const int TrBuff5 = 4, TcBuff5=4, Tr5=8, Tc5=8, TmBuff5=4, TnBuff5=4, Tm5=4, Tn5=4, Tk5=7, Tri5=14, Tci5=14;
//num read ports = num of bram banks

template <int Tr, int Tc, int Tn>
void read_ifmap_conv2d(data_type feature_temp[Tn][Tr][Tc], dma_data* feature, int tr, int ti, int tc, 
                       int H, 
                       int C, 
                       //int K, 
                       ap_uint<32> Base_addr2){
#pragma HLS INLINE off
	dma_data tmp;
    int dim_start=0;
    // int dim_start = (int)(K/2);
    // int dim_end = (int)(C+K/2-1);
	int trr, tcc, tii;
	for (tii = 0; tii < Tn; tii+=2) {
		for (trr = 0; trr < Tr; trr++) {
			for (tcc = 0; tcc < Tc; tcc++) {
				#pragma HLS PIPELINE
                // if ((tr+trr) < dim_start || (tr+trr) > dim_end || (tc+tcc) < dim_start || (tc+tcc) > dim_end){
                    // feature_temp[tii][trr][tcc]=0;
                    // feature_temp[tii+1][trr][tcc]=0;
                // }
                // else{
                    tmp=feature[(tii+ti)/2*C*C + (tr+trr-dim_start)*C +(tc+ tcc-dim_start)+Base_addr2/dataw];
                    feature_temp[tii][trr][tcc] = tmp.data.data0;
                    feature_temp[tii+1][trr][tcc] = tmp.data.data1;
                    feature_temp[tii+2][trr][tcc] = tmp.data.data2;
                    feature_temp[tii+3][trr][tcc] = tmp.data.data3;
                // }
			}
		}
	}

}



template < int Tk, int Tm,int Tn>
void read_wek(data_type weight_temp[Tm][Tn][Tk][Tk],dma_data* weight, int to, int ti,
             int K, int N, ap_uint<32> Base_addr1){
#pragma HLS INLINE
	int too,tii, tkk1,tkk2;
	dma_data tmp;

	for (too = 0; too < Tm; too+=4) {
        for (tii = 0; tii < Tn; tii++) {
            for(tkk1 =0; tkk1<K; tkk1++){
                for(tkk2 =0; tkk2<K; tkk2++){
                    #pragma HLS PIPELINE
                    tmp= weight[(too + to)/4*N*K*K + (tii+ti)*K*K+tkk1*K +tkk2 +Base_addr1/dataw];
                    weight_temp[too][tii][tkk1][tkk2] = tmp.data.data0;
                    weight_temp[too+1][tii][tkk1][tkk2] = tmp.data.data1;
                    weight_temp[too+2][tii][tkk1][tkk2] = tmp.data.data2;
                    weight_temp[too+3][tii][tkk1][tkk2] = tmp.data.data3;
                }
            }
        }
	}
}



//read_ifmap3
//tr level parallel loading
template <int Tr, int Tc, int Tn>
void read_ifmap3_conv2d(data_type feature_temp[Tn][Tr][Tc], dma_data* feature, int tr, int ti, int tc, 
                       int H, 
                       int C, 
                       //int K, 
                       ap_uint<32> Base_addr2){
#pragma HLS INLINE off
	dma_data tmp;
    int dim_start=0;
    // int dim_start = (int)(K/2);
    // int dim_end = (int)(C+K/2-1);
	int trr, tcc, tii;
	for (tii = 0; tii < Tn; tii++) {
		for (tcc = 0; tcc < Tc; tcc++) {
            for (trr = 0; trr < Tr; trr+=4) {
				#pragma HLS PIPELINE
                // if ((tr+trr) < dim_start || (tr+trr) > dim_end || (tc+tcc) < dim_start || (tc+tcc) > dim_end){
                    // feature_temp[tii][trr][tcc]=0;
                    // feature_temp[tii+1][trr][tcc]=0;
                // }
                // else{
                    tmp=feature[(tii+ti)*C*C + ((tr+trr)/4-dim_start)*C +(tc+ tcc-dim_start)+Base_addr2/dataw];
                    feature_temp[tii][trr][tcc] = tmp.data.data0;
                    feature_temp[tii][trr+1][tcc] = tmp.data.data1;
                    feature_temp[tii][trr+2][tcc] = tmp.data.data2;
                    feature_temp[tii][trr+3][tcc] = tmp.data.data3;
                // }
			}
		}
	}

}


//
//void comp_engine(
//				 data_type weight_temp[Tm][Tn][K][K], data_type feature_temp[Tr+K-S][Tn][Tc+K-S],data_type output_core_temp[Tm][Tr][Tc],
//				 data_type input_tmp_buffer[Tm][Tr]){
//#pragma HLS INLINE off
//	int k1,k2,too, tcc, tii, trr;
//    data_type tmp0;
//	//TODO: balanced unrolling input channel and output channel
//    for (tii = 0; tii < Tn; ++tii) {
//		for (tcc = 0; tcc < Tc; ++tcc) {
//			for(k2=0; k2<K;k2++){
//				for(k1=0; k1<K;k1++){
//#pragma HLS PIPELINE
//
//					for (trr = 0; trr < Tr; ++trr){
//						#pragma HLS UNROLL
//						tmp0=feature_temp[trr+k1][tii][tcc+k2];
//						for (too = 0; too < Tm; ++too){
//							input_tmp_buffer[too][trr]=tmp0;
//						}
//					}
//
//					for (too = 0; too < Tm; ++too) {
////#pragma HLS DEPENDENCE variable=weight_temp intra false
//#pragma HLS UNROLL
//						tmp0=weight_temp[too][tii][k1][k2];
//						for (trr = 0; trr < Tr; ++trr) {
////#pragma HLS DEPENDENCE variable=output_core_temp intra false
////#pragma HLS DEPENDENCE variable=output_core_temp inter false
////#pragma HLS DEPENDENCE variable=feature_temp intra false
//#pragma HLS UNROLL
//
//							output_core_temp[too][trr][tcc]+=input_tmp_buffer[too][trr]*tmp0;
//						}
//					}
//				}
//			}
//		}
//    }
//
//}


template <int Tr, int Tc, int TmBuff, int TnBuff,int Tm, int Tn,int TmW, int TnW, int Tri, int Tci,int Tk>
void comp_engine1(
				 data_type weight_temp[TmW][TnW][Tk][Tk], data_type feature_temp[TnBuff][Tri][Tci],data_type output_core_temp[TmBuff][Tr][Tc],
                 int K , int S
                 ){
#pragma HLS INLINE off
	int too, tcc, tii, trr,tkk1,tkk2,tmcomp,tncomp;
    data_type tmp0,tmp1;
	//TODO: balanced unrolling input channel and output channel
    for(tncomp=0;tncomp <TnBuff;tncomp+=Tn){
        for(tmcomp=0;tmcomp <TmBuff;tmcomp+=Tm){    
            for (tkk1=0; tkk1<K; tkk1++){
                for(tkk2=0; tkk2<K; tkk2++){
                    for (tcc = 0; tcc < Tc; ++tcc) {
                        for (trr = 0; trr < Tr; ++trr) {
                            for (tii = 0; tii < Tn; ++tii) {
                                #pragma HLS PIPELINE
                                tmp1=feature_temp[tncomp+tii][trr*S+tkk1][tcc*S+tkk2];
                                for (too = 0; too < Tm; ++too) {
                                #pragma HLS DEPENDENCE variable=output_core_temp inter false
                #pragma HLS UNROLL
                                    output_core_temp[tmcomp+too][trr][tcc]+=
                                    tmp1*weight_temp[tmcomp+too][tncomp+tii][tkk1][tkk2];
                                    
                                }
                            }
                            
                        }
                    }
                }
            }
        }
    }
}



template <int Tr, int Tc, int TmBuff, int TnBuff,int Tm, int Tn,int TmW, int TnW, int Tri, int Tci,int Tk>
void comp_engine2(
				 data_type weight_temp[TmW][TnW][Tk][Tk], data_type feature_temp[TnBuff][Tri][Tci],data_type output_core_temp[TmBuff][Tr][Tc],
                 int K , int S
                 ){
#pragma HLS INLINE off
	int too, tcc, tii, trr,tkk1,tkk2,tncomp,tmcomp;
    data_type tmp0,tmp1;
	//TODO: balanced unrolling input channel and output channel
    for(tncomp=0;tncomp <TnBuff;tncomp+=Tn){
        for(tmcomp=0;tmcomp <TmBuff;tmcomp+=Tm){    
            for (tkk1=0; tkk1<K; tkk1++){
                for(tkk2=0; tkk2<K; tkk2++){
                    for (tcc = 0; tcc < Tc; ++tcc) {
                        for (trr = 0; trr < Tr; ++trr) {
                #pragma HLS PIPELINE
                            for (tii = 0; tii < Tn; ++tii) {
                #pragma HLS UNROLL
                                #pragma HLS DEPENDENCE variable=feature_temp inter false
                                tmp1=feature_temp[tncomp+tii][trr*S+tkk1][tcc*S+tkk2];
                                for (too = 0; too < Tm; ++too) {
                                #pragma HLS DEPENDENCE variable=output_core_temp inter false
                #pragma HLS UNROLL
                                    output_core_temp[tmcomp+too][trr][tcc]+=
                                    tmp1*weight_temp[tmcomp+too][tncomp+tii][tkk1][tkk2];
                                    
                                }
                            }
                            
                        }
                    }
                }
            }
        }
    }
}



template <int Tr, int Tc, int TrBuff, int TcBuff, int Tm, int Tn,int TmW, int TnW, int Tri, int Tci,int Tk>
void comp_engine3(
				 data_type weight_temp[TmW][TnW][Tk][Tk], data_type feature_temp[Tn][Tri][Tci],data_type output_core_temp[Tm][TrBuff][TcBuff],
                 int K , int S
                 ){
#pragma HLS INLINE off
	int too, tcc, tii, trr,tkk1,tkk2,trcomp,tccomp;
    data_type tmp0,tmp1;

    for(trcomp=0;trcomp <TrBuff;trcomp+=Tr){
        for(tccomp=0;tccomp <TcBuff;tccomp+=Tc){    
            for (tkk1=0; tkk1<K; tkk1++){
                for(tkk2=0; tkk2<K; tkk2++){
                    for (tcc = 0; tcc < Tc; ++tcc) {
                        for (too = 0; too < Tm; ++too){
                            for (tii = 0; tii < Tn; ++tii) {
                                #pragma HLS PIPELINE
                                tmp1=weight_temp[too][tii][tkk1][tkk2];
                                for (trr = 0; trr < Tr; ++trr) {
                                #pragma HLS DEPENDENCE variable=output_core_temp inter false
                                #pragma HLS DEPENDENCE variable=feature_temp inter false
                                #pragma HLS UNROLL
                                    output_core_temp[too][trr+trcomp][tcc+tccomp]+=
                                    tmp1*feature_temp[tii][(trr+trcomp)*S+tkk1][(tcc+tccomp)*S+tkk2];
                                    
                                }
                            }
                            
                        }
                    }
                }
            }
        }
    }
}




template <
int TmBuff, int TnBuff, int TrBuff, int TcBuff,  int Tr, int Tc, int Tm, int Tn, int TmW, int TnW, int Tk,int Tri,int Tci
>
void conv3_3_1(
		dma_data* weight,
		dma_data* feature,
		dma_data* output_core,
	int con,
	ap_uint<32> Base_addr1,
	ap_uint<32>  Base_addr2,
	ap_uint<32>  Base_addr3,
    int M,int N, int H, int C, int K , int S) {



//#pragma HLS INTERFACE axis port=weight
//#pragma HLS INTERFACE axis port=feature
//#pragma HLS INTERFACE axis port=output_core

	dma_data tmp;
	int tr,tc;
	int to, ti, trr, tcc, too, tii;
    int tc_r, tr_r, to_r, ti_r;
	int lr_i=0;
	data_type output_core_temp[TmBuff][Tr][Tc] = { 0 };
	#pragma HLS RESOURCE variable=output_core_temp core=RAM_2P_BRAM
	data_type weight_temp[TmW][TnW][Tk][Tk] = { 0}, feature_temp[TnBuff][Tri][Tci] = { 0 };
	#pragma HLS RESOURCE variable=feature_temp core=RAM_2P_BRAM
	#pragma HLS RESOURCE variable=weight_temp core=RAM_2P_BRAM
	data_type feature_temp1[TnBuff][Tri][Tci] = { 0 };
	#pragma HLS RESOURCE variable=feature_temp1 core=RAM_2P_BRAM

		//partition anchor




        //partition finished
	if(con==0x00000001){
		//TODO: buffer initialization
		read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp, feature,0,0,0,H,C,Base_addr2);
		//read_we(weight_temp,weight,0,0);
		for (tc=0; tc<C; tc+=Tc){
			for (tr=0; tr <C; tr+=Tr){
				for (to = 0; to < M; to += TmBuff) {
					for (ti = 0; ti < N; ti += TnBuff) {
						read_wek<Tk,TmW,TnW>(weight_temp,weight,to,ti,K,N,Base_addr1);
						if (lr_i==0){


							//ping pong logic for index shifting
							//ti_r=ti;
							to_r=to;
							tc_r=tc;
							tr_r=tr;
							ti_r=ti+TnBuff;
							if (ti_r==N){
								ti_r=0;
								tr_r=tr+Tr;
								if(tr_r==C){
									tr_r=0;
									tc_r=tc_r+Tc;
									if(tc_r==C){
										tc_r=0;
									}
								}
							}
							//TODO: controlling port to switch
							read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp1, feature,tr_r,ti_r,tc_r,H,C,Base_addr2);
                            comp_engine1<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp,output_core_temp,K,S);
							lr_i=1-lr_i;
						}
						else{


							//ping pong logic for index shifting
							//ti_r=ti;
							to_r=to;
							tc_r=tc;
							tr_r=tr;
							ti_r=ti+TnBuff;
							if (ti_r==N){
								ti_r=0;
								tr_r=tr+Tr;
								if(tr_r==C){
									tr_r=0;
									tc_r=tc_r+Tc;
									if(tc_r==C){
										tc_r=0;
									}
								}
							}
							//TODO: controlling port to switch
							read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp, feature,tr_r,ti_r,tc_r,H,C,Base_addr2);
                            comp_engine1<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp1,output_core_temp,K,S);
							lr_i=1-lr_i;

						}

					}

                    for (too = 0; too < TmBuff; too+=4) {
                        for (trr = 0; trr < Tr; trr++) {
                            for (tcc = 0; tcc < Tc; tcc++) {
                                #pragma HLS PIPELINE
                                tmp.data.data0=output_core_temp[too][trr][tcc];
                                tmp.data.data1=output_core_temp[too+1][trr][tcc];
                                tmp.data.data2=output_core_temp[too+2][trr][tcc];
                                tmp.data.data3=output_core_temp[too+3][trr][tcc];
                                output_core[(too + to)/4*C*C + (tr+trr)*C +tc+ tcc+Base_addr3/dataw]=tmp;
                            }
                        }
                    }
                        
                        
                    for (trr = 0; trr < Tr; ++trr) {
                        for (tcc = 0; tcc < Tc; ++tcc) {
                            #pragma HLS PIPELINE
                            for (too = 0; too < TmBuff; ++too) {
                                #pragma HLS UNROLL
                                    output_core_temp[too][trr][tcc] = 0;
                            }
                        }
                    }




				}
			}
		}
	}


};




template <
int TmBuff, int TnBuff, int TrBuff, int TcBuff,  int Tr, int Tc, int Tm, int Tn, int TmW, int TnW, int Tk,int Tri,int Tci
>
void conv3_3_2(
		dma_data* weight,
		dma_data* feature,
		dma_data* output_core,
	int con,
	ap_uint<32> Base_addr1,
	ap_uint<32>  Base_addr2,
	ap_uint<32>  Base_addr3,
    int M,int N, int H, int C, int K , int S) {



//#pragma HLS INTERFACE axis port=weight
//#pragma HLS INTERFACE axis port=feature
//#pragma HLS INTERFACE axis port=output_core

	dma_data tmp;
	int tr,tc;
	int to, ti, trr, tcc, too, tii;
    int tc_r, tr_r, to_r, ti_r;
	int lr_i=0;
	data_type output_core_temp[TmBuff][Tr][Tc] = { 0 };
	#pragma HLS RESOURCE variable=output_core_temp core=RAM_2P_BRAM

	data_type weight_temp[TmW][TnW][Tk][Tk] = { 0}, feature_temp[TnBuff][Tri][Tci] = { 0 };
	#pragma HLS RESOURCE variable=feature_temp core=RAM_2P_BRAM
	#pragma HLS RESOURCE variable=weight_temp core=RAM_2P_BRAM


	data_type feature_temp1[TnBuff][Tri][Tci] = { 0 };
	#pragma HLS RESOURCE variable=feature_temp1 core=RAM_2P_BRAM
        //partition anchor




        //partition finished
	if(con==0x00000001){
		//TODO: buffer initialization
		read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp, feature,0,0,0,H,C,Base_addr2);
		//read_we(weight_temp,weight,0,0);
		for (tc=0; tc<C; tc+=Tc){
			for (tr=0; tr <C; tr+=Tr){
				for (to = 0; to < M; to += TmBuff) {
					for (ti = 0; ti < N; ti += TnBuff) {
						read_wek<Tk,TmW,TnW>(weight_temp,weight,to,ti,K,N,Base_addr1);
						if (lr_i==0){


							//ping pong logic for index shifting
							//ti_r=ti;
							to_r=to;
							tc_r=tc;
							tr_r=tr;
							ti_r=ti+TnBuff;
							if (ti_r==N){
								ti_r=0;
								tr_r=tr+Tr;
								if(tr_r==C){
									tr_r=0;
									tc_r=tc_r+Tc;
									if(tc_r==C){
										tc_r=0;
									}
								}
							}
							//TODO: controlling port to switch
							read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp1, feature,tr_r,ti_r,tc_r,H,C,Base_addr2);
                            comp_engine2<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp,output_core_temp,K,S);
							lr_i=1-lr_i;
						}
						else{


							//ping pong logic for index shifting
							//ti_r=ti;
							to_r=to;
							tc_r=tc;
							tr_r=tr;
							ti_r=ti+TnBuff;
							if (ti_r==N){
								ti_r=0;
								tr_r=tr+Tr;
								if(tr_r==C){
									tr_r=0;
									tc_r=tc_r+Tc;
									if(tc_r==C){
										tc_r=0;
									}
								}
							}
							//TODO: controlling port to switch
							read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp, feature,tr_r,ti_r,tc_r,H,C,Base_addr2);
                            comp_engine2<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp1,output_core_temp,K,S);
							lr_i=1-lr_i;

						}

					}

                    for (too = 0; too < TmBuff; too+=4) {
                        for (trr = 0; trr < Tr; trr++) {
                            for (tcc = 0; tcc < Tc; tcc++) {
                                #pragma HLS PIPELINE
                                tmp.data.data0=output_core_temp[too][trr][tcc];
                                tmp.data.data1=output_core_temp[too+1][trr][tcc];
                                tmp.data.data2=output_core_temp[too+2][trr][tcc];
                                tmp.data.data3=output_core_temp[too+3][trr][tcc];
                                output_core[(too + to)/4*C*C + (tr+trr)*C +tc+ tcc+Base_addr3/dataw]=tmp;
                            }
                        }
                    }
                        
                        
                    for (trr = 0; trr < Tr; ++trr) {
                        for (tcc = 0; tcc < Tc; ++tcc) {
                            #pragma HLS PIPELINE
                            for (too = 0; too < TmBuff; ++too) {
                                #pragma HLS UNROLL
                                    output_core_temp[too][trr][tcc] = 0;
                            }
                        }
                    }




				}
			}
		}
	}


};



template <
int TmBuff, int TnBuff, int TrBuff, int TcBuff,  int Tr, int Tc, int Tm, int Tn, int TmW, int TnW, int Tk,int Tri,int Tci
>
void conv3_3_3(
		dma_data* weight,
		dma_data* feature,
		dma_data* output_core,
	int con,
	ap_uint<32> Base_addr1,
	ap_uint<32>  Base_addr2,
	ap_uint<32>  Base_addr3,
    int M,int N, int H, int C, int K , int S) {



//#pragma HLS INTERFACE axis port=weight
//#pragma HLS INTERFACE axis port=feature
//#pragma HLS INTERFACE axis port=output_core

	dma_data tmp;
	int tr,tc;
	int to, ti, trr, tcc, too, tii;
    int tc_r, tr_r, to_r, ti_r;
	int lr_i=0;
	data_type output_core_temp[Tm][TrBuff][TcBuff] = { 0 };
	#pragma HLS RESOURCE variable=output_core_temp core=RAM_2P_BRAM

	data_type weight_temp[TmW][TnW][Tk][Tk] = { 0}, feature_temp[Tn][Tri][Tci] = { 0 };
	#pragma HLS RESOURCE variable=feature_temp core=RAM_2P_BRAM
	#pragma HLS RESOURCE variable=weight_temp core=RAM_2P_BRAM


	data_type feature_temp1[Tn][Tri][Tci] = { 0 };
	#pragma HLS RESOURCE variable=feature_temp1 core=RAM_2P_BRAM
        //partition anchor




        //partition finished
	if(con==0x00000001){
		//TODO: buffer initialization
		read_ifmap_conv2d<Tri,Tci,Tn>(feature_temp, feature,0,0,0,H,C,Base_addr2);
		//read_we(weight_temp,weight,0,0);
		for (tc=0; tc<C; tc+=TcBuff){
			for (tr=0; tr <C; tr+=TrBuff){
				for (to = 0; to < M; to += Tm) {
					for (ti = 0; ti < N; ti += Tn) {
						read_wek<Tk,TmW,TnW>(weight_temp,weight,to,ti,K,N,Base_addr1);
						if (lr_i==0){


							//ping pong logic for index shifting
							//ti_r=ti;
							to_r=to;
							tc_r=tc;
							tr_r=tr;
							ti_r=ti+Tn;
							if (ti_r==N){
								ti_r=0;
								tr_r=tr+TrBuff;
								if(tr_r==C){
									tr_r=0;
									tc_r=tc_r+TcBuff;
									if(tc_r==C){
										tc_r=0;
									}
								}
							}
							//TODO: controlling port to switch
							read_ifmap3_conv2d<Tri,Tci,Tn>(feature_temp1, feature,tr_r,ti_r,tc_r,H,C,Base_addr2);
                            comp_engine3<Tr,Tc,TrBuff,TcBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp,output_core_temp,K,S);
							lr_i=1-lr_i;
						}
						else{


							//ping pong logic for index shifting
							//ti_r=ti;
							to_r=to;
							tc_r=tc;
							tr_r=tr;
							ti_r=ti+Tn;
							if (ti_r==N){
								ti_r=0;
								tr_r=tr+TrBuff;
								if(tr_r==C){
									tr_r=0;
									tc_r=tc_r+TcBuff;
									if(tc_r==C){
										tc_r=0;
									}
								}
							}
							//TODO: controlling port to switch
							read_ifmap3_conv2d<Tri,Tci,Tn>(feature_temp, feature,tr_r,ti_r,tc_r,H,C,Base_addr2);
                            comp_engine3<Tr,Tc,TrBuff,TcBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp1,output_core_temp,K,S);
							lr_i=1-lr_i;

						}

					}

                    for (too = 0; too < Tm; too++) {
                        for (trr = 0; trr < TrBuff; trr+=4) {
                            for (tcc = 0; tcc < TcBuff; tcc++) {
                                #pragma HLS PIPELINE
                                tmp.data.data0=output_core_temp[too][trr][tcc];
                                tmp.data.data1=output_core_temp[too][trr+1][tcc];
                                tmp.data.data2=output_core_temp[too][trr+2][tcc];
                                tmp.data.data3=output_core_temp[too][trr+3][tcc];
                                output_core[(too + to)*C*C + (tr+trr)/4*C +tc+ tcc+Base_addr3/dataw]=tmp;
                            }
                        }
                    }
                        
                        
                    for (too = 0; too < Tm; ++too){
                        for (tcc = 0; tcc < TcBuff; ++tcc){
                            #pragma HLS PIPELINE
                            for (trr = 0; trr < TrBuff; ++trr) {

                                #pragma HLS UNROLL
                                    output_core_temp[too][trr][tcc] = 0;
                            }
                        }
                    }




				}
			}
		}
	}


};

void conv3_3 (
//dma_data* weight1,
//dma_data* feature1,
//dma_data* output_core1,
dma_data* weight2,
dma_data* feature2,
dma_data* output_core2,
//dma_data* weight3,
//dma_data* feature3,
//dma_data* output_core3,
//dma_data* weight4,
//dma_data* feature4,
//dma_data* output_core4,
//dma_data* weight5,
//dma_data* feature5,
//dma_data* output_core5,
//dma_data* weight6,
//dma_data* feature6,
//dma_data* output_core6,
//dma_data* weight7,
//dma_data* feature7,
//dma_data* output_core7,
//dma_data* weight8,
//dma_data* feature8,
//dma_data* output_core8,
//dma_data* weight9,
//dma_data* feature9,
//dma_data* output_core9,
//dma_data* weight10,
//dma_data* feature10,
//dma_data* output_core10,
//dma_data* weight11,
//dma_data* feature11,
//dma_data* output_core11,
//dma_data* weight12,
//dma_data* feature12,
//dma_data* output_core12,
//dma_data* weight13,
//dma_data* feature13,
//dma_data* output_core13,
int con,
ap_uint<32> Base_addr1,
ap_uint<32>  Base_addr2,
ap_uint<32>  Base_addr3,
//ap_uint<32> Base_addr4,
//ap_uint<32>  Base_addr5,
//ap_uint<32>  Base_addr6,
//ap_uint<32> Base_addr7,
//ap_uint<32>  Base_addr8,
//ap_uint<32>  Base_addr9,
//ap_uint<32> Base_addr10,
//ap_uint<32>  Base_addr11,
//ap_uint<32>  Base_addr12,
//ap_uint<32> Base_addr13,
//ap_uint<32>  Base_addr14,
//ap_uint<32>  Base_addr15,
//ap_uint<32> Base_addr16,
//ap_uint<32>  Base_addr17,
//ap_uint<32>  Base_addr18,
//ap_uint<32> Base_addr19,
//ap_uint<32>  Base_addr20,
//ap_uint<32>  Base_addr21,
//ap_uint<32> Base_addr22,
//ap_uint<32>  Base_addr23,
//ap_uint<32>  Base_addr24,
//ap_uint<32> Base_addr25,
//ap_uint<32>  Base_addr26,
//ap_uint<32>  Base_addr27,
ap_uint<32> Base_addr28,
ap_uint<32>  Base_addr29,
ap_uint<32>  Base_addr30,
ap_uint<32> Base_addr31,
ap_uint<32>  Base_addr32,
ap_uint<32>  Base_addr33,
ap_uint<32> Base_addr34,
ap_uint<32>  Base_addr35,
ap_uint<32>  Base_addr36,
ap_uint<32> Base_addr37,
ap_uint<32>  Base_addr38,
ap_uint<32>  Base_addr39){
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=con bundle=CRTL_BUS

//#pragma HLS INTERFACE s_axilite port=Base_addr37 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr38 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr39 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M1*N1*K1*K1/4 port=weight1
//#pragma HLS INTERFACE m_axi depth=N1*H1*H1/4 port=feature1
//#pragma HLS INTERFACE m_axi depth=M1*C1*C1/4 port=output_core1
//#pragma HLS data_pack variable=weight1
//#pragma HLS data_pack variable=feature1
//#pragma HLS data_pack variable=output_core1


#pragma HLS INTERFACE s_axilite port=Base_addr34 bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=Base_addr35 bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=Base_addr36 bundle=CRTL_BUS
#pragma HLS INTERFACE m_axi depth=M2*N2*K2*K2/4 port=weight2
#pragma HLS INTERFACE m_axi depth=N2*H2*H2/4 port=feature2
#pragma HLS INTERFACE m_axi depth=M2*C2*C2/4 port=output_core2
#pragma HLS data_pack variable=weight2
#pragma HLS data_pack variable=feature2
#pragma HLS data_pack variable=output_core2

//#pragma HLS INTERFACE s_axilite port=Base_addr31 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr32 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr33 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M3*N3*K3*K3/4 port=weight3
//#pragma HLS INTERFACE m_axi depth=N3*H3*H3/4 port=feature3
//#pragma HLS INTERFACE m_axi depth=M3*C3*C3/4 port=output_core3
//#pragma HLS data_pack variable=weight3
//#pragma HLS data_pack variable=feature3
//#pragma HLS data_pack variable=output_core3


//#pragma HLS INTERFACE s_axilite port=Base_addr28 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr29 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr30 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M4*N4*K4*K4/4 port=weight4
//#pragma HLS INTERFACE m_axi depth=N4*H4*H4/4 port=feature4
//#pragma HLS INTERFACE m_axi depth=M4*C4*C4/4 port=output_core4
//#pragma HLS data_pack variable=weight4
//#pragma HLS data_pack variable=feature4
//#pragma HLS data_pack variable=output_core4


//#pragma HLS INTERFACE s_axilite port=Base_addr1 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr2 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr3 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M5*N5*K5*K5/4 port=weight5
//#pragma HLS INTERFACE m_axi depth=N5*H5*H5/4 port=feature5
//#pragma HLS INTERFACE m_axi depth=M5*C5*C5/4 port=output_core5
//#pragma HLS data_pack variable=weight5
//#pragma HLS data_pack variable=feature5
//#pragma HLS data_pack variable=output_core5

//#pragma HLS INTERFACE s_axilite port=Base_addr4 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr5 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr6 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M6*N6*K6*K6/4 port=weight6
//#pragma HLS INTERFACE m_axi depth=N6*H6*H6/4 port=feature6
//#pragma HLS INTERFACE m_axi depth=M6*C6*C6/4 port=output_core6
//#pragma HLS data_pack variable=weight6
//#pragma HLS data_pack variable=feature6
//#pragma HLS data_pack variable=output_core6
//
//#pragma HLS INTERFACE s_axilite port=Base_addr7 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr8 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr9 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M7*N7*K7*K7/4 port=weight7
//#pragma HLS INTERFACE m_axi depth=N7*H7*H7/4 port=feature7
//#pragma HLS INTERFACE m_axi depth=M7*C7*C7/4 port=output_core7
//#pragma HLS data_pack variable=weight7
//#pragma HLS data_pack variable=feature7
//#pragma HLS data_pack variable=output_core7
//
//#pragma HLS INTERFACE s_axilite port=Base_addr10 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr11 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr12 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M8*N8*K8*K8/4 port=weight8
//#pragma HLS INTERFACE m_axi depth=N8*H8*H8/4 port=feature8
//#pragma HLS INTERFACE m_axi depth=M8*C8*C8/4 port=output_core8
//#pragma HLS data_pack variable=weight8
//#pragma HLS data_pack variable=feature8
//#pragma HLS data_pack variable=output_core8
//
//
//#pragma HLS INTERFACE s_axilite port=Base_addr13 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr14 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr15 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M8*N8*K8*K8/4 port=weight9
//#pragma HLS INTERFACE m_axi depth=N8*H8*H8/4 port=feature9
//#pragma HLS INTERFACE m_axi depth=M8*C8*C8/4 port=output_core9
//#pragma HLS data_pack variable=weight9
//#pragma HLS data_pack variable=feature9
//#pragma HLS data_pack variable=output_core9
//
//#pragma HLS INTERFACE s_axilite port=Base_addr16 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr17 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr18 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M8*N8*K8*K8/4 port=weight10
//#pragma HLS INTERFACE m_axi depth=N8*H8*H8/4 port=feature10
//#pragma HLS INTERFACE m_axi depth=M8*C8*C8/4 port=output_core10
//#pragma HLS data_pack variable=weight10
//#pragma HLS data_pack variable=feature10
//#pragma HLS data_pack variable=output_core10
//
//#pragma HLS INTERFACE s_axilite port=Base_addr19 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr20 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr21 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M11*N11*K11*K11/4 port=weight11
//#pragma HLS INTERFACE m_axi depth=N11*H11*H11/4 port=feature11
//#pragma HLS INTERFACE m_axi depth=M11*C11*C11/4 port=output_core11
//#pragma HLS data_pack variable=weight11
//#pragma HLS data_pack variable=feature11
//#pragma HLS data_pack variable=output_core11
//
//#pragma HLS INTERFACE s_axilite port=Base_addr22 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr23 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr24 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M11*N11*K11*K11/4 port=weight12
//#pragma HLS INTERFACE m_axi depth=N11*H11*H11/4 port=feature12
//#pragma HLS INTERFACE m_axi depth=M11*C11*C11/4 port=output_core12
//#pragma HLS data_pack variable=weight12
//#pragma HLS data_pack variable=feature12
//#pragma HLS data_pack variable=output_core12
//
//#pragma HLS INTERFACE s_axilite port=Base_addr25 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr26 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr27 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M11*N11*K11*K11/4 port=weight13
//#pragma HLS INTERFACE m_axi depth=N11*H11*H11/4 port=feature13
//#pragma HLS INTERFACE m_axi depth=M11*C11*C11/4 port=output_core13
//#pragma HLS data_pack variable=weight13
//#pragma HLS data_pack variable=feature13
//#pragma HLS data_pack variable=output_core13
//	conv1:conv3_3_1<M1,N1,H1,C1,K1,S1,Tr1,Tc1,Tm1,Tn1>(weight1,feature1,output_core1,con,Base_addr37,Base_addr38,Base_addr39);

	conv2:conv3_3_1<TmBuff2,TnBuff2,TrBuff2,TcBuff2,Tr2,Tc2,Tm2,Tn2,TmBuff2,TnBuff2,Tk2,Tri2,Tri2>(weight2,feature2,output_core2,con,Base_addr34,Base_addr35,Base_addr36,M2,H2,C2,K2,H2,S2);
//	conv3:conv3_3_1<M3,N3,H3,C3,K3,S3,Tr3,Tc3,Tm3,Tn3>(weight3,feature3,output_core3,con,Base_addr31,Base_addr32,Base_addr33);
//	conv4:conv3_3_1<M4,N4,H4,C4,K4,S4,Tr4,Tc4,Tm4,Tn4>(weight4,feature4,output_core4,con,Base_addr28,Base_addr29,Base_addr30);
//	conv5:conv3_3_1<M5,N5,H5,C5,K5,S5,Tr5,Tc5,Tm5,Tn5>(weight5,feature5,output_core5,con,Base_addr1,Base_addr2,Base_addr3);
//	conv6:conv3_3_1<M6,N6,H6,C6,K6,S6,Tr6,Tc6,Tm6,Tn6>(weight6,feature6,output_core6,con,Base_addr4,Base_addr5,Base_addr6);
//	conv7:conv3_3_1<M7,N7,H7,C7,K7,S7,Tr7,Tc7,Tm7,Tn7>(weight7,feature7,output_core7,con,Base_addr7,Base_addr8,Base_addr9);
//	conv8:conv3_3_1<M8,N8,H8,C8,K8,S8,Tr8,Tc8,Tm8,Tn8>(weight8,feature8,output_core8,con,Base_addr10,Base_addr11,Base_addr12);
//	conv9:conv3_3_1<M8,N8,H8,C8,K8,S8,Tr8,Tc8,Tm8,Tn8>(weight9,feature9,output_core9,con,Base_addr13,Base_addr14,Base_addr15);
//	conv10:conv3_3_1<M8,N8,H8,C8,K8,S8,Tr8,Tc8,Tm8,Tn8>(weight10,feature10,output_core10,con,Base_addr16,Base_addr17,Base_addr18);
//	conv11:conv3_3_1<M11,N11,H11,C11,K11,S11,Tr11,Tc11,Tm11,Tn11>(weight11,feature11,output_core11,con,Base_addr19,Base_addr20,Base_addr21);
//	conv12:conv3_3_1<M11,N11,H11,C11,K11,S11,Tr11,Tc11,Tm11,Tn11>(weight12,feature12,output_core12,con,Base_addr22,Base_addr23,Base_addr24);
//	conv13:conv3_3_1<M11,N11,H11,C11,K11,S11,Tr11,Tc11,Tm11,Tn11>(weight13,feature13,output_core13,con,Base_addr25,Base_addr26,Base_addr27);
}

