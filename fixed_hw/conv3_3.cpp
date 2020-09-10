#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "conv3x3.h"
#include <ap_int.h>
using namespace std;



int main(){


    //mem allocation
    data_type ****weight=(data_type ****)malloc(M*sizeof(data_type***));
    for(unsigned int i=0; i<M;i++){
        weight[i]=(data_type ***)malloc(N*sizeof(data_type**));
        for(unsigned int j=0; j<N;j++){
            weight[i][j]=(data_type **)malloc(K*sizeof(data_type*));
            for(unsigned int k=0; k<K;k++){
                weight[i][j][k]=(data_type *)malloc(K*sizeof(data_type));
            }
        }
    }


    data_type ***input=(data_type ***)malloc(N*sizeof(data_type**));
    for(unsigned int i=0; i<N; i++){
        input[i]=(data_type **)malloc(H*sizeof(data_type*));
        for(unsigned int j=0;j<H;j++){
            input[i][j]=(data_type *)malloc(H*sizeof(data_type));
        }
    }


    data_type ***output=(data_type ***)malloc(M*sizeof(data_type**));
    for(unsigned int i=0; i<M; i++){
        output[i]=(data_type **)malloc(C*sizeof(data_type*));
        for(unsigned int j=0;j<C;j++){
            output[i][j]=(data_type *)malloc(C*sizeof(data_type));
        }
    }



    for(unsigned int i=0; i<N;i++)
        for(unsigned int j=1; j<(H-1);j++)
            for(unsigned int k=1; k<(H-1); k++)
            	//input[i][j][k]=0.021;
                input[i][j][k]=(float)(j-1)/100.0;

    for(unsigned int i=0; i<M;i++)
        for(unsigned int j=0; j<C;j++)
            for(unsigned int k=0; k<C; k++)
                output[i][j][k]=0;

    //padding zero
    for(unsigned int i=0; i<N;i++)
        for(unsigned int j=0; j<H;j++)
        	//input[i][j][0]=0.021;
        	input[i][j][0]=0;

    for(unsigned int i=0; i<N;i++)
        for(unsigned int j=0; j<H;j++)
        	//input[i][0][j]=0.021;
            input[i][0][j]=0;




    for(unsigned int i=0; i<M;i++)
        for(unsigned int j=0; j<N;j++)
            for(unsigned int k=0; k<K; k++)
                for(unsigned int l=0; l<K; l++ )
                	//weight[i][j][k][l]=0.055;
                    weight[i][j][k][l]=(float)k/100.0;




    cout<<"Finished generating the data\n";


    //flatten the matrix
   dma_data * flatten_weight=(dma_data *)malloc(M*N*K*K/4*sizeof(dma_data));
   dma_data * flatten_input=(dma_data *)malloc(N*H*H/4*sizeof(dma_data));
   dma_data * flatten_output=(dma_data *)malloc(M*C*C/4*sizeof(dma_data));
   dma_data * src=(dma_data *)malloc((M*N*K*K+N*H*H+M*C*C)/4*sizeof(dma_data));
   for(unsigned int i=0; i<M/4;i++)
       for(unsigned int j=0; j<N;j++)
           for(unsigned int k=0; k<K; k++)
               for(unsigned int l=0; l<K; l++ )
               {
            	   flatten_weight[i*N*K*K+j*K*K+k*K+l].data.data0
				   =(data_type)weight[i*4][j][k][l];
            	   flatten_weight[i*N*K*K+j*K*K+k*K+l].data.data1
				   =(data_type)weight[i*4+1][j][k][l];
            	   flatten_weight[i*N*K*K+j*K*K+k*K+l].data.data2
				   =(data_type)weight[i*4+2][j][k][l];
            	   flatten_weight[i*N*K*K+j*K*K+k*K+l].data.data3
				   =(data_type)weight[i*4+3][j][k][l];
               }

   for(unsigned int i=0; i<N; i++)
	   for(unsigned int j=0; j<H/4; j++)
		   for(unsigned int k=0; k<H; k++){
			   flatten_input[j*H*N+i*H+k].data.data0
			   =(data_type)input[i][j*4][k];
			   flatten_input[j*H*N+i*H+k].data.data1
			   =(data_type)input[i][j*4+1][k];
			   flatten_input[j*H*N+i*H+k].data.data2
			   =(data_type)input[i][j*4+2][k];
			   flatten_input[j*H*N+i*H+k].data.data3
			   =(data_type)input[i][j*4+3][k];
		   }

   for(unsigned int i=0; i<M/4; i++)
	   for(unsigned int j=0; j<C; j++)
		   for(unsigned int k=0; k<C; k++){
			   flatten_output[i*C*C+j*C+k].data.data0
			   =(data_type)output[i*4][j][k];
			   flatten_output[i*C*C+j*C+k].data.data1
			   =(data_type)output[i*4+1][j][k];
			   flatten_output[i*C*C+j*C+k].data.data2
			   =(data_type)output[i*4+2][j][k];
			   flatten_output[i*C*C+j*C+k].data.data3
			   =(data_type)output[i*4+3][j][k];
		   }


   cout<<"Finished converting the data\n";


   //module
   //possible solution, specify regular r/w pattern
   conv3_3(flatten_weight,flatten_input,flatten_output,1,0,0,0);
   std::cout<<M*C*C/4<<endl;
    for(unsigned int j=0; j<K;j++){
        for(unsigned int k=0; k<K; k++){
            cout<<flatten_weight[K*K*N+j*K+k].data.data1;
            cout<<", ";
        }
            cout<<"==========\n";
    }

    cout<<"===========================================================================\n=============================================================\n";

    for(unsigned int j=0; j<C;j++){
        for(unsigned int k=0; k<C; k++){
            cout<<flatten_output[C*C*0+j*C+k].data.data0;
            cout<<", ";
        }
            cout<<"==========\n";
    }
    cout<<"===========================================================================\n=============================================================\n";

    for(unsigned int j=0; j<C;j++){
        for(unsigned int k=0; k<C; k++){
            cout<<flatten_output[C*C*1+j*C+k].data.data0;
            cout<<", ";
        }
            cout<<"==========\n";
    }
    for(unsigned int j=0; j<C;j++){
        for(unsigned int k=0; k<C; k++){
            cout<<flatten_output[C*C*2+j*C+k].data.data0;
            cout<<", ";
        }
            cout<<"==========\n";
    }

    return 0;

}
