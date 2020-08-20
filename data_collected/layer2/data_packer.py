import numpy as np
import os

files=os.listdir("./")
print('data to pack',files)


#const int M1 = 96, N1 = 3, C1 = 32, S1 = 4, K1 = 11;
#const int H1 = C1*S1 + K1-1;
#const int M2 = 256, N2 = 48, C2 = 16, S2 = 1, K2 = 5;
#const int H2 = C2*S2 + K2-1;
#const int M3 = 384, N3 = 256, C3 = 8, S3 = 1, K3 = 3;
#const int H3 = C3*S3 + K3-1;
#const int M4 = 384, N4 = 192, C4 = 8, S4 = 1, K4 = 3;
#const int H4 = C4 - S4 + K4;
#const int M5 = 256, N5 = 192, C5 = 8, S5 = 1, K5 = 3;
#const int H5 = C5 - S5 + K5;




#input channel; output channel; output_dim; kernel_dim, stride
layer_structure=[48,256,16,5,1]

packed_data=[]
for f in files:
    if('npy' in f and 'packed_data' not in f):
        raw=np.load(f,allow_pickle=True)
        for dp in raw:
            packed_data.append([layer_structure+dp[0][0]+[dp[0][1]]+dp[0][2]+dp[0][3],dp[1][0]])


np.save('packed_data1.npy',packed_data)
