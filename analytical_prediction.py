import os 
import numpy as np
import math
dnn_structure=[\
              [4,96,32,11,4],\
              [48,256,16,5,1],\
              [256,384,8,3,1],\
              [192,384,8,3,1],\
              [192,256,8,3,1],\
              [48,256,112,5,1],\
              [4,48,224,5,1]

]


def pack_data(fn,keyword):
    files=os.listdir(fn)
    packed_data=[]
    for f in files:
        if(keyword in f):
            raw=np.load(fn+f,allow_pickle=True)
            for dp in raw:
                packed_data.append([dp[0][0][:]+[dp[0][1]],dp[1][0]])
    return packed_data


def comp_engine_lat(comp_mode,input_params,net_struct):
    result_lat=1
    if comp_mode==0:
        result_lat*=(input_params[2]*input_params[3]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[6]) 
    elif comp_mode==1:
        result_lat*=(input_params[2]*input_params[3]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[6]/input_params[7])
    elif comp_mode==2:
        result_lat*=(input_params[2]*input_params[3]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[0])        
    print(result_lat)
    return result_lat


def read_if_lat(input_params,net_struct):
    tri=max(input_params[4]+net_struct[3]-1,input_params[0])
    tci=max(input_params[5]+net_struct[3]-1,input_params[1])
    return input_params[3]*tci*math.ceil(tri/4)


def read_we_lat(input_params,net_struct):
    return input_params[2]*input_params[3]*net_struct[3]*net_struct[3]/4


def write_ofmap(input_params,net_struct):
    read_write_1=input_params[2]*input_params[0]*input_params[1]/4
    clear_buffer=input_params[0]*input_params[1]
    return read_write_1+clear_buffer


def combined_latency(comp_mode, input_params,net_struct):
    outer_loop_tc=net_struct[2]/input_params[1]
    outer_loop_tr=net_struct[2]/input_params[0]
    outer_loop_tm=net_struct[0]/input_params[2]
    outer_loop_tn=net_struct[1]/input_params[3]
    read_if_we_comp=max(comp_engine_lat(comp_mode,input_params,net_struct), read_if_lat(input_params,net_struct))+read_we_lat(input_params,net_struct)
    read_if_we_comp_tn=read_if_we_comp*outer_loop_tn
    inner_lat=write_ofmap(input_params,net_struct)+read_if_we_comp_tn
    return inner_lat*outer_loop_tc*outer_loop_tr*outer_loop_tm


def resource_consumption(input_params):
    dsp=0
    bram=0
    return (dsp,bram)

raw=np.load('fixed_hw_cp1_data4.npy',allow_pickle=True)
print(raw[0][0])
print(raw[0][1], (raw[0][0][13]))
print(combined_latency(raw[0][0][13],raw[0][0][5:13],dnn_structure[3]))

#lat=combined_latency(2,[8,4,8,8,8,4,8,8],dnn_structure[3])
#print(lat)