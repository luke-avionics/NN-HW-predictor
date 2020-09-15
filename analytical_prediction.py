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
                     *net_struct[3]/input_params[4])        
    #print('comp lat ', result_lat)
    return result_lat

def dw_comp_engine_lat(comp_mode,input_params,net_struct):
    if input_params[3] !=1:
        raise Exception('input channel & corresponding tiling needs to be set as one for dw conv')
    result_lat=1
    if comp_mode==0:
        result_lat*=(input_params[2]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[6]) 
    elif comp_mdoe==1:
        result_lat*=(input_params[2]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[4])  
    else:
        raise Exception('non-supported comp mode')
    
    return result_lat

def read_if_lat(input_params,net_struct,quant=16):
    tri=max(input_params[4]+net_struct[3]-1,input_params[0])
    tci=max(input_params[5]+net_struct[3]-1,input_params[1])
    return input_params[3]*tci*math.ceil(tri/4)*(quant/16)
    
def dw_read_if_lat(input_params,net_struct,quant=16):
    tri=max(input_params[4]+net_struct[3]-1,input_params[0])
    tci=max(input_params[5]+net_struct[3]-1,input_params[1])
    return input_params[2]*tci*math.ceil(tri/4)*(quant/16)


def read_we_lat(comp_mode,input_params,net_struct,quant=16):
    if comp_mode==2:
        #print('weight loading',input_params[2]*input_params[3]*net_struct[3] )
        return input_params[2]*input_params[3]*net_struct[3]*(quant/16)
    else: 
        return input_params[2]*input_params[3]*net_struct[3]*net_struct[3]/4*(quant/16)

def dw_read_we_lat(comp_mode,input_params,net_struct,quant=16):
    if input_params[3] !=1:
        raise Exception('input channel & corresponding tiling needs to be set as one for dw conv')
    if comp_mode==1:
        return input_params[2]*input_params[3]*net_struct[3]*(quant/16)
    else:
        return input_params[2]*input_params[3]*net_struct[3]*net_struct[3]/4*(quant/16)

def write_ofmap(comp_mode,input_params,net_struct,quant=16):
    if comp_mode==2:
        read_write_1=input_params[2]*input_params[0]*input_params[1]/4
        clear_buffer=input_params[1]*input_params[2]*(input_params[0]/input_params[4])
    else:
        read_write_1=input_params[2]*input_params[0]*input_params[1]/4
        clear_buffer=input_params[0]*input_params[1]
    #print('clear output', read_write_1, clear_buffer)
    return (read_write_1+clear_buffer)*(quant/16)


def dw_write_ofmap(comp_mode,input_params,net_struct,quant=16):
    if comp_mode==1:
        read_write_1=input_params[2]*input_params[0]*input_params[1]/4
        clear_buffer=input_params[1]*input_params[2]*(input_params[0]/input_params[4])
    else:
        read_write_1=input_params[2]*input_params[0]*input_params[1]/4
        clear_buffer=input_params[0]*input_params[1]
    #print('clear output', read_write_1, clear_buffer)
    return (read_write_1+clear_buffer)*(quant/16)



def combined_latency(comp_mode, input_params,net_struct):
    outer_loop_tc=net_struct[2]/input_params[1]
    outer_loop_tr=net_struct[2]/input_params[0]
    outer_loop_tm=net_struct[1]/input_params[2]
    outer_loop_tn=net_struct[0]/input_params[3]
    read_if_we_comp=max(comp_engine_lat(comp_mode,input_params,net_struct), read_if_lat(input_params,net_struct))+read_we_lat(comp_mode,input_params,net_struct)
    read_if_we_comp_tn=read_if_we_comp*outer_loop_tn
    inner_lat=write_ofmap(comp_mode,input_params,net_struct)+read_if_we_comp_tn
    return inner_lat*outer_loop_tc*outer_loop_tr*outer_loop_tm
    
    
def dw_combined_latency(comp_mode, input_params,net_struct):
    outer_loop_tc=net_struct[2]/input_params[1]
    outer_loop_tr=net_struct[2]/input_params[0]
    outer_loop_tm=net_struct[1]/input_params[2]
    read_if_we_comp=max(dw_comp_engine_lat(comp_mode,input_params,net_struct),\
                        dw_read_if_lat(input_params,net_struct))+\
                        dw_read_we_lat(comp_mode,input_params,net_struct)+\
                        dw_write_ofmap(comp_mode,input_params,net_struct)
    return outer_loop_tc*outer_loop_tr*outer_loop_tm*read_if_we_comp

def resource_consumption(comp_mode,input_params,net_struct,dw=False,quant=16):
    max_bank_size=1125*16
    if not dw:
        if comp_mode==0:
            #TODO: cases using completely LUT
            dsp=input_params[6]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            
            input_bank_size=tri*tci*(input_params[3]/input_params[7])
            input_bram=input_params[7]*math.ceil(input_bank_size*quant/max_bank_size)*2
            
            output_bank_size=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
            output_bram=input_params[6]*math.ceil(output_bank_size*quant/max_bank_size)
            #TODO: in output channel tiling only; input channel is still tiled, fix in auto_runner side
            weight_bank_size=net_struct[3]*net_struct[3]*input_params[3]*(input_params[2]/input_params[6])
            weight_bram=input_params[6]*math.ceil(weight_bank_size*quant/max_bank_size)
            total_bram=input_bram+output_bram+weight_bram
            
        elif comp_mode==1:
            dsp=input_params[6]*input_params[7]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            
            input_bank_size=tri*tci*(input_params[3]/input_params[7])
            input_bram=input_params[7]*math.ceil(input_bank_size*quant/max_bank_size)*2
            
            output_bank_size=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
            output_bram=input_params[6]*math.ceil(output_bank_size*quant/max_bank_size)
            
            weight_bank_size=net_struct[3]*net_struct[3]*(input_params[3]/input_params[7])*(input_params[2]/input_params[6])
            weight_bram=input_params[6]*input_params[7]*math.ceil(weight_bank_size*quant/max_bank_size)
            total_bram=input_bram+output_bram+weight_bram
            
        elif comp_mode==2:
            #TODO: adding additional adder tree cost
            dsp=input_params[4]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            input_bank_size=tci*input_params[3]
            input_bram=tri*math.ceil(input_bank_size*quant/max_bank_size)*2
            print(input_bram)
            
            output_bank_size=input_params[1]*input_params[2]
            output_bram=input_params[4]*math.ceil(output_bank_size*quant/max_bank_size)
            print(output_bram)
            weight_bank_size=net_struct[3]*input_params[2]*input_params[3]
            weight_bram=net_struct[3]*math.ceil(weight_bank_size*quant/max_bank_size)
            print(weight_bram)
            total_bram=input_bram+output_bram+weight_bram
    else:
        if comp_mode==0:
            
    return (dsp,total_bram)


#files=['fixed_hw_cp1_data4.npy','fixed_hw_cp2_data4.npy','fixed_hw_cp2_data7.npy']
files=['fixed_hw_cp3_data7.npy']
for i,fn in enumerate(files):
    if i ==0:
        raw=np.load(fn,allow_pickle=True)
    else:
        raw=np.concatenate((raw,np.load(fn,allow_pickle=True))) 

#raw=np.load('fixed_hw_cp1_data4.npy',allow_pickle=True)
raw_len=len(raw)
print(raw_len)
print(raw[0][1], (raw[0][0][13]))

error_list=[]
ctr=0
for i, dp in enumerate(raw):
    absolute_truth=float(dp[1])
    predicted_perf=combined_latency(dp[0][13],dp[0][5:13],dp[0][0:5])
    if(np.abs(absolute_truth-predicted_perf)/absolute_truth>0.2):
        print('truth: ',absolute_truth,'  predicted: ', predicted_perf)
        print('tiling')
        print(dp[0][5:13])
        print('net structure')
        print(dp[0][0:5])
        ctr+=1
    error_list.append(np.abs(absolute_truth-predicted_perf)/absolute_truth)
print(np.mean(error_list))
print(ctr/raw_len)


consumption=resource_consumption(2,[112,28,3,2,28,28,3,2],dnn_structure[6])
print(consumption)