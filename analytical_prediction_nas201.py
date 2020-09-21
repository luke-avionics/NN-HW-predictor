import os 
import numpy as np
import math
import re
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
        print(input_params)
        raise Exception('input channel & corresponding tiling needs to be set as one for dw conv')
    result_lat=1
    if comp_mode==0:
        result_lat*=(input_params[2]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[6]) 
    elif comp_mode==1:
        result_lat*=(input_params[2]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[4])  
    else:
        raise Exception('non-supported comp mode')
    return result_lat

def read_if_lat(comp_mode,input_params,net_struct,quant=16):
    tri=max(input_params[4]+net_struct[3]-1,input_params[0])
    tci=max(input_params[5]+net_struct[3]-1,input_params[1])
    if comp_mode==2:
        return math.ceil(input_params[3]*tci*tri/max(min(4,tri),2))*(quant/16)
    else:
        return math.ceil(input_params[3]*tci*tri/max(min(4,input_params[7]),2))*(quant/16)
    
def dw_read_if_lat(comp_mode,input_params,net_struct,quant=16):
    tri=max(input_params[4]+net_struct[3]-1,input_params[0])
    tci=max(input_params[5]+net_struct[3]-1,input_params[1])
    if comp_mode==2:
        return math.ceil(input_params[2]*tci*tri/max(min(4,tri),2))*(quant/16)
    else:
        return math.ceil(input_params[2]*tci*tri/max(min(4,input_params[6]),2))*(quant/16)


def read_we_lat(comp_mode,input_params,net_struct,quant=16):
    if comp_mode==2:
        #print('weight loading',input_params[2]*input_params[3]*net_struct[3] )
        return input_params[2]*input_params[3]*net_struct[3]*(quant/16)
    else: 
        return math.ceil(input_params[2]*input_params[3]*net_struct[3]*net_struct[3]/max(min(4,input_params[6]),2))*(quant/16)

def dw_read_we_lat(comp_mode,input_params,net_struct,quant=16):
    if input_params[3] !=1:
        raise Exception('input channel & corresponding tiling needs to be set as one for dw conv')
    if comp_mode==1:
        return input_params[2]*input_params[3]*net_struct[3]*(quant/16)
    else:
        return math.ceil(input_params[2]*input_params[3]*net_struct[3]*net_struct[3]/max(min(4,input_params[6]),2))*(quant/16)

def write_ofmap(comp_mode,input_params,net_struct,quant=16):
    if comp_mode==2:
        read_write_1=math.ceil(input_params[2]*input_params[0]*input_params[1]/max(min(4,input_params[4]),2))
        clear_buffer=input_params[1]*input_params[2]*(input_params[0]/input_params[4])
    else:
        read_write_1=math.ceil(input_params[2]*input_params[0]*input_params[1]/max(min(4,input_params[6]),2))
        clear_buffer=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
    #print('clear output', read_write_1, clear_buffer)
    
    return (read_write_1+clear_buffer)*(quant/16)


def dw_write_ofmap(comp_mode,input_params,net_struct,quant=16):
    if comp_mode==1:
        read_write_1=math.ceil(input_params[2]*input_params[0]*input_params[1]/max(min(4,input_params[4]),2))
        clear_buffer=input_params[1]*input_params[2]*(input_params[0]/input_params[4])
    else:
        read_write_1=math.ceil(input_params[2]*input_params[0]*input_params[1]/max(min(4,input_params[6]),2))
        clear_buffer=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
    #print('clear output', read_write_1, clear_buffer)
    return (read_write_1+clear_buffer)*(quant/16)



def combined_latency(comp_mode, input_params,net_struct,quant=16):
    outer_loop_tc=net_struct[2]/input_params[1]
    outer_loop_tr=net_struct[2]/input_params[0]
    outer_loop_tm=net_struct[1]/input_params[2]
    outer_loop_tn=net_struct[0]/input_params[3]
    read_if_we_comp=max(comp_engine_lat(comp_mode,input_params,net_struct), read_if_lat(comp_mode,input_params,net_struct,quant=quant))+read_we_lat(comp_mode,input_params,net_struct,quant=quant)
    read_if_we_comp_tn=read_if_we_comp*outer_loop_tn
    inner_lat=write_ofmap(comp_mode,input_params,net_struct,quant=quant)+read_if_we_comp_tn

    return inner_lat*outer_loop_tc*outer_loop_tr*outer_loop_tm
    
    
def dw_combined_latency(comp_mode, input_params,net_struct,quant=16):
    outer_loop_tc=net_struct[2]/input_params[1]
    outer_loop_tr=net_struct[2]/input_params[0]
    outer_loop_tm=net_struct[1]/input_params[2]
    read_if_we_comp=max(dw_comp_engine_lat(comp_mode,input_params,net_struct),\
                        dw_read_if_lat(comp_mode,input_params,net_struct,quant=quant))+\
                        dw_read_we_lat(comp_mode,input_params,net_struct,quant=quant)+\
                        dw_write_ofmap(comp_mode,input_params,net_struct,quant=quant)
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
            #      separate the parallel choice for kernels input_channel and output_channel
            weight_bank_size=net_struct[3]*net_struct[3]*input_params[3]*(input_params[2]/input_params[6])
            weight_bram=input_params[6]*math.ceil(weight_bank_size*quant/max_bank_size)
            total_bram=input_bram+output_bram
            
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
            total_bram=input_bram+output_bram
            
        elif comp_mode==2:
            #TODO: adding additional adder tree cost
            dsp=input_params[4]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            input_bank_size=tci*input_params[3]
            input_bram=tri*math.ceil(input_bank_size*quant/max_bank_size)*2
            
            
            output_bank_size=input_params[1]*input_params[2]
            output_bram=input_params[4]*math.ceil(output_bank_size*quant/max_bank_size)
            
            weight_bank_size=net_struct[3]*input_params[2]*input_params[3]
            weight_bram=net_struct[3]*math.ceil(weight_bank_size*quant/max_bank_size)
            
            total_bram=input_bram+output_bram+weight_bram
    else:
        if comp_mode==0:
            #TODO: cases using completely LUT
            dsp=input_params[6]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            
            input_bank_size=tri*tci*(input_params[2]/input_params[6])
            input_bram=input_params[6]*math.ceil(input_bank_size*quant/max_bank_size)*2
            output_bank_size=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
            output_bram=input_params[6]*math.ceil(output_bank_size*quant/max_bank_size)
            #TODO: in output channel tiling only; input channel is still tiled, fix in auto_runner side
            #      separate the parallel choice for kernels input_channel and output_channel
            weight_bank_size=net_struct[3]*net_struct[3]*(input_params[2]/input_params[6])
            weight_bram=input_params[6]*math.ceil(weight_bank_size*quant/max_bank_size)
            total_bram=input_bram+output_bram
        elif comp_mode==1:
            dsp=input_params[4]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            input_bank_size=tci*input_params[3]
            input_bram=tri*math.ceil(input_bank_size*quant/max_bank_size)*2
            
            
            output_bank_size=input_params[1]*input_params[2]
            output_bram=input_params[4]*math.ceil(output_bank_size*quant/max_bank_size)
            
            weight_bank_size=net_struct[3]*input_params[2]
            weight_bram=net_struct[3]*math.ceil(weight_bank_size*quant/max_bank_size)
            
            total_bram=input_bram+output_bram+weight_bram
            
            
    return (dsp,total_bram)

def sys_latency(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget):
    #input_params_set
    #[[comp_mode,fw,fh,of,if,f(fw),f(fh),f(of),f(if),quant]...]
    #net_struct
    #[[]....]
    #accelerator_alloc
    #{layer_num:accelerator_num}
    latency_break_down={}
    layer_wise_break_down_to_accel={}
    layer_wise_break_down=[]
    for i in input_params_set.keys():
        latency_break_down[i]=0
        layer_wise_break_down_to_accel[i]=[]
    for i, layer_struct in enumerate(net_struct):
        input_params=input_params_set[accelerator_alloc[i]]
        if dw[i]:
            tmp_lat=dw_combined_latency(input_params[0],input_params[1:9],layer_struct,quant=input_params[-1])
            latency_break_down[accelerator_alloc[i]]+=tmp_lat
            layer_wise_break_down_to_accel[accelerator_alloc[i]].append(tmp_lat)
            layer_wise_break_down.append(tmp_lat)
        else:
            if layer_struct[0]==-1:
                tmp_lat=zeroize_lat(layer_struct)
            elif layer_struct[0]==-2:
                tmp_lat=average_pool_lat(layer_struct)
            elif layer_struct[0]==-3:
                tmp_lat=addition_lat(layer_struct,layer_struct[-1])
            else: 
                tmp_lat=combined_latency(input_params[0],input_params[1:9],layer_struct,quant=input_params[-1])
            latency_break_down[accelerator_alloc[i]]+=tmp_lat
            layer_wise_break_down_to_accel[accelerator_alloc[i]].append(tmp_lat)
            layer_wise_break_down.append(tmp_lat)
    bottleneck_latency=0
    for i in latency_break_down.keys(): 
        if latency_break_down[i] >bottleneck_latency:
            bottleneck_latency=latency_break_down[i]
    return bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,layer_wise_break_down

def sys_consumption(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget,platform_specs):
    #input_params_set
    #[[comp_mode,fw,fh,of,if,f(fw),f(fh),f(of),f(if),quant]...]
    #net_struct
    #[[]....]
    #accelerator_alloc
    #{layer_num:accelerator_num}
    consumption_breakdown={}
    for i in input_params_set.keys():
        consumption_breakdown[i]=[0,0]
    for i, layer_struct in enumerate(net_struct):
        input_params=input_params_set[accelerator_alloc[i]]
        if layer_struct[0]==-1:
            res_i=zeroize_consumption(layer_struct)
        elif layer_struct[0]==-2:
            res_i=average_pool_consumption(layer_struct)
        elif layer_struct[0]==-3:
            res_i=addition_consumption(layer_struct,layer_struct[-1])
        else:
            res_i=resource_consumption(input_params[0],input_params[1:9],\
                                                         layer_struct,dw=dw[i],quant=input_params[-1])
        consumption_breakdown[accelerator_alloc[i]]= [max(consumption_breakdown[accelerator_alloc[i]][0],\
                                                         res_i[0]),\
                                                      max(consumption_breakdown[accelerator_alloc[i]][1],\
                                                         res_i[1])]
                
    total_dsp_used=0
    total_bram_used=0
    for i in consumption_breakdown.keys():
        total_dsp_used+=consumption_breakdown[i][0]
        total_bram_used+=consumption_breakdown[i][1]
    if total_dsp_used>platform_specs['dsp']:
        raise Exception('dsp limit exceeded')
    elif total_bram_used>platform_specs['bram']:
        raise Exception('bram exceeded')
    
    for i in accelerator_wise_budget.keys():
        if consumption_breakdown[i][0] > accelerator_wise_budget[i]['dsp']:
            print("Warning: accelerator "+str(i)+" dsp budget exceeded")
        elif consumption_breakdown[i][1]> accelerator_wise_budget[i]['bram']: 
            print("Warning: accelerator "+str(i)+" bram budget exceeded")
    return (total_dsp_used,total_bram_used), consumption_breakdown


def allocate_layers(net_struct,cell_layer):
    accelerator_alloc={}
    accelerator_wise_budget={}
    accelerator_types=['a0','a1','a2']
    for i, layer_struct in enumerate(net_struct):
        accelerator_alloc[i]="a"+str(cell_layer[i])
    return accelerator_alloc, accelerator_types, accelerator_wise_budget
    
# def allocate_layers(net_struct,quant_list,dw,platform_specs,cifar=True):
    # dw_quantization_bins={}
    # std_quantization_bins={}
    # accelerator_alloc={}
    # accelerator_wise_budget={}
    # accelerator_types=[]
    # for i, layer_struct in enumerate(net_struct):
        # if dw[i]:
            # if quant_list[i] not in dw_quantization_bins.keys():
                # #initiate the bins
                # dw_quantization_bins[quant_list[i]]=[i]
            # else:
                # #add layers to the corresponding bins
                # dw_quantization_bins[quant_list[i]].append(i)
        # else:
            # if quant_list[i] not in std_quantization_bins.keys():
                # #initiate the bins
                # std_quantization_bins[quant_list[i]]=[i]
            # else:
                # #add layers to the corresponding bins
                # std_quantization_bins[quant_list[i]].append(i)
    # if cifar:    
        # for i, quant_bit in enumerate(std_quantization_bins.keys()):
            # for layer in std_quantization_bins[quant_bit]:
                # if net_struct[layer][2]>=16:
                    # if "a0"+"q"+str(quant_bit) not in accelerator_types:
                        # accelerator_types.append("a0"+"q"+str(quant_bit))
                    # accelerator_alloc[layer]="a0"+"q"+str(quant_bit)
                # else:
                    # if "a1"+"q"+str(quant_bit) not in accelerator_types:
                        # accelerator_types.append("a1"+"q"+str(quant_bit))
                    # accelerator_alloc[layer]="a1"+"q"+str(quant_bit)
                    
        # for i, quant_bit in enumerate(dw_quantization_bins.keys()):
            # for layer in dw_quantization_bins[quant_bit]:
                # if net_struct[layer][2]>=16:
                    # if "dwa0"+"q"+str(quant_bit) not in accelerator_types:
                        # accelerator_types.append("dwa0"+"q"+str(quant_bit))
                    # accelerator_alloc[layer]="dwa0"+"q"+str(quant_bit)
                # else:
                    # if "dwa1"+"q"+str(quant_bit) not in accelerator_types:
                        # accelerator_types.append("dwa1"+"q"+str(quant_bit))
                    # accelerator_alloc[layer]="dwa1"+"q"+str(quant_bit)
    # else:
        # for i, quant_bit in enumerate(std_quantization_bins.keys()):
            # for layer in std_quantization_bins[quant_bit]:
                # if net_struct[layer][2]>=28:
                    # if "a0"+"q"+str(quant_bit) not in accelerator_types:
                        # accelerator_types.append("a0"+"q"+str(quant_bit))
                    # accelerator_alloc[layer]="a0"+"q"+str(quant_bit)
                # else:
                    # if "a1"+"q"+str(quant_bit) not in accelerator_types:
                        # accelerator_types.append("a1"+"q"+str(quant_bit))
                    # accelerator_alloc[layer]="a1"+"q"+str(quant_bit)
                    
        # for i, quant_bit in enumerate(dw_quantization_bins.keys()):
            # for layer in dw_quantization_bins[quant_bit]:
                # if net_struct[layer][2]>=28:
                    # if "dwa0"+"q"+str(quant_bit) not in accelerator_types:
                        # accelerator_types.append("dwa0"+"q"+str(quant_bit))
                    # accelerator_alloc[layer]="dwa0"+"q"+str(quant_bit)
                # else:
                    # if "dwa1"+"q"+str(quant_bit) not in accelerator_types:
                        # accelerator_types.append("dwa1"+"q"+str(quant_bit))
                    # accelerator_alloc[layer]="dwa1"+"q"+str(quant_bit)
    # # print("="*20)     
    # # print(len(net_struct))
    # # print(len(list(accelerator_alloc.keys())))
    # # print(accelerator_alloc)
    # # print("="*20)  
    # #return None
    # return accelerator_alloc, accelerator_types, accelerator_wise_budget


def zeroize_lat(net_struct):
    return net_struct[1]*net_struct[2]**2/4/(16/32)
def average_pool_lat(net_struct):
    #read and write *2
    return net_struct[1]*net_struct[2]**2*2/4/(16/32)
def addition_lat(net_struct, num_input):
    return net_struct[1]*net_struct[2]**2/4/(16/32)*num_input
def zeroize_consumption(net_struct):
    return (1,1)
def average_pool_consumption(net_struct):
    #assuming 3x3 pooling and same padding
    return (9,9)
def addition_consumption(net_struct, num_input): 
    return (num_input,1)
def cell_to_layers(cell_def):
    #-1 zeroize  -2 avg pool -3 addition(with numbers in the end denoting addtion number)
    net_struct=[[4,16,32,3,1]]
    cell_layer=[0]
    #first stage
    for i in range(5):
        for j in range(6):
            if cell_def[j]==0: 
                net_struct.append([-1,16,32,0,0])
                cell_layer.append(0)
            elif cell_def[j]==2:
                net_struct.append([16,16,32,1,1])
                cell_layer.append(0)
            elif cell_def[j]==3:
                net_struct.append([16,16,32,3,1])
                cell_layer.append(0)
            elif cell_def[j]==4:
                net_struct.append([-2,16,32,0,0])
                cell_layer.append(0)
        net_struct.append([-3,16,32,0,2])
        cell_layer.append(0)
        net_struct.append([-3,16,32,0,3])
        cell_layer.append(0)
    #residue block 1
    net_struct.append([16,32,16,3,2])
    cell_layer.append(1)
    net_struct.append([32,32,16,3,1])
    cell_layer.append(1)
    net_struct.append([32,32,16,1,1])
    cell_layer.append(1)
    net_struct.append([-2,32,16,0,0])
    cell_layer.append(1)
    #second stage
    for i in range(5):
        for j in range(6):
            if cell_def[j]==0: 
                net_struct.append([-1,32,16,0,0])
                cell_layer.append(1)
            elif cell_def[j]==2:
                net_struct.append([32,32,16,1,1])
                cell_layer.append(1)
            elif cell_def[j]==3:
                net_struct.append([32,32,16,3,1])
                cell_layer.append(1)
            elif cell_def[j]==4:
                net_struct.append([-2,32,16,0,0])
                cell_layer.append(1)
        net_struct.append([-3,32,16,0,2])
        cell_layer.append(1)
        net_struct.append([-3,32,16,0,3])
        cell_layer.append(1)
    #residue block 2
    net_struct.append([32,64,8,3,2] )
    net_struct.append([64,64,8,3,1])
    net_struct.append([64,64,8,1,1])
    net_struct.append([-2,64,8,0,0])
    cell_layer+=([2]*4)
    #second stage
    for i in range(5):
        for j in range(6):
            if cell_def[j]==0: 
                net_struct.append([-1,64,8,0,0])
            elif cell_def[j]==2:
                net_struct.append([64,64,8,1,1])
            elif cell_def[j]==3:
                net_struct.append([64,64,8,3,1])   
            elif cell_def[j]==4:
                net_struct.append([-2,64,8,0,0])   
        net_struct.append([-3,64,8,0,2] )
        net_struct.append([-3,64,8,0,3] )
    cell_layer+=([2]*40)
    return net_struct,cell_layer
        
cell_def_test=[2,2,2,1,1,2]
platform_specs={'dsp':900,'bram':1000}
input_params_set={'a0':[1,32,32,16,16,32,32,8,8,32],'a1':[1,16,16,32,32,16,16,8,8,32],'a2':[1,8,8,32,32,8,8,16,16,32]}


net_struct,cell_layer=cell_to_layers(cell_def_test)

quant_list=[32]*len(net_struct)
dw=[0]*len(net_struct)
print(net_struct)
accelerator_alloc, accelerator_types, accelerator_wise_budget=allocate_layers(net_struct,cell_layer)
(total_dsp_used,total_bram_used), consumption_breakdown=sys_consumption(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget,platform_specs)
bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,layer_wise_break_down=sys_latency(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget)
print(bottleneck_latency)