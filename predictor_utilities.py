import os 
import numpy as np
import math
import re
import time 
from  multiprocessing import Queue
import multiprocessing
######################################
##common utitiltes
######################################

def closestMultiple(n, x): 
    if x > n: 
        return x; 
    z = (int)(x / 2); 
    n = n + z; 
    n = n - (n % x); 
    return n; 

def multi_p(func,args,output_q,num_worker_threads,dump_yard):
    #routine to distribute workers to multi cores
    #BETTER leave it

    #length of args has to be the multiple of num_worker_threads
    args=list(args)
    run_ites=int((len(args))//num_worker_threads)
    for run_ite in range(run_ites):
        processes = [multiprocessing.Process(target=func, args=([args[i]])) for i in range(run_ite*num_worker_threads,(run_ite+1)*num_worker_threads)]
        #print(len(processes))
        #print('queue size: ',score_pair.qsize())
        for p in processes:
            p.start()
            time.sleep(0.01)
        print('all job started')
        while not output_q.empty():
            pair=output_q.get()
            dump_yard.append(pair)
        for p in processes:
            p.join()
        print('all job finishes')
    while not output_q.empty():
        pair=output_q.get()
        dump_yard.append(pair)
    return None
    
    
######################################
##FPGA performance predictor specific
######################################
def model_profiler(net_struct,layer_block_corr=None):
    param_size=0
    param_size_bits=0
    mac_size=1
    block_wise_performance={}
    if layer_block_corr!= None:
        for key in layer_block_corr.keys():
            block_wise_performance[key]=0
    for i, layer_struct in enumerate(net_struct):
        mac_size+=(layer_struct[0]*layer_struct[1]*(layer_struct[2]**2)*(layer_struct[3]**2))
        param_size+=(layer_struct[0]*layer_struct[1]*(layer_struct[3]**2))
        if layer_block_corr!= None:
            for key in layer_block_corr.keys():
                if i in layer_block_corr[key]:
                    block_wise_performance[key]+=(layer_struct[0]*layer_struct[1]*(layer_struct[2]**2)*(layer_struct[3]**2))
                    #break
    #print(mac_size)
    return param_size,mac_size,block_wise_performance

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
            if quant > 16:
                dsp=input_params[6]*2 
            elif quant <=16 and quant > 8:
                dsp=input_params[6]
            elif quant <= 8:
                dsp=max(1,input_params[6]//2)
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
            if quant > 16:
                dsp=input_params[6]*input_params[7]*2 
            elif quant <=16 and quant > 8:
                dsp=input_params[6]*input_params[7]
            elif quant <= 8:
                dsp=max(1,input_params[6]*input_params[7]//2)
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
            if quant > 16:
                dsp=input_params[4]*2 
            elif quant <=16 and quant > 8:
                dsp=input_params[4]
            elif quant <= 8:
                dsp=max(1,input_params[4]//2)
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
            if quant > 16:
                dsp=input_params[6]*2 
            elif quant <=16 and quant > 8:
                dsp=input_params[6]
            elif quant <= 8:
                dsp=max(1,input_params[6]//2)
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
            if quant > 16:
                dsp=input_params[4]*2 
            elif quant <=16 and quant > 8:
                dsp=input_params[4]
            elif quant <= 8:
                dsp=max(1,input_params[4]//2)
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
        consumption_breakdown[accelerator_alloc[i]]= [max(consumption_breakdown[accelerator_alloc[i]][0],\
                                                         resource_consumption(input_params[0],input_params[1:9],\
                                                         layer_struct,dw=dw[i],quant=input_params[-1])[0]),\
                                                      max(consumption_breakdown[accelerator_alloc[i]][1],\
                                                         resource_consumption(input_params[0],input_params[1:9],\
                                                         layer_struct,dw=dw[i],quant=input_params[-1])[1])]
                
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


def allocate_layers(net_struct,quant_list,dw,platform_specs,layer_block_corr,cifar=True,edd=False,channel_part=False):
    dw_quantization_bins={}
    std_quantization_bins={}
    accelerator_alloc={}
    accelerator_wise_budget={}
    accelerator_types=[]
    for i, layer_struct in enumerate(net_struct):
        if dw[i]:
            if quant_list[i] not in dw_quantization_bins.keys():
                #initiate the bins
                dw_quantization_bins[quant_list[i]]=[i]
            else:
                #add layers to the corresponding bins
                dw_quantization_bins[quant_list[i]].append(i)
        else:
            if quant_list[i] not in std_quantization_bins.keys():
                #initiate the bins
                std_quantization_bins[quant_list[i]]=[i]
            else:
                #add layers to the corresponding bins
                std_quantization_bins[quant_list[i]].append(i)
    if not channel_part:
        if cifar:    
            for i, quant_bit in enumerate(std_quantization_bins.keys()):
                for layer in std_quantization_bins[quant_bit]:
                    if net_struct[layer][2]>=16:
                        if "a0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a0"+"q"+str(quant_bit)
                    else:
                        if "a1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a1"+"q"+str(quant_bit)
                        
            for i, quant_bit in enumerate(dw_quantization_bins.keys()):
                for layer in dw_quantization_bins[quant_bit]:
                    if net_struct[layer][2]>=16:
                        if "dwa0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa0"+"q"+str(quant_bit)
                    else:
                        if "dwa1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa1"+"q"+str(quant_bit)
        else:
            for i, quant_bit in enumerate(std_quantization_bins.keys()):
                for layer in std_quantization_bins[quant_bit]:
                    if net_struct[layer][2]>=28:
                        if "a0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a0"+"q"+str(quant_bit)
                    else:
                        if "a1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a1"+"q"+str(quant_bit)
                        
            for i, quant_bit in enumerate(dw_quantization_bins.keys()):
                for layer in dw_quantization_bins[quant_bit]:
                    if net_struct[layer][2]>=28:
                        if "dwa0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa0"+"q"+str(quant_bit)
                    else:
                        if "dwa1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa1"+"q"+str(quant_bit)
    else:
    #applies specifically to Yonggan's space
        if not edd:    
            for i, quant_bit in enumerate(std_quantization_bins.keys()):
                for layer in std_quantization_bins[quant_bit]:
                    if layer in layer_block_corr[0] or layer in layer_block_corr[1] or\
                       layer in layer_block_corr[2] or layer in layer_block_corr[3] or\
                       layer in layer_block_corr[4]:
                        if "a0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a0"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[5] or layer in layer_block_corr[6] or\
                         layer in layer_block_corr[7] or layer in layer_block_corr[8]:
                        if "a1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a1"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[9] or layer in layer_block_corr[10] or\
                         layer in layer_block_corr[11] or layer in layer_block_corr[12]:
                        if "a2"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a2"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a2"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[13] or layer in layer_block_corr[14] or\
                         layer in layer_block_corr[15] or layer in layer_block_corr[16]:
                        if "a3"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a3"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a3"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[17] or layer in layer_block_corr[18] or\
                       layer in layer_block_corr[19] or layer in layer_block_corr[20] or\
                       layer in layer_block_corr[21]:
                        if "a4"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a4"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a4"+"q"+str(quant_bit)
            for i, quant_bit in enumerate(dw_quantization_bins.keys()):
                for layer in dw_quantization_bins[quant_bit]:
                    if layer in layer_block_corr[0] or layer in layer_block_corr[1] or\
                       layer in layer_block_corr[2] or layer in layer_block_corr[3] or\
                       layer in layer_block_corr[4]:
                        if "dwa0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa0"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[5] or layer in layer_block_corr[6] or\
                         layer in layer_block_corr[7] or layer in layer_block_corr[8]:
                        if "dwa1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa1"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[9] or layer in layer_block_corr[10] or\
                         layer in layer_block_corr[11] or layer in layer_block_corr[12]:
                        if "dwa2"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa2"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa2"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[13] or layer in layer_block_corr[14] or\
                         layer in layer_block_corr[15] or layer in layer_block_corr[16]:
                        if "dwa3"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa3"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa3"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[17] or layer in layer_block_corr[18] or\
                       layer in layer_block_corr[19] or layer in layer_block_corr[20] or\
                       layer in layer_block_corr[21]:
                        if "dwa4"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa4"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa4"+"q"+str(quant_bit)
        else:
            for i, quant_bit in enumerate(std_quantization_bins.keys()):
                for layer in std_quantization_bins[quant_bit]:
                    if layer in layer_block_corr[0] or layer in layer_block_corr[1] or\
                       layer in layer_block_corr[2] or layer in layer_block_corr[3]:
                        if "a0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a0"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[4] or layer in layer_block_corr[5]or\
                         layer in layer_block_corr[6] or layer in layer_block_corr[7]:
                        if "a1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a1"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[8] or layer in layer_block_corr[9]or\
                         layer in layer_block_corr[10] or layer in layer_block_corr[11]:
                        if "a2"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a2"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a2"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[12] or layer in layer_block_corr[13]or\
                         layer in layer_block_corr[14] or layer in layer_block_corr[15]:
                        if "a3"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a3"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a3"+"q"+str(quant_bit)
      
            for i, quant_bit in enumerate(dw_quantization_bins.keys()):
                for layer in dw_quantization_bins[quant_bit]:
                    if layer in layer_block_corr[0] or layer in layer_block_corr[1]or\
                       layer in layer_block_corr[2] or layer in layer_block_corr[3]:
                        if "dwa0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa0"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[4] or layer in layer_block_corr[5]or\
                         layer in layer_block_corr[6] or layer in layer_block_corr[7]:
                        if "dwa1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa1"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[8] or layer in layer_block_corr[9]or\
                         layer in layer_block_corr[10] or layer in layer_block_corr[11]:
                        if "dwa2"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa2"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa2"+"q"+str(quant_bit)
                    elif layer in layer_block_corr[12] or layer in layer_block_corr[13]or\
                         layer in layer_block_corr[14] or layer in layer_block_corr[15]:
                        if "dwa3"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa3"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa3"+"q"+str(quant_bit)
            
                    
    # print("="*20)     
    # print(len(net_struct))
    # print(len(list(accelerator_alloc.keys())))
    # print(accelerator_alloc)
    # print("="*20)  
    #return None
    return accelerator_alloc, accelerator_types, accelerator_wise_budget



def cifar_convert_to_layers(block_info,quant_list,cifar=True,edd=False):
    #TODO: include EDD cases
    if cifar:
        output_dim=[32]+[32]*4+[16]*4+[8]*4+[8]*4+[4]*4+[4]
        num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1]
        #currently only support 1 
        #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
        #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
        num_channel_list = [16]+[24]*4+[32]*4+[64]*4+[112]*4+[192]*4+[352]
        stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1]
        
    else: 
        output_dim=[112]+[56]*4+[28]*4+[14]*4+[14]*4+[7]*4+[7]
        num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1]
        #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
        #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
        num_channel_list = [16]+[24]*4+[32]*4+[64]*4+[112]*4+[192]*4+[352]
        stride_list = [1, 2,1,1,1, 2,1,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1]
    if edd:
        output_dim=[56,28,28,28,28,14,14,14,14,14,14,7,7,7,7,7]
        num_layer_list= [1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1]
        num_channel_list =[32,48,48,48, 96,96,96,96, 128,128,128,256, 256,256,256,320]
        stride_list=[2,2,1,1,  1,2,1,1,  1,1,1,2,  1,1,1,1]
    net_struct=[]
    dw=[]
    layer_wise_quant=[]
    layer_block_corr={}
    for i in range(sum(num_layer_list)):
        layer_block_corr[i]=[]
    layer_num=0
    for i, rep_times in enumerate(num_layer_list):
        if "g" not in block_info[i] and block_info[i] != "skip":
            k=int(block_info[i][1])
            e=int(block_info[i][4])
            if num_layer_list[i]==1:
                if i==0:
                    #TODO: confirm if the layer dimension is right
                    net_struct.append([16,16*e,output_dim[0],1,1])
                    net_struct.append([1,16*e,output_dim[0],k,1])
                    net_struct.append([16*e,16,output_dim[0],1,1])
                    dw+=[False,True,False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit,quant_bit,quant_bit]
                    layer_block_corr[0]+=[0,1,2]
                    layer_num+=3
                else:
                    net_struct.append([num_channel_list[i-1],num_channel_list[i-1]*e,output_dim[i-1],1,stride_list[i]])
                    net_struct.append([1,num_channel_list[i-1]*e,output_dim[i],k,1])
                    net_struct.append([num_channel_list[i-1]*e,num_channel_list[i],output_dim[i],1,1])  
                    dw+=[False,True,False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit,quant_bit,quant_bit]
                    layer_block_corr[i]+=[layer_num,layer_num+1,layer_num+2]
                    layer_num+=3
            else:
                raise Exception('Currently not supporting repetive block info input')
        elif "g" in  block_info[i]:
            k=int(block_info[i][1])
            e=int(block_info[i][4])
            if num_layer_list[i]==1:
                if i==0:
                    #TODO: confirm if the layer dimension is right
                    net_struct.append([16/2,16*e/2,output_dim[0],1,1])
                    net_struct.append([16/2,16*e/2,output_dim[0],1,1])
                    net_struct.append([1,16*e,output_dim[0],k,1])
                    net_struct.append([16*e/2,16/2,output_dim[0],1,1])
                    net_struct.append([16*e/2,16/2,output_dim[0],1,1])
                    dw+=[False,False,True,False,False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit,quant_bit,quant_bit,quant_bit,quant_bit]
                    layer_block_corr[0]+=[0,1,2,3,4]
                    layer_num+=5
                else:
                    net_struct.append([num_channel_list[i-1]/2,num_channel_list[i-1]*e/2,output_dim[i-1],1,stride_list[i]])
                    net_struct.append([num_channel_list[i-1]/2,num_channel_list[i-1]*e/2,output_dim[i-1],1,stride_list[i]])
                    net_struct.append([1,num_channel_list[i-1]*e,output_dim[i],k,1])
                    net_struct.append([num_channel_list[i-1]*e/2,num_channel_list[i]/2,output_dim[i],1,1])  
                    net_struct.append([num_channel_list[i-1]*e/2,num_channel_list[i]/2,output_dim[i],1,1])
                    dw+=[False,False,True,False,False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit,quant_bit,quant_bit,quant_bit,quant_bit]
                    layer_block_corr[i]+=[layer_num,layer_num+1,layer_num+2,layer_num+3,layer_num+4]
                    layer_num+=5
            else:
                raise Exception('Currently not supporting repetive block info input')
    return net_struct,dw,layer_wise_quant,layer_block_corr



########################
##DNA specific utilities
########################

# def cifar_convert_to_layers(block_info,quant_list,cifar=True,edd=False):
    # #TODO: include EDD cases
    # if cifar:
        # output_dim=[32]+[32]*4+[16]*4+[8]*4+[8]*4+[4]*4+[4]
        # num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1]
        # #currently only support 1 
        # #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
        # #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
        # num_channel_list = [16]+[24]*4+[32]*4+[64]*4+[112]*4+[192]*4+[352]
        # stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1]
        
    # else: 
        # output_dim=[112]+[56]*4+[28]*4+[14]*4+[14]*4+[7]*4+[7]
        # num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1]
        # #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
        # #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
        # num_channel_list = [16]+[24]*4+[32]*4+[64]*4+[112]*4+[192]*4+[352]
        # stride_list = [1, 2,1,1,1, 2,1,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1]
    # if edd:
        # output_dim=[56,28,28,28,28,14,14,14,14,14,14,7,7,7,7,7]
        # num_layer_list= [1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1]
        # num_channel_list =[32,48,48,48, 96,96,96,96, 128,128,128,256, 256,256,256,320]
        # stride_list=[2,2,1,1,  1,2,1,1,  1,1,1,2,  1,1,1,1]
    # net_struct=[]
    # dw=[]
    # layer_wise_quant=[]
    # layer_block_corr={}
    # for i in range(sum(num_layer_list)):
        # layer_block_corr[i]=[]
    # layer_num=0
    # for i, rep_times in enumerate(num_layer_list):
        # if "g" not in block_info[i] and block_info[i] != "skip":
            # k=int(block_info[i][1])
            # e=int(block_info[i][4])
            # if num_layer_list[i]==1:
                # if i==0:
                    # #TODO: confirm if the layer dimension is right
                    # net_struct.append([16,16*e,output_dim[0],1,1])
                    # net_struct.append([1,16*e,output_dim[0],k,1])
                    # net_struct.append([16*e,16,output_dim[0],1,1])
                    # dw+=[False,True,False]
                    # quant_bit=quant_list.pop(0)
                    # layer_wise_quant+=[quant_bit,quant_bit,quant_bit]
                    # layer_block_corr[0]+=[0,1,2]
                    # layer_num+=3
                # else:
                    # net_struct.append([num_channel_list[i-1],num_channel_list[i-1]*e,output_dim[i],1,stride_list[i]])
                    # net_struct.append([1,num_channel_list[i-1]*e,output_dim[i],k,1])
                    # net_struct.append([num_channel_list[i-1]*e,num_channel_list[i],output_dim[i],1,1])  
                    # dw+=[False,True,False]
                    # quant_bit=quant_list.pop(0)
                    # layer_wise_quant+=[quant_bit,quant_bit,quant_bit]
                    # layer_block_corr[i]+=[layer_num,layer_num+1,layer_num+2]
                    # layer_num+=3
            # else:
                # raise Exception('Currently not supporting repetive block info input')
        # elif "g" in  block_info[i]:
            # k=int(block_info[i][1])
            # e=int(block_info[i][4])
            # if num_layer_list[i]==1:
                # if i==0:
                    # #TODO: confirm if the layer dimension is right
                    # net_struct.append([16/2,16*e/2,output_dim[0],1,1])
                    # net_struct.append([16/2,16*e/2,output_dim[0],1,1])
                    # net_struct.append([1,16*e,output_dim[0],k,1])
                    # net_struct.append([16*e/2,16/2,output_dim[0],1,1])
                    # net_struct.append([16*e/2,16/2,output_dim[0],1,1])
                    # dw+=[False,False,True,False,False]
                    # quant_bit=quant_list.pop(0)
                    # layer_wise_quant+=[quant_bit,quant_bit,quant_bit,quant_bit,quant_bit]
                    # layer_block_corr[0]+=[0,1,2,3,4]
                    # layer_num+=5
                # else:
                    # net_struct.append([num_channel_list[i-1]/2,num_channel_list[i-1]*e/2,output_dim[i],1,stride_list[i]])
                    # net_struct.append([num_channel_list[i-1]/2,num_channel_list[i-1]*e/2,output_dim[i],1,stride_list[i]])
                    # net_struct.append([1,num_channel_list[i-1]*e,output_dim[i],k,1])
                    # net_struct.append([num_channel_list[i-1]*e/2,num_channel_list[i]/2,output_dim[i],1,1])  
                    # net_struct.append([num_channel_list[i-1]*e/2,num_channel_list[i]/2,output_dim[i],1,1])
                    # dw+=[False,False,True,False,False]
                    # quant_bit=quant_list.pop(0)
                    # layer_wise_quant+=[quant_bit,quant_bit,quant_bit,quant_bit,quant_bit]
                    # layer_block_corr[i]+=[layer_num,layer_num+1,layer_num+2,layer_num+3,layer_num+4]
                    # layer_num+=5
            # else:
                # raise Exception('Currently not supporting repetive block info input')
    # return net_struct,dw,layer_wise_quant,layer_block_corr


def design_choice_gen(cifar=True,edd=False,channel_part=False):
    #TODO: include imagenet cases
    if not channel_part:
        if cifar:
            acc1_space={'comp_mode':[0,1,2],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
            acc2_space={'comp_mode':[0,1,2],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
            dw_acc1_space={'comp_mode':[0,1],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[8,4,2,1],'tn':[1]}
            dw_acc2_space={'comp_mode':[0,1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
        else:
            acc1_space={'comp_mode':[0,1,2],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
            acc2_space={'comp_mode':[0,1,2],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
            dw_acc1_space={'comp_mode':[0,1],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[8,4,2,1],'tn':[1]}
            dw_acc2_space={'comp_mode':[0,1],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
        #design_choices: {comp_mode:[0,1,2],fw:[2,4,6,8]...}
        if edd:
            acc1_space={'comp_mode':[0,1,2],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[16,8,4,2,1],'tnbuff':[16,8,4,2,1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[16,8,4,2,1],'tn':[16,8,4,2,1]}
            acc2_space={'comp_mode':[0,1,2],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
            dw_acc1_space={'comp_mode':[0,1],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[16,8,4,2,1],'tnbuff':[1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[16,8,4,2,1],'tn':[1]}
            dw_acc2_space={'comp_mode':[0,1],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
        return (acc1_space,acc2_space,dw_acc1_space,dw_acc2_space)
    else:   
        if cifar:
            acc1_space={'comp_mode':[0,1,2],'trbuff':[32,16,8,4,2,1],'tcbuff':[32,16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[32,16,8,4,2,1],'tc':[32,16,8,4,2,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
            acc2_space={'comp_mode':[0,1,2],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
            acc3_space={'comp_mode':[0,1,2],'trbuff':[8,4,2,1],'tcbuff':[8,4,2,1],'tmbuff':[64,32,16,8,4,2,1],'tnbuff':[64,32,16,8,4,2,1], 'tr':[8,4,2,1],'tc':[8,4,2,1],'tm':[64,32,16,8,4,2,1],'tn':[64,32,16,8,4,2,1]}
            acc4_space={'comp_mode':[0,1,2],'trbuff':[8,4,2,1],'tcbuff':[8,4,2,1],'tmbuff':[112,56,28,14,7,1],'tnbuff':[112,56,28,14,7,1], 'tr':[8,4,2,1],'tc':[8,4,2,1],'tm':[112,56,28,14,7,1],'tn':[112,56,28,14,7,1]}
            acc5_space={'comp_mode':[0,1,2],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}

            
            dw_acc1_space={'comp_mode':[0,1],'trbuff':[32,16,8,4,2,1],'tcbuff':[32,16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[32,16,8,4,2,1],'tc':[32,16,8,4,2,1],'tm':[8,4,2,1],'tn':[1]}
            dw_acc2_space={'comp_mode':[0,1],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
            dw_acc3_space={'comp_mode':[0,1],'trbuff':[8,4,2,1],'tcbuff':[8,4,2,1],'tmbuff':[64,32,16,8,4,2,1],'tnbuff':[1], 'tr':[8,4,2,1],'tc':[8,4,2,1],'tm':[64,32,16,8,4,2,1],'tn':[1]}
            dw_acc4_space={'comp_mode':[0,1],'trbuff':[8,4,2,1],'tcbuff':[8,4,2,1],'tmbuff':[112,56,28,14,7,1],'tnbuff':[1], 'tr':[8,4,2,1],'tc':[8,4,2,1],'tm':[112,56,28,14,7,1],'tn':[1]}
            dw_acc5_space={'comp_mode':[0,1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
            return (acc1_space,acc2_space,acc3_space,acc4_space,acc5_space,dw_acc1_space,dw_acc2_space,dw_acc3_space,dw_acc4_space,dw_acc5_space)

        else:
            acc1_space={'comp_mode':[0,1,2],'trbuff':[56,28,14,7,1],'tcbuff':[56,28,14,7,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[56,28,14,7,1],'tc':[56,28,14,7,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
            acc2_space={'comp_mode':[0,1,2],'trbuff':[28,14,7,1],'tcbuff':[28,14,7,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[28,14,7,1],'tc':[28,14,7,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
            acc3_space={'comp_mode':[0,1,2],'trbuff':[14,7,1],'tcbuff':[14,7,1],'tmbuff':[64,32,16,8,4,2,1],'tnbuff':[64,32,16,8,4,2,1], 'tr':[14,7,1],'tc':[14,7,1],'tm':[64,32,16,8,4,2,1],'tn':[64,32,16,8,4,2,1]}
            acc4_space={'comp_mode':[0,1,2],'trbuff':[14,7,1],'tcbuff':[14,7,1],'tmbuff':[112,56,28,14,7,1],'tnbuff':[112,56,28,14,7,1], 'tr':[14,7,1],'tc':[14,7,1],'tm':[112,56,28,14,7,1],'tn':[112,56,28,14,7,1]}
            acc5_space={'comp_mode':[0,1,2],'trbuff':[7,1],'tcbuff':[7,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[7,1],'tc':[7,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}

            
            dw_acc1_space={'comp_mode':[0,1],'trbuff':[56,28,14,7,1],'tcbuff':[56,28,14,7,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[56,28,14,7,1],'tc':[56,28,14,7,1],'tm':[8,4,2,1],'tn':[1]}
            dw_acc2_space={'comp_mode':[0,1],'trbuff':[28,14,7,1],'tcbuff':[28,14,7,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[28,14,7,1],'tc':[28,14,7,1],'tm':[32,16,8,4,2,1],'tn':[1]}
            dw_acc3_space={'comp_mode':[0,1],'trbuff':[14,7,1],'tcbuff':[14,7,1],'tmbuff':[64,32,16,8,4,2,1],'tnbuff':[1], 'tr':[14,7,1],'tc':[14,7,1],'tm':[64,32,16,8,4,2,1],'tn':[1]}
            dw_acc4_space={'comp_mode':[0,1],'trbuff':[14,7,1],'tcbuff':[14,7,1],'tmbuff':[112,56,28,14,7,1],'tnbuff':[1], 'tr':[14,7,1],'tc':[14,7,1],'tm':[112,56,28,14,7,1],'tn':[1]}
            dw_acc5_space={'comp_mode':[0,1],'trbuff':[7,1],'tcbuff':[7,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[7,1],'tc':[7,1],'tm':[32,16,8,4,2,1],'tn':[1]}
            return (acc1_space,acc2_space,acc3_space,acc4_space,acc5_space,dw_acc1_space,dw_acc2_space,dw_acc3_space,dw_acc4_space,dw_acc5_space)
       
        if edd: 
            acc1_space={'comp_mode':[0,1,2],'trbuff':[28,14,7,1],'tcbuff':[28,14,7,1],'tmbuff':[16,8,4,2,1],'tnbuff':[16,8,4,2,1], 'tr':[28,14,7,1],'tc':[28,14,7,1],'tm':[16,8,4,2,1],'tn':[16,8,4,2,1]}
            acc2_space={'comp_mode':[0,1,2],'trbuff':[14,7,1],'tcbuff':[14,7,1],'tmbuff':[96,48,24,12,8,4,3,2,1],'tnbuff':[96,48,24,12,8,4,3,2,1], 'tr':[14,7,1],'tc':[14,7,1],'tm':[96,48,24,12,8,4,3,2,1],'tn':[96,48,24,12,8,4,3,2,1]}
            
            acc3_space={'comp_mode':[0,1,2],'trbuff':[14,7,1],'tcbuff':[14,7,1],'tmbuff':[128,64,32,16,8,4,2,1],'tnbuff':[128,64,32,16,8,4,2,1], 'tr':[14,7,1],'tc':[14,7,1],'tm':[128,64,32,16,8,4,2,1],'tn':[128,64,32,16,8,4,2,1]}
           #acc3_space={'comp_mode':[0,1,2],'trbuff':[7,1],'tcbuff':[7,1],'tmbuff':[64,32,16,8,4,2,1],'tnbuff':[64,32,16,8,4,2,1], 'tr':[7,1],'tc':[7,1],'tm':[64,32,16,8,4,2,1],'tn':[64,32,16,8,4,2,1]}

            acc4_space={'comp_mode':[0,1,2],'trbuff':[7,1],'tcbuff':[7,1],'tmbuff':[64,32,16,8,4,2,1],'tnbuff':[64,32,16,8,4,2,1], 'tr':[7,1],'tc':[7,1],'tm':[64,32,16,8,4,2,1],'tn':[64,32,16,8,4,2,1]}

            
            dw_acc1_space={'comp_mode':[0,1],'trbuff':[28,14,7,1],'tcbuff':[28,14,7,1],'tmbuff':[16,8,4,2,1],'tnbuff':[1], 'tr':[28,14,7,1],'tc':[28,14,7,1],'tm':[16,8,4,2,1],'tn':[1]}
            dw_acc2_space={'comp_mode':[0,1],'trbuff':[14,7,1],'tcbuff':[14,7,1],'tmbuff':[96,48,24,12,8,4,3,2,1],'tnbuff':[1], 'tr':[14,7,1],'tc':[14,7,1],'tm':[96,48,24,12,8,4,3,2,1],'tn':[1]}
            dw_acc3_space={'comp_mode':[0,1],'trbuff':[14,7,1],'tcbuff':[14,7,1],'tmbuff':[128,64,32,16,8,4,2,1],'tnbuff':[1], 'tr':[14,7,1],'tc':[14,7,1],'tm':[128,64,32,16,8,4,2,1],'tn':[1]}
            dw_acc4_space={'comp_mode':[0,1],'trbuff':[7,1],'tcbuff':[7,1],'tmbuff':[64,32,16,8,4,2,1],'tnbuff':[1], 'tr':[7,1],'tc':[7,1],'tm':[64,32,16,8,4,2,1],'tn':[1]}
            return (acc1_space,acc2_space,acc3_space,acc4_space,dw_acc1_space,dw_acc2_space,dw_acc3_space,dw_acc4_space)
            
        

def random_sample(input_dict):
    np.random.seed()
    result_sample=[]
    result_sample_dict={}
    for key in input_dict.keys():
        tmp=input_dict[key][np.random.randint(len(input_dict[key]))]
        if "tr"== key or "tc"==key or "tm" == key or "tn" ==key :
            #tmp=np.random.randint(len(input_dict[key]))
            while tmp > result_sample_dict[key+"buff"]:
                tmp=input_dict[key][np.random.randint(len(input_dict[key]))]
            result_sample.append(tmp)
            result_sample_dict[key]=tmp
        else:
            result_sample.append(tmp)
            result_sample_dict[key]=tmp
    return result_sample

def mac_calc(net_struct):
    mac=0
    for i, layer in enumerate(net_struct):
        mac+=layer[0]*layer[1]*layer[2]*layer[2]*layer[3]*layer[3]
    return mac
