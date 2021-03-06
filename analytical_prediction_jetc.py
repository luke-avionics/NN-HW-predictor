import os 
import numpy as np
import math
import re
import copy
from  multiprocessing import Queue
import multiprocessing
from predictor_utilities import *
def arch2spec(arch):
    block_info=[]
    block_info_all=[]
    quant_list=[]
    block_depth=arch[0]['d']
    std_w_q=arch[1]['pw_w_bits_setting']
    std_a_q=arch[1]['pw_a_bits_setting']
    dw_w_q=arch[1]['dw_w_bits_setting']
    dw_a_q=arch[1]['dw_a_bits_setting']
    block_depth[-1]=1
    for i, info in enumerate(zip(arch[0]['ks'],arch[0]['e'])):
        block_info_all.append("k"+str(info[0])+"e"+str(info[1]))
    for i in range(21):
        nowd = i % 4
        stg = i // 4   
        if nowd < block_depth[stg]:
            block_info.append(block_info_all[i])
            quant_list.append(((std_w_q[i]+std_a_q[i])/2,(dw_w_q[i]+dw_a_q[i])/2))
    return block_info, quant_list,block_depth


def cifar_convert_to_layers_mixed(block_info,quant_list,cifar=True):
    if cifar:
        raise Exception('Not supported')
    else:
        output_dim=[4]+[4]*4+[4]*4+[4]*4+[4]
        num_layer_list=(14)*[1]
        num_channel_list=[32]+[64]*4+[64]*4+[64]*4+[128]
        stride_list=[1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1]

        
    net_struct=[]
    dw=[]
    layer_wise_quant=[]
    layer_block_corr={}
    for i in range(sum(num_layer_list)*3+4):
        layer_block_corr[i]=[]
    layer_num=0


    net_struct.append([24,48,8,7,2])
    dw+=[False]
    prec_value=quant_list[0][0]
    layer_wise_quant+=[quant_list[0][0]]
    layer_block_corr[0]+=[0]
    layer_num+=1


    net_struct.append([48,96,4,5,2])
    dw+=[False]
    prec_value=quant_list[0][0]
    layer_wise_quant+=[quant_list[0][0]]
    layer_block_corr[1]+=[1]
    layer_num+=1


    for i, rep_times in enumerate(num_layer_list):
        if "g" not in block_info[i] and block_info[i] != "skip" and 'c' not in block_info[i]:
            k=int(block_info[i][1])
            e=float(block_info[i].split('e')[1])
            if num_layer_list[i]==1:
                if i==0:
                    #TODO: confirm if the layer dimension is right
                    expanded_size=int(32*e)
                    net_struct.append([32,expanded_size,output_dim[0],1,1])
                    net_struct.append([1,expanded_size,output_dim[0],k,1])
                    net_struct.append([expanded_size,16,output_dim[0],1,1])
                    dw+=[False,True,False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit[0],quant_bit[1],quant_bit[0]]
                    layer_block_corr[2]+=[2,3,4]
                    layer_num+=3
                else:
                    expanded_size=int(num_channel_list[i-1]*e)
                    net_struct.append([num_channel_list[i-1],expanded_size,output_dim[i-1],1,stride_list[i]])
                    net_struct.append([1,expanded_size,output_dim[i],k,1])
                    net_struct.append([expanded_size,num_channel_list[i],output_dim[i],1,1])  
                    dw+=[False,True,False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit[0],quant_bit[1],quant_bit[0]]
                    layer_block_corr[i+2]+=[layer_num,layer_num+1,layer_num+2]
                    layer_num+=3
            elif 'c' in block_info[i]:
                k=int(block_info[i][1])
                if i==0:
                    #TODO: confirm if the layer dimension is right
                    net_struct.append([32,16,output_dim[0],k,1])
                    dw+=[False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit[0]]
                    layer_block_corr[2]+=[2]
                    layer_num+=1
                else:
                    expanded_size=int(num_channel_list[i-1]*e)
                    net_struct.append([num_channel_list[i-1],num_channel_list[i],output_dim[i],k,stride_list[i]])
                    dw+=[False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit[0]]
                    layer_block_corr[i+2]+=[layer_num]
                    layer_num+=1

            else:
                raise Exception('Currently not supporting repetive block info input')
        elif "g" in  block_info[i]:
            raise Exception('Current APQ structure should not have g')

    net_struct.append([128,48,8,5,2])
    dw+=[False]
    layer_wise_quant+=[prec_value]
    layer_block_corr[layer_num-1]+=[layer_num]
    layer_num+=1

    net_struct.append([48,24,16,7,2])
    dw+=[False]
    layer_wise_quant+=[prec_value]
    layer_block_corr[layer_num-1]+=[layer_num]
    layer_num+=1


    return net_struct,dw,layer_wise_quant,layer_block_corr   


def allocate_layers_rl(net_struct,quant_list,dw,platform_specs,layer_block_corr,cifar=True,edd=False,channel_part=False):
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
                    if layer < 2:
                        if "a0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a0"+"q"+str(quant_bit)
                    elif net_struct[layer][1] % 24 !=0:
                        if "a1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a1"+"q"+str(quant_bit)
                    else:
                        if "a2"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("a2"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="a2"+"q"+str(quant_bit)
                        
            for i, quant_bit in enumerate(dw_quantization_bins.keys()):
                for layer in dw_quantization_bins[quant_bit]:
                    if layer < 2:
                        if "dwa0"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa0"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa0"+"q"+str(quant_bit)
                    elif net_struct[layer][1] % 24 !=0:
                        if "dwa1"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa1"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa1"+"q"+str(quant_bit)
                    else:
                        if "dwa2"+"q"+str(quant_bit) not in accelerator_types:
                            accelerator_types.append("dwa2"+"q"+str(quant_bit))
                        accelerator_alloc[layer]="dwa2"+"q"+str(quant_bit)
    else:
    #applies specifically to Yonggan's space
        raise Exception('allocation mode not supported for this variant')
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




def capsuled_predictor(input_params_set, block_info_test,quant_list,cifar):
    
    #generate the layer wise structure, if_layer_is_dw, layer_wise_quant
    net_struct,dw,layer_wise_quant,layer_block_corr=cifar_convert_to_layers_mixed(block_info_test,copy.deepcopy(quant_list),cifar=cifar)

    #print(len(net_struct),len(dw))
    #print(mac_calc(net_struct))
    #exit()
    #allocate each layer with its corresponding accelerator
    #{layer_num: <accelerator_type>}
    accelerator_alloc, accelerator_types, accelerator_wise_budget=allocate_layers_rl(net_struct,layer_wise_quant,dw,None,layer_block_corr,cifar=cifar)
    # print(dw)
    # print(accelerator_alloc)
    # print(accelerator_types)

    platform_specs={'dsp':450,'bram':700}
    bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,layer_wise_break_down=sys_latency(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget)
    consumption_used, consumption_breakdown=sys_consumption(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget,platform_specs)
    bs=min(math.floor(platform_specs['dsp']/consumption_used[0]),math.floor(platform_specs['bram']/consumption_used[1]))
    bs=1
    bottleneck_latency=bottleneck_latency/bs
    for key in latency_break_down.keys():
        latency_break_down[key]=latency_break_down[key]/bs
        consumption_breakdown[key][0]=consumption_breakdown[key][0]*bs
        consumption_breakdown[key][1]=consumption_breakdown[key][1]*bs
        layer_wise_break_down_to_accel[key]=[i/bs for i in layer_wise_break_down_to_accel[key]]
    layer_wise_break_down=[i/bs for i in layer_wise_break_down]
    consumption_used=[i*bs for i in consumption_used]
    block_wise_performance=[]
    for key in layer_block_corr.keys():
        tmp_block_lat=0
        for layer_num in layer_block_corr[key]:
            tmp_block_lat+=layer_wise_break_down[layer_num]
        block_wise_performance.append(tmp_block_lat)
    #print(block_wise_performance)
        
    return bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
           layer_wise_break_down,consumption_used, consumption_breakdown,\
           accelerator_alloc,bs,block_wise_performance,net_struct  


def design_choice_gen_apq(cifar): 
    if cifar:
        raise Exception('Not supported')
        # acc1_space={'comp_mode':[0,1,2],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
        # acc2_space={'comp_mode':[0,1,2],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
        # dw_acc1_space={'comp_mode':[0,1],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[8,4,2,1],'tn':[1]}
        # dw_acc2_space={'comp_mode':[0,1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
    else:
        acc1_space={'comp_mode':[1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[48,24,16,12,8,6,4,3,2,1],'tnbuff':[48,24,16,12,8,6,4,3,2,1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[48,24,16,12,8,6,4,3,2,1],'tn':[48,24,16,12,8,6,4,3,2,1]}
        acc2_space={'comp_mode':[1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
        acc3_space={'comp_mode':[1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[48,24,16,12,8,6,4,3,2,1],'tnbuff':[48,24,16,12,8,6,4,3,2,1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[48,24,16,12,8,6,4,3,2,1],'tn':[48,24,16,12,8,6,4,3,2,1]}        
        dw_acc1_space={'comp_mode':[0,1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[48,24,16,12,8,6,4,3,2,1],'tnbuff':[1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[48,24,16,12,8,6,4,3,2,1],'tn':[1]}
        dw_acc2_space={'comp_mode':[0,1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
        dw_acc3_space={'comp_mode':[0,1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[48,24,16,12,8,6,4,3,2,1],'tnbuff':[1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[48,24,16,12,8,6,4,3,2,1],'tn':[1]}

    return (acc1_space,acc2_space,acc3_space, dw_acc1_space,dw_acc2_space, dw_acc3_space)
output_q=Queue()
def worker(id,block_info_test):   
    #4
    #test_arch=[{"wid": None, "ks": [3, 3, 7, 7, 5, 5, 3, 3, 7, 7, 7, 3, 5, 5, 3, 3, 7, 5, 3, 3, 3], "e": [5.5, 4.333333333333333, 6.0, 5.0, 4.333333333333333, 5.0, 5.6, 5.8, 4.8, 5.1, 4.4, 5.0, 5.4, 5.583333333333333, 5.0, 5.0, 4.083333333333333, 5.958333333333333, 5.958333333333333, 4.75, 5.666666666666667], "d": [4, 4, 4, 4, 4, 1]}, {"pw_w_bits_setting": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], "pw_a_bits_setting": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], "dw_w_bits_setting": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], "dw_a_bits_setting": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]}]
    #1
    #test_arch=[{"wid": None, "ks": [5, 3, 5, 5, 5, 3, 5, 7, 7, 3, 5, 5, 5, 3, 5, 7, 7, 3, 5, 3, 3], "e": [5.0, 4.0, 4.666666666666667, 4.666666666666667, 5.333333333333333, 5.8, 4.0, 5.2, 4.8, 5.1, 5.2, 4.7, 5.6, 5.583333333333333, 5.0, 6.0, 4.666666666666667, 5.875, 5.833333333333333, 4.125, 6.0], "d": [4, 4, 4, 4, 4, 1]}, {"pw_w_bits_setting": [8, 8, 8, 4, 4, 6, 8, 6, 6, 6, 4, 6, 8, 4, 6, 6, 4, 8, 8, 4, 8], "pw_a_bits_setting": [6, 6, 4, 8, 8, 6, 6, 8, 6, 8, 8, 4, 8, 4, 6, 8, 6, 6, 6, 6, 4], "dw_w_bits_setting": [4, 4, 4, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 8, 6, 4, 8, 8, 8, 6], "dw_a_bits_setting": [4, 4, 4, 8, 4, 4, 6, 8, 8, 4, 4, 4, 4, 4, 4, 6, 6, 8, 4, 4, 4]}]
    quant_options=[16]
    
    #block_info_test=['k3_e1', 'skip', 'skip', 'skip', 'k5_e1', 'k3_e1', 'k3_e1', 'skip', 'k5_e1', 'k5_e1', 'k5_e1', 'k5_e1']
    
    quant_list=len(block_info_test)*[(16,16)]
    cifar=False
    acc1_space,acc2_space,acc3_space, dw_acc1_space,dw_acc2_space,dw_acc3_space=design_choice_gen_apq(cifar=cifar)
    latency_list=[]
    best_throughput=0 
    for _ in range(500000):
        design_choice_integrity=False
        while not design_choice_integrity:
            input_params_set={}
            for quant_option in quant_options:
                input_params_set["a0q"+str(quant_option)]=random_sample(acc1_space)+[quant_option]
                input_params_set["a1q"+str(quant_option)]=random_sample(acc2_space)+[quant_option]
                input_params_set["a2q"+str(quant_option)]=random_sample(acc3_space)+[quant_option]
                input_params_set["dwa0q"+str(quant_option)]=random_sample(dw_acc1_space)+[quant_option]
                input_params_set["dwa1q"+str(quant_option)]=random_sample(dw_acc2_space)+[quant_option]
                input_params_set["dwa2q"+str(quant_option)]=random_sample(dw_acc3_space)+[quant_option]
            for accel in input_params_set.keys():
                if input_params_set[accel][1] < input_params_set[accel][5] or\
                   input_params_set[accel][2] < input_params_set[accel][6] or\
                   input_params_set[accel][3] < input_params_set[accel][7] or\
                   input_params_set[accel][4] < input_params_set[accel][8]:
                    design_choice_integrity=False
                    break
                else:
                    design_choice_integrity=True\




        try:
            bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
            layer_wise_break_down,consumption_used, consumption_breakdown,\
            accelerator_alloc,bs,block_wise_performance,net_struct=capsuled_predictor(input_params_set, block_info_test,quant_list,cifar=cifar)
            param_size,mac_size,block_wise_mac=model_profiler(net_struct)
        except Exception as e:
            #print(e)
            pass

        if 1/(bottleneck_latency/200e6)> best_throughput: 
            best_throughput=1/(bottleneck_latency/200e6)
            best_consumption_used=consumption_used
            best_consumption_breakdown=consumption_breakdown
            best_latency_break_down=latency_break_down
            best_input_params_set=input_params_set
            best_accelerator_alloc=accelerator_alloc
            best_net_struct=net_struct
            best_bs=bs
            best_layer_wise_break_down=layer_wise_break_down
            best_block_wise_performance=block_wise_performance
            best_layer_wise_break_down_to_accel=layer_wise_break_down_to_accel
        #print(best_throughput)
        #print(best_consumption_used)
        #print('param size: ',param_size ,'mac size: ', mac_size)
    # print('throughput: ', best_throughput)
    # print('best_bs: ', best_bs)
    # print('latency_break_down: ', best_latency_break_down)
    # print('layer_wise_break_down: ',best_layer_wise_break_down)
    # print('consumption_used: ', best_consumption_used)
    # print('consumption_breakdown: ', best_consumption_breakdown)
    # print('accelerator_alloc', best_accelerator_alloc)
    # print('input_params',best_input_params_set)
    # print('net_struct', net_struct)
    print(best_throughput)
    print(best_consumption_used)
    print(best_accelerator_alloc)
    print(best_latency_break_down)
    print(best_layer_wise_break_down)
    print(best_layer_wise_break_down_to_accel)
    output_q.put((id,(best_throughput,best_consumption_used)))
    # return {id:(best_throughput,best_consumption_used)}

start=time.time()


net_pool=[
['k3_e1', 'skip', 'skip', 'skip', 'k5_e1', 'k3_e1', 'k3_e1', 'skip', 'k5_e1', 'k5_e1', 'k5_e1', 'k5_e1'],\
['k5_e1', 'k3_e1', 'skip', 'skip', 'k5_e1', 'k3_e1', 'k5_e1', 'k3_e1', 'k5_e1', 'k5_e1', 'k5_e1', 'k5_e1'],\
['k3_e1', 'k3_e1', 'k3_e1', 'k3_e1', 'k3_e1', 'k3_e1', 'k3_e1', 'k3_e1', 'k5_e1', 'k5_e1', 'k5_e1', 'k5_e1'],\
['k5_e3', 'k5_e1', 'k5_e1', 'k3_e1', 'k5_e1', 'k3_e1', 'k3_e1', 'k3_e1', 'k5_e1', 'k5_e1', 'k5_e1', 'k5_e1'],\
['k5_e3', 'k3_e1', 'k3_e1', 'skip', 'k5_e3', 'k5_e1', 'k3_e1', 'k3_e1', 'k5_e1', 'k5_e3', 'k5_e1', 'k5_e1'],\
['k3_e3', 'k3_e1', 'k3_e1', 'k5_e1', 'k5_e1', 'k3_e1', 'k3_e1', 'k5_e1', 'k5_e3', 'k5_e3', 'k5_e3', 'k5_e3'],\
['k5_e6', 'k5_e3', 'k5_e3', 'k5_e3', 'k5_e3', 'k3_e3', 'k5_e3', 'k3_e3', 'k5_e6', 'k5_e3', 'k5_e3', 'k5_e3'],\
['k3_e6', 'k5_e6', 'k3_e3', 'k5_e3', 'k5_e6', 'k5_e3', 'k5_e1', 'k3_e1', 'k5_e3', 'k5_e6', 'k5_e3', 'k5_e6'],\
['k5_e6', 'k5_e6', 'k3_e3', 'k3_e1', 'k5_e6', 'k5_e3', 'k5_e3', 'k3_e3', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6'],\
['k5_e3', 'k5_e6', 'k3_e3', 'k5_e3', 'k5_e6', 'k5_e6', 'k3_e3', 'k3_e6', 'k5_e6', 'k5_e6', 'k5_e3', 'k5_e6'],\
['k5_e3', 'k3_e6', 'k5_e6', 'k3_e6', 'k5_e6', 'k5_e3', 'k5_e6', 'k3_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e3'],\
['k5_e6', 'k5_e3', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e1', 'k5_e6', 'k5_e3'],\
['k3_e6', 'k3_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k3_e6', 'k3_e6', 'k3_e3', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6'],\
['k5_e6', 'k5_e6', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e6', 'k5_e6'],\
['k3_e1', 'skip',  'skip',  'skip',  'k3_e1', 'k3_e1', 'skip',  'skip',  'k5_e1', 'k5_e1', 'k5_e1', 'k5_e1'],\
]



net_pool=[
#['k3_e1', 'k5_e3', 'k5_e6', 'k3_e3', 'k5_e6', 'k5_e6', 'k3_e6', 'k3_e1', 'k3_e1', 'k3_e3', 'k5_e6', 'k3_e3', 'k5_e1', 'k5_e3']
#['k3_e1', 'k5_e6', 'skip', 'k5_e3', 'skip', 'skip', 'c5', 'k3_e3', 'c3', 'k3_e3', 'k3_e1', 'k5_e3', 'k3_e1', 'k5_e1']
#['k3_e1', 'skip', 'skip', 'k3_e1', 'skip', 'k3_e1', 'skip', 'skip', 'skip', 'skip', 'skip', 'skip', 'skip', 'skip']
['k5_e3', 'skip', 'skip', 'skip', 'k3_e1', 'skip', 'skip', 'skip', 'skip', 'skip', 'skip', 'skip', 'skip', 'skip']
]

for net in net_pool:
    worker(0,net)



# dump_yard=[]
# args=list(range(1))
# num_worker_threads=1
# multi_p(worker,args,output_q,num_worker_threads,dump_yard)
#print(dump_yard)

# best_throughput=0
# best_data=None 
# for i, tmp in enumerate(dump_yard):
#     if tmp[1][0]>best_throughput:
#         best_throughput=tmp[1][0]
#         best_data=copy.deepcopy(tmp)
# print(best_data)        
print('takes: ',time.time()-start)
exit()

