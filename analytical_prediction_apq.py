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


def cifar_convert_to_layers_mixed(block_info,quant_list,block_depth,cifar=True):
    if cifar:
        output_dim=[32]*block_depth[0]+[16]*block_depth[1]+[8]*block_depth[2]+[8]*block_depth[3]+\
                   [4]*block_depth[4]+[4]*block_depth[5]
        num_layer_list=sum(block_depth)*[1]
        num_channel_list=[24]*block_depth[0]+[32]*block_depth[1]+[64]*block_depth[2]+[96]*block_depth[3]+\
                         [160]*block_depth[4]+[320]*block_depth[5]
        stride_list=[2]+[1]*(block_depth[0]-1)+[2]+[1]*(block_depth[1]-1)+[2]+[1]*(block_depth[2]-1)+[1]*block_depth[3]+\
                    [2]+[1]*(block_depth[4]-1)+[1]*block_depth[5]
    else:
        output_dim=[56]*block_depth[0]+[28]*block_depth[1]+[14]*block_depth[2]+[14]*block_depth[3]+\
                   [7]*block_depth[4]+[7]*block_depth[5]
        num_layer_list=sum(block_depth)*[1]
        num_channel_list=[24]*block_depth[0]+[32]*block_depth[1]+[64]*block_depth[2]+[96]*block_depth[3]+\
                         [160]*block_depth[4]+[320]*block_depth[5]
        stride_list=[2]+[1]*(block_depth[0]-1)+[2]+[1]*(block_depth[1]-1)+[2]+[1]*(block_depth[2]-1)+[1]*block_depth[3]+\
                    [2]+[1]*(block_depth[4]-1)+[1]*block_depth[5]
        
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
            e=float(block_info[i].split('e')[1])
            if num_layer_list[i]==1:
                if i==0:
                    #TODO: confirm if the layer dimension is right
                    expanded_size=closestMultiple(int(16*e),8)
                    net_struct.append([16,expanded_size,output_dim[0],1,1])
                    net_struct.append([1,expanded_size,output_dim[0],k,1])
                    net_struct.append([expanded_size,16,output_dim[0],1,1])
                    dw+=[False,True,False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit[0],quant_bit[1],quant_bit[0]]
                    layer_block_corr[0]+=[0,1,2]
                    layer_num+=3
                else:
                    expanded_size=closestMultiple(int(num_channel_list[i-1]*e),8)
                    net_struct.append([num_channel_list[i-1],expanded_size,output_dim[i-1],1,stride_list[i]])
                    net_struct.append([1,expanded_size,output_dim[i],k,1])
                    net_struct.append([expanded_size,num_channel_list[i],output_dim[i],1,1])  
                    dw+=[False,True,False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit[0],quant_bit[1],quant_bit[0]]
                    layer_block_corr[i]+=[layer_num,layer_num+1,layer_num+2]
                    layer_num+=3
            else:
                raise Exception('Currently not supporting repetive block info input')
        elif "g" in  block_info[i]:
            raise Exception('Current APQ structure should not have g')
    return net_struct,dw,layer_wise_quant,layer_block_corr   


def capsuled_predictor(input_params_set, block_info_test,quant_list,block_depth,cifar):
    
    #generate the layer wise structure, if_layer_is_dw, layer_wise_quant
    net_struct,dw,layer_wise_quant,layer_block_corr=cifar_convert_to_layers_mixed(block_info_test,copy.deepcopy(quant_list),block_depth,cifar=cifar)

    #print(len(net_struct),len(dw))
    #print(mac_calc(net_struct))
    #exit()
    #allocate each layer with its corresponding accelerator
    #{layer_num: <accelerator_type>}
    accelerator_alloc, accelerator_types, accelerator_wise_budget=allocate_layers(net_struct,layer_wise_quant,dw,None,layer_block_corr,cifar=cifar)
    # print(dw)
    # print(accelerator_alloc)
    # print(accelerator_types)

    platform_specs={'dsp':900,'bram':700}
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
        acc1_space={'comp_mode':[0,1,2],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
        acc2_space={'comp_mode':[0,1,2],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
        dw_acc1_space={'comp_mode':[0,1],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[8,4,2,1],'tn':[1]}
        dw_acc2_space={'comp_mode':[0,1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
    else:
        acc1_space={'comp_mode':[0,1,2],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
        acc2_space={'comp_mode':[0,1,2],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
        dw_acc1_space={'comp_mode':[0,1],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[8,4,2,1],'tn':[1]}
        dw_acc2_space={'comp_mode':[0,1],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}

    return (acc1_space,acc2_space,dw_acc1_space,dw_acc2_space)
output_q=Queue()
def worker(id):   
    #4
    test_arch=[{"wid": None, "ks": [3, 3, 7, 7, 5, 5, 3, 3, 7, 7, 7, 3, 5, 5, 3, 3, 7, 5, 3, 3, 3], "e": [5.5, 4.333333333333333, 6.0, 5.0, 4.333333333333333, 5.0, 5.6, 5.8, 4.8, 5.1, 4.4, 5.0, 5.4, 5.583333333333333, 5.0, 5.0, 4.083333333333333, 5.958333333333333, 5.958333333333333, 4.75, 5.666666666666667], "d": [4, 4, 4, 4, 4, 1]}, {"pw_w_bits_setting": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], "pw_a_bits_setting": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], "dw_w_bits_setting": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], "dw_a_bits_setting": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]}]
    #1
    #test_arch=[{"wid": None, "ks": [5, 3, 5, 5, 5, 3, 5, 7, 7, 3, 5, 5, 5, 3, 5, 7, 7, 3, 5, 3, 3], "e": [5.0, 4.0, 4.666666666666667, 4.666666666666667, 5.333333333333333, 5.8, 4.0, 5.2, 4.8, 5.1, 5.2, 4.7, 5.6, 5.583333333333333, 5.0, 6.0, 4.666666666666667, 5.875, 5.833333333333333, 4.125, 6.0], "d": [4, 4, 4, 4, 4, 1]}, {"pw_w_bits_setting": [8, 8, 8, 4, 4, 6, 8, 6, 6, 6, 4, 6, 8, 4, 6, 6, 4, 8, 8, 4, 8], "pw_a_bits_setting": [6, 6, 4, 8, 8, 6, 6, 8, 6, 8, 8, 4, 8, 4, 6, 8, 6, 6, 6, 6, 4], "dw_w_bits_setting": [4, 4, 4, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 8, 6, 4, 8, 8, 8, 6], "dw_a_bits_setting": [4, 4, 4, 8, 4, 4, 6, 8, 8, 4, 4, 4, 4, 4, 4, 6, 6, 8, 4, 4, 4]}]
    quant_options=[]
    block_info_test,quant_list,block_depth=arch2spec(test_arch)
    for i in quant_list:
        for j in i:
            if j not in quant_options:
                quant_options.append(j)
    #print(quant_options)

    cifar=False
    acc1_space,acc2_space,dw_acc1_space,dw_acc2_space=design_choice_gen_apq(cifar=cifar)
    latency_list=[]
    best_throughput=0 
    for _ in range(250000):
        design_choice_integrity=False
        while not design_choice_integrity:
            input_params_set={}
            for quant_option in quant_options:
                input_params_set["a0q"+str(quant_option)]=random_sample(acc1_space)+[quant_option]
                input_params_set["a1q"+str(quant_option)]=random_sample(acc2_space)+[quant_option]
                input_params_set["dwa0q"+str(quant_option)]=random_sample(dw_acc1_space)+[quant_option]
                input_params_set["dwa1q"+str(quant_option)]=random_sample(dw_acc2_space)+[quant_option]
            for accel in input_params_set.keys():
                if input_params_set[accel][1] < input_params_set[accel][5] or\
                   input_params_set[accel][2] < input_params_set[accel][6] or\
                   input_params_set[accel][3] < input_params_set[accel][7] or\
                   input_params_set[accel][4] < input_params_set[accel][8]:
                    design_choice_integrity=False
                    break
                else:
                    design_choice_integrity=True
        try:
            bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
            layer_wise_break_down,consumption_used, consumption_breakdown,\
            accelerator_alloc,bs,block_wise_performance,net_struct=capsuled_predictor(input_params_set, block_info_test,quant_list,block_depth,cifar=cifar)
            param_size,mac_size,block_wise_mac=model_profiler(net_struct)
        except Exception as e:
            print(e)
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
    output_q.put((id,(best_throughput,best_consumption_used)))
    # return {id:(best_throughput,best_consumption_used)}

start=time.time()
data=worker(1)
dump_yard=[]
args=list(range(2))
num_worker_threads=2
multi_p(worker,args,output_q,num_worker_threads,dump_yard)
#print(dump_yard)

best_throughput=0
best_data=None 
for i, tmp in enumerate(dump_yard):
    if tmp[1][0]>best_throughput:
        best_throughput=tmp[1][0]
        best_data=copy.deepcopy(tmp)
print(best_data)        
print('takes: ',time.time()-start)
exit()

