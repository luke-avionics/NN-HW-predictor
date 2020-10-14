import os 
import numpy as np
import math
import re
import copy
from predictor_utilities import *

def cifar_convert_to_layers_mixed(block_info,quant_list,cifar=True,small=True):
    #TODO: include EDD cases
    if not small:
        if cifar:
            output_dim=[32]*2+[16]*3+[8]*8+[4]*6
            num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1]
            #currently only support 1 
            #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
            #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
            num_channel_list = [16]+[24]*2+[32]*3+[64]*4+[128]*4+[160]*4+[256]
            stride_list = [2, 1,2,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1,1]
            
        else: 
            output_dim=[56]*2+[28]*3+[14]*8+[7]*6
            num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1]
            #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
            #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
            num_channel_list = [16]+[24]*2+[32]*3+[64]*4+[128]*4+[160]*4+[256]
            stride_list = [2, 1,2,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1,1]
    else:
        if cifar:
            output_dim=[32]*2+[16]*3+[8]*7+[4]*7
            num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1]
            #currently only support 1 
            #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
            #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
            num_channel_list = [16]+[24]*2+[32]*3+[64]*4+[128]*4+[160]*4+[256]
            stride_list = [2, 1,2,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1,1]
            
        else: 
            output_dim=[56]*2+[28]*3+[14]*7+[7]*7
            num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1]
            #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
            #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
            num_channel_list = [16]+[24]*2+[32]*3+[64]*3+[128]*4+[160]*5+[256]
            stride_list = [2, 1,2,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1,1]
    
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
                    layer_wise_quant+=[quant_bit[0],quant_bit[1],quant_bit[0]]
                    layer_block_corr[0]+=[0,1,2]
                    layer_num+=3
                else:
                    net_struct.append([num_channel_list[i-1],num_channel_list[i-1]*e,output_dim[i-1],1,stride_list[i]])
                    net_struct.append([1,num_channel_list[i-1]*e,output_dim[i],k,1])
                    net_struct.append([num_channel_list[i-1]*e,num_channel_list[i],output_dim[i],1,1])  
                    dw+=[False,True,False]
                    quant_bit=quant_list.pop(0)
                    layer_wise_quant+=[quant_bit[0],quant_bit[1],quant_bit[0]]
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
                    layer_wise_quant+=[quant_bit[0],quant_bit[0],quant_bit[1],quant_bit[0],quant_bit[0]]
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
                    layer_wise_quant+=[quant_bit[0],quant_bit[0],quant_bit[1],quant_bit[0],quant_bit[0]]
                    layer_block_corr[i]+=[layer_num,layer_num+1,layer_num+2,layer_num+3,layer_num+4]
                    layer_num+=5
            else:
                raise Exception('Currently not supporting repetive block info input')
    return net_struct,dw,layer_wise_quant,layer_block_corr


def capsuled_predictor(input_params_set, block_info_test,quant_list,cifar,small):
    
    #generate the layer wise structure, if_layer_is_dw, layer_wise_quant
    net_struct,dw,layer_wise_quant,layer_block_corr=cifar_convert_to_layers_mixed(block_info_test,copy.deepcopy(quant_list),cifar=cifar,small=small)

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
           
def design_choice_gen_mixed(cifar,small): 

    if not small:
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

    else:
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
    quant_options=[2,4,6]
    cifar=False
    small=True
    acc1_space,acc2_space,dw_acc1_space,dw_acc2_space=design_choice_gen_mixed(cifar=cifar,small=small)
    latency_list=[]
    best_throughput=0 
    for _ in range(500000):
        if small:
            quant_list=[(6,2),(6,4),(4,2),(2,2), (4,2),(2,2) , (2,2),(4,4),(2,2),(2,2),  (4,2),(4,4),(4,2),(2,2), (4,2),(4,4),(4,2),(2,2), (2,2) ]
            block_info_test=['k3_e1','k3_e6','k5_e1','k5_e6',\
                             'k3_e1','k3_e1','k3_e6','k5_e1',\
                             'k5_e3','k5_e3','k3_e1','k3_e1',\
                             'k3_e1','k5_e6','k3_e1','k3_e1',\
                             'k3_e1','k5_e6','k5_e6']
        else:
            quant_list=[(6,2),(6,4),(4,2),(2,2), (4,2),(2,2) , (2,2),(4,4),(4,2),(2,2),  (4,2),(4,4),(2,2),(4,2), (2,2),(4,4),(4,2),(4,2), (2,2) ]
            block_info_test=['k3_e1','k3_e6','k3_e3','k5_e6',\
                             'k5_e3','k5_e3','k5_e6','k5_e3',\
                             'k5_e1','k5_e6','k5_e6','k5_e1',\
                             'k5_e3','k3_e1','k5_e6','k5_e6',\
                             'k5_e3','k5_e6','k5_e6']
        #print(block_info_test)
        #generate sample input
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
            accelerator_alloc,bs,block_wise_performance,net_struct=capsuled_predictor(input_params_set, block_info_test,quant_list,cifar=cifar,small=small)
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
        # print(best_throughput)
        # print(best_consumption_used)
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


data=worker(1)
dump_yard=[]
args=list(range(4))
num_worker_threads=4
multi_p(worker,args,output_q,num_worker_threads,dump_yard)
#print(dump_yard)

best_throughput=0
best_data=None 
for i, tmp in enumerate(dump_yard):
    if tmp[1][0]>best_throughput:
        best_throughput=tmp[1][0]
        best_data=copy.deepcopy(tmp)
print(best_data) 

exit()
