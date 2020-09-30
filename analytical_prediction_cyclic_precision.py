import os 
import numpy as np
import math
import re
import copy
from predictor_utilities import *
from resnet_def import *


def capsuled_predictor(input_params_set, net_struct,quant_list,cifar,edd,channel_part=False):
    accelerator_alloc, accelerator_types, accelerator_wise_budget=allocate_layers(net_struct,quant_list,[0]*len(net_struct),None,None,cifar=cifar,edd=edd,channel_part=channel_part)
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
    return bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
           layer_wise_break_down,consumption_used, consumption_breakdown,\
           accelerator_alloc,bs

def design_choice_gen_resnet(cifar):
    if cifar:
        acc1_space={'comp_mode':[0,1,2],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[16,8,4,2,1],'tnbuff':[16,8,4,2,1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[16,8,4,2,1],'tn':[16,8,4,2,1]}
        acc2_space={'comp_mode':[0,1,2],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
        dw_acc1_space={'comp_mode':[0,1],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[16,8,4,2,1],'tnbuff':[1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[16,8,4,2,1],'tn':[1]}
        dw_acc2_space={'comp_mode':[0,1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
    else:
        acc1_space={'comp_mode':[0,1,2],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
        acc2_space={'comp_mode':[0,1,2],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
        dw_acc1_space={'comp_mode':[0,1],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[8,4,2,1],'tn':[1]}
        dw_acc2_space={'comp_mode':[0,1],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
    return (acc1_space,acc2_space,dw_acc1_space,dw_acc2_space)

bit=8
quant_options=[bit]    
net_struct=copy.deepcopy(resnet164)
quant_list=[bit]*len(net_struct)
dw=[0]*len(net_struct)
channel_part=False
cifar=True
edd=False

if not channel_part:
    acc1_space,acc2_space,dw_acc1_space,dw_acc2_space=design_choice_gen(cifar=cifar,edd=edd,channel_part=channel_part)
else:
    if edd: 
        (acc1_space,acc2_space,acc3_space,acc4_space,dw_acc1_space,dw_acc2_space,dw_acc3_space,dw_acc4_space)=design_choice_gen(cifar=cifar,edd=edd,channel_part=channel_part)
    else:
        (acc1_space,acc2_space,acc3_space,acc4_space,acc5_space,dw_acc1_space,dw_acc2_space,dw_acc3_space,dw_acc4_space,dw_acc5_space)=design_choice_gen_resnet(cifar=cifar,edd=edd,channel_part=channel_part)
acc1_space={'comp_mode':[1],'trbuff':[16],'tcbuff':[16],'tmbuff':[16/(bit/3)**(0.5)],'tnbuff':[16/(bit/3)**(0.5)], 'tr':[16],'tc':[16],'tm':[16/(bit/3)**(0.5)],'tn':[16/(bit/3)**(0.5)]}
acc2_space={'comp_mode':[1],'trbuff':[4],'tcbuff':[4],'tmbuff':[32/(bit/3)**(0.5)],'tnbuff':[32/(bit/3)**(0.5)], 'tr':[4],'tc':[4],'tm':[32/(bit/3)**(0.5)],'tn':[32/(bit/3)**(0.5)]}

latency_list=[]
best_throughput=0 
for _ in range(10):
    if not channel_part:
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
    else: 
        if not edd:
            design_choice_integrity=False
            while not design_choice_integrity:
                input_params_set={}
                for quant_option in quant_options:
                    input_params_set["a0q"+str(quant_option)]=random_sample(acc1_space)+[quant_option]
                    input_params_set["a1q"+str(quant_option)]=random_sample(acc2_space)+[quant_option]
                    input_params_set["a2q"+str(quant_option)]=random_sample(acc3_space)+[quant_option]
                    input_params_set["a3q"+str(quant_option)]=random_sample(acc4_space)+[quant_option]
                    input_params_set["a4q"+str(quant_option)]=random_sample(acc5_space)+[quant_option]
                    
                    input_params_set["dwa0q"+str(quant_option)]=random_sample(dw_acc1_space)+[quant_option]
                    input_params_set["dwa1q"+str(quant_option)]=random_sample(dw_acc2_space)+[quant_option]
                    input_params_set["dwa2q"+str(quant_option)]=random_sample(dw_acc3_space)+[quant_option]
                    input_params_set["dwa3q"+str(quant_option)]=random_sample(dw_acc4_space)+[quant_option]
                    input_params_set["dwa4q"+str(quant_option)]=random_sample(dw_acc5_space)+[quant_option]
                for accel in input_params_set.keys():
                    if input_params_set[accel][1] < input_params_set[accel][5] or\
                       input_params_set[accel][2] < input_params_set[accel][6] or\
                       input_params_set[accel][3] < input_params_set[accel][7] or\
                       input_params_set[accel][4] < input_params_set[accel][8]:
                        design_choice_integrity=False
                        break
                    else:
                        design_choice_integrity=True
        else:
            design_choice_integrity=False
            while not design_choice_integrity:
                input_params_set={}
                for quant_option in quant_options:
                    input_params_set["a0q"+str(quant_option)]=random_sample(acc1_space)+[quant_option]
                    input_params_set["a1q"+str(quant_option)]=random_sample(acc2_space)+[quant_option]
                    input_params_set["a2q"+str(quant_option)]=random_sample(acc3_space)+[quant_option]
                    input_params_set["a3q"+str(quant_option)]=random_sample(acc4_space)+[quant_option]
  
                    
                    input_params_set["dwa0q"+str(quant_option)]=random_sample(dw_acc1_space)+[quant_option]
                    input_params_set["dwa1q"+str(quant_option)]=random_sample(dw_acc2_space)+[quant_option]
                    input_params_set["dwa2q"+str(quant_option)]=random_sample(dw_acc3_space)+[quant_option]
                    input_params_set["dwa3q"+str(quant_option)]=random_sample(dw_acc4_space)+[quant_option]

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
        accelerator_alloc,bs=capsuled_predictor(input_params_set, net_struct,quant_list,cifar,edd)
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
    print(best_throughput)
print('throughput: ', best_throughput)
print('best_bs: ', best_bs)
print('latency_break_down: ', best_latency_break_down)
print('layer_wise_break_down: ',best_layer_wise_break_down)
print('consumption_used: ', best_consumption_used)
print('consumption_breakdown: ', best_consumption_breakdown)
print('accelerator_alloc', best_accelerator_alloc)
print('input_params',best_input_params_set)
print('net_struct', net_struct)
exit()