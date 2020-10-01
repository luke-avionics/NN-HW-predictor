import os 
import numpy as np
import math
import re
from predictor_utilities import *
dnn_structure=[\
              [4,96,32,11,4],\
              [48,256,16,5,1],\
              [256,384,8,3,1],\
              [192,384,8,3,1],\
              [192,256,8,3,1],\
              [48,256,112,5,1],\
              [4,48,224,5,1]
]






def capsuled_predictor(input_params_set, block_info_test,quant_list,cifar,edd,channel_part):
    
    #generate the layer wise structure, if_layer_is_dw, layer_wise_quant
    net_struct,dw,layer_wise_quant,layer_block_corr=cifar_convert_to_layers(block_info_test,quant_list,cifar=cifar,edd=edd)

    #print(len(net_struct),len(dw))
    #print(mac_calc(net_struct))
    #exit()
    #allocate each layer with its corresponding accelerator
    #{layer_num: <accelerator_type>}
    accelerator_alloc, accelerator_types, accelerator_wise_budget=allocate_layers(net_struct,layer_wise_quant,dw,None,layer_block_corr,cifar=cifar,edd=edd,channel_part=channel_part)
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
    param_size,mac_size,block_wise_performance=model_profiler(net_struct,layer_block_corr)
    print(block_wise_performance)
    return bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
           layer_wise_break_down,consumption_used, consumption_breakdown,\
           accelerator_alloc,bs,block_wise_performance,net_struct



#print(combined_latency(2,[14, 7, 16, 4, 14, 7, 16, 1],[96, 480, 14, 1, 2]))
#exit()
############################
##front end testing
############################
block_options=['k3_e1','k3_e3','k3_e6','k5_e1','k5_e6','k5_e3','skip','k3_e1_g2','k5_e1_g2']
#quant_options=[4,6,8]
quant_options=[4,6,8]
channel_part=True
cifar=False 
edd=False

if not channel_part:
    acc1_space,acc2_space,dw_acc1_space,dw_acc2_space=design_choice_gen(cifar=cifar,edd=edd,channel_part=channel_part)
else:
    if edd: 
        (acc1_space,acc2_space,acc3_space,acc4_space,dw_acc1_space,dw_acc2_space,dw_acc3_space,dw_acc4_space)=design_choice_gen(cifar=cifar,edd=edd,channel_part=channel_part)
    else:
        (acc1_space,acc2_space,acc3_space,acc4_space,acc5_space,dw_acc1_space,dw_acc2_space,dw_acc3_space,dw_acc4_space,dw_acc5_space)=design_choice_gen(cifar=cifar,edd=edd,channel_part=channel_part)

  

latency_list=[]
best_throughput=0 
for _ in range(1000000):
    block_info_test=[]
    quant_list=[]
    #block info 
    for i in range(22):
        block_info_test.append(block_options[np.random.randint(9)])
    #block wise quantization choice
    for i in range(22):
        quant_list.append(quant_options[np.random.randint(len(quant_options))])
    
    ##Yongan's model
    #block_info_test= ['k5_e6', 'k5_e6', 'k3_e1', 'k5_e3', 'k5_e3', 'k5_e6', 'k5_e1_g2', 'k5_e1', 'k5_e6', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e1_g2', 'k3_e6', 'k5_e6', 'k3_e3', 'k5_e6', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e3', 'k5_e6']
    #quant_list= [6, 8, 6, 6, 4, 8, 6, 6, 6, 8, 6, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6]
    block_info_test=['k5_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e1', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6','k5_e3', 'k5_e1_g2', 'k5_e3','k3_e6', 'k5_e3', 'k3_e6', 'k3_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6', 'k5_e6']
    quant_list=[8, 8, 6, 6, 6, 8, 8, 8, 8, 8, 6, 8, 6, 8, 6, 6, 8, 8, 8, 6, 6, 8]          
    block_info_test=['skip', 'k5_e1', 'skip', 'skip', 'skip', 'k5_e3', 'skip', 'skip', 'k5_e1_g2', 'k5_e6', 'k5_e1_g2', 'k5_e1', 'k5_e1_g2', 'k5_e6', 'k5_e6', 'k3_e3', 'k5_e3', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e3', 'k5_e6']
    quant_list=[8]*22

    ##EDDnet3
    #block_info_test= ['k5_e5', 'k5_e4', 'k5_e4', 'k3_e5',    'k5_e4', 'k5_e5', 'k5_e6', 'k5_e6','k5_e6',    'k3_e4', 'k3_e4', 'k5_e4', 'k3_e4',    'k3_e4', 'k3_e4', 'k5_e6']
    #quant_list=[16]*16
    
    ##Mixed Precision Neural Architecture Search
    
    #print(block_info_test)
    #generate sample input
    # input_dict={}
    # for quant_option in quant_options:
        # input_dict[quant_option]=[random_sample(acc1_space),random_sample(acc2_space),random_sample(dw_acc1_space),random_sample(dw_acc2_space)]
    
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
            

    #print(input_params_set)
    #can be capsuled


    #!!!!!
    #pay attention to the format of input_params_set there is a [quant_option] in the end


    #!!!!
    #bottleneck_latency and block_wise_performance are what you want

    # bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
    # layer_wise_break_down,consumption_used, consumption_breakdown,\
    # accelerator_alloc,bs,block_wise_performance,net_struct=capsuled_predictor(input_params_set, block_info_test,quant_list,cifar=False,edd=True,channel_part=True)

    try:
        bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
        layer_wise_break_down,consumption_used, consumption_breakdown,\
        accelerator_alloc,bs,block_wise_performance,net_struct=capsuled_predictor(input_params_set, block_info_test,quant_list,cifar=cifar,edd=edd,channel_part=channel_part)
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
    print(best_consumption_used)
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
#EDD centric comparison
#    net_struct,dw,layer_wise_quant=cifar_convert_to_layers(block_info_test,quant_list)


#############################
##predictor backend testing
#############################
files=[
       'fixed_hw_cp1_data4.npy',\
       'fixed_hw_cp2_data4.npy',\
       'fixed_hw_cp2_data7.npy',\
       'fixed_hw_cp3_data7.npy',\
       ]
#files=['fixed_hw_cp3_data7.npy']
for i,fn in enumerate(files):
    if i ==0:
        raw=np.load(fn,allow_pickle=True)
    else:
        raw=np.concatenate((raw,np.load(fn,allow_pickle=True))) 

#raw=np.load('fixed_hw_cp1_data4.npy',allow_pickle=True)
raw_len=len(raw)
print(raw_len)

#take largest factor and lower ones 
#[comp_mode,fw,fh,of,if,f(fw),f(fh),f(of),f(if),quant]

#depthwise in one chunk -> so one option for depthwise

#bandwitdh -> accelerator numbers
#allocation of accelerators 


error_list=[]
ctr=0
for i, dp in enumerate(raw):
    absolute_truth=float(dp[1])
    predicted_perf=combined_latency(dp[0][13],dp[0][5:13],dp[0][0:5])
    if(np.abs(absolute_truth-predicted_perf)/absolute_truth>0.2):
        # print('truth: ',absolute_truth,'  predicted: ', predicted_perf)
        # print('tiling')
        # print(dp[0][5:13])
        # print('net structure')
        # print(dp[0][0:5])
        ctr+=1
    error_list.append(np.abs(absolute_truth-predicted_perf)/absolute_truth)
print('cycles estimate error: ', np.mean(error_list))
print('extremely wrong answers: ', ctr/raw_len)

dsp_error=[]
bram_error=[]
for i, dp in enumerate(raw):
    true_consumption=dp[2]
    true_dsp=true_consumption['dsp']
    true_bram=true_consumption['bram']
    predicted_dsp,predicted_bram=resource_consumption(dp[0][13],dp[0][5:13],dp[0][0:5],dw=False,quant=16)
    dsp_error.append(np.abs(true_dsp-predicted_dsp)/true_dsp)
    bram_error.append(np.abs(true_bram-predicted_bram)/true_bram)

print('dsp prediction error: ', np.mean(dsp_error))
print('bram prediction error: ', np.mean(bram_error))

test_id=10
dp=raw[test_id]
print(dp)
print(combined_latency(dp[0][13],dp[0][5:13],dp[0][0:5]))
print(resource_consumption(dp[0][13],dp[0][5:13],dp[0][0:5],dw=False,quant=16))
