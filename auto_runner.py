import time
import random
import subprocess as sbp
import os
import xml.etree.ElementTree as ET
import shutil 
import re 
import numpy as np

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


layer_struct=[\
              [4,96,32,4,11],\
              [48,256,16,1,5],\
              [256,384,8,1,3],\
              [192,384,8,1,3],\
              [192,256,8,1,3]
]



class auto_hls_run:
    #note1: did you already run the source to include the vivado?
    #note2: is base solution1 source present?  
    def __init__(self,project_dir,sub_project_dir,top_module,vivado_path,bash=True): 
        self.project_dir=project_dir
        self.sub_project_dir=sub_project_dir
        self.vivado_path=vivado_path
        self.bash=bash
        self.top_module=top_module
        self.input_design=[] 
        #self.sys_setup()
        #self.license_config()
    def license_config(self):
        pass
        return None
    def sys_setup(self):
        #TODO:source not working yet
        #source before hand instead
        if not self.bash:
            os.system("source "+self.vivado_path+"settings64.csh")
        else:
            os.system("source "+self.vivado_path+"settings64.sh")
        return None

    def modify_tiling_comp_mode(self,fn,layer_num,tiling_list,comp_mode):
        reading_file = open(fn, "r")
        new_file_content = []
        tk=layer_struct[layer_num-1][4]
        tri=max(tiling_list[4]+layer_struct[layer_num-1][4]-1,tiling_list[0])
        tci=max(tiling_list[5]+layer_struct[layer_num-1][4]-1,tiling_list[1])
        lines=reading_file.readlines()
        for line in lines:
            if "const int TrBuff"+str(layer_num)+" =" in line:
            #under construction 
                new_line="const int TrBuff"+str(layer_num)+" = "+str(tiling_list[0])+", TcBuff"+str(layer_num)+" = "+str(tiling_list[1])+\
                                 ", TmBuff"+str(layer_num)+" = "+str(tiling_list[2])+", TnBuff"+str(layer_num)+" = "+str(tiling_list[3])+\
                                 ", Tr"+str(layer_num)+" = "+str(tiling_list[4])+", Tc"+str(layer_num)+" = "+str(tiling_list[5])+\
                                 ", Tm"+str(layer_num)+" = "+str(tiling_list[6])+", Tn"+str(layer_num)+" = "+str(tiling_list[7])+\
                                 ", Tk"+str(layer_num)+" = "+str(tk)+\
                                 ", Tri"+str(layer_num)+" = "+str(tri)+", Tci"+str(layer_num)+" = "+str(tci)+\
                                 ";\n"
            elif "conv"+str(layer_num)+":conv3_3" in line:
                tmp_comp_mode=re.findall(r'conv\d+:conv3_3_\d+',line)[0]
                new_line=line.replace(tmp_comp_mode, "conv"+str(layer_num)+":conv3_3_"+str(comp_mode+1))
            else:
                new_line=line
            new_file_content.append(new_line)
        reading_file.close()
        writing_file = open(fn, "w")
        writing_file.writelines(new_file_content)
        writing_file.close()

        return None
    def modify_pragma_unroll(self,fn,layer_num,comp_mode,pragma_info,kernel_mem_info):
        #currently unrolling only allow dimension specific unrolling, instead of be specific to each data type buffer
        #For instance,
        #              if you want to unroll input_channel, you have to unroll both the input channel at 
        #                                                                                           weight and input
        reading_file = open(fn, "r")
        new_file_content = []
        lines=reading_file.readlines()
        new_line=""
        right_conv_mode=False
        for line in lines:
            #check if conv mode is right
            if "void conv3_3_"+str(comp_mode+1) in line:
                right_conv_mode=True
            else:
                pass
            if right_conv_mode and "partition anchor" in line:
                if pragma_info[0]==1:
                    new_line+="#pragma HLS ARRAY_PARTITION variable=feature_temp complete dim=2\n"
                    new_line+="#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=2\n"
                    new_line+="#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=2\n"
                if pragma_info[1]==1:
                    new_line+="#pragma HLS ARRAY_PARTITION variable=feature_temp complete dim=3\n"
                    new_line+="#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=3\n"
                    new_line+="#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=3\n"
                if pragma_info[2]==1:
                    new_line+="#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=1\n"
                    new_line+="#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=1\n"
                if pragma_info[3]==1:
                    new_line+="#pragma HLS ARRAY_PARTITION variable=feature_temp complete dim=1\n"
                    new_line+="#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=1\n"
                    new_line+="#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=2\n"
                if kernel_mem_info[1]==1:
                    new_line+="#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=3\n"
                if kernel_mem_info[2]==1:
                    new_line+="#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=4\n"
            #deal with kernel memory type
            elif right_conv_mode and "#pragma HLS RESOURCE variable=weight_temp" in line:
                if kernel_mem_info[0]==0:
                    new_line="#pragma HLS RESOURCE variable=weight_temp core=RAM_2P_BRAM\n"
                elif kernel_mem_info[0]==1:
                    new_line="#pragma HLS RESOURCE variable=weight_temp core=RAM_2P_LUTRAM\n"
            else:
                new_line=line
            #clear right_conv_mode flag
            if right_conv_mode and "partition finished" in line:
                right_conv_mode=False 
            new_file_content.append(new_line)
        reading_file.close()
        writing_file = open(fn, "w")
        writing_file.writelines(new_file_content)
        writing_file.close()
        return None

    def modify_loop_ordering(self,fn,layer_num,comp_mode):
        #ordering of the loops outside of the comp engine 
        #ordering of the loops inside of the comp engine  


       # reading_file = open(fn, "r")
       # new_file_content = []
       # lines=reading_file.readlines()
       # new_line=""
       # for line in lines:
             
        return None

    def string_replace(self,fn,str1,str2):
        reading_file = open(fn, "r")
        new_file_content = []
        lines=reading_file.readlines()
        for line in lines:
            new_line = line.replace(str1, str2)
            new_file_content.append(new_line)
        reading_file.close()

        writing_file = open(fn, "w")
        writing_file.writelines(new_file_content)
        writing_file.close()
        return None
    def solution_creater(self,starting_soln, ending_soln):
        if starting_soln==1 or ending_soln==1:
            raise Exception('why are u trying to creat the overwrite the base solution1 source? check ur starting/ending range')
        base_soln_path=self.project_dir+self.sub_project_dir+"solution1/"
        files_to_cp=os.listdir(base_soln_path)
        for i in range(starting_soln,ending_soln+1):    
            try:
                shutil.rmtree(self.project_dir+self.sub_project_dir+"solution"+str(i)+"/")
            except:
                pass
            #script.tcl solutionx.aps needs under go modification 
            os.mkdir(self.project_dir+self.sub_project_dir+"solution"+str(i)+"/")
            shutil.copy(base_soln_path+"directives.tcl",self.project_dir+self.sub_project_dir+"solution"+str(i)+"/"+"directives.tcl")
            shutil.copy(base_soln_path+"script.tcl",self.project_dir+self.sub_project_dir+"solution"+str(i)+"/"+"script.tcl")
            shutil.copy(base_soln_path+"solution1.aps",self.project_dir+self.sub_project_dir+"solution"+str(i)+"/"+"solution"+str(i)+".aps")
            shutil.copy(base_soln_path+"solution1.directive",self.project_dir+self.sub_project_dir+"solution"+str(i)+"/"+"solution"+str(i)+".directive")
            #modify for different solutions
            self.string_replace(self.project_dir+self.sub_project_dir+"solution"+str(i)+"/"+"solution"+str(i)+".aps",'solution1',"solution"+str(i))
            self.string_replace(self.project_dir+self.sub_project_dir+"solution"+str(i)+"/"+"solution"+str(i)+".directive",'solution1',"solution"+str(i))
            self.string_replace(self.project_dir+self.sub_project_dir+"solution"+str(i)+"/"+"script.tcl",'solution1',"solution"+str(i))
        return None 
    def r_factors(self,x):
        #find the factors of a number
        factor_list=[]
        #starting number 2 here in order to favor of the pipeline
        for i in range(1, x + 1):
            if x % i == 0:
                factor_list.append(i)
        return factor_list
    



    def source_modifier(self, dnn_layer_all,dnn_layer_x,batch_id, batch_size=10):
        #tiling options
        tr1=[32,16,8,4,2]
        tc1=[32,16,8,4,2]
        tm1=[96,48,32,16,12,8,6,4,3,2]
        tn1=[3]
        print(len(tr1)*len(tc1)*len(tm1)*len(tn1))
        tr2=sorted(self.r_factors(16),reverse=True)[0:-1]
        tc2=sorted(self.r_factors(16),reverse=True)[0:-1]
        tm2=sorted(self.r_factors(256),reverse=True)[0:-1]
        tn2=sorted(self.r_factors(48),reverse=True)[0:-1]
        print(len(tr2)*len(tc2)*len(tm2)*len(tn2))
        tr3=sorted(self.r_factors(layer_struct[dnn_layer_x-1][2]),reverse=True)[0:-1]
        tc3=sorted(self.r_factors(layer_struct[dnn_layer_x-1][2]),reverse=True)[0:-1]
        tm3=sorted(self.r_factors(layer_struct[dnn_layer_x-1][1]),reverse=True)[0:-1]
        tn3=sorted(self.r_factors(layer_struct[dnn_layer_x-1][0]),reverse=True)[0:-1]
        print(len(tr3)*len(tc3)*len(tm3)*len(tn3))
        tr4=sorted(self.r_factors(8),reverse=True)[0:-1]
        tc4=sorted(self.r_factors(8),reverse=True)[0:-1]
        tm4=sorted(self.r_factors(384),reverse=True)[0:-1]
        tn4=sorted(self.r_factors(192),reverse=True)[0:-1]
        print(len(tr4)*len(tc4)*len(tm4)*len(tn4))
        tr5=sorted(self.r_factors(8),reverse=True)[0:-1]
        tc5=sorted(self.r_factors(8),reverse=True)[0:-1]
        tm5=sorted(self.r_factors(256),reverse=True)[0:-1]
        tn5=sorted(self.r_factors(192),reverse=True)[0:-1]
        print(len(tr5)*len(tc5)*len(tm5)*len(tn5))
        
        
        comp_mode_num=3
        
        entire_space=[[tr1,tc1,tm1,tn1],\
                      [tr2,tc2,tm2,tn2],\
                      [tr3,tc3,tm3,tn3],\
                      [tr4,tc4,tm4,tn4],\
                      [tr5,tc5,tm5,tn5],\
                      ]
                   

        #option pool
        option_pool=[]
        #denoting the unroll pragma
        for tr_p in range(2):
            for tc_p in range(2):
                for tm_p in range(2):
                    for tn_p in range(2):
                        for kernel_mem_type in range(2):
                            for kernel_unroll_row in range(2):
                                for kernel_unroll_col in range(2):
                                    for t_cm in range(comp_mode_num):
                                        #buff decision
                                        if t_cm==0 or t_cm==1:
                                            for t_trbuff in entire_space[dnn_layer_x-1][0]:
                                                for t_tcbuff in entire_space[dnn_layer_x-1][1]:
                                                    for t_tmbuff in entire_space[dnn_layer_x-1][2]:
                                                        for t_tnbuff in entire_space[dnn_layer_x-1][3]:
                                                            for t_tm in sorted(self.r_factors(t_tmbuff),reverse=True)[0:-1]:
                                                                for t_tn in sorted(self.r_factors(t_tnbuff),reverse=True)[0:-1]:
                                                                    option_pool.append(([t_trbuff,t_tcbuff,t_tmbuff,t_tnbuff,t_trbuff,t_tcbuff,t_tm,t_tn],t_cm,[tr_p,tc_p,tm_p,tn_p],[kernel_mem_type,kernel_unroll_row,kernel_unroll_col]))
                                                                    #option_pool.append(([t_tr,t_tc,t_tm,t_tn],t_cm,[1,1,1,1],[1,kernel_unroll_row,kernel_unroll_col]))
                                        else:
                                            for t_trbuff in entire_space[dnn_layer_x-1][0]:
                                                for t_tcbuff in entire_space[dnn_layer_x-1][1]:
                                                    for t_tmbuff in entire_space[dnn_layer_x-1][2]:
                                                        for t_tnbuff in entire_space[dnn_layer_x-1][3]:
                                                            for t_tr in sorted(self.r_factors(t_trbuff),reverse=True)[0:-1]:
                                                                for t_tc in sorted(self.r_factors(t_tcbuff),reverse=True)[0:-1]:
                                                                    option_pool.append(([t_trbuff,t_tcbuff,t_tmbuff,t_tnbuff,t_tr,t_tc,t_tmbuff,t_tnbuff],t_cm,[tr_p,tc_p,tm_p,tn_p],[kernel_mem_type,kernel_unroll_row,kernel_unroll_col]))
                                                                    #option_pool.append(([t_tr,t_tc,t_tm,t_tn],t_cm,[1,1,1,1],[1,kernel_unroll_row,kernel_unroll_col]))
    
        random.Random(4).shuffle(option_pool)
        #copy and remove previous source
        #print(option_pool[0:batch_id*batch_size+batch_size])
        for i in range(0,batch_size):
            #remove file
            try:
                shutil.rmtree(self.project_dir+"src"+str(i+2))
            except:
                pass
            #try:
            #    os.remove(self.project_dir+"conv3x3_"+str(i+2)+".h")
            #except:
            #    pass
            os.mkdir(self.project_dir+"src"+str(i+2))
            #create new
            shutil.copy(self.project_dir+"conv3x3.cpp", self.project_dir+"src"+str(i+2)+"/conv3x3.cpp")
            shutil.copy(self.project_dir+"conv3x3.h", self.project_dir+"src"+str(i+2)+"/conv3x3.h")
            chosen_design=option_pool[batch_id*batch_size+i] 
            self.input_design.append(chosen_design)
            # layer info needed here, start from 1 
            self.modify_tiling_comp_mode(self.project_dir+"src"+str(i+2)+"/conv3x3.cpp",dnn_layer_x,chosen_design[0],chosen_design[1])
            self.modify_pragma_unroll(self.project_dir+"src"+str(i+2)+"/conv3x3.cpp",dnn_layer_x,chosen_design[1],chosen_design[2],chosen_design[3])
            #self.modify_loop_ordering()
            #modify script file for each solution 
            self.string_replace(self.project_dir+self.sub_project_dir+"solution"+str(i+2)+"/"+"script.tcl","conv3x3.cpp","src"+str(i+2)+"/conv3x3.cpp")
            self.string_replace(self.project_dir+self.sub_project_dir+"solution"+str(i+2)+"/"+"script.tcl","conv3x3.h","src"+str(i+2)+"/conv3x3.h")
        return None

    def script_starter(self, solution):
        print("vivado_hls -f "+self.project_dir+self.sub_project_dir+"solution"+str(solution)+"/script.tcl -l vivado_soln"+str(solution)+".log")
        return_st=sbp.Popen(["vivado_hls -f "+self.project_dir+self.sub_project_dir+"solution"+str(solution)+"/script.tcl -l vivado_soln"+str(solution)+".log"],shell=True, stdout=sbp.DEVNULL, stderr=sbp.DEVNULL,cwd=self.project_dir)
        #print(return_st)
        return return_st
    
    def results_collector(self, solution):
        report_path=self.project_dir+self.sub_project_dir+"solution"+str(solution)+"/syn/report/"+self.top_module+"_csynth.xml"
        max_res={'dsp':0, 'bram':0,'lut':0, 'ff': 0}
        used_res={'dsp':0, 'bram':0,'lut':0, 'ff': 0}
        tree = ET.parse(report_path)
        root = tree.getroot()
        
        performance_analysis=root.find('PerformanceEstimates')
        timing_stat=performance_analysis.find('SummaryOfTimingAnalysis')
        timing_unit=timing_stat.find('unit').text    
        timing_period=float(timing_stat.find('EstimatedClockPeriod').text)
        latency_stat=performance_analysis.find('SummaryOfOverallLatency')
        max_interval=latency_stat.find('Interval-max').text
        
        area_est=root.find("AreaEstimates")
        max_a_est=area_est.find("AvailableResources")
        used_a_est=area_est.find('Resources')
        max_res['bram']=int(max_a_est.find('BRAM_18K').text)
        max_res['dsp']=int(max_a_est.find('DSP48E').text)   
        max_res['lut']=int(max_a_est.find('LUT').text)
        max_res['ff']=int(max_a_est.find('FF').text)
        used_res['bram']=int(used_a_est.find('BRAM_18K').text)
        used_res['dsp']=int(used_a_est.find('DSP48E').text)
        used_res['lut']=int(used_a_est.find('LUT').text)
        used_res['ff']=int(used_a_est.find('FF').text)
        res_percent={} 
        for i in max_res:
            res_percent[str(i)]=used_res[str(i)]/max_res[str(i)]*100  

        return max_interval, used_res, max_res, res_percent
    def group_results_collector(self, starting_soln, ending_soln,batch_id):
        results_pack=[]
        ctr=0
        for i in range(starting_soln, ending_soln+1):
            try:
                results_pack.append((self.input_design[ctr],self.results_collector(i))) 
            except:
                pass
            ctr+=1
        np.save('results_pack'+str(batch_id)+'.npy',results_pack)
        return results_pack



test=auto_hls_run('/home/yz87/hls_dir/auto_runner_test/','energy_mode_check/',
                 'conv3_3','/home/yz87/Xilinx/Vivado/2018.3/') 

#iterate over the layers    
for batch_id in range(0,9):
    #change cpp source for different layer
    batch_size=100
    dnn_layer_all=5
    dnn_layer_x=3
    
    test.solution_creater(2,batch_size+1)
    
    test.source_modifier(dnn_layer_all,dnn_layer_x,batch_id,batch_size=batch_size)
    
    
    try:
        os.remove(test.project_dir+test.sub_project_dir+"vivado_hls.app")
    except:
        pass
    
    p_list=[]
    for i in range(2,batch_size+2):
        p=test.script_starter(i)
        p_list.append(p)
        time.sleep(10)
        try:
            os.remove(test.project_dir+test.sub_project_dir+"vivado_hls.app")
        except:
            pass
    
    #exit_codes = [p.wait() for p in p_list]
    
    
    threshold=180
    waiting_period={}
    for p in range(len(p_list)):
        waiting_period[p]=0
    
    
    finish_flag=False
    while not finish_flag:
        for p in range(len(p_list)):
            if p_list[p].poll()==None and  waiting_period[p]<=threshold:
                waiting_period[p]+=1
            elif p_list[p].poll()==None and  waiting_period[p] > threshold:
                p_list[p].kill()
            else:
                pass 
        unfinished_pool=[]
        for p in range(len(p_list)):
            if p_list[p].poll()==None:
                unfinished_pool.append(p)
            #print(p_list[p].poll()!=None,end = '')
        print(unfinished_pool)
        print(len(unfinished_pool))
        print("====="*20)
        finished_num=0
        for p in range(len(p_list)):
            if p_list[p].poll()!=None:
                finished_num+=1
        if finished_num==len(p_list):
            break
        time.sleep(10)
    
    results=test.group_results_collector(2,batch_size+1,batch_id)
    print(results)
    os.system("pkill -u yz87 vivado_hls")
    os.system("rm -r src*")
    os.system("rm *.log")
    #move the npy
    #rt=sbp.call(['vivado_hls'],shell=True, stdout=sbp.PIPE, stderr=sbp.STDOUT,cwd=test.project_dir)
    #print(rt)
