import paramiko
import xml.etree.ElementTree as ET
import re
import time
import copy
from scp import SCPClient
from predictor_utilities import *
def insert_str2str (source_str, insert_str, pos):
    return source_str[:pos]+insert_str+source_str[pos:]


ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.connect('10.129.1.80', username='hy34', password='youhaoran')
# SCPCLient takes a paramiko transport as an argument
scp = SCPClient(ssh.get_transport())
print('connection established')

#take the input from Yonggan's space and convert it 
block_options=['k3_e1','k3_e3','k3_e6','k5_e1','k5_e6','k5_e3','skip','k3_e1_g2','k5_e1_g2']
#quant_options=[4,6,8]
quant_options=[4,6,8]
dnn_dimensions="NPQRSCM"
channel_part=True
cifar=False
edd=False
##Yongan's model
block_info_test= ['k5_e6', 'k5_e6', 'k3_e1', 'k5_e3', 'k5_e3', 'k5_e6', 'k5_e1_g2', 'k5_e1', 'k5_e6', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e1_g2', 'k3_e6', 'k5_e6', 'k3_e3', 'k5_e6', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e3', 'k5_e6']
quant_list= [16]*22
                                              #Split, ... ordering....
params_input ={'PE_size':168,"dummy_buffer_split":5,
                              "shared_glb_split":6,
                              "dummy_buffer_order":[0,1,2,3,4,5,6],\
                              "psum_spad_order":[0,1,2,3,4,5,6],\
                              "ifmap_spad_order":[0,1,2,3,4,5,6],\
                              "weights_spad_order":[0,1,2,3,4,5,6],\
                              "dummy_buffer2_order":[0,1,2,3,4,5,6],\
                              "shared_glb_temporal_order":[0,1,2,3,4,5,6],\
                              "DRAM_order":[0,1,2,3,4,5,6]}



work_space="/home/hy34/accelergy_luke/timeloop-accelergy-exercises/exercises/timeloop/06-mapper-convlayer-eyeriss/"
network_config="prob/VGG02_layer5.yaml"
hw_arch_config="arch/eyeriss_like.yaml"
mapper_config="mapper/mapper.yaml"
conda_bin="/home/hy34/anaconda3/bin/conda"
accelergy_bin="/home/hy34/anaconda3/bin/accelergy"
cacti_path="/home/hy34/anaconda3/bin/"
arch_constraints="constraints/eyeriss_like_arch_constraints.yaml"
map_constraints="constraints/eyeriss_like_map_constraints.yaml"

def capsulated_asic_model(block_info_test,quant_list,params_input,channel_part,cifar,edd):
    global work_space,network_config,hw_arch_config,mapper_config,conda_bin,accelergy_bin,cacti_path,arch_constraints,map_constraints
    global ssh,scp
    net_struct,dw,layer_wise_quant,layer_block_corr=cifar_convert_to_layers(block_info_test,copy.deepcopy(quant_list),cifar=cifar,edd=edd)
    #modify arch PE number buffer size bit precision 
    scp.get(work_space+hw_arch_config)
    quant_option=quant_list[0]
    reading_file = open(hw_arch_config.split('/')[1], "r")
    new_file_content = []
    lines=reading_file.readlines()
    for line_idx, line in enumerate(lines):
        new_line=""
        #modify bit precision 
        if line_idx==14 or line_idx==27 or line_idx==35 or line_idx==47 or line_idx==57 or line_idx==68 or line_idx==75 or\
           line_idx==34 or line_idx==45 or line_idx==55 or line_idx==65:
            new_line+=line.split(': ')[0]+": "+str(quant_option)+"\n"
        elif line_idx==13 or line_idx==26:
            new_line+=line.split(': ')[0]+": "+str(int(64/quant_option))+"\n"
        #modify PE sizes
        elif line_idx==39:
            new_line+=line.split('[')[0]+"[0.."+str(params_input['PE_size']-1)+"]\n"
        #modify global buffer size: will follow the PE size: the more PE you have, the smaller it is
        elif line_idx==23:
            new_line+=line.split(': ')[0]+": "+str(int(16384*168/params_input['PE_size']))+"\n"
        else:
            new_line=copy.deepcopy(line)
        new_file_content.append(new_line)
    reading_file.close()
    writing_file = open(hw_arch_config.split('/')[1], "w")
    writing_file.writelines(new_file_content)
    writing_file.close()  
    scp.put(hw_arch_config.split('/')[1],work_space+hw_arch_config)

    #modify loop ordering in constraint file
    scp.get(work_space+arch_constraints)
    quant_option=quant_list[0]
    reading_file = open(arch_constraints.split('/')[1], "r")
    new_file_content = []
    lines=reading_file.readlines()
    for line_idx, line in enumerate(lines):
        new_line=""
        #dummy buffer split + permutation
        if line_idx==28:
            new_line+=line.split(': ')[0]+": "+str(params_input["dummy_buffer_split"])+"\n"
        elif line_idx==29:
            tmp_order_str=""
            for i in range(0,7):
                tmp_order_str+=dnn_dimensions[params_input["dummy_buffer_order"][i]]
            tmp_order_str=insert_str2str(tmp_order_str," ",params_input["dummy_buffer_split"])
            new_line+=line.split(': ')[0]+": "+tmp_order_str+"\n"
        #modify shared_glb split
        elif line_idx==34:
            new_line+=line.split(': ')[0]+": "+str(params_input["shared_glb_split"])+"\n"
        #modify shared_glb spatial
        # elif line_idx==36:
            # tmp_order_str=""
            # for i in range(0,7):
                # tmp_order_str+=dnn_dimensions[params_input["shared_glb_order"][i]]
            # tmp_order_str=insert_str2str(tmp_order_str," ",params_input["shared_glb_split"])
            # new_line+=line.split(': ')[0]+": "+tmp_order_str+"\n"
        #modify psum_spad
        elif line_idx==40:
            tmp_order_str=""
            for i in range(0,7):
                tmp_order_str+=dnn_dimensions[params_input["psum_spad_order"][i]]
            new_line+=line.split(': ')[0]+": "+tmp_order_str+"\n"
        #modify weights_spad
        elif line_idx==45:
            tmp_order_str=""
            for i in range(0,7):
                tmp_order_str+=dnn_dimensions[params_input["weights_spad_order"][i]]
            new_line+=line.split(': ')[0]+": "+tmp_order_str+"\n"
        #modify ifmap_spad
        elif line_idx==49:
            tmp_order_str=""
            for i in range(0,7):
                tmp_order_str+=dnn_dimensions[params_input["ifmap_spad_order"][i]]
            new_line+=line.split(': ')[0]+": "+tmp_order_str+"\n"
        else:
            new_line=copy.deepcopy(line)
        new_file_content.append(new_line)
    reading_file.close()
    writing_file = open(arch_constraints.split('/')[1], "w")
    writing_file.writelines(new_file_content)
    writing_file.close()  

    scp.get(work_space+map_constraints)
    quant_option=quant_list[0]
    reading_file = open(map_constraints.split('/')[1], "r")
    new_file_content = []
    lines=reading_file.readlines()
    for line_idx, line in enumerate(lines):
        new_line=""
        if line_idx==11:
            tmp_order_str=""
            for i in range(0,7):
                tmp_order_str+=dnn_dimensions[params_input["dummy_buffer2_order"][i]]
            new_line+=line.split(': ')[0]+": "+tmp_order_str+"\n"
        elif line_idx==16:
            tmp_order_str=""
            for i in range(0,7):
                tmp_order_str+=dnn_dimensions[params_input["shared_glb_temporal_order"][i]]
            new_line+=line.split(': ')[0]+": "+tmp_order_str+"\n"
        elif line_idx==21:
            tmp_order_str=""
            for i in range(0,7):
                tmp_order_str+=dnn_dimensions[params_input["DRAM_order"][i]]
            new_line+=line.split(': ')[0]+": "+tmp_order_str+"\n"
        else:
            new_line=copy.deepcopy(line)
        new_file_content.append(new_line)
    reading_file.close()
    writing_file = open(map_constraints.split('/')[1], "w")
    writing_file.writelines(new_file_content)
    writing_file.close()  
    scp.put(arch_constraints.split('/')[1],work_space+arch_constraints)
    scp.put(map_constraints.split('/')[1],work_space+map_constraints)

    #reading area info
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("source /home/hy34/.bashrc;\
                                                          PATH=$PATH:"+cacti_path+";\
                                                          export PATH;\
                                                          source /home/hy34/accelergy_luke/timeloop/env/setup-env.bash;\
                                                          cd "+work_space+"; "+
                                                          accelergy_bin+" arch/ -o output")  
    exit_code = ssh_stdout.channel.recv_exit_status() 
    stdout = []
    for line in ssh_stdout:
        stdout.append(line.strip())

    stderr = []
    for line in ssh_stderr:
        stderr.append(line.strip())

    area_info={'PE':0, 'DummyBuffer':0,'ifmap_spad':0,'weights_spad':0, 'psum_spad':0,'shared_glb':0}
    scp.get(work_space+"output/ART.yaml")
    reading_file = open('ART.yaml', "r")
    new_file_content = []
    lines=reading_file.readlines()
    for i, line in enumerate(lines):
        if 'area:' in line:
            tmp_area=float(line.split(': ')[1])
            if 'PE' in lines[i-1] and 'mac' in lines[i-1]:
                area_info['PE']=max(area_info['PE'],tmp_area)*params_input["PE_size"]
            elif 'DummyBuffer' in lines[i-1]:
                area_info['DummyBuffer']=max(area_info['DummyBuffer'],tmp_area)
            elif 'ifmap_spad' in lines[i-1]:
                area_info['ifmap_spad']=max(area_info['ifmap_spad'],tmp_area)*params_input["PE_size"]
            elif 'weights_spad' in lines[i-1]:
                area_info['weights_spad']=max(area_info['weights_spad'],tmp_area)*params_input["PE_size"]
            elif 'psum_spad' in lines[i-1]:
                area_info['psum_spad']=max(area_info['psum_spad'],tmp_area)*params_input["PE_size"]      
            elif 'shared_glb' in lines[i-1]:
                area_info['shared_glb']=max(area_info['shared_glb'],tmp_area) 
    reading_file.close()  
    #update total area
    total_area=0
    for component in area_info.keys():
        total_area+=area_info[component]
    print('area breakdown', area_info)
    print('total area', total_area)

    #iterate over layers

    for i, layer_struct in enumerate(net_struct):
        print('optimizing: ', layer_struct)
        if i >3:
            break
        #modify and transfer the config file
        #fetch the files
        scp.get(work_space+network_config)
        scp.get(work_space+mapper_config)
        #modify net_struct
        reading_file = open(network_config.split('/')[1], "r")
        new_file_content = []
        lines=reading_file.readlines()
        dw_scaling=1
        for line in lines:
            new_line=""
            if "C:" in line:
                new_line+="    C: "+str(layer_struct[0])+"\n"
            elif "Hstride:" in line:
                new_line+="    Hstride: "+str(layer_struct[4])+"\n"
            elif "M:" in line:
                if layer_struct[0]==1:
                    dw_scaling=layer_struct[1]
                    new_line+="    M: 1\n"
                else:
                    new_line+="    M: "+str(layer_struct[1])+"\n"
            elif "P:" in line:
                new_line+="    P: "+str(layer_struct[2])+"\n"
            elif "Q:" in line:
                new_line+="    Q: "+str(layer_struct[2])+"\n"
            elif "R:" in line:
                new_line+="    R: "+str(layer_struct[3])+"\n"
            elif "S:" in line:
                new_line+="    S: "+str(layer_struct[3])+"\n"
            else:
                new_line=copy.deepcopy(line)
            new_file_content.append(new_line)
        reading_file.close()
        writing_file = open(network_config.split('/')[1], "w")
        writing_file.writelines(new_file_content)
        writing_file.close()
        
        scp.put(network_config.split('/')[1],work_space+network_config)
        scp.put(mapper_config.split('/')[1],work_space+mapper_config)
        print('finished modifying and transfering the file')


        #run timeloop remotely
        start_time=time.time()
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("source /home/hy34/.bashrc;\
                                                              source /home/hy34/accelergy_luke/timeloop/env/setup-env.bash;\
                                                              cd "+work_space+";\
                                                              accelergy arch/ -o output/;\
                                                              timeout -s SIGINT 60s /home/hy34/accelergy_luke/timeloop/build/timeloop-mapper "+network_config+" arch/components/*.yaml "+hw_arch_config+" constraints/*.yaml "+mapper_config+" > /dev/null 2>&1;\
                                                              cat "+work_space+"timeloop-mapper.stats.txt")    
        exit_code = ssh_stdout.channel.recv_exit_status() 
        stdout = []
        for line in ssh_stdout:
            stdout.append(line.strip())

        stderr = []
        for line in ssh_stderr:
            stderr.append(line.strip())


                    
        print('Evaluation & transmission takes: ', time.time() - start_time)
        print('Will have dw scaling: ', dw_scaling)
        for i, line in enumerate(stdout):
            if 'Cycles:' in line:
                cycles=int(re.findall(r'\d+',line)[0])*dw_scaling
            elif 'Energy:' in line:
                energy=float(re.findall(r'\d+\.\d+',line)[0])*dw_scaling


        print('cycles: ',cycles)
        print('energy: ', energy)

        
    # Clean up elements
    ssh.close()
    del ssh, ssh_stdin, ssh_stdout, ssh_stderr


capsulated_asic_model(block_info_test,quant_list,params_input,channel_part,cifar,edd)