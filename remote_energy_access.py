import paramiko
import xml.etree.ElementTree as ET
import re
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.connect('10.129.1.80', username='hy34', password='youhaoran')
print('connection established')
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('source /home/hy34/accelergy_luke/timeloop/env/setup-env.bash;\
                                                      cd /home/hy34/accelergy_luke/timeloop-accelergy-exercises/exercises/timeloop/06-mapper-convlayer-eyeriss;\
                                                      /home/hy34/accelergy_luke/timeloop/build/timeloop-mapper prob/VGG02_layer5.yaml arch/components/*.yaml arch/eyeriss_like.yaml constraints/*.yaml mapper/mapper.yaml > /dev/null 2>&1;\
                                                      cat /home/hy34/accelergy_luke/timeloop-accelergy-exercises/exercises/timeloop/06-mapper-convlayer-eyeriss/timeloop-mapper.stats.txt')
                                                      
exit_code = ssh_stdout.channel.recv_exit_status() 

stdout = []
for line in ssh_stdout:
    stdout.append(line.strip())

stderr = []
for line in ssh_stderr:
    stderr.append(line.strip())

# Clean up elements
ssh.close()
del ssh, ssh_stdin, ssh_stdout, ssh_stderr
print(stdout)
print(stderr)

for i, line in enumerate(stdout):
    if 'Cycles:' in line:
        cycles=int(re.findall(r'\d+',line)[0])
    elif 'Energy:' in line:
        energy=float(re.findall(r'\d+\.\d+',line)[0])


print('cycles: ',cycles)
print('energy: ', energy)

