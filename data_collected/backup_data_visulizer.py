import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import sys
from tqdm import tqdm
from tqdm._utils import _term_move_up
import matplotlib
def data_loader(fn):
    raw=np.load(fn,allow_pickle=True)
    return raw
def data_combiner():
    combined_data=[]
    for f in os.listdir("./"):
        if 'packed_data' in f:
            raw=data_loader(f)
            for i in raw:
                combined_data.append(i)
    return np.array(combined_data)



eval_mode=False
ckpt_path='tmp_weight.pth'
eval_data_path='layer3/layer3_results_pack_6_50.npy'
#needs to be changed
layer_structure=[256,384,8,3,1]
#layer_structure=[48,256,16,5,1]

writer=SummaryWriter("800_1e-3_0_99_80000_logging.log")


#raw_data=data_loader('combined_data.npy')
raw_data=data_combiner()
#print(len(raw_data))
#exit()
np.random.shuffle(raw_data)
train_data=raw_data[0:800,:]
train_x=np.array([np.array(i) for i in train_data[:,0]],dtype='float32')
max_train_x=np.amax(train_x[:,5:],axis=0)
min_train_x=np.amin(train_x[:,5:],axis=0)
#train_x[:,5:]=(train_x[:,5:]-min_train_x)/(max_train_x-min_train_x)
#train_x[:,0:5]=train_x[:,0:5]/512
train_x[:,5]=train_x[:,5]-1
train_y=np.array(train_data[:,1],dtype='float32').reshape(train_data.shape[0],1)
max_train_y=np.amax(train_y,axis=0)
min_train_y=np.amin(train_y,axis=0)
#train_y=(train_y-min_train_y)/(max_train_y-min_train_y)
test_data=raw_data[800:,:]
#print(len(train_data[:,0][0]))
#exit()
train_set=TensorDataset(torch.tensor(train_x),torch.tensor(train_y))
train_loader=DataLoader(train_set,batch_size=800,num_workers=29)


test_x=np.array([np.array(i) for i in test_data[:,0]],dtype='float32')
#test_x[:,5:]=(test_x[:,5:]-min_train_x)/(max_train_x-min_train_x)
#test_x[:,0:5]=test_x[:,0:5]/512
test_x[:,5]=test_x[:,5]-1
test_y=np.array(test_data[:,1],dtype='float32').reshape(test_data.shape[0],1)
#test_y=(test_y-min_train_y)/(max_train_y-min_train_y)
#print(test_y)
test_x=torch.tensor(test_x)
test_y=torch.tensor(test_y)

print(train_y.shape)

#exit()
for x_index in range(17): 
    fname1="input_output_vis"+str(x_index)+".png"
    fig, ax1 = plt.subplots()
    #ax1.scatter (list(range(train_y.shape[0])), train_y, 2.2)
    #ax1.scatter (list(range(train_x.shape[0])), train_x[:,x_index], 2.2)
    ax1.scatter (train_x[:,x_index],train_y,2.2)
    ax1.grid(True)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname1)







