from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import torch
import sys 
from tqdm import tqdm
from tqdm._utils import _term_move_up
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


layer="234"
initial_lr=1e-3
lr_decay=0.95
lr_step=40
weight_decay=1e-1
total_epochs=120000
train_data_ratio=0.8
data_normalize=True
net_arch=[17,1,1024,3]
drop_out=0
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

def error_loss(outputs, labels, mean_y,std_y):
    loss=torch.mean(torch.div(torch.abs((outputs*std_y+mean_y)-(labels*std_y+mean_y)),(labels*std_y+mean_y)))
    return loss 


#seperate computation mode? ...
class Net(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,layer_num,use_relu=True,drop_out=0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        middle=[]
        for _ in range(0,layer_num):
            middle.append(nn.Linear(hidden_dim, hidden_dim))
            if use_relu:
                middle.append(nn.ReLU())
            if drop_out!=0:
                middle.append(nn.Dropout(p=drop_out))
        self.fc2=nn.Sequential(*middle)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim,bias=True)
        self.relu=nn.ReLU()
        
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_dim=hidden_dim
        self.layer_num=layer_num
        self.use_relu=use_relu
    def forward(self, x):
        x =self.fc1(x)
        if self.use_relu:
            x=self.relu(x)
        x =self.fc2(x)
        #for _ in range(0,self.layer_num):
        #    x=self.fc2(x)
        #    if self.use_relu:
        #        x=self.relu(x)
        x = self.fc3(x)
        return x


#    def __init__(self,input_dim,output_dim,hidden_dim,layer_num,use_relu=True):
#        super(Net, self).__init__()
#        self.fc1 = nn.Linear(input_dim,hidden_dim)
#        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#        self.fc3 = nn.Linear(hidden_dim, output_dim)
#        self.relu=nn.ReLU()
#        
#        self.input_dim=input_dim
#        self.output_dim=output_dim
#        self.hidden_dim=hidden_dim
#        self.layer_num=layer_num
#        self.use_relu=use_relu
#
#    def forward(self, x):
#        x =self.fc1(x)
#        if self.use_relu:
#            x=self.relu(x)
#        for _ in range(0,self.layer_num):
#            x=self.fc2(x)
#            if self.use_relu:
#                x=self.relu(x)
#        x = self.fc3(x)
#        return x



eval_mode=False
ckpt_path='tmp_weight.pth'
eval_data_path='layer3/layer3_results_pack_6_50.npy'
#needs to be changed
layer_structure=[256,384,8,3,1]
#layer_structure=[48,256,16,5,1]

ites=100
for  ite in range(ites):

    #raw_data=data_loader('combined_data.npy')
    raw_data=data_combiner()
    #print(len(raw_data))
    #exit()
    np.random.shuffle(raw_data)
    td_sz= int(train_data_ratio*len(raw_data))
    train_data=raw_data[0:td_sz,:]
    train_x=np.array([np.array(i) for i in train_data[:,0]],dtype='float32')
    max_train_x=np.amax(train_x[:,:],axis=0)
    min_train_x=np.amin(train_x[:,:],axis=0)
    mean_train_x=np.mean(train_x[:,:],axis=0)
    std_train_x=np.std(train_x[:,:],axis=0)
    if data_normalize:
        train_x=(train_x-mean_train_x)/std_train_x
        train_x[:,4].fill(1)

    #train_x[:,5:]=(train_x[:,5:]-min_train_x)/(max_train_x-min_train_x)
    #train_x[:,0:5]=train_x[:,0:5]/512
    #train_x[:,5]=train_x[:,5]-1

    train_y=np.array(train_data[:,1],dtype='float32').reshape(train_data.shape[0],1)
    max_train_y=np.amax(train_y,axis=0)
    min_train_y=np.amin(train_y,axis=0)
    mean_train_y=np.mean(train_y,axis=0)
    std_train_y=np.std(train_y,axis=0)
    if data_normalize:
        train_y=(train_y-mean_train_y)/std_train_y
    #train_y=(train_y-min_train_y)/(max_train_y-min_train_y)
    test_data=raw_data[td_sz:,:]
    #print(len(train_data[:,0][0]))
    #exit()
    train_set=TensorDataset(torch.tensor(train_x),torch.tensor(train_y))
    train_loader=DataLoader(train_set,batch_size=td_sz,num_workers=29)


    test_x=np.array([np.array(i) for i in test_data[:,0]],dtype='float32')
    #test_x[:,5:]=(test_x[:,5:]-min_train_x)/(max_train_x-min_train_x)
    #test_x[:,0:5]=test_x[:,0:5]/512
    test_y=np.array(test_data[:,1],dtype='float32').reshape(test_data.shape[0],1)

    print("x_mean_comparison")
    print(mean_train_x)
    print(np.mean(test_x,axis=0))
    print("=="*15)
    print("y_mean_comparison")
    print(mean_train_y)
    print(np.mean(test_y,axis=0))
    print("=="*15)
    print("x_std_comparison")
    print(std_train_x)
    print(np.std(test_x,axis=0))
    print("=="*15)
    print("y_std_comparison")
    print(std_train_y)
    print(np.std(test_y,axis=0))
    print("=="*15)
    if ite==0:
        x_mean_diff=np.abs((mean_train_x-np.mean(test_x,axis=0))/mean_train_x)
        x_std_diff=np.abs((std_train_x-np.std(test_x,axis=0))/std_train_x)
        y_mean_diff=np.abs((mean_train_y-np.mean(test_y,axis=0))/mean_train_y)
        y_std_diff=np.abs((std_train_y-np.std(test_y,axis=0))/std_train_y)
    else:
        x_mean_diff+= np.abs((mean_train_x - np.mean(test_x, axis=0)) / mean_train_x)
        x_std_diff+= np.abs((std_train_x - np.std(test_x, axis=0)) / std_train_x)
        y_mean_diff+=np.abs((mean_train_y-np.mean(test_y,axis=0))/mean_train_y)
        y_std_diff+=np.abs((std_train_y-np.std(test_y,axis=0))/std_train_y)

    if data_normalize:
        test_x=(test_x-mean_train_x)/std_train_x
        test_x[:,4].fill(1)
        test_y=(test_y-mean_train_y)/std_train_y

x_mean_diff=x_mean_diff/ites
x_std_diff=x_std_diff/ites
y_mean_diff=y_mean_diff/ites
y_std_diff=y_std_diff/ites

print(x_mean_diff)
print(x_std_diff)
print(y_mean_diff)
print(y_std_diff)

pca = PCA(n_components=1)
pca.fit(train_x)
reduced_train_x=pca.transform(train_x)
reduced_test_x=pca.transform(test_x)
print(pca.singular_values_)
print("=="*15)
print(np.mean(reduced_train_x,axis=0))
print(np.mean(reduced_test_x,axis=0))
print("=="*15)
print(np.std(reduced_train_x,axis=0))
print(np.std(reduced_test_x,axis=0))
#fname1 = "input_output_vis_pca.png"
#fig, ax1 = plt.subplots()
#ax1 = fig.add_subplot(111, projection='3d')
#ax1.scatter(reduced_train_x[:,0],reduced_train_x[:,1] , train_y)


fname2 = "singular_val.png"
fig, ax2 = plt.subplots()
ax2.plot(list(range(len(pca.singular_values_))),pca.singular_values_)
plt.savefig(fname2)
#plt.show()
