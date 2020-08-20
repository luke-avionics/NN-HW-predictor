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


layer="234"
initial_lr=1e-4
lr_decay=0.95
lr_step=100
weight_decay=1e-2
total_epochs=120000
train_data_ratio=0.8
data_normalize=True
net_arch=[17,1,1024,3]
drop_out=0
bs=1
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
    loss=torch.mean(torch.true_divide(torch.abs((outputs*std_y+mean_y)-(labels*std_y+mean_y)),(labels*std_y+mean_y)))
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
train_loader=DataLoader(train_set,batch_size=int(td_sz*bs),num_workers=29)


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

if data_normalize:
    test_x=(test_x-mean_train_x)/std_train_x
    test_x[:,4].fill(1)
    test_y=(test_y-mean_train_y)/std_train_y


#test_y=(test_y-min_train_y)/(max_train_y-min_train_y)
#print(test_y)
#exit()
test_x=torch.tensor(test_x)
test_y=torch.tensor(test_y)

writer=SummaryWriter("error_loss_net"+str(net_arch)+"drop_out"+str(drop_out)+"_layer"+str(layer)+"_"+"{:.0e}".format(weight_decay)+"_"+str(lr_decay).replace(".","_")+"initial_lr_"+str(initial_lr)+"_lr_step"+str(lr_step)+"_"+str(total_epochs)+"_logging.log")

net=Net(net_arch[0],net_arch[1],net_arch[2],net_arch[3],drop_out=drop_out)
net = torch.nn.DataParallel(net).cuda()

#criterion=nn.KLDivLoss()criterion=nn.L1Loss()
criterion=nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr,weight_decay=weight_decay)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=3e-3, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,lr_step, eta_min=1e-7) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,lr_step,lr_decay)
train_loss_logged=0


test_x = Variable(test_x, volatile=True).cuda()
test_y = Variable(test_y, volatile=True).cuda()




if eval_mode:
    eval_data=[]
    raw_eval=np.load(eval_data_path,allow_pickle=True)
    for dp in raw_eval:
        eval_data.append([layer_structure+dp[0][0]+[dp[0][1]],dp[1][0]])
    eval_data=np.array(eval_data)
    eval_inputs=torch.tensor(np.array([np.array(i) for i in eval_data[:,0]],dtype='float32'))
    eval_targets=torch.tensor(np.array(eval_data[:,1],dtype='float32').reshape(eval_data.shape[0],1))
    net.load_state_dict(torch.load(ckpt_path))
    net.eval()
    eval_outputs=net(eval_inputs)
    print(eval_outputs.data.numpy())
    print(eval_targets)
    print(torch.mean(torch.true_divide(torch.abs(eval_outputs-eval_targets),eval_targets)).item())
    exit()


pbar = tqdm(range(total_epochs))
border = "="*50
clear_border = _term_move_up() + "\r" + " "*len(border) + "\r"
for epoch in pbar:  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        #loss = criterion(outputs, labels)
        loss=error_loss(outputs, labels, mean_train_y[0],std_train_y[0])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1 == 0:    
            pbar.write('[%d, %5d] loss: %.3f ' %
                  (epoch + 1, i + 1, running_loss / 1))
            train_loss_logged=running_loss
            running_loss = 0.0
            if data_normalize:
                pbar.write('train_acc1: '+str(torch.mean(torch.true_divide(torch.abs((outputs*std_train_y[0]+mean_train_y[0])-(labels*std_train_y[0]+mean_train_y[0])),(labels*std_train_y[0]+mean_train_y[0]))).item()))
                pbar.write('train_acc2: '+ str(torch.mean(torch.true_divide(torch.abs(outputs-labels),labels)).item()))
            else:
                pbar.write('train_acc1: '+ str(torch.mean(torch.true_divide(torch.abs(outputs-labels),labels)).item()))

    pbar.write("="*15)
    #eval
    tested_outputs=net(test_x)
    #for ite in range(85):
    #    pbar.write(str((torch.true_divide(torch.abs(tested_outputs-test_y),test_y)[ite]))+" ;;;  "+str(test_x[ite]))
    #pbar.write(str((tested_outputs*std_train_y[0]+mean_train_y[0]).data))
    if data_normalize:
        pbar.write(str(torch.mean(torch.true_divide(torch.abs((tested_outputs*std_train_y[0]+mean_train_y[0])-(test_y*std_train_y[0]+mean_train_y[0])),(test_y*std_train_y[0]+mean_train_y[0]))).item()))
        pbar.write(str(torch.mean(torch.true_divide(torch.abs(tested_outputs-test_y),test_y)).item()))
    else:
    	pbar.write(str(torch.mean(torch.true_divide(torch.abs(tested_outputs-test_y),test_y)).item()))
    test_loss=criterion(tested_outputs, test_y)
    pbar.write('test loss: %.3f' % (test_loss.item()))
    scheduler.step()
    writer.add_scalar('Loss/train', train_loss_logged, epoch)
    writer.add_scalar('Loss/test',  test_loss.item(), epoch)
    if data_normalize:
        writer.add_scalar('Accuracy/test',torch.mean(torch.true_divide(torch.abs((tested_outputs*std_train_y[0]+mean_train_y[0])-(test_y*std_train_y[0]+mean_train_y[0])),(test_y*std_train_y[0]+mean_train_y[0]))).item(), epoch)
    else:
        writer.add_scalar('Accuracy/test', torch.mean(torch.true_divide(torch.abs(tested_outputs-test_y),test_y)).item(), epoch)
    if epoch % 20==19:
        torch.save(net.state_dict(), "error_loss_net"+str(net_arch)+"drop_out"+str(drop_out)+"_layer"+str(layer)+"_"+"{:.0e}".format(weight_decay)+"_"+str(lr_decay).replace(".","_")+"initial_lr_"+str(initial_lr)+"_lr_step"+str(lr_step)+"_"+str(total_epochs)+"_tmp_weight.pth")
    pbar.update()
print('Finished Training')

