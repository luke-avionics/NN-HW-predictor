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
#evaluation mode has not been modified



layer="234"
initial_lr=500
lr_decay=0.991
lr_step=100
weight_decay=1e-3
total_epochs=120000
train_data_ratio=0.8
data_normalize=True
EFF_hidden_layer=[24]
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


def sample_part(samples):
    #separate the input into:
    #                        Nmac, pf, pf_related_param, Nipt, Nopt, Nwht
    Nmac=[] 
    pf=[]
    pf_related_param=[]
    Nipt=[]
    Nopt=[]
    Nwht=[]
    for i in range(samples.shape[0]):
        #Num of MACs
        Nmac.append(samples[i,0]*samples[i,1]*samples[i,2]*samples[i,2]*samples[i,3]*samples[i,3])
        #potential parallel factors num
        if samples[i,10]==1:
            tr=samples[i,5]
        else:
            tr=1
        if samples[i,11]==1:
            tc=samples[i,6]
        else:
            tc=1
        if samples[i,12]==1:
            tm=samples[i,7]
        else:
            tm=1
        if samples[i,13]==1:
            tn=samples[i,8]
        else:
            tn=1
        if samples[i,15]==1:
            tk1=samples[i,3]
        else:
            tk1=1
        if samples[i,16]==1:
            tk2=samples[i,3]
        else:
            tk2=1
        pf.append(tr*tc*tm*tn*tk1*tk2)
        #parameters will influence the pf efficiency comp mode, kernel mem type
        #to do: one hot comp mode
        #       outer loop for Nipt as multiplier
        pf_related_param.append([samples[i,9],samples[i,14]])
        Nipt.append((samples[i,4]*samples[i,5]+samples[i,3]-1)*(samples[i,4]*samples[i,6]+samples[i,3]-1)*samples[i,8])
        Nopt.append(samples[i,5]*samples[i,6]*samples[i,7])
        Nwht.append(samples[i,3]*samples[i,3]*samples[i,7]*samples[i,8])
        
    return {"Nmac": np.asarray(Nmac,dtype="float").reshape(len(Nmac),1), "pf": np.asarray(pf,dtype="float").reshape(len(pf),1), "pf_related_param": np.asarray(pf_related_param,dtype="float"), "Nipt": np.asarray(Nipt,dtype="float").reshape(len(Nipt),1), "Nopt":np.asarray(Nopt,dtype="float").reshape(len(Nopt),1), "Nwht": np.asarray(Nwht,dtype="float").reshape(len(Nwht),1)  }


#seperate computation mode? ...
class Net(nn.Module):
    def __init__(self,EFF_hidden_layer=[],use_relu=True):
        super(Net, self).__init__()
        self.tcomp=nn.Parameter(torch.randn(1,1))
        self.tcomp.requires_grad = True
        self.correction= nn.Parameter(torch.randn(1,1))
        self.correction.requires_grad = True
        self.b=nn.Parameter(torch.randn(1,1))
        self.b.requires_grad = True
        self.c=nn.Parameter(torch.randn(1,1))
        self.c.requires_grad = True
        self.d=nn.Parameter(torch.randn(1,1))
        self.d.requires_grad = True
        self.relu=nn.ReLU()
        if len(EFF_hidden_layer)==0:
            self.fc1 = nn.Linear(2,1)
        else:
            middle=[]
            middle.append(nn.Linear(2,EFF_hidden_layer[0]))
            if use_relu:
                middle.append(nn.ReLU())
            lnum=0
            for i in EFF_hidden_layer:
                if lnum == len(EFF_hidden_layer)-1:
                    middle.append(nn.Linear(i, 1))
                else:
                    middle.append(nn.Linear(i, EFF_hidden_layer[lnum+1]))
                    if use_relu:
                        middle.append(nn.ReLU())
                lnum+=1
            self.fc1=nn.Sequential(*middle)
        self.use_relu=use_relu
    def forward(self, Nmac, pf, pf_related_param, Nipt, Nopt, Nwht):
        x= Nmac*self.tcomp/pf
        #x=torch.transpose(x,0,1)
        x=x*pf_related_param
        x=self.fc1(x)
        x=self.relu(x)
        x=x+Nipt*self.b+Nopt*self.c+Nwht*self.d+self.correction
        x=self.relu(x)
        return x


eval_mode=False
ckpt_path='tmp_weight.pth'
eval_data_path='layer3/layer3_results_pack_6_50.npy'
#needs to be changed
layer_structure=[256,384,8,3,1]
#layer_structure=[48,256,16,5,1]

writer=SummaryWriter("analytical_version2_EFF_hidden_layer_"+str(EFF_hidden_layer)+"_layer"+str(layer)+"_"+"{:.0e}".format(weight_decay)+"_"+str(lr_decay).replace(".","_")+"initial_lr_"+str(initial_lr)+"_lr_step"+str(lr_step)+"_"+str(total_epochs)+"_logging.log")


#raw_data=data_loader('combined_data.npy')
raw_data=data_combiner()
#print(len(raw_data))
#exit()
np.random.shuffle(raw_data)
td_sz= int(train_data_ratio*len(raw_data))
train_data=raw_data[0:td_sz,:]
test_data=raw_data[td_sz:,:]
train_x=np.array([np.array(i) for i in train_data[:,0]],dtype='float32')
train_y=np.array(train_data[:,1],dtype='float32').reshape(train_data.shape[0],1)
test_x=np.array([np.array(i) for i in test_data[:,0]],dtype='float32')
test_y=np.array(test_data[:,1],dtype='float32').reshape(test_data.shape[0],1)

#dict containing partitioned data
train_x_dict=sample_part(train_x)
test_x_dict=sample_part(test_x)


print(test_x_dict['pf_related_param'])
print(np.mean(test_x_dict['Nmac'],axis=0), end=';  ')
print(np.mean(train_x_dict['Nmac'],axis=0))
print(np.std(test_x_dict['Nmac'],axis=0), end=';   ')
print(np.std(train_x_dict['Nmac'],axis=0))

print("==="*10)

print(np.mean(test_x_dict['pf'],axis=0), end=';  ')
print(np.mean(train_x_dict['pf'],axis=0))
print(np.std(test_x_dict['pf'],axis=0), end=';   ')
print(np.std(train_x_dict['pf'],axis=0))

print("==="*10)


print(np.mean(test_x_dict['pf_related_param'],axis=0), end=';  ')
print(np.mean(train_x_dict['pf_related_param'],axis=0))
print(np.std(test_x_dict['pf_related_param'],axis=0), end=';   ')
print(np.std(train_x_dict['pf_related_param'],axis=0))

print("==="*10)


print(np.mean(test_x_dict['Nipt'],axis=0), end=';  ')
print(np.mean(train_x_dict['Nipt'],axis=0))
print(np.std(test_x_dict['Nipt'],axis=0), end=';   ')
print(np.std(train_x_dict['Nipt'],axis=0))

print("==="*10)


print(np.mean(test_x_dict['Nopt'],axis=0), end=';  ')
print(np.mean(train_x_dict['Nopt'],axis=0))
print(np.std(test_x_dict['Nopt'],axis=0), end=';   ')
print(np.std(train_x_dict['Nopt'],axis=0))

print("==="*10)


print(np.mean(test_x_dict['Nwht'],axis=0), end=';  ')
print(np.mean(train_x_dict['Nwht'],axis=0))
print(np.std(test_x_dict['Nwht'],axis=0), end=';   ')
print(np.std(train_x_dict['Nwht'],axis=0))


print("==="*10)


print(np.mean(test_y,axis=0), end=';  ')
print(np.mean(train_y,axis=0))
print(np.std(test_y,axis=0), end=';   ')
print(np.std(train_y,axis=0))




#calculate data stat for normalization
print(train_x_dict["Nipt"])
mean_train_x_dict={}
std_train_x_dict={}
mean_train_x_dict["Nmac"]=np.mean(train_x_dict["Nmac"],axis=0)
mean_train_x_dict["pf"]=np.mean(train_x_dict["pf"],axis=0)
mean_train_x_dict["pf_related_param"]=np.mean(train_x_dict["pf_related_param"],axis=0)
mean_train_x_dict["Nipt"]=np.mean(train_x_dict["Nipt"],axis=0)
mean_train_x_dict["Nopt"]=np.mean(train_x_dict["Nopt"],axis=0)
mean_train_x_dict["Nwht"]=np.mean(train_x_dict["Nwht"],axis=0)

std_train_x_dict["Nmac"]=np.std(train_x_dict["Nmac"],axis=0)
std_train_x_dict["pf"]=np.std(train_x_dict["pf"],axis=0)
std_train_x_dict["pf_related_param"]=np.std(train_x_dict["pf_related_param"],axis=0)
std_train_x_dict["Nipt"]=np.std(train_x_dict["Nipt"],axis=0)
std_train_x_dict["Nopt"]=np.std(train_x_dict["Nopt"],axis=0)
std_train_x_dict["Nwht"]=np.std(train_x_dict["Nwht"],axis=0)

##data normalization
if data_normalize:
    train_x_dict["Nmac"]=(train_x_dict["Nmac"]-mean_train_x_dict["Nmac"])/std_train_x_dict["Nmac"]
    train_x_dict["pf"]=(train_x_dict["pf"]-mean_train_x_dict["pf"])/std_train_x_dict["pf"]
    train_x_dict["pf_related_param"]=(train_x_dict["pf_related_param"]-mean_train_x_dict["pf_related_param"])/std_train_x_dict["pf_related_param"]
    train_x_dict["Nipt"]=(train_x_dict["Nipt"]-mean_train_x_dict["Nipt"])/std_train_x_dict["Nipt"]
    train_x_dict["Nopt"]=(train_x_dict["Nopt"]-mean_train_x_dict["Nopt"])/std_train_x_dict["Nopt"]
    train_x_dict["Nwht"]=(train_x_dict["Nwht"]-mean_train_x_dict["Nwht"])/std_train_x_dict["Nwht"]

#train_x[:,5:]=(train_x[:,5:]-min_train_x)/(max_train_x-min_train_x)
#train_x[:,0:5]=train_x[:,0:5]/512
#train_x[:,5]=train_x[:,5]-1

max_train_y=np.amax(train_y,axis=0)
min_train_y=np.amin(train_y,axis=0)
mean_train_y=np.mean(train_y,axis=0)
std_train_y=np.std(train_y,axis=0)
if data_normalize:
    train_y=(train_y-mean_train_y)/std_train_y
#train_y=(train_y-min_train_y)/(max_train_y-min_train_y)



if data_normalize:
    test_x_dict["Nmac"]=(test_x_dict["Nmac"]-mean_train_x_dict["Nmac"])/std_train_x_dict["Nmac"]
    test_x_dict["pf"]=(test_x_dict["pf"]-mean_train_x_dict["pf"])/std_train_x_dict["pf"]
    test_x_dict["pf_related_param"]=(test_x_dict["pf_related_param"]-mean_train_x_dict["pf_related_param"])/std_train_x_dict["pf_related_param"]
    test_x_dict["Nipt"]=(test_x_dict["Nipt"]-mean_train_x_dict["Nipt"])/std_train_x_dict["Nipt"]
    test_x_dict["Nopt"]=(test_x_dict["Nopt"]-mean_train_x_dict["Nopt"])/std_train_x_dict["Nopt"]
    test_x_dict["Nwht"]=(test_x_dict["Nwht"]-mean_train_x_dict["Nwht"])/std_train_x_dict["Nwht"]
    test_y=(test_y-mean_train_y)/std_train_y

#test_y=(test_y-min_train_y)/(max_train_y-min_train_y)
#print(test_y)
#exit()



train_set=TensorDataset(torch.tensor(train_x_dict["Nmac"]),\
                        torch.tensor(train_x_dict["pf"]),\
                        torch.tensor(train_x_dict["pf_related_param"]),\
                        torch.tensor(train_x_dict["Nipt"]),\
                        torch.tensor(train_x_dict["Nopt"]),\
                        torch.tensor(train_x_dict["Nwht"]),\
                        torch.tensor(train_y))
train_loader=DataLoader(train_set,batch_size=td_sz,num_workers=29)
test_x_dict["Nmac"]=torch.tensor(test_x_dict["Nmac"])
test_x_dict["pf"]=torch.tensor(test_x_dict["pf"])
test_x_dict["pf_related_param"]=torch.tensor(test_x_dict["pf_related_param"])
test_x_dict["Nipt"]=torch.tensor(test_x_dict["Nipt"])
test_x_dict["Nopt"]=torch.tensor(test_x_dict["Nopt"])
test_x_dict["Nwht"]=torch.tensor(test_x_dict["Nwht"])
test_y=torch.tensor(test_y)
net=Net(EFF_hidden_layer).double()
for name, param in net.named_parameters():
    if param.requires_grad:
        print(name,end=";  ")
        print(param.type())
net = torch.nn.DataParallel(net).cuda()

#criterion=nn.KLDivLoss()criterion=nn.L1Loss()
criterion=nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4,weight_decay=weight_decay)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=3e-3, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,lr_step,lr_decay)
train_loss_logged=0


test_x_dict["Nmac"] = Variable(test_x_dict["Nmac"], volatile=True).cuda()
test_x_dict["pf"] = Variable(test_x_dict["pf"], volatile=True).cuda()
test_x_dict["pf_related_param"] = Variable(test_x_dict["pf_related_param"], volatile=True).cuda()
test_x_dict["Nipt"] = Variable(test_x_dict["Nipt"], volatile=True).cuda()
test_x_dict["Nopt"] = Variable(test_x_dict["Nopt"], volatile=True).cuda()
test_x_dict["Nwht"] = Variable(test_x_dict["Nwht"], volatile=True).cuda()
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
        Nmac,pf,pf_related_param,Nipt,Nopt,Nwht, labels = data
        Nmac = Variable(Nmac, volatile=True).cuda()
        pf = Variable(pf, volatile=True).cuda()
        pf_related_param = Variable(pf_related_param, volatile=True).cuda()
        Nipt = Variable(Nipt, volatile=True).cuda()
        Nopt = Variable(Nopt, volatile=True).cuda()
        Nwht = Variable(Nwht, volatile=True).cuda()
        labels = Variable(labels, volatile=True).double().cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(Nmac,pf,pf_related_param,Nipt,Nopt,Nwht)
        #loss = criterion(outputs, labels)
        loss=error_loss(outputs, labels, mean_train_y[0],std_train_y[0])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1 == 0:    
            pbar.write('[%d, %5d] loss: %.3f ' %
                  (epoch + 1, i + 1, running_loss / 0))
            train_loss_logged=running_loss
            running_loss = 0.0
            if data_normalize:
                pbar.write('train_acc1: '+str(torch.mean(torch.true_divide(torch.abs((outputs*std_train_y[0]+mean_train_y[0])-(labels*std_train_y[0]+mean_train_y[0])),(labels*std_train_y[0]+mean_train_y[0]))).item()))
                pbar.write('train_acc2: '+ str(torch.mean(torch.true_divide(torch.abs(outputs-labels),labels)).item()))
            else:
                pbar.write('train_acc1: '+ str(torch.mean(torch.true_divide(torch.abs(outputs-labels),labels)).item()))

    pbar.write("="*15)
    #eval
    tested_outputs=net(test_x_dict["Nmac"],\
                       test_x_dict["pf"],\
                       test_x_dict["pf_related_param"],\
                       test_x_dict["Nipt"],\
                       test_x_dict["Nopt"],\
                       test_x_dict["Nwht"])
    #for ite in range(85):
    #    pbar.write(str((torch.true_divide(torch.abs(tested_outputs-test_y),test_y)[ite]))+" ;;;  "+str(test_x[ite]))
    #pbar.write(str((tested_outputs*std_train_y[0]+mean_train_y[0]).data))
    for i in range(100):
        print((tested_outputs*std_train_y[0]+mean_train_y[0])[i].data,(test_y*std_train_y[0]+mean_train_y[0])[i].data)
    
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
        torch.save(net.state_dict(), "analytical_version2_EFF_hidden_layer_"+str(EFF_hidden_layer)+"_layer"+str(layer)+"_"+"{:.0e}".format(weight_decay)+"_"+str(lr_decay).replace(".","_")+"initial_lr_"+str(initial_lr)+"_lr_step"+str(lr_step)+"_"+str(total_epochs)+"_tmp_weight.pth")
    pbar.update()
print('Finished Training')

