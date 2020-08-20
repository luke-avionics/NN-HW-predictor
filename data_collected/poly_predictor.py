import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch
import sys 
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

#seperate computation mode? ...
class Net(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,layer_num,use_relu=True):
        super(Net, self).__init__()
        self.fc11 = nn.Linear(input_dim,hidden_dim)
        self.fc12 = nn.Linear(input_dim,hidden_dim)
        self.fc13 = nn.Linear(input_dim,hidden_dim)
        middle=[]
        for _ in range(0,layer_num):
            middle.append(nn.Linear(hidden_dim, hidden_dim))
            if use_relu:
                middle.append(nn.ReLU())
        #self.fc2=nn.Sequential(*middle)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim,bias=True)
        self.relu=nn.ReLU()
        
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_dim=hidden_dim
        self.layer_num=layer_num
        self.use_relu=use_relu

    def forward(self, x):
        x2=x*x;
        x3=x*x*x;
        x= self.fc2(self.fc11(x))+self.fc12(x2)+self.fc13(x3)
        if self.use_relu:
            x=self.relu(x)
        #x =self.fc2(x)
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

writer=SummaryWriter("logging2.log")


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
train_loader=DataLoader(train_set,batch_size=400,num_workers=29)


test_x=np.array([np.array(i) for i in test_data[:,0]],dtype='float32')
#test_x[:,5:]=(test_x[:,5:]-min_train_x)/(max_train_x-min_train_x)
#test_x[:,0:5]=test_x[:,0:5]/512
test_x[:,5]=test_x[:,5]-1
test_y=np.array(test_data[:,1],dtype='float32').reshape(test_data.shape[0],1)
#test_y=(test_y-min_train_y)/(max_train_y-min_train_y)
#print(test_y)
#test_x=torch.tensor(test_x)
#test_y=torch.tensor(test_y)



from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_Poly = poly_reg.fit_transform(train_x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_Poly, train_y)
pred_y=lin_reg_2.predict(poly_reg.fit_transform(test_x))
print(np.mean(np.abs(pred_y-test_y)/test_y))
exit()



net=Net(17,1,256,2)
#criterion=nn.KLDivLoss()
criterion=nn.MSELoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=3e-3, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,500,0.95)
train_loss_logged=0



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



for epoch in range(20000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1 == 0:    
            print('[%d, %5d] loss: %.3f ' %
                  (epoch + 1, i + 1, running_loss / 5))
            train_loss_logged=running_loss
            running_loss = 0.0
    print("="*15)
    #eval
    tested_outputs=net(test_x)
    print(torch.mean(torch.true_divide(torch.abs(tested_outputs-test_y),test_y)).item())
    test_loss=criterion(tested_outputs, test_y)
    print('test loss: %.3f' % (test_loss.item()))
    scheduler.step()
    writer.add_scalar('Loss/train', train_loss_logged, epoch)
    writer.add_scalar('Loss/test',  test_loss.item(), epoch)
    writer.add_scalar('Accuracy/test', torch.mean(torch.true_divide(torch.abs(tested_outputs-test_y),test_y)).item(), epoch)
    if epoch % 20==19:
        torch.save(net.state_dict(), 'tmp_weight.pth')
print('Finished Training')

