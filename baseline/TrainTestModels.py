# %%
import torch
import torchvision
from torchvision.transforms import Resize
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import json
import os
from os.path import join
import sys
from collections import defaultdict
import random
import pandas as pd
import string
import timeit
import wandb
from collections import OrderedDict
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import json
import os
from os.path import join
import sys
from collections import defaultdict
import random
import pandas as pd
import string
import timeit

# %%
class Utility():
    num_classes = 2
    device = torch.device("cuda")
    seed=42
    epochs=10
    batch_size=8
    train_env=['1-Back Rooms', '3-Theatre', '2-Loft',"4-Lincoln's Inn Drawing Room",'5-Church Of St Peter Stourton', '9-Modern_House',
               '10-the-dive-shop-matterport','11-san-simeon-a6-model-2','14-japan_house','15-vr-store']
    val_env=['6-Industry','13-impact-hub-startup-accelerator-trieste-italy','12-a-house-my-mate-built']
    test_env=['8-DMI','7-Mirabilia']



# %%
def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec
class BlendedDataset(Dataset):
    def __init__(self,pandas_dataset,num_classes,transform=None) -> None:
        super().__init__()
        self.pandas_dataset=pandas_dataset
        self.num_classes=num_classes
        self.transform=transform
    def __len__(self):
        return len(self.pandas_dataset)
    def __getitem__(self, index) -> any:
        rgb_path=self.pandas_dataset.loc[index,"RGB"]
        floor_path=self.pandas_dataset.loc[index,"Floor"]
        rgb_im=cv2.imread(rgb_path)
        floor_im=cv2.imread(floor_path)
        h,w,c=floor_im.shape
        
        for y in range(h):
            for x in range(w):
                x=floor_im[y,x]
                x[0]=0
                x[1]=0
                if x[2]>0:
                    x[2]=200
        image=cv2.addWeighted(rgb_im, 1, floor_im, 1, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = image[:,:,:3]
        image = Image.fromarray(image)
        
        
        labels=self.pandas_dataset.loc[index,"class"]
        target = onehot(Utility.num_classes, labels)
        #print("label tensor: ",target)
        if self.transform:
            image = self.transform(image)
        return {'image': image,
                'labels': target
                }
    
class BinarySegmentationDataset(Dataset):
    def __init__(self,pandas_dataset,num_classes,transform=None) -> None:
        super().__init__()
        self.pandas_dataset=pandas_dataset
        self.num_classes=num_classes
        self.transform=transform
    def __len__(self):
        return len(self.pandas_dataset)
    def __getitem__(self, index) -> any:
        rgb_path=self.pandas_dataset.loc[index,"RGB"]
        floor_path=self.pandas_dataset.loc[index,"Floor"]
        rgb_im=cv2.imread(rgb_path)
        floor_im=cv2.imread(floor_path,cv2.IMREAD_GRAYSCALE)
        h,w=floor_im.shape
        
        for y in range(h):
            for x in range(w):
                if floor_im[y,x]>0:
                    floor_im[y,x]=255
                    
        image = cv2.merge((rgb_im,floor_im))
        image = Image.fromarray(image)
        labels=self.pandas_dataset.loc[index,"class"]
        target = onehot(2, labels)
        
        if self.transform:
            image = self.transform(image)
        return {'image': image,
                'labels': target
                }

class RGBDataset(Dataset):
    def __init__(self,pandas_dataset,num_classes,transform=None) -> None:
        super().__init__()
        self.pandas_dataset=pandas_dataset
        self.num_classes=num_classes
        self.transform=transform
        self.resize=Resize((224, 224))
    def __len__(self):
        return len(self.pandas_dataset)
    def __getitem__(self, index) -> any:
        rgb_path=self.pandas_dataset.loc[index,"RGB"]
       
        image=cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = image[:,:,:3]
        image = Image.fromarray(image)
        image=self.resize(image)
        labels=self.pandas_dataset.loc[index,"class"]
        target = onehot(Utility.num_classes, labels)
        
        if self.transform:
            image = self.transform(image)
        return {'image': image,
                'labels': target
                }


# %%
class AverageValueMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0
        
    def add(self, value, num):
        self.sum += value*num
        self.num += num
        
    def value(self):
        try:
            return self.sum/self.num
        except:
            return None    

# %%
def train_classifier(model, train_loader, validation_loader, exp_name='experiment', lr=0.01, epochs=10, momentum=0.99, logdir='logs'):
    criterion = nn.CrossEntropyLoss() 
    optimizer = SGD(model.parameters(), lr, momentum=momentum) 
    #optimizer= torch.optim.Adam(model.parameters(),lr,(0.9,0.999))
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    precision_meter = AverageValueMeter()
    f1_meter = AverageValueMeter()
    recall_meter = AverageValueMeter()

    writer = SummaryWriter(join(logdir, exp_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    loader = {
    'train' : train_loader,
    'validation' : validation_loader
    }


    global_step = 0
    for e in range(epochs):
        print("epoch: ",e,end='\n')
        

        for mode in ['train','validation']:
            loss_meter.reset()
            acc_meter.reset()
            precision_meter.reset()
            f1_meter.reset()
            recall_meter.reset()
            
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'):
                for i, batch in enumerate(loader[mode]):
                    
                    x=batch["image"].to(device) 
                    y=batch["labels"].to(device)
                    output = model(x)
                    print("Model output: ",output)
                    print("Groundtruth: ",y)
                    print("Fixed Groundtruth: ",torch.max(y, 1)[1])
                    n = x.shape[0]
                    print(mode," iter:",i,"/",len(loader[mode]),end='\r')
                    global_step += n
                    
                    l = criterion(output,torch.max(y, 1)[1])
                    if mode=='train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    acc = accuracy_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
                    prec=precision_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
                    rec=recall_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
                    f1=f1_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])

                    loss_meter.add(l.item(),n)
                    acc_meter.add(acc,n)
                    precision_meter.add(prec,n)
                    recall_meter.add(rec,n)
                    f1_meter.add(f1,n)
                    
                    
                    if mode=='train':
                        wandb.log({"loss/train": loss_meter.value(), 'accuracy/train': acc_meter.value(),'precision/train': precision_meter.value(),'recall/train': recall_meter.value(),'f1/train': f1_meter.value() })
                       

            print("accuracy: ",acc_meter.value())
            print("precision: ",precision_meter.value())
            print("recall: ",recall_meter.value())
            print("F1: ",f1_meter.value())
            wandb.log({"loss/"+mode: loss_meter.value(), 'accuracy/'+mode: acc_meter.value(),'precision/'+mode: precision_meter.value(),'recall/'+mode: recall_meter.value(),'f1/'+mode: f1_meter.value() })
            
        
        torch.save(model.state_dict(),'%s-%d.pth'%(exp_name,e+1))
    return model

# %%
def test_classifier(model, test_loader, exp_name='experiment',test_name="test", logdir='logs'):
    criterion = nn.CrossEntropyLoss() 

    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    precision_meter = AverageValueMeter()
    f1_meter = AverageValueMeter()
    recall_meter = AverageValueMeter()
    

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loss_meter.reset()
    acc_meter.reset()
    model.eval()
    global_step=0
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(test_loader):
            
            x=batch["image"].to(device) 
            y=batch["labels"].to(device)
            output = model(x)
            
            
            n = x.shape[0] 
            print("Test"," iter:",i,"/",len(test_loader),end='\r')
            global_step += n
            
            l = criterion(output,torch.max(y, 1)[1])
            acc = accuracy_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
            prec=precision_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
            rec=recall_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
            f1=f1_score(y.to('cpu').max(1)[1],output.to('cpu').max(1)[1])
            loss_meter.add(l.item(),n)
            acc_meter.add(acc,n)
            precision_meter.add(prec,n)
            recall_meter.add(rec,n)
            f1_meter.add(f1,n)
        wandb.log({"loss/"+test_name: loss_meter.value(), 'accuracy/'+test_name: acc_meter.value(),'precision/'+test_name: precision_meter.value(),'recall/'+test_name: recall_meter.value(),'f1/'+test_name: f1_meter.value() })
        print("Test Accuracy: ",acc_meter.value())
    

# %%

def build_df(df):
    RGB=[]
    Unity_floor=[]
    classes=[]
    for row in df.iterrows():
        if row[1]["Difficulty"]=="medium":
            RGB.append(row[1]["RGB"])
            Unity_floor.append(row[1]["Wrong_Floor_Seg"])
            classes.append(1)
            RGB.append(row[1]["RGB"])
            Unity_floor.append(row[1]["Good_Floor_Seg"])
            classes.append(0)
    df_c=pd.DataFrame()
    df_c["RGB"]=RGB
    df_c["Floor"]=Unity_floor
    df_c["class"]=[int(x) for x in classes]
    df_c=df_c.sample(frac=1,random_state=200).reset_index(drop=True)
    return df_c


# %%
def get_models():
    model_list=[]
    
    
    
    resnet18_FC= models.resnet18(weights="ResNet18_Weights.DEFAULT")
    model_in=resnet18_FC.fc.in_features
    resnet18_FC.fc=nn.Linear(model_in,2)
    model_list.append({'name': 'resnet18_FC', 'model': resnet18_FC, 'batch_size': 64,'type':'RGB','normalize':True})
    
    resnet18_SM= models.resnet18(weights="ResNet18_Weights.DEFAULT")
    model_in=resnet18_SM.fc.in_features
    resnet18_SM.fc=nn.Linear(model_in,2)
    resnet18_SM=nn.Sequential(resnet18_SM,nn.Softmax(dim=1))
    model_list.append({'name': 'resnet18_SM', 'model': resnet18_SM, 'batch_size': 64,'type':'RGB','normalize':True})
    
    resnet18_FC2= models.resnet18(weights="ResNet18_Weights.DEFAULT")
    model_in=resnet18_FC2.fc.in_features
    resnet18_FC2.fc=nn.Linear(model_in,2)
    model_list.append({'name': 'resnet18_FC', 'model': resnet18_FC2, 'batch_size': 64,'type':'RGB','normalize':False})
    
    resnet18_SM2= models.resnet18(weights="ResNet18_Weights.DEFAULT")
    model_in=resnet18_SM2.fc.in_features
    resnet18_SM2.fc=nn.Linear(model_in,2)
    resnet18_SM2=nn.Sequential(resnet18_SM2,nn.Softmax(dim=1))
    model_list.append({'name': 'resnet18_SM', 'model': resnet18_SM2, 'batch_size': 64,'type':'RGB','normalize':False})
    

    resnet18_SM_segmentation2= models.resnet18(weights="ResNet18_Weights.DEFAULT")
    save_weights=resnet18_SM_segmentation2.conv1.state_dict()
    state_dict = resnet18_SM_segmentation2.state_dict()
    name,param= list(state_dict.items())[0]
    newparam=np.zeros((64,4,7,7))
    newparam[:, :param.cpu().numpy().shape[1]] = param.cpu().numpy()
    mean_value = np.mean(param.cpu().numpy())
    newparam[:, :param.cpu().numpy().shape[1]] = mean_value
    t=torch.from_numpy(newparam)
    resnet18_SM_segmentation2.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    new_state_dict = OrderedDict()
    new_state_dict['weight'] = t
    resnet18_SM_segmentation2.conv1.load_state_dict(new_state_dict)
    model_in=resnet18_SM_segmentation2.fc.in_features
    resnet18_SM_segmentation2.fc=nn.Linear(model_in,2)
    resnet18_SM_segmentation2=nn.Sequential(resnet18_SM_segmentation2,nn.Softmax(dim=1))
    model_list.append({'name': 'resnet18_SM', 'model': resnet18_SM_segmentation2, 'batch_size': 64,'type':'RGB+S','normalize':False})
    
    resnet18_FC_segmentation2= models.resnet18(weights="ResNet18_Weights.DEFAULT")
    save_weights=resnet18_FC_segmentation2.conv1.state_dict()
    state_dict = resnet18_FC_segmentation2.state_dict()
    name,param= list(state_dict.items())[0]
    newparam=np.zeros((64,4,7,7))
    newparam[:, :param.cpu().numpy().shape[1]] = param.cpu().numpy()
    mean_value = np.mean(param.cpu().numpy())
    newparam[:, :param.cpu().numpy().shape[1]] = mean_value
    t=torch.from_numpy(newparam)
    resnet18_FC_segmentation2.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    new_state_dict = OrderedDict()
    new_state_dict['weight'] = t
    resnet18_FC_segmentation2.conv1.load_state_dict(new_state_dict)
    model_in=resnet18_FC_segmentation2.fc.in_features
    resnet18_FC_segmentation2.fc=nn.Linear(model_in,2)
    model_list.append({'name': 'resnet18_FC', 'model': resnet18_FC_segmentation2, 'batch_size': 64,'type':'RGB+S','normalize':True})

    resnet18_SM_segmentation= models.resnet18(weights="ResNet18_Weights.DEFAULT")
    save_weights=resnet18_SM_segmentation.conv1.state_dict()
    state_dict = resnet18_SM_segmentation.state_dict()
    name,param= list(state_dict.items())[0]
    newparam=np.zeros((64,4,7,7))
    newparam[:, :param.cpu().numpy().shape[1]] = param.cpu().numpy()
    mean_value = np.mean(param.cpu().numpy())
    newparam[:, :param.cpu().numpy().shape[1]] = mean_value
    t=torch.from_numpy(newparam)
    resnet18_SM_segmentation.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    new_state_dict = OrderedDict()
    new_state_dict['weight'] = t
    resnet18_SM_segmentation.conv1.load_state_dict(new_state_dict)
    model_in=resnet18_SM_segmentation.fc.in_features
    resnet18_SM_segmentation.fc=nn.Linear(model_in,2)
    resnet18_SM_segmentation=nn.Sequential(resnet18_SM_segmentation,nn.Softmax(dim=1))
    model_list.append({'name': 'resnet18_SM', 'model': resnet18_SM_segmentation, 'batch_size': 64,'type':'RGB+S','normalize':True})
    
    resnet18_FC_segmentation= models.resnet18(weights="ResNet18_Weights.DEFAULT")
    save_weights=resnet18_FC_segmentation.conv1.state_dict()
    state_dict = resnet18_FC_segmentation.state_dict()
    name,param= list(state_dict.items())[0]
    newparam=np.zeros((64,4,7,7))
    newparam[:, :param.cpu().numpy().shape[1]] = param.cpu().numpy()
    mean_value = np.mean(param.cpu().numpy())
    newparam[:, :param.cpu().numpy().shape[1]] = mean_value
    t=torch.from_numpy(newparam)
    resnet18_FC_segmentation.conv1=nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    new_state_dict = OrderedDict()
    new_state_dict['weight'] = t
    resnet18_FC_segmentation.conv1.load_state_dict(new_state_dict)
    model_in=resnet18_FC_segmentation.fc.in_features
    resnet18_FC_segmentation.fc=nn.Linear(model_in,2)
    model_list.append({'name': 'resnet18_FC', 'model': resnet18_FC_segmentation, 'batch_size': 64,'type':'RGB+S','normalize':False})
    
    return model_list

# %%
def get_transform_RGB_normalizer():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    val_test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

def get_transform_segmentation_normalizer():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406,0.449],std=[0.229, 0.224, 0.225,0.226])
    ])
    val_test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406,0.449],std=[0.229, 0.224, 0.225,0.226])
    ])
    return train_transform, val_test_transform

def get_transform():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val_test_transform=transforms.Compose([
        transforms.ToTensor()
    ])
    return train_transform, val_test_transform

def get_ptDataset_RGB(df_train,df_val,df_test,train_transform,val_test_transform):
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    train_dataset = BlendedDataset(df_train,
                                    num_classes=Utility.num_classes,
                                    transform=train_transform)
    validation_dataset = BlendedDataset(df_val,
                                    num_classes=Utility.num_classes,
                                    transform=val_test_transform)
    test_dataset = BlendedDataset(df_test,
                                    num_classes=Utility.num_classes,
                                    transform=val_test_transform)
    return train_dataset, validation_dataset, test_dataset
def get_ptDataset_Real(df,val_test_transform):
    df.reset_index(drop=True, inplace=True)
    
    real_dataset = RGBDataset(df,
                                    num_classes=Utility.num_classes,
                                    transform=val_test_transform)
    
    return real_dataset
def get_ptDataset_segmentation(df_train,df_val,df_test,train_transform,val_test_transform):
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    train_dataset = BinarySegmentationDataset(df_train,
                                    num_classes=Utility.num_classes,
                                    transform=train_transform)
    validation_dataset = BinarySegmentationDataset(df_val,
                                    num_classes=Utility.num_classes,
                                    transform=val_test_transform)
    test_dataset = BinarySegmentationDataset(df_test,
                                    num_classes=Utility.num_classes,
                                    transform=val_test_transform)
    return train_dataset, validation_dataset, test_dataset

def get_loaders(train_dataset,validation_dataset,test_dataset):
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=Utility.batch_size,
                                                    shuffle=True,
                                                    num_workers=4)

    
    validation_dataset_loader=torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=Utility.batch_size,
                                                    shuffle=True,
                                                    num_workers=4)

    
                                    
    test_dataset_loader=torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=Utility.batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
    return train_dataset_loader,validation_dataset_loader,test_dataset_loader


def get_torch_loader_RGB_normalizer(df_train,df_val,df_test):
    train_transform,val_test_transform=get_transform_RGB_normalizer()
    train_dataset,validation_dataset,test_dataset=get_ptDataset_RGB(df_train,df_val,df_test,train_transform,val_test_transform)
    return get_loaders(train_dataset,validation_dataset,test_dataset)

def get_torch_loader_RGB(df_train,df_val,df_test):
    train_transform,val_test_transform=get_transform()
    train_dataset,validation_dataset,test_dataset=get_ptDataset_RGB(df_train,df_val,df_test,train_transform,val_test_transform)
    return get_loaders(train_dataset,validation_dataset,test_dataset)

def get_torch_loader_segmentation(df_train,df_val,df_test):
    train_transform,val_test_transform=get_transform()
    train_dataset,validation_dataset,test_dataset=get_ptDataset_segmentation(df_train,df_val,df_test,train_transform,val_test_transform)
    return get_loaders(train_dataset,validation_dataset,test_dataset)

def get_torch_loader_segmentation_normalizer(df_train,df_val,df_test):
    train_transform,val_test_transform=get_transform_segmentation_normalizer()
    train_dataset,validation_dataset,test_dataset=get_ptDataset_segmentation(df_train,df_val,df_test,train_transform,val_test_transform)
    return get_loaders(train_dataset,validation_dataset,test_dataset)   

def get_torch_loader_real_normalizer(df):
    _,val_test_transform=get_transform_RGB_normalizer()
    real_dataset=get_ptDataset_Real(df,val_test_transform)

    return torch.utils.data.DataLoader(real_dataset,
                                                    batch_size=Utility.batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
def get_torch_loader_real(df):
    _,val_test_transform=get_transform()
    real_dataset=get_ptDataset_Real(df,val_test_transform)

    return torch.utils.data.DataLoader(real_dataset,
                                                    batch_size=Utility.batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
# %%
def log_init(exp_name, model_name,lr, batch_size,test_name,epochs):
    wandb.init(
        # set the wandb project where this run will be logged
        project="misalignment-detection-MultipleEnviroments",
        name=exp_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "batch_size":Utility.batch_size,
            "architecture": model_name,
            "test": test_name,
            "epochs": Utility.epochs,
        },resume=True)

# %%
df=pd.read_csv('VirtualData_CSV.csv') 
df_train=df.loc[df["Model_name"].isin(Utility.train_env)]
df_val=df.loc[df["Model_name"].isin(Utility.val_env)]
df_test=df.loc[df["Model_name"].isin(Utility.test_env)]
df_train_c=build_df(df_train)
df_val_c=build_df(df_val)
df_test_c=build_df(df_test)

print(len(df_train))
print(len(df_val))
print(len(df_test))


# %%
models=get_models()
df_real=pd.read_csv('RealFrame.csv').sample(frac=1,random_state=200).reset_index(drop=True)

#%%

# %%
for model in models:
    for lr in [0.01]:
        Utility.batch_size=model["batch_size"]*2
        if model['normalize'] and model['type']=='RGB':
            train_dataset_loader, validation_dataset_loader, test_dataset_loader=get_torch_loader_RGB_normalizer(df_train_c,df_val_c,df_test_c)
            real_dataset_loader=get_torch_loader_real_normalizer(df_real)
        elif model['normalize']==False and model['type']=='RGB':
            train_dataset_loader, validation_dataset_loader, test_dataset_loader=get_torch_loader_RGB(df_train_c,df_val_c,df_test_c)
            real_dataset_loader=get_torch_loader_real(df_real)
        elif model['normalize']==False and model['type']=='RGB+S':
            train_dataset_loader, validation_dataset_loader, test_dataset_loader=get_torch_loader_segmentation(df_train_c,df_val_c,df_test_c)
        elif model['normalize']==True and model['type']=='RGB+S':
            train_dataset_loader, validation_dataset_loader, test_dataset_loader=get_torch_loader_segmentation_normalizer(df_train_c,df_val_c,df_test_c)
        stringa="_normalized_" if model["normalize"]==True else "_unnormalized_"
        exp_name=model["name"]+"_"+model["type"]+stringa+str(lr)
        
        trained_model = train_classifier(model["model"], train_dataset_loader, validation_dataset_loader,exp_name, epochs = Utility.epochs,lr=lr)
        
        test_classifier(trained_model, test_dataset_loader, exp_name=exp_name,test_name="testMisalignedMediummix")
        wandb.finish()

