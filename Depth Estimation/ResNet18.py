import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
import torch.nn.functional as F

import csv

TRAIN=1690
TEST=654
E=9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
learningrate=1e-3

class depthDataset(Dataset):
    def __init__(self, path, mode, transform=None):
        self.path = path
        self.mode =mode
    def __getitem__(self, idx):
        length=len(str(idx))
        idx1='0'*(6-length)+str(idx)
        image_name = self.path+idx1+'_color.jpg'
        depth_name = self.path+idx1+'_depth.png'
        image = cv2.imread(image_name)
        depth = cv2.imread(depth_name,-1)
        image = cv2.resize(image,(224,224))
        depth = cv2.resize(depth,(112,112))
        depth = np.array([depth])
        image = np.swapaxes(image,0,2)
        image = np.swapaxes(image,1,2)
        image = image.astype(np.float32) 
        if self.mode==TEST:
            depth = depth.astype(np.float32) / 1000.
        if self.mode==TRAIN:
            depth = depth.astype(np.float32) / 255.*10.
        sample = {'image': image, 'depth': depth}
        return sample
    def __len__(self):
        return self.mode

def getTrainingData(batch_size=64):
    dataloader_training=DataLoader(depthDataset('./nyuv2/train/',TRAIN),batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)
    return dataloader_training

def getTestingData(batch_size=64):
    dataloader_testing=DataLoader(depthDataset('./nyuv2/test/',TEST),batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)
    return dataloader_testing

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv2 = nn.Conv2d(512,512,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def upsample(self, x, in_channels,out_channels,a=2):
        conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False).to(device)
        bn = nn.BatchNorm2d(out_channels).to(device)
        x = F.interpolate(input=x,scale_factor=a,mode='bilinear',align_corners=True)
        x = conv1(x)
        x = bn(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        b5 = self.conv2(b4)
        b5 = self.bn2(b5)
        b5 = self.relu(b5)

        up1 = self.upsample(b5,512,256)
        up2 = self.upsample(up1,256,128)
        up3 = self.upsample(up2,128,64)
        up4 = self.upsample(up3,64,32)

        up5 = self.upsample(b1,64,8)
        up6 = self.upsample(b2,128,8,4)
        up7 = self.upsample(b3,256,8,8)
        up8 = self.upsample(b4,512,8,16)

        MFF_output = torch.cat([up5,up6,up7,up8],dim=1)
        MFF_output = self.conv3(MFF_output)
        MFF_output = self.bn3(MFF_output)
        out1 = self.relu(MFF_output)

        out2 = torch.cat([out1,up4],dim=1)
        out2 = self.conv4(out2)
        out2 = self.bn4(out2)
        out3 = self.relu(out2)

        out3 = self.conv5(out3)
        out3 = self.bn5(out3)
        out4 = self.relu(out3)

        out4 = self.conv6(out4)
        out4 = self.bn6(out4)
        out = self.relu(out4)
        return out

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def update_lr(optimizer,lr):
    for i in optimizer.param_grops:
        i['lr']=lr

def train():
    training_data=getTrainingData(2)
    res=resnet18().to(device)
    cur_lr=learningrate
    los=nn.L1Loss().to(device)
    optimizer=torch.optim.Adam(res.parameters(),lr=cur_lr)
    for epoch in range(6):
        for i,sample in enumerate(training_data):
            image=sample['image'].to(device)
            depth=sample['depth'].to(device)
            output=res(image)
            loss=los(output,depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, E, i+1, len(training_data), loss.item()))
    torch.save(res,'res_6')
    print("6finish")
    for i,sample in enumerate(training_data):
        image=sample['image'].to(device)
        depth=sample['depth'].to(device)
        output=res(image)
        loss=los(output,depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(res,'res_7')
    print("7finish")
    for i,sample in enumerate(training_data):
        image=sample['image'].to(device)
        depth=sample['depth'].to(device)
        output=res(image)
        loss=los(output,depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(res,'res_8')
    print("8finish")
    for i,sample in enumerate(training_data):
        image=sample['image'].to(device)
        depth=sample['depth'].to(device)
        output=res(image)
        loss=los(output,depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(res,'res_9')
    print("9finish")

def test():
    res=torch.load('res_6')
    testing_data=getTestingData(2)
    res.eval()
    with torch.no_grad():
        R=[]
        L=[]
        for i,sample in enumerate(testing_data):
            image=sample['image']
            depth=sample['depth']
            image=image.to(device)
            output=res(image)
            output=output.data.cpu().numpy()+0.0001
            depth=depth.numpy()+0.0001

            REL=(np.abs(output-depth)/depth).mean()
            LOG=np.abs(np.log10(output)-np.log10(depth)).mean()
            R.append(REL)
            L.append(LOG)
        R=np.array(R)
        L=np.array(L)
        print("Rel:",R.mean())
        print("Log:",L.mean())

if __name__ == '__main__':
    # train()
    test()
    
    

