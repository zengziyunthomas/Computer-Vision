import cv2 as cv
import numpy as np 
import scipy.cluster.vq
import math
from libsvm.python.libsvm.svmutil import *
from libsvm.python.libsvm.svm import *

CLASS_NUM = 256
TRAIN_NUM = 15
SIZE = 48
K = 200

norm=lambda x1,x2,x3,x4:math.sqrt((x1-x3)**2+(x2-x4)**2)

def read_features(path):
    with open(path,'rb') as f:
        return np.load(path)

def rho(vector1,vector2):
    dot=np.dot(vector1,vector2)
    norm=np.linalg.norm(vector1)*np.linalg.norm(vector2)
    return dot/norm

def detectandcompute(image):
    sift=cv.xfeatures2d.SIFT_create()
    kp,des=sift.detectAndCompute(image,None)
    return kp,des

def load_label(path):
    with open(path,'r') as f:
        names=f.readlines()
    return [n.strip() for n in names]

def bagsoffeatures(train_path,model_path,codebook,length,indexs,images):
    bof=[]
    kp,des=detectandcompute(images[0])
    for i in range(length):
        kp,des=detectandcompute(images[i])
        feature1=[[0 for i in range(K)]]
        if type(des) == np.ndarray:
            for j in range(des.shape[0]):
                maximum=0 
                count=0
                for k in range(codebook.shape[0]):
                    rho1=rho(des[j],codebook[k])
                    if rho1 > maximum:
                        maximum,count=rho1,k
                feature1[0][count] += 1
            total=0
            for p in range(len(feature1[0])):
                total+=feature1[0][p]
            for p in range(len(feature1[0])):
                if total!=0:
                    feature1[k][p]=feature1[0][p]/total
        bof.append(feature1)
    with open(model_path,'w') as f:
        for i in range(len(bof)):
            f.write(str(int(indexs[i])+1))
            for j in range(1):
                for k in range(K):
                    if bof[i][j][k] != 0:
                        f.write(' '+str(K*j+k+1)+':'+str(bof[i][j][k]))
            if i < len(bof)-1:
                f.write('\n')  

def bagsoffeatures_spm(train_path,model_path,codebook,length,indexs,images):
    bof=[]
    kp,des=detectandcompute(images[0])
    for i in range(length):
        kp,des=detectandcompute(images[i])
        feature1=[[0 for i in range(K)] for j in range(21)]
        if type(des) == np.ndarray:
            for j in range(des.shape[0]):
                maximum=0 
                count=0
                for k in range(codebook.shape[0]):
                    rho1=rho(des[j],codebook[k])
                    if rho1 > maximum:
                        maximum,count=rho1,k
                x,y=kp[j].pt 
                x1,y1 = int(x/int(SIZE/2)),int(y/int(SIZE/2))
                x2,y2 = int(x/int(SIZE/4)),int(y/int(SIZE/4))
                feature1[0][count] += 1
                feature1[2*x1+y1+1][count] += 1
                feature1[4*x2+y2+5][count] += 1
            for k in range(21):
                total=0
                for p in range(len(feature1[k])):
                    total+=feature1[k][p]
                for p in range(len(feature1[k])):
                    if total!=0:
                        feature1[k][p]=feature1[k][p]/total
                        
                    if k<=4:
                        feature1[k][p]=feature1[k][p]/4
                    else:
                        feature1[k][p]=feature1[k][p]/2
        bof.append(feature1)
    with open(model_path,'w') as f:
        for i in range(len(bof)):
            f.write(str(int(indexs[i])+1))
            for j in range(21):
                for k in range(K):
                    if bof[i][j][k] != 0:
                        f.write(' '+str(K*j+k+1)+':'+str(bof[i][j][k]))
            if i < len(bof)-1:
                f.write('\n')  

def cal_features(images,length):
    _,des=detectandcompute(images[0])
    for i in range(length):
        if i>0:
            _,des1=detectandcompute(images[i])
            if type(des1) == np.ndarray:
                if des1.shape[1]==128:
                    des=np.vstack((des,des1))
    return des

def load_train_or_test(path):
    with open(path,'r') as f:
        data=f.readlines()
    images=[]
    indexs=[]
    count=0
    for i in data:
        path,index=i.strip().split(' ')
        image=cv.imread(path,0)
        if image.shape[0]>SIZE or image.shape[1]>SIZE:
            image=cv.resize(image,(SIZE,SIZE))
        images.append(image)
        indexs.append(index)
        count+=1
    return images,indexs

if __name__ == '__main__':
    images1,indexs1=load_train_or_test("train_15.txt")
    features1=cal_features(images1,CLASS_NUM*TRAIN_NUM)
    images2,indexs2=load_train_or_test("test_15.txt")
    with open('test_15.txt','r') as f:
        length=len(f.readlines())
    features2=cal_features(images2,length)

    codebook,distortion=scipy.cluster.vq.kmeans(features1,K)
    bagsoffeatures_spm('train_15.txt','model_15.txt',codebook,CLASS_NUM*TRAIN_NUM,indexs1,images1)
    bagsoffeatures_spm('test_15.txt','predict_15.txt',codebook,len(indexs2),indexs2,images2)