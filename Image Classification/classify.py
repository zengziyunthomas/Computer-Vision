import cv2 as cv
import numpy as np 
import scipy.cluster.vq
import math
from libsvm.python.libsvm.svmutil import *
from libsvm.python.libsvm.svm import *

SIZE = 100

if __name__ == '__main__':
    train_y,train_x=svm_read_problem('model_15.txt')
    test_y,test_x=svm_read_problem('predict_15.txt')
    m=svm_train(train_y,train_x,'-t 0 -c 10000')
    length=int(len(test_y)/SIZE)
    total=SIZE*len(test_y)
    acc=[]
    for i in range(length):
        a,p_acc,b=svm_predict(test_y[SIZE*i:SIZE*(i+1)],test_x[SIZE*i:SIZE*(i+1)],m)
        acc.append(p_acc)
    _,p_acc1,_=svm_predict(test_y[SIZE*length:],test_x[SIZE*length:],m)
    for i in acc:
        total+=SIZE*i[0]
    total+=(len(test_y)-length*SIZE)*p_acc1[0]
    final_acc=total/len(test_y)
    print('The accuracy of classification is '+str(final_acc)+'%')