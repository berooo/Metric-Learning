# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
import time
from threading import Thread
import functools
import random

# (global) variable definition here
A = 0
TRAINING_TIME_LIMIT = 60*10
# class definition here
# function definition here
def timeout(timeout):

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

@timeout(TRAINING_TIME_LIMIT)
def train(traindata):
    # 在此处完成你的训练函数，注意训练时间不要超过TRAINING_TIME_LIMIT(秒)。
    #time.sleep(1) # 这行仅用于测试训练超时，运行时请删除这行，否则你的TRAINING_TIME_LIMIT将-1s。
    train_data = traindata[0] # instances of training data
    train_lable = traindata[1] # labels of training data
    n, d = train_data.shape
    global A
    # Initialize A to a scaling matrix
    A = np.eye(d)#np.random.random([d,d])
    learning_rate = 0.001
    length = train_data.shape[1]
    for it in range(700):
        length = train_data.shape[0]
        a = range(0,length)

        n = random.sample(a,30)
        X=train_data[n]
        labels = train_lable[n]
        # Run NCA
        dX = X[:,None] - X[None]  # shape (n, n, d)
        tmp = np.einsum('...i,...j->...ij', dX, dX)  # shape (n, n, d, d)
        #print(labels[:,None] == labels[None] )
        masks = labels[:,None] == labels[None]
        for i, label in enumerate(labels):
            mask = masks[i]
            Ax = A.dot(X.T).T  # shape (n, d)
            softmax = np.exp(-((Ax[i] - Ax)**2).sum(axis=1))  # shape (n)
            softmax[i] = 0
            #sum = softmax.sum()
           # if sum == 0: sum = 0.0001
            softmax /= softmax.sum()
            t = softmax[:, None, None] * tmp[i]  # shape (n, d, d)
            #print(softmax[:, None, None] )
            d = softmax[mask].sum() * t.sum(axis=0) - t[mask].sum(axis=0)
            A += 2*learning_rate * A.dot(d)
    return 0

def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)

def distance(inst_a, inst_b):
    dist = A.dot(inst_a-inst_b).T.dot((A.dot(inst_a-inst_b)))
    return dist

# main program here
if  __name__ == '__main__':

    pass
