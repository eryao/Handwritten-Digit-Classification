#!/usr/local/bin/python
from keras.datasets import mnist
# I used this package to import MINST datasets, I didn't use any other functions in this package to train my network
import matplotlib.pyplot as plt
import numpy as np
import time
# load the MNIST dataset
(train_image, train_label), (test_image, test_label) = mnist.load_data()
pixels=train_image.shape[1]*train_image.shape[2]
#print train_image.shape (60000, 28, 28)
x=train_image.reshape(train_image.shape[0],pixels).astype('float32')
xprime=test_image.reshape(test_image.shape[0],pixels).astype('float32')
#normalize the input form 0-255 to 0-1
x=x/255
xprime=xprime/255
#print train_ima.shape (60000, 784)
#print train_label.shape (60000,)
#print train_label[0]# 5
#change train_label to 10 elements matrix
d=np.zeros((train_label.shape[0],10))
for i in range(train_label.shape[0]):
    d[i][train_label[i]]=1
dprime=np.zeros((test_label.shape[0],10))
for i in range(test_label.shape[0]):
    dprime[i][test_label[i]]=1
#print d[3] [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
#print train_label[3] 1
#print dprime[5] [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
#print test_label[5] 1
def sigmoid(x):
    return 1/(1+np.exp(-x))
def dsigmoid(y):
    return y*(1-y)
def tanh(x):
    return np.tanh(x)
def dtanh(y):
    return 1-y*y
def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist
#normalize weight
def initialize(input_num,hidden_num,output_num):
    input_d=1.0/(input_num)**(1/2)
    hidden_d=1.0/(hidden_num)**(1/2)
    wi=np.random.normal(loc=0,scale=input_d,size=(input_num,hidden_num))
    wo=np.random.normal(loc=0,scale=hidden_d,size=(hidden_num,output_num))
    bi=np.ones(hidden_num)
    bo=np.ones(output_num)
    vi=np.zeros((input_num,hidden_num))
    vo=np.zeros((hidden_num,output_num))
    vbi=np.zeros(hidden_num)
    vbo=np.zeros(output_num)
    return wi,wo,bi,bo,vi,vo,vbi,vbo
#wi,wo=initialize(784,300,10)
#print wi.shape (784, 300)
#print wo.shape (300, 10)
def feedforward(wi,wo,x,bi,bo):
    v=np.dot(wi.T,x)+bi
    #v (20,)
    y=tanh(v)
    vprime=np.dot(wo.T,y)+bo
    #print 'vprime',vprime
    #vprime (3,)
    yprime=softmax(vprime)
    yprime=np.round(yprime,2)
    return v,y,vprime,yprime
def backpropagate(d,x,v,vprime,y,yprime,momentum,u,vo,vi,vbo,vbi,bo,bi,wo,wi):
    err=2*(d-yprime)
    #print 'err', err.shape,err
    delta_prime=err
    #print 'delta_prime' ,delta_prime.shape
    #print 'v',v
    delta=dtanh(y)*np.dot(wo,delta_prime)
    #print 'delta',delta.shape,delta
    delta_wo=-1*delta_prime*np.reshape(y,(y.shape[0],1))
    #print 'delta_wo',delta_wo.shape,delta_wo
    delta_wi=-1*delta*np.reshape(x,(x.shape[0],1))
    #print 'delta_wi',delta_wi.shape,delta_wi
    delta_bo=-1*delta_prime
    #print 'delta_bo',delta_bo,delta_bo
    delta_bi=-1*delta
    #print 'delta_bi', delta_bi.shape,delta_bi
    wo=wo-u*delta_wo-momentum*vo
    vo=delta_wo
    wi=wi-u*delta_wi-momentum*vi
    vi=delta_wi
    bo=bo-u*delta_bo-momentum*vbo
    vbo=delta_bo
    bi=bi-u*delta_bi-momentum*vbi
    vbi=delta_bi
    return wo,wi,bo,bi
##################################
# main 
##################################
######################
# initialize variables
#######################
wi,wo,bi,bo,vi,vo,vbi,vbo=initialize(x.shape[1],500,10)
#print 'wi',wi
#print 'wo',wo
u=0.01
MSE=[2]
miss=[]
miss1=[]
mse1=[]
#print x.shape (60000, 784)
######################
#start training
######################
# take 60000 samples for tarining
start=time.time()
for i in range(100):
    print i
    #calculate the first error
    mse=[0]*600
    v,y,vprime,yprime=feedforward(wi,wo,x[0],bi,bo)
    wo,wi,bo,bi=backpropagate(d[0],x[0],v,vprime,y,yprime,0,u,vo,vi,vbo,vbi,bo,bi,wo,wi)
    print 'yprime',yprime
    print 'd',d[0]
    error_begin=sum((d[0]-yprime)**2)
    mse[0]=error_begin
    mis=0
    mis1=0
    errorcost=[0]*10000
    for j in range(1,600):
        start_loop=time.time()
        v,y,vprime,yprime=feedforward(wi,wo,x[j],bi,bo)
        if sum(d[j]-yprime)!=0:
            wo,wi,bo,bi=backpropagate(d[j],x[j],v,vprime,y,yprime,0,u,vo,vi,vbo,vbi,bo,bi,wo,wi)
            mis=mis+1
        error=sum((d[j]-yprime)**2)
        mse[j]=(error)
        end_loop=time.time()
    miss.append(mis)
    print 'training time',end_loop-start_loop
    MSE.append(sum(mse)/600.0)
    print 'losscost',MSE[i]
    for k in range(10000):
        v,y,vprime,yprime=feedforward(wi,wo,xprime[k],bi,bo)
        if sum(dprime[k]-yprime)!=0:
            mis1=mis1+1
        errorcost[k]=(sum((dprime[k]-yprime)**2))
    mse1.append(sum(errorcost)/10000.0)
    miss1.append(mis1)
    if MSE[i]>MSE[i-1]:
        u=0.9*u
    if abs(MSE[i]-MSE[i-1])<1e-4:
        break
end=time.time()
print 'MSE',MSE[1:]
print 'training time', end-start
print 'the number of classification errors',miss
print 'the number of classification errors in training',miss1
print 'mse1',mse1
######################
#start testing
######################



fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].plot( MSE[1:], 'k')
axes[0, 0].set_title(' train epoch VS MSE')
axes[0, 0].set_xlabel('epoch')
axes[0, 0].set_ylabel('MSE')
axes[0, 1].plot( mse1, 'r')
axes[0, 1].set_title('test epoch VS MSE')
axes[0, 1].set_xlabel('epoch')
axes[0, 1].set_ylabel('MSE')
axes[1, 0].plot( miss, 'k')
axes[1, 0].set_title('train epoch VS number of classification errors')
axes[1, 0].set_xlabel('epoch')
axes[1, 0].set_ylabel('number of classification errors')
axes[1, 1].plot( miss1, 'r')
axes[1, 1].set_title('test epoch VS number of classification errors')
axes[1, 1].set_xlabel('epoch')
axes[1, 1].set_ylabel('number of classification errors')
plt.show()

