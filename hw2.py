#!/usr/bin/python
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from pylab import *
import random
import matplotlib.pyplot as plt

def load_mnist(dataset="training", digits=np.arange(10)):

    if dataset == "training":
        fname_img = os.path.join('/Users/ou/Dropbox/course/559 neural networks/HW/hw2/train-images-idx3-ubyte')
        fname_lbl = os.path.join('/Users/ou/Dropbox/course/559 neural networks/HW/hw2/train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join('/Users/ou/Dropbox/course/559 neural networks/HW/hw2/t10k-images-idx3-ubyte')
        fname_lbl = os.path.join('/Users/ou/Dropbox/course/559 neural networks/HW/hw2/t10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)

    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels
#imshow(images.mean(axis=0), cmap=cm.gray)
#show()
#print images[1]
#images /= 255.0
def get_weights(m,n):
	list_x=[0]*784
	list_w=[[ 0 for i in range(10)] for j in range(784)]
	for i in range(1,784):
		for j in range(1,10):
			list_w[i][j]=random.uniform(m,n)
	return  list_w
#image=[[ 0 for i in range(28)] for j in range(28)]
def get_x(image):
	list_x=[]
	for i in range(28):
		for j in range(28):
			list_x.append(image[i][j])
	return list_x	
###################### initalize all global variable ########################	
vout=[]
x=[]
####################### all functions #######################################
def sample(condition):
	images=[]
	labels=[]
	if condition=='training':
		for n in range(10):
			image,label =load_mnist('training',digits=[n])#the label n
			images.append(image)
			labels.append(label)
	elif condition=='testing':
		for n in range(10):
                        image,label =load_mnist('testing',digits=[n])#the label n
                        images.append(image)
                        labels.append(label)
	else:
		print'the condition is wrong'
	return images,labels
#print len(images),len(images1),len(images[0][1]),len(images1[0][1]),len(images[1]),len(images1[1])
def equals(i,j,w):
	dout=[0]*10
	x=get_x(images[i][j])
	
	v = np.dot(x,w)
	#print 'w:',len(w),len(w[0])
	v = v.tolist()
	max_index=v.index(max(v))
        #print 'output label:',max_index
	#desired_output
	dout[i]=1
	#print 'input lable:',dout
	if max_index==i:
		return 1,x,v,dout
	else:
		return 0,x,v,dout
# current_output	
def current_output(vout):
	cout=[]
	for i in range(len(vout)):
		cout.append(1 if vout[i] >= 0 else 0)
	return cout

# PTA
def pta(u,dout,cout,x,w):
	for i in range(784):
		for j in range(10):
			w[i][j]=w[i][j]+u*(dout[j]-cout[j])*x[i]
	return w
def train(w,u,n,minmum):
	error_epoch=[]
	epoch = 0
	while epoch<=50:
		# step 1 count errors
		error = 0
		for j in range(n/10):
		       for i in range(10):
				(m,x,v,d)=equals(i,j,w)
				#print m
				c=current_output(v)
			 	if (m==0):
					error = error + 1
			 	elif (m==1):
					error = error + 0

		error_epoch.append(error)		
		print "@epoch", epoch, "# errors = ", error
		# step 2 add epoch
		epoch = epoch + 1
		# step 3 update W
		for j in range(n/10):
			for i in range(10):
				(m,x,v,d)=equals(i,j,w)
				c=current_output(v)
				w=pta(u,d,c,x,w)
		# step 4 check if continue
		# 0 should work
		if (error_epoch[epoch-1])<= minmum*n:
			break 
	return w
def test(w,u,n):
        #epoch = 0
        #while True:
                # step 1 count errors
                error = 0
                for j in range(n/10):
                       for i in range(10):
                                (m,x,v,d)=equals(i,j,w)
                                #print m
                                c=current_output(v)
                                if (m==0):
                                        error = error + 1
                                elif (m==1):
                                        error = error + 0

                #error_epoch.append(error)
                #print  "# errors = ", error
                # step 2 add epoch
                #epoch = epoch + 1
		#if (error_epoch[epoch-1]) <= 0:
                       # break
        	return error

print'##################### u=1 min=0 ################################'
print'############ Q(f) 50 d ##################'
images,labels=sample('training')
w=get_weights(-1,1)
w_f=train(w,1,50,0)
"""
print'########### Q(f) 10000 e ################'
images,labels=sample('testing')
#IndexError: index 892 is out of bounds for axis 0 with size 892
error=test(w_f,1,8920)+test(w_f,1,1080)
print  "# errors of 50  = ", error

print'########### Q(g) 1000 d #################'
images,labels=sample('training')
w=get_weights(-1,1)
w_g=train(w,1,1000,0)
print'########### Q(g) 10000 e ################'
images,labels=sample('testing')
#IndexError: index 892 is out of bounds for axis 0 with size 892
error=test(w_g,1,8920)+test(w_g,1,1080)
print  "# errors of 1000 = ", error

print'########### Q(h) 60000 d ################'
images,labels=sample('training')
w=get_weights(-1,1)
#IndexError: index 5421 is out of bounds for axis 0 with size 5421
w_n=train(w,1,54210,0)
w_h=train(w_n,1,5790,0)
print'########### Q(h) 10000 e ################'
images,labels=sample('testing')
#IndexError: index 892 is out of bounds for axis 0 with size 892
error=test(w_h,1,8920)+test(w_h,1,1080)
print  "# errors of 60000= ", error
print '############## Repeat the following two subitems three times #################'
print'########### first time u=10 minimum=0.01 ################'
images,labels=sample('training')
w=get_weights(-1,1)
#IndexError: index 5421 is out of bounds for axis 0 with size 5421
w_n=train(w,10,54210,0.01)
w_h=train(w_n,10,5790,0.01)
images,labels=sample('testing')
#IndexError: index 892 is out of bounds for axis 0 with size 892
error=test(w_h,10,8920)+test(w_h,10,1080)
print  "# errors of 60000= ", error
print'########### second time u=0.1 minimum=0.01  ################'
images,labels=sample('training')
w=get_weights(-1,1)
#IndexError: index 5421 is out of bounds for axis 0 with size 5421
w_n=train(w,0.1,54210,0.01)
w_h=train(w_n,0.1,5790,0.01)
images,labels=sample('testing')
#IndexError: index 892 is out of bounds for axis 0 with size 892
error=test(w_h,0.1,8920)+test(w_h,0.1,1080)
print  "# errors of 60000= ", error
print'########### third time u=10 minimum=0.1 ################'
images,labels=sample('training')
w=get_weights(-1,1)
#IndexError: index 5421 is out of bounds for axis 0 with size 5421
w_n=train(w,10,54210,0.1)
w_h=train(w_n,10,5790,0.1)
images,labels=sample('testing')
#IndexError: index 892 is out of bounds for axis 0 with size 892
error=test(w_h,10,8920)+test(w_h,10,1080)
print  "# errors of 60000= ", error
"""
