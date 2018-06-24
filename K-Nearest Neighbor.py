import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_CIFAR_batch
cifar_batch_train = 'cifar-10-batches-py/data_batch_1'
X_train,Y_train = load_CIFAR_batch(cifar_batch_train)

k = 5
correct = 0
incorr_index = []
corr_index = []
#print (X_train.shape, Y_train.shape) 

cifar_batch_test = 'cifar-10-batches-py/test_batch'
X_test,Y_test = load_CIFAR_batch(cifar_batch_test)

X_train.reshape(X_train.shape[0],-1)
X_test.reshape(X_test.shape[0],-1)
#print (X_test.shape)
num_train = 9000
num_test = 500
dist = np.zeros((num_test,num_train))
label_arange = np.array([0]*k)
#print (label_arange.shape)
Y_predict = np.array([0]*num_test)
for i in range (num_test):
    for j in range (num_train):
        dist[i][j] = np.linalg.norm(X_test[i]-X_train[j])


for i in range (num_test):
    
    label_arange = np.argsort(dist[i])[:k]
    index = np.argmax(np.bincount(label_arange))
    Y_predict[i] = Y_train[index]
    #print (Y_predict[i])
    
    if (Y_predict[i] == Y_test[i]):
        correct += 1
        corr_index.append(i)
    else:
        incorr_index.append(i)

    
print (corr_index)
print (incorr_index)
print ('Accuracy of %d testcase = %f' % (num_test,(correct*100)/num_test))







