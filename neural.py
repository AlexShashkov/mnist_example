import os
import cv2
import pickle
import numpy as np

from matplotlib import pyplot as plt

from neural_methods import *
from datamanager import *

def plot(X, xlabel, ylabel):
    plt.figure(1)
    for i in range(len(X)):
        plt.subplot(1, 5, i+1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(X[i])
    plt.show()

m = 60000
m_test = 10000
n = 784

alpha = 0.4
iterations = 200

training, testing, y_train, y_test = loadData(m, m_test, n, 'training', 'testing')
training = training[:, 1:training.shape[1]]
training /= 255
testing /= 255
print(y_test)

y = np.zeros((m, 10), dtype=np.float64)
print(y.shape, y_train.shape)
for i in range(m):
    y[i, y_train[i] ] = 1


epsilon = 0.12

theta1 = np.random.rand(n, 25) * (2*epsilon)-epsilon
theta2 = np.random.rand(25, 10) * (2*epsilon)-epsilon

bias1 = np.ones((1, 25))
bias2 = np.ones((1, 10))
J = []

y_train = y

# lambda = 0
j, theta1, theta2, bias1, bias2 = backProp(training, y_train, theta1, theta2, bias1, bias2, alpha, iterations)
J.append(j)
with open('neural.pkl', 'wb') as f:
        pickle.dump([theta1, theta2, bias1, bias2], f)

# lambda = 1
j, theta1, theta2, bias1, bias2 = backProp(training, y_train, theta1, theta2, bias1, bias2, alpha, iterations, 1)
J.append(j)
with open('neural_1.pkl', 'wb') as f:
        pickle.dump([theta1, theta2, bias1, bias2], f)

# lambda = 2
j, theta1, theta2, bias1, bias2 = backProp(training, y_train, theta1, theta2, bias1, bias2, alpha, iterations, 2)
J.append(j)
with open('neural_2.pkl', 'wb') as f:
        pickle.dump([theta1, theta2, bias1, bias2], f)

# lambda = 3
j, theta1, theta2, bias1, bias2 = backProp(training, y_train, theta1, theta2, bias1, bias2, alpha, iterations, 3)
J.append(j)
with open('neural_3.pkl', 'wb') as f:
        pickle.dump([theta1, theta2, bias1, bias2], f)

# lambda = 10
j, theta1, theta2, bias1, bias2 = backProp(training, y_train, theta1, theta2, bias1, bias2, alpha, iterations, 10)
J.append(j)
with open('neural_10.pkl', 'wb') as f:
        pickle.dump([theta1, theta2, bias1, bias2], f)

plot(J, 'epoch', 'cost')

for i in range(len(J)):
    print(f'last lambd param of #{i} is', J[i][len(J[i])-1])