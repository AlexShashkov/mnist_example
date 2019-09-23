import os
import cv2
import pickle
import numpy as np

from log_methods import *
from datamanager import *

m = 60000
m_test = 10000
n = 784

alpha = 0.3
iterations = 500

training, testing, y_train, y_test = loadData(m, m_test, n, 'training', 'testing')

theta = None
theta2 = None
theta3 = None
theta10 = None

if not os.path.isfile('theta_log.pkl'): 
    lamb = 1
    theta = np.zeros([n+1, 10], dtype=np.float64)
    print('Theta shape', theta.shape)

    print(costFunction(training, y_train, theta, 1))
    print(costFunction(training, np.array([1 if y_train[j] == 0 else 0 for j in range(m)]), theta[:, 0], 1))
    print('training')
    for i in range(10):
        theta[:, i], J = gradientDescent(training, np.array([1 if y_train[j] == i else 0 for j in range(m)], dtype=int), theta[:, i], alpha, iterations, lamb)
        #plot(J)
    with open('theta_log.pkl', 'wb') as f:
        pickle.dump(theta, f)
else:
    with open('theta_log.pkl', 'rb') as f:
        theta = pickle.load(f)

if not os.path.isfile('theta_log_l2.pkl'): 
    lamb = 2
    theta2 = np.zeros([n+1, 10], dtype=np.float64)
    print('Theta shape', theta2.shape)

    print(costFunction(training, y_train, theta2, 1))
    print(costFunction(training, np.array([1 if y_train[j] == 0 else 0 for j in range(m)]), theta2[:, 0], 1))
    print('training')
    for i in range(10):
        theta2[:, i], J = gradientDescent(training, np.array([1 if y_train[j] == i else 0 for j in range(m)], dtype=int), theta2[:, i], alpha, iterations, lamb)
        #plot(J)
    with open('theta_log_l2.pkl', 'wb') as f:
        pickle.dump(theta2, f)
else:
    with open('theta_log_l2.pkl', 'rb') as f:
        theta2 = pickle.load(f)

if not os.path.isfile('theta_log_l3.pkl'): 
    lamb = 3
    theta3 = np.zeros([n+1, 10], dtype=np.float64)
    print('Theta shape', theta3.shape)

    print(costFunction(training, y_train, theta3, 1))
    print(costFunction(training, np.array([1 if y_train[j] == 0 else 0 for j in range(m)]), theta3[:, 0], 1))
    print('training')
    for i in range(10):
        theta3[:, i], J = gradientDescent(training, np.array([1 if y_train[j] == i else 0 for j in range(m)], dtype=int), theta3[:, i], alpha, iterations, lamb)
        #plot(J)
    with open('theta_log_l3.pkl', 'wb') as f:
        pickle.dump(theta3, f)
else:
    with open('theta_log_l3.pkl', 'rb') as f:
        theta3 = pickle.load(f)
        
if not os.path.isfile('theta_log_l10.pkl'): 
    lamb = 10
    theta10 = np.zeros([n+1, 10], dtype=np.float64)
    print('Theta shape', theta10.shape)

    print(costFunction(training, y_train, theta10, 1))
    print(costFunction(training, np.array([1 if y_train[j] == 0 else 0 for j in range(m)]), theta10[:, 0], 1))
    print('training')
    for i in range(10):
        theta10[:, i], J = gradientDescent(training, np.array([1 if y_train[j] == i else 0 for j in range(m)], dtype=int), theta10[:, i], alpha, iterations, lamb)
        #plot(J)
    with open('theta_log_l10.pkl', 'wb') as f:
        pickle.dump(theta, f)
else:
    with open('theta_log_l10.pkl', 'rb') as f:
        theta10 = pickle.load(f)

print('Параметры загружены')

# Проверка

test1 = np.argmax(training.dot(theta), axis=1)
test2 = np.argmax(testing.dot(theta), axis=1)

res1 = [1 if test1[i] == y_train[i] else 0 for i in range(test1.shape[0])]
res2 = [1 if test2[i] == y_test[i] else 0 for i in range(test2.shape[0])]

res1 = sum(res1)/len(res1)
res2 = sum(res2)/len(res2)

print(f'Точность на тренировочной выборке:{res1}, lambda:= 1')
print(f'Точность на тестовой выборке:{res2}, lambda:= 1')

test1 = np.argmax(training.dot(theta2), axis=1)
test2 = np.argmax(testing.dot(theta2), axis=1)

res1 = [1 if test1[i] == y_train[i] else 0 for i in range(test1.shape[0])]
res2 = [1 if test2[i] == y_test[i] else 0 for i in range(test2.shape[0])]

res1 = sum(res1)/len(res1)
res2 = sum(res2)/len(res2)

print(f'Точность на тренировочной выборке:{res1}, lambda:= 2')
print(f'Точность на тестовой выборке:{res2}, lambda:= 2')

test1 = np.argmax(training.dot(theta3), axis=1)
test2 = np.argmax(testing.dot(theta3), axis=1)

res1 = [1 if test1[i] == y_train[i] else 0 for i in range(test1.shape[0])]
res2 = [1 if test2[i] == y_test[i] else 0 for i in range(test2.shape[0])]

res1 = sum(res1)/len(res1)
res2 = sum(res2)/len(res2)

print(f'Точность на тренировочной выборке:{res1}, lambda:= 3')
print(f'Точность на тестовой выборке:{res2}, lambda:= 3')

test1 = np.argmax(training.dot(theta10), axis=1)
test2 = np.argmax(testing.dot(theta10), axis=1)

res1 = [1 if test1[i] == y_train[i] else 0 for i in range(test1.shape[0])]
res2 = [1 if test2[i] == y_test[i] else 0 for i in range(test2.shape[0])]

res1 = sum(res1)/len(res1)
res2 = sum(res2)/len(res2)

print(f'Точность на тренировочной выборке:{res1}, lambda:= 10')
print(f'Точность на тестовой выборке:{res2}, lambda:= 10')



