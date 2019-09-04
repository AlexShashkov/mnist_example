import os
import cv2
import pickle
import numpy as np

from log_methods import *

m = 60000
m_test = 10000
n = 784

alpha = 0.3
iterations = 500

training = np.empty([m, n], dtype=np.float64)
testing = np.empty([m_test, n], dtype=np.float64)

path = 'training'
y_train = np.array([int(directory) for directory in os.listdir(path) for file in os.listdir(path+'/'+directory)])#.reshape([m, 1])
path = 'testing'
y_test = np.array([int(directory) for directory in os.listdir(path) for file in os.listdir(path+'/'+directory)])#.reshape([m_test, 1])

if not os.path.isfile('training.pkl'): 
    path = 'training'
    training_path = [path+'/'+directory+'/'+file for directory in os.listdir(path) for file in os.listdir(path+'/'+directory)]
    y_train = [directory for directory in os.listdir(path) for file in os.listdir(path+'/'+directory)]
    for i in range(m):
        training[i, :] = cv2.imread(training_path[i],cv2.IMREAD_GRAYSCALE).flatten()

    training = np.c_[np.ones(m), training]

    with open('training.pkl', 'wb') as f:
        pickle.dump(training, f)
else:
    with open('training.pkl', 'rb') as f:
        training = pickle.load(f)


if not os.path.isfile('testing.pkl'): 
    path = 'testing'
    testing_path = [path+'/'+directory+'/'+file for directory in os.listdir(path) for file in os.listdir(path+'/'+directory)]
    for i in range(m_test):
        testing[i, :] = cv2.imread(testing_path[i],cv2.IMREAD_GRAYSCALE).flatten()

    testing = np.c_[np.ones(m_test), testing]

    with open('testing.pkl', 'wb') as f:
        pickle.dump(testing, f)
else:
    with open('testing.pkl', 'rb') as f:
        testing = pickle.load(f)

print('Выборки загружены.')
print('Train set', training.shape)
print('Test set', testing.shape)
print('Train labels', y_train.shape)
print('Test labels', y_test.shape)

plotRandom(training, 'Train set')
plotRandom(testing, 'Test set')

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



