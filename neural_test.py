import pickle
import numpy as np

from neural_methods import feedForward
from datamanager import *

m = 60000
m_test = 10000
n = 784

training, testing, y_train, y_test = loadData(m, m_test, n, 'training', 'testing')
training = training[:, 1:training.shape[1]]
testing = testing[:, 1:testing.shape[1]]
training /= 255
testing /= 255

y = np.zeros((m, 10), dtype=np.float64)
for i in range(m):
    y[i, y_train[i] ] = 1

theta1, theta2, bias1, bias2 = None, None, None, None

with open('neural.pkl', 'rb') as f:
        theta1, theta2, bias1, bias2 = pickle.load(f)


print(testing.shape,theta1.shape,theta2.shape,bias1.shape,bias2.shape)
_, _, a = feedForward(training, theta1, theta2, bias1, bias2)
print(a.shape)
test1 = np.argmax(a, axis=1)
_, _, a = feedForward(testing, theta1, theta2, bias1, bias2)
test2 = np.argmax(a, axis=1)

res1 = [1 if test1[i] == y_train[i] else 0 for i in range(test1.shape[0])]
res2 = [1 if test2[i] == y_test[i] else 0 for i in range(test2.shape[0])]

res1 = sum(res1)/len(res1)
res2 = sum(res2)/len(res2)

print(f'Точность на тренировочной выборке:{res1}, lambda:= 0')
print(f'Точность на тестовой выборке:{res2}, lambda:= 0')


with open('neural_1.pkl', 'rb') as f:
        theta1, theta2, bias1, bias2 = pickle.load(f)

print(testing.shape,theta1.shape,theta2.shape,bias1.shape,bias2.shape)
_, _, a = feedForward(training, theta1, theta2, bias1, bias2)
print(a.shape)
test1 = np.argmax(a, axis=1)
_, _, a = feedForward(testing, theta1, theta2, bias1, bias2)
test2 = np.argmax(a, axis=1)

res1 = [1 if test1[i] == y_train[i] else 0 for i in range(test1.shape[0])]
res2 = [1 if test2[i] == y_test[i] else 0 for i in range(test2.shape[0])]

res1 = sum(res1)/len(res1)
res2 = sum(res2)/len(res2)

print(f'Точность на тренировочной выборке:{res1}, lambda:= 1')
print(f'Точность на тестовой выборке:{res2}, lambda:= 1')

with open('neural_2.pkl', 'rb') as f:
        theta1, theta2, bias1, bias2 = pickle.load(f)

print(testing.shape,theta1.shape,theta2.shape,bias1.shape,bias2.shape)
_, _, a = feedForward(training, theta1, theta2, bias1, bias2)
test1 = np.argmax(a, axis=1)
_, _, a = feedForward(testing, theta1, theta2, bias1, bias2)
test2 = np.argmax(a, axis=1)

res1 = [1 if test1[i] == y_train[i] else 0 for i in range(test1.shape[0])]
res2 = [1 if test2[i] == y_test[i] else 0 for i in range(test2.shape[0])]

res1 = sum(res1)/len(res1)
res2 = sum(res2)/len(res2)

print(f'Точность на тренировочной выборке:{res1}, lambda:= 2')
print(f'Точность на тестовой выборке:{res2}, lambda:= 2')

with open('neural_3.pkl', 'rb') as f:
        theta1, theta2, bias1, bias2 = pickle.load(f)

print(testing.shape,theta1.shape,theta2.shape,bias1.shape,bias2.shape)
_, _, a = feedForward(training, theta1, theta2, bias1, bias2)
test1 = np.argmax(a, axis=1)
_, _, a = feedForward(testing, theta1, theta2, bias1, bias2)
test2 = np.argmax(a, axis=1)

res1 = [1 if test1[i] == y_train[i] else 0 for i in range(test1.shape[0])]
res2 = [1 if test2[i] == y_test[i] else 0 for i in range(test2.shape[0])]

res1 = sum(res1)/len(res1)
res2 = sum(res2)/len(res2)

print(f'Точность на тренировочной выборке:{res1}, lambda:= 3')
print(f'Точность на тестовой выборке:{res2}, lambda:= 3')

with open('neural_10.pkl', 'rb') as f:
        theta1, theta2, bias1, bias2 = pickle.load(f)

print(testing.shape,theta1.shape,theta2.shape,bias1.shape,bias2.shape)
_, _, a = feedForward(training, theta1, theta2, bias1, bias2)
test1 = np.argmax(a, axis=1)
_, _, a = feedForward(testing, theta1, theta2, bias1, bias2)
test2 = np.argmax(a, axis=1)

res1 = [1 if test1[i] == y_train[i] else 0 for i in range(test1.shape[0])]
res2 = [1 if test2[i] == y_test[i] else 0 for i in range(test2.shape[0])]

res1 = sum(res1)/len(res1)
res2 = sum(res2)/len(res2)

print(f'Точность на тренировочной выборке:{res1}, lambda:= 10')
print(f'Точность на тестовой выборке:{res2}, lambda:= 10')