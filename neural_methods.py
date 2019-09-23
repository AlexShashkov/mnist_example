import random
import numpy as np

def costFunction(X, y, theta1, theta2, bias1, bias2, lambd=0):
    m = y.shape[0]
    _, _, hypo = feedForward(X, theta1, theta2, bias1, bias2)
    a = np.zeros((m, 1))
    b = np.zeros((m, 1))
    #print(a.shape, hypo.shape, y.shape)
    for i in range(m):
        #print(y[i, :].reshape((10,1)).T.shape, hypo[i, :].reshape((10,1)).shape)
        ans = y[i, :].reshape((10,1)).T
        got = hypo[i, :].reshape((10,1))

        a[i] = -ans.dot(np.log(got))
        b[i] = (1-ans).dot(np.log(1-got))
    a = np.sum(a)
    b = np.sum(b)
    J = (1/m)*(a-b)
    J += (lambd/(2*m)) * (np.sum(np.sum(theta1**2)) + np.sum(np.sum(theta2**2)))
    return J

def backProp(X, y, theta1, theta2, bias1, bias2, alpha, iters, lambd=0):
    m = X.shape[0]
    J = []
    for i in range(iters):
        print(f'Running epoch {i}')
        #print(theta1)
        #print(theta2)
        Z1, Z2, A = feedForward(X, theta1, theta2, bias1, bias2)
        print(A, 'y')
        #print('Z1 shape, A shape')
        #print(Z1.shape, A.shape)

        #print('Sigmoid derivative shape\n', sigmoid_derivative(Z2).shape)
        d_Z2 = A-y

        #print('Z2 shape, its derivative shape')
        #print(Z2.shape, d_Z2.shape)

        #print('Theta 2 shape, Derivative of Z2 shape, A1 shape')
        #print(theta2.shape, d_Z2.shape, sigmoid(Z1).shape)
        d_W2 = sigmoid(Z1).T.dot(d_Z2)
        #print(d_Z2, 'derivative of Z2')
        d_b2 = np.sum(d_Z2, axis=0, keepdims=True)

        #print('derivative of theta 2 shape, derivative of Z2 shape')
        #print(d_W2.shape, Z2.shape)
        d_A1 = theta2.dot(d_Z2.T)

        #print('Derivative of A1 shape, derivative of sigmoid shape')
        #print(d_A1.shape, sigmoid_derivative(Z1).shape)
        d_Z1 = d_A1.T*sigmoid_derivative(Z1)

        #print('Derivative of Z1 shape, input shape')
        #print(Z1.shape, X.shape)
        d_W1 = d_Z1.T.dot(X)
        d_b1 = np.sum(d_Z1, axis=0, keepdims=True)

        #print('Derivative of theta 1 shape, theta 1 shape')
        #print(d_W1.shape, theta1.shape)
        
        #print('Derivative of theta 2 shape, theta 2 shape')
        #print(d_W2.shape, theta2.shape)

        theta1 -= alpha*(1/m)*(d_W1+(lambd*d_W1)).T
        theta2 -= alpha*(1/m)*(d_W2+(lambd*d_W2))
        #print(bias2, 'bias 2')
        #print((alpha/m)*d_b2, 'its derivative')
        bias1 -= (alpha/m)*d_b1
        bias2 -= (alpha/m)*d_b2

        j = costFunction(X, y, theta1, theta2, bias1, bias2, lambd)
        J.append(j)
        print(f'J: {j}')
    return (J, theta1, theta2, bias1, bias2)

def feedForward(X, theta1, theta2, bias1, bias2):
    # 1 слой
    Z = X.dot(theta1) + bias1
    Z1 = Z
    A = sigmoid(Z)
    # 2 слой
    Z = A.dot(theta2) + bias2
    Z2 = Z
    A = sigmoid(Z)
    return (Z1, Z2, A)

def sigmoid(z):
    z[z > 100] = 100
    z[z < -100] = -100
    z = np.exp(-z)
    return 1/(1+z)

def sigmoid_derivative(z):
    z = sigmoid(z)*(1-sigmoid(z))
    #print(z.shape, 'derivative shape')
    return z