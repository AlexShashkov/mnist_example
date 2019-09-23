import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def predict(img, theta, needOne=False):
    if needOne:
        img = np.r_[1, img]
    return np.argmax(theta.T.dot(img), axis=0)

def loadImage(path, inv=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    if inv:
        img = cv2.bitwise_not(img)
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img.flatten()

def plotRandom(X, name, iter=10):
    rand = [random.randrange(0, X.shape[0], 1) for i in range(iter)]
    plt.figure(1)
    for i in range(1, iter+1, 1):
        plt.subplot(2, 5, i)
        plt.imshow(X[rand[i-1], 1:].reshape([28, 28]), cmap='gray', vmin=0, vmax=255)
    plt.xlabel(name)
    plt.show()

def plot(X, y=None):
    print(X)
    if y is None:
        plt.plot(X)
    else:
        plt.plot(X, y)
    plt.show()

def sigmoid(z):
    z[z > 100] = 100
    z[z < -100] = -100
    z = np.exp(-z)
    return 1/(1+z)

def costFunction(X, y, theta, lamb):
    m = y.shape[0]
    hypo = sigmoid(X.dot(theta))
    sum = (1/m) * ( (-y.T.dot(np.log(hypo))) - (1-y.T).dot(np.log(1-hypo)) )
    #reguralisation
    sum += (lamb/(2*m))*np.sum(theta[1:]**2) 
    #print(sum)
    return sum

def gradientDescent(X, y, theta, alpha, iterations, lamb):
    J_hist = []
    print(y)
    m = y.shape[0]
    print(m)
    for i in range(iterations):
        hypo = sigmoid(X.dot(theta))
        #print(hypo.shape, y.shape, X.T.shape)
        #print(hypo)
        hypo = X.T.dot(hypo - y)
        theta -= (alpha/m) * hypo
        #reguralisation
        theta[1:] += (lamb/m)*theta[1:]
        J_hist.append(costFunction(X, y, theta, lamb))
    return (theta, J_hist)