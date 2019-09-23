import os
import cv2
import numpy as np
import pickle

def loadData(m, m_test, n, train_path, test_path):
    training = np.empty([m, n], dtype=np.float64)
    testing = np.empty([m_test, n], dtype=np.float64)
    y_train = np.array([int(directory) for directory in os.listdir(train_path) for file in os.listdir(train_path+'/'+directory)])#.reshape([m, 1])
    y_test = np.array([int(directory) for directory in os.listdir(test_path) for file in os.listdir(test_path+'/'+directory)])#.reshape([m_test, 1])

    if not os.path.isfile('training.pkl'):
        training_path = [train_path+'/'+directory+'/'+file for directory in os.listdir(train_path) for file in os.listdir(train_path+'/'+directory)]
        y_train = [directory for directory in os.listdir(train_path) for file in os.listdir(train_path+'/'+directory)]
        for i in range(m):
            training[i, :] = cv2.imread(training_path[i],cv2.IMREAD_GRAYSCALE).flatten()

        training = np.c_[np.ones(m), training]

        with open('training.pkl', 'wb') as f:
            pickle.dump(training, f)
    else:
        with open('training.pkl', 'rb') as f:
            training = pickle.load(f)


    if not os.path.isfile('testing.pkl'): 
        testing_path = [test_path+'/'+directory+'/'+file for directory in os.listdir(test_path) for file in os.listdir(test_path+'/'+directory)]
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

    #plotRandom(training, 'Train set')
    #plotRandom(testing, 'Test set')

    return (training, testing, y_train, y_test)
