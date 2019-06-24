import pandas as pd
import numpy as np

train = pd.read_csv('C:/Users/meeta/Desktop/TSWE2019/sign-language-recognition/dataset/sign_mnist_train.csv', header = 0)
test = pd.read_csv('dataset/sign_mnist_test.csv', header = 0)

y_train = train['label'].values
y_test = test['label'].values

x_train = train.drop(['label'],axis=1)
x_test = test.drop(['label'], axis=1)

x_train = np.array(x_train.iloc[:])
x_test = np.array(x_test.iloc[:])

import matplotlib.pyplot as plt
from sklearn import svm

x_train = x_train/255

clf = svm.SVC(gamma = 0.0001, C = 100)

clf.fit(x_train, y_train)

print('Prediction: ', clf.predict(x_test[20:21]))

x_test = np.array([np.reshape(i, (28, 28)) for i in x_test])
plt.imshow(np.array(x_test[20]), interpolation='nearest')
plt.show()
