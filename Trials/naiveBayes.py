import pandas as pd
import numpy as np

train = pd.read_csv('dataset/sign_mnist_train.csv', header = 0)
test = pd.read_csv('dataset/sign_mnist_test.csv', header = 0)

y_train = train['label'].values
y_test = test['label'].values

x_train = train.drop(['label'],axis=1)
x_test = test.drop(['label'], axis=1)

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(x_train, y_train)
gnb_predictions = gnb.predict(x_test)

accuracy = gnb.score(x_test, y_test)
print(accuracy)
