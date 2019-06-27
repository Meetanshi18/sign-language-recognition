import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('dataset/sign_mnist_train.csv', header = 0)
test = pd.read_csv('dataset/sign_mnist_test.csv', header = 0)

y_train = train['label'].values
y_test = test['label'].values

x_train = train.drop(['label'],axis=1)
x_test = test.drop(['label'], axis=1)

x_train = np.array(x_train.iloc[:])
x_test = np.array(x_test.iloc[:])

# print(x_train[:5])

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5).fit(x_train[:5000], y_train[:5000])

from sklearn.metrics import accuracy_score
print('accuracy is: ')
print(accuracy_score(y_test[:100], clf.predict(x_test[:100])))

accuracy_values = []

for x in range(2,6):
	clf=KNeighborsClassifier(n_neighbors=x).fit(x_train[:5000],y_train[:5000])
	accuracy=accuracy_score(y_test[:100],clf.predict(x_test[:100]))
	accuracy_values.append([x,accuracy])

import numpy as np
accuracy_values=np.array(accuracy_values)

plt.plot(accuracy_values[:,0],accuracy_values[:,1])
plt.xlabel("K")
plt.ylabel("accuracy")
plt.show()