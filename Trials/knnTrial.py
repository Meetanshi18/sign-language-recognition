from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
# print(iris.data)
# print(iris.target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.33)

# -------------------------------------------------------------
# x_train = np.array([np.reshape(i, (2, 2)) for i in x_train])
# for i in range(5):   
#     plt.subplot(1, 5, i+1)
#     plt.title(i+1)
#     plt.imshow(x_train[i], interpolation='nearest')
# plt.show()
# -------------------------------------------------------------

print(y_train[:5])
print(len(y_test))

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)

from sklearn.metrics import accuracy_score
print('accuracy is')
print(accuracy_score(y_test, clf.predict(x_test)))

accuracy_values = []

for x in range(1,x_train.shape[0]):
	clf=KNeighborsClassifier(n_neighbors=x).fit(x_train,y_train)
	accuracy=accuracy_score(y_test,clf.predict(x_test))
	accuracy_values.append([x,accuracy])

import numpy as np
accuracy_values=np.array(accuracy_values)

plt.plot(accuracy_values[:,0],accuracy_values[:,1])
plt.xlabel("K")
plt.ylabel("accuracy")
plt.show()