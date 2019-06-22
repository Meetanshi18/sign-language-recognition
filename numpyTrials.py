import numpy as np

a = np.array([1,2,3])
# print(a)

a = np.arange(1, 12, 2)
# print(a)

a = np.linspace(1, 12, 6)
# print(a)

a = a.reshape(3,2)
# print(a)

# print(a.size)
# print(a.shape)
# print(a.dtype)
# print(a.itemsize)

b = np.array([(1,2,3) , (4,5,6)])
# print(b)
# print( b > 2 )
# print( b * 3 )

zeroMatrix = np.zeros((3, 2))
# print(zeroMatrix)
# print(np.ones((10)))

a = np.array([2,3,4], dtype = np.int16)
# print(a.dtype)
# print(np.random.random((2,3)))

a = np.random.randint(0,10,6)
a = a.reshape(3,2)
# print(a)
# print(a.sum( axis = 1 )) # sum(), mean(), var(), std() all accept axis parameter
# print(a.sum( axis = 0 ))

a = [[1,2,3,4,5,6],[7,8,9,10,11,12]]
a = np.array([np.reshape(i, (3,2)) for i in a])
# print(a)

a = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
a = np.reshape(a, -1)
print(a)