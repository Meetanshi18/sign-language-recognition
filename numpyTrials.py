import numpy as np

a = np.array([1,2,3])
print(a)

a = np.arange(1, 12, 2)
print(a)

a = np.linspace(1, 12, 6)
print(a)

a = a.reshape(3,2)
print(a)

print(a.size)
print(a.shape)
print(a.dtype)
print(a.itemsize)

b = np.array([(1,2,3) , (4,5,6)])
print(b)
print( b > 2 )