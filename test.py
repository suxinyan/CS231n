#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import tensorflow as tf

import numpy as np

a = np.array([10, 20, 30, 40])
print(list(enumerate(a)))

print(a == 10) # np的array可以和数字直接比较

# 自带Python数组不可以这样
# myarray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# mask = range(5)
# print(mask)
# print(myarray[mask])

# np.array可以这样
myarray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
mask = range(5)
print(mask)
print(myarray[mask])

for i in np.arange(10):
	print(i)

for i in range(10):
	print(i)

print()
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])
print(np.sum(a == b))

# numpy广播测试
a = np.array([1, 2, 3, 4])
b = np.array([[1], [2], [3], [4]])
print(a + b)

print(a.reshape(4, 1))

# numpy二维数组array_split测试
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
print(np.array_split(a, 2))
print(a[:, 0])

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(np.array_split(a, 2))
# print(a[:, 0])

b = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
print(b[1][2])

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
print(np.concatenate(a))

# sorted函数测试
a = {1: 3, 3: 4, 7: 9, 6: 1}
for i in sorted(a):
	print("%d %d" % (i, a[i]))

a = np.array([[1, 2], [3, 4]])
b = np.array([[0, 1], [1, 0]])
print(a.dot(b))
print(np.dot(a, b))

a = np.array([1, 3, 5, 7, 9])
print(list(a))

a = np.array([[1, 2], [3, 4, 5]])
print(a.shape)

print(5E4)

# 测试向量的叠加
a = np.zeros((3, 2))
b = np.zeros(2,)
print(a)
print(b)
print(b.shape)
print(np.array([1, 2]).shape)
print(a + b)
c = np.zeros((2, 1))
print(c)
# print(a + c)

a = np.zeros((3, 2))
print(*a.shape)


a = 100 if 1 != 0 else 1000
print(a)

a = 5
print(a ** 3)


a = np.zeros(5)
print(a[None, :, None, None])
print(a[:,None, None, None])




