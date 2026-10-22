import torch


#1-D tensor

# x = torch.tensor([1,2,3,4,5])
# print(x)

# # 2-D Tensor

# y = torch.tensor([
#     [1,2],
#     [3,4]
# ])
# print(y)

"Zeros Tensor"

# z = torch.zeros(3,2)
# print(z)

"Ones Tensor"
# a = torch.ones(2,6)
# print(a)

"Random decimal values Tensor"

# x = torch.rand(3,4)
# print(x)

"Indexing"

# i = torch.tensor([23,43,63,99])
# print(i[0])
# print(i[3])

# i = torch.tensor([
#     [10,20,30],
#     [40,50,60],
# ])
    

# print(i[1,2])


"Simple Operations"

# a = torch.tensor([1,4,7])
# b = torch.tensor([7,29,5])

# print(a+b)
# print(a*b)
# print(a-b)


"------------Tensor and Numpy share memoriessss--------------"

import numpy as np

arr = np.array([1,2,3,4,5])
a = torch.from_numpy(arr)
print(a)

