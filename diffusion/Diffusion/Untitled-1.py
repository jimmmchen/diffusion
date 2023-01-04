import torch
x = torch.ones(4, 1, 1, 1)
y = torch.randn(2,3,2)
print(x)
print(y)
print((x * y).shape)