from torch import nn
import torch
m= nn.AdaptiveAvgPool3d((5,7,9))
input=torch.randn(1,64,8,9,10)
output=m(input)
a=torch.tensor([[2,3]])
print(a.size())
