import torch
a=torch.tensor([[[5,5,5,5], [6,6,6,6], [7,7,7,7]], [[1,1,1,1],[2,2,2,2],[3,3,3,3]]])
print(a)
print(a.size())
print("#############################################3")

mask = torch.ByteTensor([[[1],[1],[0]],[[0],[1],[1]]])
print(mask.size())
b = a.masked_fill(mask, value=torch.tensor(-1e9))
print(b)
print(b.size())

mask2 = torch.ByteTensor(a)
print(mask2.shape)
c = a.masked_fill(mask2, value=torch.tensor(-1e9))
print (c)
print (c.shape)
