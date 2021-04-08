import torch

n = 4
img = torch.tensor(list(range(n)))
img = img.unsqueeze(0)
img = img.expand(n,n)
img = img + n*torch.tensor(list(range(n))).unsqueeze(1) + 1
print(1, img.t().flatten())


"""img = torch.tensor_split(img, 2, dim=-1)
img = torch.stack(img, dim=-1)
print(2, img)
img = torch.tensor_split(img, 2, dim=-2)
img = torch.stack(img, dim=-1)
print(3, img.shape)
img = img.flatten(start_dim=-4, end_dim=-3)
print(4, img)"""

