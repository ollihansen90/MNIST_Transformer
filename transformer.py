import torch
from torch import nn
from patchify import patchify

"""class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=64):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size//patch_size)**2
        
        self.projection = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
"""
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.linear(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0):
        super(Attention, self).__init__()
        inner_dim = dim_head*heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.sftmx = nn.Softmax(dim=-1)
        self.to_qkf = nn.Linear(dim, inner_dim*3, bias=False)

        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # TODO
        b, n, _ = x.shape
        return x

class Normlayer(nn.Module):
    def __init__(self):
        super(Normlayer, self).__init__()
        self.norm = nn.Layernorm()

    def forward(self, x):
        out = x+self.norm(x)
        return out

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # TODO

    def forward(self, x):
        # TODO
        return x

class VisualTransformer(nn.Module):
    def __init__(self):
        super(VisualTransformer, self).__init__()
        self.projector = MLP(dim=49, hidden_dim=49*2, dropout=0.5)
        self.class_token = nn.Parameter(torch.randn(1, 1, 49))
        self.pos_emb = nn.Parameter(torch.randn(1, 16+1, 49))
        self.normlayer = Normlayer()
        self.transfomer = Transformer() # TODO: Parameter

        self.dropout = nn.Dropout()

    def forward(self, img):
        x = patchify(img)
        x = self.projector(x)
        x = 
        x = self.normlayer(x)

        return x
