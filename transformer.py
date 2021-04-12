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
    def __init__(self, in_dim=49, hidden_dim=64, dropout=0.0):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
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
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        dots = torch.matmul(q, k.transpose(-1, -2))*self.scale
        attn = self.sftmx(dots)
        out = torch.matmul(attn, v)

        return out

class PreNorm(nn.Module):
    def __init__(self, dim, func):
        super(PreNorm, self).__init__()
        self.norm = nn.Layernorm()
        self.func = func

    def forward(self, x, **kwargs):
        out = self.func(self.norm(x), **kwargs)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth=3, heads=8, dim_head=64, mlp_dim=128, dropout=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, MLP(in_dim=dim, hidden_dim=mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)+x
            x = mlp(x)+x
        return x

class VisualTransformer(nn.Module):
    def __init__(self, inner_dim=49*2):
        super(VisualTransformer, self).__init__()
        self.inner_dim = inner_dim
        self.projector = nn.Linear(49, inner_dim) # hier stimmt die 49
        self.outMLP = nn.Linear(inner_dim, 10) # inner_dim auf 10 Klassen (da 10 Ziffern)
        
        self.class_token = nn.Parameter(torch.randn(1, 1, inner_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 16+1, inner_dim))
        
        self.transfomer = Transformer() # TODO: Parameter
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img):
        # b, _, _ = img.shape # b ist die Batchsize (später)
        x = patchify(img) # x ist jetzt ein "Stapel" von Matrizen mit zeilenweise geflatteten Patches
        x = self.projector(x) # Kann der mit dem Patches arbeiten? nn.Linear müsste eigentlich mit der letzten Dimension arbeiten (y)
        x = torch.cat((self.class_token, x), dim=-2) # hier fehlt später noch die Batchsize b mit repeat oder sowas
        x += self.pos_emb                               # hier auch
        x = self.transfomer(x)  # das hier funktioniert noch überhaupt nicht
        x = self.outMLP(x)
        # hier fehlt vermutlich noch ein Softmax oder sowas

        return x
