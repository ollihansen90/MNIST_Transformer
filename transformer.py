import torch
from torch import nn
from torch.nn.functional import softmax
from patchify import patchify
from WeavedMLP import WeavedMLP
from datetime import datetime as dt

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

class rConv2d(nn.Module):
    def __init__(self, n_x=7, n_y=7, n_c=1):
        super(rConv2d, self).__init__()
        self.n_c, self.n_x, self.n_y = n_c, n_x, n_y
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, padding_mode="reflect", groups=3)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.view(-1, self.n_c, self.n_x, self.n_y)
        print(x.shape)
        return self.conv(x).flatten(start_dim=-2, end_dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
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
        x = x.unsqueeze(-3)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = torch.cat(q.chunk(self.heads, dim=-1), dim=-3)
        k = torch.cat(k.chunk(self.heads, dim=-1), dim=-3)
        v = torch.cat(v.chunk(self.heads, dim=-1), dim=-3)
        dots = torch.matmul(q, k.transpose(-1, -2))*self.scale
        attn = self.sftmx(dots)
        #print(attn.shape)
        #out = torch.matmul(attn.transpose(-2,-1), v)
        out = torch.matmul(attn, v)
        out = out.transpose(-3,-2).flatten(start_dim=-2, end_dim=-1).squeeze()
        return self.out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, func):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.func = func

    def forward(self, x, **kwargs):
        out = self.func(self.norm(x), **kwargs)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth=3, heads=8, dim_head=64, mlp_dim=128, mlp_groups=2, dropout=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                #PreNorm(dim, MLP(in_dim=dim, hidden_dim=mlp_dim, dropout=dropout))
                PreNorm(dim, WeavedMLP(dim_in=dim, 
                                        hidden_dims=mlp_dim, 
                                        dim_out=dim, 
                                        dropout=dropout, 
                                        n_groups=mlp_groups
                                    )
                        )
            ]))

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)+x
            x = mlp(x)+x
        return x

class VisualTransformer(nn.Module):
    def __init__(
                    self, 
                    n_patches=16,
                    inner_dim=49*2, 
                    transformer_depth=5, # Größe des Stapels an Transformern (werden nacheinander durchiteriert)
                    attn_heads=8, # Anzahl Attention Heads
                    dim_head=64, # eigene Dimension für Attention
                    mlp_dim=128, # Dimension des MLPs im Transformer
                    mlp_groups=1,
                    transformer_dropout=0., # Dropout des MLP im Transformer
                    num_classes=10 # Anzahl Klassen (max=10)
                ):
        super(VisualTransformer, self).__init__()
        self.n_patches = n_patches
        print("num_classes", num_classes)
        self.projector = nn.Linear(int(28**2/n_patches), inner_dim, bias=False) # int(28**2/n_patches) ist hier die vektorisierte Patchgröße
        
        self.class_token = nn.Parameter(torch.randn(1, 1, inner_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 16+1, inner_dim))
        
        self.transfomer = Transformer(
                            dim=inner_dim, 
                            depth=transformer_depth, 
                            heads=attn_heads, 
                            dim_head=dim_head, 
                            mlp_dim=mlp_dim,
                            mlp_groups=mlp_groups,
                            dropout=transformer_dropout
                        ) 
        self.dropout = nn.Dropout(p=0.)
        self.outMLP = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, num_classes)
        )
    def forward(self, img):
        #starttime = dt.now()
        while len(img.shape)<4:
            img = img.unsqueeze(0)
        b, *_ = img.shape # batchsize
        
        x = patchify(img, n_patches=self.n_patches) # x ist jetzt ein "Stapel" von Matrizen mit zeilenweise geflatteten Patches
        x = self.projector(x) # x wird in inner_dim-dimensionalen Vektorraum projiziert

        cls_token = self.class_token.repeat([b,1,1]) # Klassentoken
        pos_emb = self.pos_emb.repeat([b,1,1]) # Positionembedding
        if self.n_patches==1: # Im Fall des gesamten Bildes fehlt bei der Ausgabe des Projektors eine Dimension, die den Stapel beschreibt
            x = x.unsqueeze(1)
        x = torch.cat((cls_token, x), dim=-2)
        x += pos_emb
        x = self.dropout(x) # Dropout für das pos_emb (hier wird auch der cls-Token gedroppt, ist das richtig?)
        
        x = self.transfomer(x)[:,0] # oder x.mean(dim=-2)
        x = self.outMLP(x)
        
        x = softmax(x, dim=-1)
        return x
