import torch
from torch import nn
from torch.nn.functional import softmax
from patchify import patchify

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
        out = torch.matmul(attn.transpose(-2,-1), v)
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
    def __init__(self, dim, depth=8, heads=8, dim_head=64, mlp_dim=128, dropout=0.0):
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
    def __init__(
                    self, 
                    inner_dim=49*2, 
                    transformer_depth=5, # Größe des Stapels an Transformern (werden nacheinander durchiteriert)
                    attn_heads=8, # Anzahl Attention Heads
                    dim_head=64, # eigene Dimension für Attention
                    mlp_dim=128, # Dimension des MLPs im Transformer
                    num_classes=10 # Anzahl Klassen (max=10)
                ):
        super(VisualTransformer, self).__init__()
        print("num_classes", num_classes)
        self.projector = nn.Linear(49, inner_dim) # hier stimmt die 49
        
        self.class_token = nn.Parameter(torch.randn(1, 1, inner_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 16+1, inner_dim))
        
        self.transfomer = Transformer(
                            dim=inner_dim, 
                            depth=transformer_depth, 
                            heads=attn_heads, 
                            dim_head=dim_head, 
                            mlp_dim=mlp_dim
                        ) 
        self.dropout = nn.Dropout(p=0.5)
        self.outMLP = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, num_classes) # inner_dim auf 10 Klassen (da 10 Ziffern)
        )
    def forward(self, img):
        while len(img.shape)<4:
            img = img.unsqueeze(0)
        b, *_ = img.shape # b ist die Batchsize (später)
        
        x = patchify(img) # x ist jetzt ein "Stapel" von Matrizen mit zeilenweise geflatteten Patches
        #print(x.shape)
        x = self.projector(x)

        cls_token = self.class_token.repeat([b,1,1])
        pos_emb = self.class_token.repeat([b,1,1])
        x = torch.cat((cls_token, x), dim=-2)
        x += pos_emb
        x = self.dropout(x)
        
        x = self.transfomer(x)[:,0]
        #print("Transformer-Output", x.shape)
        #x = x.mean(dim=-2)
        #print("Transformer-Output", x.shape)
        x = self.outMLP(x)
        # hier fehlt vermutlich noch ein Softmax oder sowas
        x = softmax(x, dim=-1)

        return x
