import torch 
from transformer import VisualTransformer

num_classes = 62
params = {"inner_dim": 5*num_classes, 
            "transformer_depth": 3, # Größe des Stapels an Transformern (werden nacheinander durchiteriert)
            "attn_heads": 16, # Anzahl Attention Heads
            "dim_head": 2*62, # eigene Dimension für Attention
            "mlp_dim": 128, # Dimension des MLPs im Transformer
            "transformer_dropout": 0.,#1, # Dropout des MLP im Transformer
            "num_classes": num_classes # Anzahl Klassen}
}
model = VisualTransformer(**params)
with open("architecture.txt", "w+") as file:
    file.write(str(params)+"\n")
    for p in model.named_parameters():
        file.write(str(p[0])+": "+str(p[1].numel())+"\n")
    file.write("Total parameter count: "+ str(sum([p.numel() for p in model.parameters()])))
    