import torch.nn as nn

class HandSignMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=126, out_features=64), 
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=6) # TODO: QUANDO è COMPLETO IL DATASET QUI DEVI METTERCI 12; HARD CODED CONCESSO PERCHé SERVE ESATTAMENTE A QUESTO OBIETTIVO
        )

    def forward(self, x):
        return self.net(x)