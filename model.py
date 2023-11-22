import torch
from opt import args
import torch.nn as nn
import torch.nn.functional as F

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class hard_sample_aware_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, act, n_num,cluster_num):
        super(hard_sample_aware_network, self).__init__()
        self.AE1 = nn.Linear(input_dim, hidden_dim)
        self.AE2 = nn.Linear(input_dim, hidden_dim)

        self.SE1 = nn.Linear(n_num, hidden_dim)
        self.SE2 = nn.Linear(n_num, hidden_dim)

        self.alpha = nn.Parameter(torch.Tensor(1, ))
        self.alpha.data = torch.tensor(0.99999).to(args.device)

        self.pos_weight = torch.ones(n_num * 2).to(args.device)
        self.pos_neg_weight = torch.ones([n_num * 2, n_num * 2]).to(args.device)

        self.cls = Prototypes(hidden_dim, cluster_num)
        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()

    def forward(self, x, A):
        Z1 = self.activate(self.AE1(x))
        Z2 = self.activate(self.AE2(x))

        Z1 = F.normalize(Z1, dim=1, p=2)
        Z2 = F.normalize(Z2, dim=1, p=2)

        E1 = F.normalize(self.SE1(A), dim=1, p=2)
        E2 = F.normalize(self.SE2(A), dim=1, p=2)

        p1 = self.cls(Z1)
        p2 = self.cls(Z2)
        return Z1, Z2, E1, E2, p1, p2
