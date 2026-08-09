from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

from .layers import MultiHeadGraphAttention

class GAT(nn.Module):
    def __init__(self,n_units,n_heads,dropout,attn_dropout):
        super(GAT,self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()
        for i in range(self.num_layer):
            self.layer_stack.append(
                MultiHeadGraphAttention(
                    n_heads[i],
                    n_units[i + 1],
                    attn_dropout,
                )
            )

    def forward(self,x,adj):
        for i,get_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x,self.dropout,training=self.training)

            x = get_layer(x, adj)
            x=x.mean(dim = 0)
            if i + 1 < self.num_layer:
                x=F.elu(x)

        return x
