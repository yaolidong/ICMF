import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,indices,values,shape,b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices,values,shape)
        ctx.save_for_backward(a,b)
        ctx.N = shape[0]
        return torch.matmul(a,b)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx,grad_output):
        a,b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0,:] * ctx.N + a._indices()[1,:]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None,grad_values,None,grad_b

class SpecialSpmm(nn.Module):
    def forward(self,indices,values,shape,b):
        return SpecialSpmmFunction.apply(indices,values,shape,b)

class MultiHeadGraphAttention(nn.Module):
    def __init__(self,n_head,f_out,attn_dropout):
        super(MultiHeadGraphAttention,self).__init__()
        self.n_head = n_head
        self.f_out = f_out
        self.w = Parameter(torch.Tensor(n_head,1,f_out))
        self.a_src_dst = Parameter(torch.Tensor(n_head,f_out * 2,1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        nn.init.ones_(self.w)
        stdv = 1. / math.sqrt(self.a_src_dst.size(1))
        nn.init.uniform_(self.a_src_dst,-stdv,stdv)


    def forward(self,input,adj):
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            edge = adj._indices()
            h = torch.mul(input,self.w[i])

            edge_h = torch.cat((h[edge[0,:],:],h[edge[1,:],:]),dim=1)
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze()))
            e_rowsum = self.special_spmm(edge,edge_e,torch.Size([N,N]),torch.ones(size=(N,1)).cuda() if next(self.parameters()).is_cuda else torch.ones(size=(N,1)))
            edge_e = F.dropout(edge_e,self.attn_dropout,training=self.training)

            h_prime = self.special_spmm(edge,edge_e,torch.Size([N,N]),h)
            h_prime = h_prime.div(e_rowsum)

            output.append(h_prime.unsqueeze(0))
        return torch.cat(output,dim = 0)

    def __repr__(self):
        return self.__class__.__name__+' ('+str(self.f_out)+' -> '+str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
