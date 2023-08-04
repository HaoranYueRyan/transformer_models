import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttentation(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadSelfAttentation,self).__init__()
        assert d_model % num_heads ==0

        self.d_k=d_model//num_heads
        self.num_heads==num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
    def attention(self,query,key,vlaue,mask=None):
        d_k=query.size(-1)

        scores=torch.matmul(query,key.transposes(-2,-1))//math.sqrt(d_k)

        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)
        p_atten=F.softmax(scores,dim=-1)
        return torch.matmul(scores,vlaue),p_atten


    def forward(self,query,key,value,maks=None):
        batch_size=query.size(0)
        query,key,value=[l(x).view(batch_size,-1,num_heads,d_k).transpose(1,2) for l,x in zip(self.linear_layers,(query,key,value))]

        x,self.atten =self.attention(query,key)

        output=x.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*self.d_k)

        return self.output_linear(output)

if __name__=='__main__':
    batch_size = 2
    sequence_length = 3
    d_model = 4
    num_heads = 2
    d_k = d_model // num_heads

    x = torch.rand(batch_size, sequence_length, d_model)
    print(x.shape)
    linear_layer = nn.Linear(d_model, d_model)
    print(linear_layer)
    output_1 = linear_layer(x)
    print(output_1.shape
          )

    output = linear_layer(x).view(batch_size, -1, num_heads, d_k).transpose(1, 2)

    print(output.shape)
