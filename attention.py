import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self,feature_dim,attention_dim,num_heads=1):
        if attention_dim%num_heads:raise ValueError("attention_dim must be divisible by num_heads")
        super().__init__();self.q=nn.Linear(feature_dim,attention_dim);self.k=nn.Linear(feature_dim,attention_dim);self.v=nn.Linear(feature_dim,attention_dim);self.out=nn.Linear(attention_dim,attention_dim);self.num_heads=num_heads;self.head_dim=attention_dim//num_heads;self.scale=self.head_dim**-0.5
    def forward(self,current,reference,return_scores=False):
        batch,current_tokens,_=current.shape;reference_tokens=reference.shape[1]
        q=self.q(current).reshape(batch,current_tokens,self.num_heads,self.head_dim).transpose(1,2);k=self.k(reference).reshape(batch,reference_tokens,self.num_heads,self.head_dim).transpose(1,2);v=self.v(reference).reshape(batch,reference_tokens,self.num_heads,self.head_dim).transpose(1,2)
        logits=(q@k.transpose(-1,-2)*self.scale).abs()
        # One reference choice per query, shared by all heads.
        scores=logits.mean(dim=1).amax(dim=-1)
        attention=logits.softmax(-1)
        context=(attention@v).transpose(1,2).reshape(batch,current_tokens,-1)
        result=(self.out(context),attention.mean(dim=1))
        return (*result,scores) if return_scores else result
