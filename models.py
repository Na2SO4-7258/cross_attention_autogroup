import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CrossAttention,ReferenceSimilarity,DynamicReferenceSelector,EditTransfer

class ConvBlock(nn.Module):
    def __init__(self,inside,outside,stride=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(inside,outside,3,stride,1),nn.GroupNorm(max(1,outside//16),outside),nn.SiLU(),nn.Conv2d(outside,outside,3,1,1),nn.GroupNorm(max(1,outside//16),outside),nn.SiLU())
    def forward(self,x):return self.net(x)

class ImageEncoder(nn.Module):
    def __init__(self,dim,base_dim=32,mid_dim=64):
        super().__init__();self.a=ConvBlock(3,base_dim);self.b=ConvBlock(base_dim,mid_dim,2);self.c=ConvBlock(mid_dim,dim,2)
    def forward(self,x):return self.c(self.b(self.a(x)))

class EditEncoder(nn.Module):
    def __init__(self,dim,base_dim=32,mid_dim=64):
        super().__init__();self.a=ConvBlock(6,base_dim);self.b=ConvBlock(base_dim,mid_dim,2);self.c=ConvBlock(mid_dim,dim,2)
    def forward(self,original,expert):return self.c(self.b(self.a(torch.cat([original,expert-original],1))))

class EnhancementUNet(nn.Module):
    def __init__(self,edit_dim,base_dim=64,mid_dim=96):
        super().__init__();self.input=ConvBlock(3+edit_dim,base_dim);self.down=ConvBlock(base_dim,mid_dim,2);self.mid=ConvBlock(mid_dim,mid_dim);self.up=ConvBlock(mid_dim,base_dim);self.output=nn.Conv2d(base_dim,3,3,1,1)
    def forward(self,original,edit):
        edit=F.interpolate(edit,size=original.shape[-2:],mode="bilinear",align_corners=False);skip=self.input(torch.cat([original,edit],1));x=self.down(skip);x=self.mid(x);x=F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False);x=self.up(x)+skip;return (original+self.output(x).tanh()*0.25).clamp(0,1)

class LatentRetouchModel(nn.Module):
    def __init__(self,feature_dim=128,attention_dim=128,edit_dim=64,cross_attention_heads=1,attention_spatial_size=64,image_encoder_base_dim=32,image_encoder_mid_dim=64,enhancer_base_dim=64,enhancer_mid_dim=96):
        super().__init__();self.attention_spatial_size=attention_spatial_size;self.image_encoder=ImageEncoder(feature_dim,image_encoder_base_dim,image_encoder_mid_dim);self.edit_encoder=EditEncoder(edit_dim,image_encoder_base_dim,image_encoder_mid_dim);self.cross_attention=CrossAttention(feature_dim,attention_dim,cross_attention_heads);self.similarity=ReferenceSimilarity();self.selector=DynamicReferenceSelector();self.transfer=EditTransfer();self.edit_project=nn.Linear(edit_dim,edit_dim);self.context_edit=nn.Linear(attention_dim,edit_dim);self.enhancer=EnhancementUNet(edit_dim,enhancer_base_dim,enhancer_mid_dim)
    @staticmethod
    def tokens(feature):return feature.flatten(2).transpose(1,2)
    def pair(self,current,ref_original,ref_expert):
        size=(self.attention_spatial_size,self.attention_spatial_size);current_feature=F.interpolate(self.image_encoder(current),size=size,mode="bilinear",align_corners=False);ref_image=F.interpolate(self.image_encoder(ref_original),size=size,mode="bilinear",align_corners=False);edit_feature=F.interpolate(self.edit_encoder(ref_original,ref_expert),size=size,mode="bilinear",align_corners=False);context,attention=self.cross_attention(self.tokens(current_feature),self.tokens(ref_image));edit=self.edit_project(self.tokens(edit_feature));transferred=self.transfer(attention,edit)+self.context_edit(context);return attention,transferred,self.similarity(attention)
    def forward(self,current,ref_originals,ref_experts,temperature):
        batch,count=ref_originals.shape[:2];scores=[];transferred=[]
        for j in range(count):
            attention,edit,score=self.pair(current,ref_originals[:,j],ref_experts[:,j]);scores.append(score);transferred.append(edit)
        scores=torch.stack(scores,1);weights=self.selector(scores,temperature);edit=sum(weights[:,j,None,None]*transferred[j] for j in range(count));edit=edit.transpose(1,2).reshape(batch,-1,self.attention_spatial_size,self.attention_spatial_size);return self.enhancer(current,edit),scores,weights
