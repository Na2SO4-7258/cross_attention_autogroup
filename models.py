import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CrossAttention

class ConvBlock(nn.Module):
    def __init__(self,inside,outside,stride=1):
        super().__init__();groups=max(1,outside//16);self.net=nn.Sequential(nn.Conv2d(inside,outside,3,stride,1),nn.GroupNorm(groups,outside),nn.SiLU(),nn.Conv2d(outside,outside,3,1,1),nn.GroupNorm(groups,outside),nn.SiLU())
    def forward(self,x):return self.net(x)

class Encoder(nn.Module):
    def __init__(self,channels,dim,base_dim=32,mid_dim=64):
        super().__init__();self.a=ConvBlock(channels,base_dim);self.b=ConvBlock(base_dim,mid_dim,2);self.c=ConvBlock(mid_dim,dim,2)
    def forward(self,x):return self.c(self.b(self.a(x)))

class EnhancementUNet(nn.Module):
    def __init__(self,edit_dim,base_dim=64,mid_dim=96):
        super().__init__();self.input=ConvBlock(3+edit_dim,base_dim);self.down=ConvBlock(base_dim,mid_dim,2);self.mid=ConvBlock(mid_dim,mid_dim);self.up=ConvBlock(mid_dim,base_dim);self.output=nn.Linear(base_dim,6)
        nn.init.normal_(self.output.weight,std=1e-3)
        with torch.no_grad():
            self.output.bias[:3].fill_(1.0);self.output.bias[3:].zero_()
    def mapping_parameters(self,original,edit):
        """Predict per-image RGB gains and offsets, shared by all pixels."""
        edit=F.interpolate(edit,size=original.shape[-2:],mode="bilinear",align_corners=False);skip=self.input(torch.cat([original,edit],1));x=self.down(skip);x=self.mid(x);x=F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False);x=self.up(x)+skip;parameters=self.output(x.mean(dim=(2,3)))
        gain,offset=parameters.chunk(2,dim=1)
        return gain[:,:,None,None],offset[:,:,None,None]
    def forward(self,original,edit):
        gain,offset=self.mapping_parameters(original,edit)
        return (gain*original+offset).clamp(0,1)

class LatentRetouchModel(nn.Module):
    def __init__(self,attention_dim=128,edit_dim=64,cross_attention_heads=1,attention_spatial_size=32,encoder_base_dim=32,encoder_mid_dim=64,enhancer_base_dim=64,enhancer_mid_dim=96,ablation_uniform_weight=False,ablation_cnn_fusion=False):
        super().__init__();self.attention_spatial_size=attention_spatial_size;self.a=Encoder(3,edit_dim,encoder_base_dim,encoder_mid_dim);self.b=Encoder(9,edit_dim,encoder_base_dim,encoder_mid_dim);self.cross_attention=CrossAttention(edit_dim,attention_dim,cross_attention_heads);self.enhancer=EnhancementUNet(edit_dim,enhancer_base_dim,enhancer_mid_dim)
        self.ablation_uniform_weight=ablation_uniform_weight;self.ablation_cnn_fusion=ablation_cnn_fusion
        if ablation_cnn_fusion:self.cnn_fusion=Encoder(12,edit_dim,encoder_base_dim,encoder_mid_dim)
    @staticmethod
    def tokens(feature):return feature.flatten(2).transpose(1,2)
    def resize(self,feature):return F.interpolate(feature,size=(self.attention_spatial_size,self.attention_spatial_size),mode="bilinear",align_corners=False)
    def edit_features(self,original,expert):return self.b(torch.cat([original,expert,expert-original],1))
    def grouping_embedding(self,original,expert):
        """One [1, n] descriptor per (original, expert) pair from the existing 9-channel encoder."""
        return self.b(torch.cat([original,expert,expert-original],1)).mean(dim=(2,3))
    def transfer_pair(self,current,reference_edit):
        encoded=self.a(current)
        context,_=self.cross_attention(self.tokens(self.resize(encoded)),self.tokens(self.resize(reference_edit)),
            values=reference_edit,query_size=(self.attention_spatial_size,)*2,key_size=(self.attention_spatial_size,)*2)
        return F.interpolate(context,size=encoded.shape[-2:],mode="bilinear",align_corners=False)

    def forward(self,current,current_expert,ref_originals,ref_experts,temperature=None):
        """Return output, per-position scores and reference masks [B, R, S*S].

        current_expert and temperature are retained for caller compatibility;
        neither participates in spatial reference selection or enhancement.
        """
        batch,count=ref_originals.shape[:2]
        if count<1:raise ValueError("At least one reference is required")
        current_encoded=self.a(current)
        current_tokens=self.tokens(self.resize(current_encoded))
        scores=[];transferred=[]
        for j in range(count):
            reference_edit=self.edit_features(ref_originals[:,j],ref_experts[:,j])
            context,_,score=self.cross_attention(current_tokens,self.tokens(self.resize(reference_edit)),return_scores=True,
                values=reference_edit,query_size=(self.attention_spatial_size,)*2,key_size=(self.attention_spatial_size,)*2)
            scores.append(score)
            if self.ablation_cnn_fusion:
                fused=torch.cat([current,ref_originals[:,j],ref_experts[:,j],ref_experts[:,j]-ref_originals[:,j]],1)
                transferred.append(self.cnn_fusion(fused))
            else:
                transferred.append(F.interpolate(context,size=current_encoded.shape[-2:],mode="bilinear",align_corners=False))
        scores=torch.stack(scores,dim=1)
        if self.ablation_uniform_weight:
            weights=torch.full_like(scores,1/count)
        else:
            # argmax uses the first reference when scores tie.
            selected=scores.argmax(dim=1)
            weights=F.one_hot(selected,num_classes=count).permute(0,2,1).to(scores.dtype)
        spatial_weights=F.interpolate(weights.reshape(batch,count,self.attention_spatial_size,self.attention_spatial_size),
            size=current_encoded.shape[-2:],mode="nearest")
        edit=(torch.stack(transferred,dim=1)*spatial_weights.unsqueeze(2)).sum(dim=1)
        return self.enhancer(current,edit),scores,weights
