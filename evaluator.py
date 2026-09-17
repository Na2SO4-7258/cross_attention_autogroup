# Lower values mark more attention links; the previous threshold was 2.0.
ATTENTION_STD_MULTIPLIER = 1.5
# Minimum multiple of uniform attention; 1.0 excludes flat attention maps.
ATTENTION_UNIFORM_MULTIPLIER = 1.0

from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from utils import psnr_per_image,ssim_per_image,save_json
import colorsys
import logging
from time import perf_counter
from tqdm import tqdm

def _attention_groups(attention,spatial_size):
    """Connected components of all strong edges in the query/reference graph."""
    values=attention.detach().float().cpu().numpy()
    count=spatial_size*spatial_size
    if values.shape!=(count,count):raise ValueError("Attention shape does not match the spatial grid")
    threshold=values.mean(axis=1,keepdims=True)+ATTENTION_STD_MULTIPLIER*values.std(axis=1,keepdims=True)
    strong=(values>threshold)&(values>ATTENTION_UNIFORM_MULTIPLIER/count)&np.isfinite(values)
    # Traverse whole frontiers with NumPy instead of Python loops over every edge.
    remaining=strong.any(axis=1)
    groups=[]
    while remaining.any():
        queries=np.zeros(count,dtype=bool);keys=np.zeros(count,dtype=bool)
        frontier=np.zeros(count,dtype=bool);frontier[np.flatnonzero(remaining)[0]]=True
        while frontier.any():
            queries|=frontier;remaining[frontier]=False
            new_keys=strong[frontier].any(axis=0)&~keys
            keys|=new_keys
            if not new_keys.any():break
            frontier=strong[:,new_keys].any(axis=1)&remaining
        groups.append((set(np.flatnonzero(queries).tolist()),set(np.flatnonzero(keys).tolist())))
    return groups


def _mark_attention_regions(tile,groups,side,spatial_size):
    pixels=(tile.detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    image=Image.fromarray(pixels).convert("RGBA");overlay=Image.new("RGBA",image.size)
    draw=ImageDraw.Draw(overlay);width,height=image.size;labels=[]
    for index,group in enumerate(groups):
        color=tuple(round(channel*255) for channel in colorsys.hsv_to_rgb((index*0.61803398875)%1,0.8,1))
        cells=group[side]
        for cell in sorted(cells):
            row,col=divmod(cell,spatial_size)
            x0,y0=col*width//spatial_size,row*height//spatial_size
            x1,y1=(col+1)*width//spatial_size,(row+1)*height//spatial_size
            draw.rectangle((x0,y0,x1-1,y1-1),fill=(*color,100))
            for neighbor,line in ((cell-1,(x0,y0,x0,y1-1)),(cell+1,(x1-1,y0,x1-1,y1-1)),(cell-spatial_size,(x0,y0,x1-1,y0)),(cell+spatial_size,(x0,y1-1,x1-1,y1-1))):
                if neighbor not in cells or (neighbor==cell-1 and col==0) or (neighbor==cell+1 and col==spatial_size-1):draw.line(line,fill=(*color,255),width=1)
        row,col=divmod(min(cells),spatial_size);labels.append((col*width//spatial_size,row*height//spatial_size,str(index+1),color))
    image=Image.alpha_composite(image,overlay).convert("RGB");draw=ImageDraw.Draw(image)
    for x,y,label,color in labels:
        box=draw.textbbox((x,y),label);draw.rectangle(box,fill="black");draw.text((x,y),label,fill=color)
    return image

def _visual_attention(model,original,reference,reference_expert):
    # CNN fusion has no spatial transfer attention to visualize.
    if model.ablation_cnn_fusion:return None
    reference_edit=model.edit_features(reference,reference_expert)
    _,attention=model.cross_attention(model.tokens(model.resize(model.a(original))),model.tokens(reference_edit))
    return attention[0]

def _save_visualization(path,original,references,reference_experts,output,target,psnr,ssim,attentions,spatial_size,weights):
    tiles=[original,references[0],reference_experts[0],output,target]
    panel=torch.cat([tile.detach().cpu() for tile in tiles],2).permute(1,2,0).numpy()
    image=Image.fromarray((panel.clip(0,1)*255).astype(np.uint8))
    count=len(references);tile_width=original.shape[2];gap=16
    header_height=46;attention_header=64
    pairs_width=2*tile_width*count+gap*(count-1)
    canvas_width=max(image.width,pairs_width)
    annotated=Image.new("RGB",(canvas_width,2*image.height+header_height+attention_header),"white")
    first_left=(canvas_width-image.width)//2
    annotated.paste(image,(first_left,header_height));draw=ImageDraw.Draw(annotated)
    draw.text((first_left+6,4),f"PSNR: {psnr:.3f}  SSIM: {ssim:.4f}",fill="black")
    for i,label in enumerate(["original","reference 1","reference expert 1","output","target"]):
        draw.text((first_left+i*tile_width+4,22),label,fill="black")
    row_top=image.height+header_height
    for j,(reference,attention) in enumerate(zip(references,attentions)):
        groups=[] if attention is None else _attention_groups(attention,spatial_size)
        left=(canvas_width-pairs_width)//2+j*(2*tile_width+gap)
        draw.text((left+4,row_top+4),f"Ref {j+1} | weight: {float(weights[j]):.6f} ({float(weights[j]):.2%})",fill="black")
        message=f"{len(groups)} groups | mean + {ATTENTION_STD_MULTIPLIER:g} std"
        if attention is None:message="No transfer attention (CNN fusion)"
        draw.text((left+4,row_top+22),message,fill="black")
        for side,(tile,label) in enumerate(((original,"original attention"),(reference,f"reference {j+1} attention"))):
            tile_left=left+side*tile_width
            draw.text((tile_left+4,row_top+42),label,fill="black")
            marked=_mark_attention_regions(tile,groups,side,spatial_size)
            annotated.paste(marked,(tile_left,row_top+attention_header))
    annotated.save(path)


def _all_visual_attentions(model,original,references,reference_experts):
    if model.ablation_cnn_fusion:return [None]*references.shape[1]
    current_tokens=model.tokens(model.resize(model.a(original)))
    attentions=[]
    for j in range(references.shape[1]):
        reference_edit=model.edit_features(references[:,j],reference_experts[:,j])
        _,attention=model.cross_attention(current_tokens,model.tokens(reference_edit))
        attentions.append(attention[0].detach().cpu())
        del attention
    return attentions

@torch.no_grad()
def save_visualizations(model,dataset,trainer,name,max_images=None,split="val"):
    model.eval();visual_dir=Path(trainer.cfg.paths()["visuals"])/name;visual_dir.mkdir(parents=True,exist_ok=True)
    limit=min(len(dataset),max(0,trainer.cfg.max_visualizations if max_images is None else max_images))
    started=perf_counter();log=logging.getLogger("fivek")
    log.info("Saving %s visualizations: %d/%d samples",name,limit,len(dataset))
    for index in tqdm(range(limit),desc=f"{name} visuals",unit="image"):
        original,expert,image_id=dataset.load(index);o=torch.from_numpy(original[None]).to(trainer.device);t=torch.from_numpy(expert[None]).to(trainer.device)
        ro,re=trainer._references([index],split)
        out,_,weights=model(o,t,ro,re,trainer.temperature);psnr=psnr_per_image(out,t).item();ssim=ssim_per_image(out,t).item()
        attentions=_all_visual_attentions(model,o,ro,re)
        _save_visualization(visual_dir/f"{image_id}.png",o[0],ro[0],re[0],out[0],t[0],psnr,ssim,attentions,model.attention_spatial_size,weights[0].detach().cpu())

    log.info("Saved %d visualizations to %s in %.1fs",limit,visual_dir,perf_counter()-started)

@torch.no_grad()
def evaluate(model,dataset,trainer,name):
    model.eval();metrics=[];visual_dir=Path(trainer.cfg.paths()["visuals"])/name;visual_dir.mkdir(parents=True,exist_ok=True)
    # 测试样本只作为查询；参考候选始终来自完整训练集。
    if name=="test":trainer.update_grouping(dataset,name,0,trainer.train_set,"train")
    else:trainer.update_grouping(dataset,name,0)
    for index in tqdm(range(len(dataset)),desc=f"{name} evaluation",unit="image"):
        original,expert,image_id=dataset.load(index);o=torch.from_numpy(original[None]).to(trainer.device);t=torch.from_numpy(expert[None]).to(trainer.device)
        ro,re=trainer._references([index],name)
        out,_,weights=model(o,t,ro,re,trainer.temperature);metrics.append((psnr_per_image(out,t).item(),ssim_per_image(out,t).item()))
        if index<trainer.cfg.max_visualizations:
            attentions=_all_visual_attentions(model,o,ro,re)
            psnr,ssim=metrics[-1];_save_visualization(visual_dir/f"{image_id}.png",o[0],ro[0],re[0],out[0],t[0],psnr,ssim,attentions,model.attention_spatial_size,weights[0].detach().cpu())
    result={"split":name,"psnr":float(np.mean([m[0] for m in metrics])),"ssim":float(np.mean([m[1] for m in metrics])),"lpips":None,"count":len(metrics)};save_json(Path(trainer.cfg.paths()["root"])/f"{name}_evaluation.json",result);return result
