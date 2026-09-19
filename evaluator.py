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
    _,attention=model.cross_attention(model.tokens(model.resize(model.a(original))),model.tokens(model.resize(reference_edit)))
    return attention[0]

def _pixel_change_maps(original,output):
    """Output minus input: RGB luma, HSV saturation, normalized R-B warmth.

    Warmth is a relative color-balance proxy, not a Kelvin temperature estimate.
    """
    def measures(tile):
        rgb=tile.detach().float().cpu().numpy().clip(0,1)
        high=rgb.max(axis=0);low=rgb.min(axis=0)
        brightness=0.2126*rgb[0]+0.7152*rgb[1]+0.0722*rgb[2]
        saturation=np.divide(high-low,high,out=np.zeros_like(high),where=high>1e-6)
        total=rgb.sum(axis=0)
        warmth=np.divide(rgb[0]-rgb[2],total,out=np.zeros_like(total),where=total>1e-6)
        return brightness,saturation,warmth
    before=measures(original);after=measures(output)
    return tuple(new-old for old,new in zip(before,after))


def _change_heatmap(change,limit=None):
    """Symmetric scale, optionally shared: blue negative, white zero, red positive."""
    limit=float(np.abs(change).max()) if limit is None else float(limit)
    signed=np.clip(change/limit,-1,1) if limit>1e-8 else np.zeros_like(change)
    rgb=np.ones((*change.shape,3),dtype=np.float32)
    rgb[...,0]-=np.maximum(-signed,0)
    rgb[...,1]-=np.abs(signed)
    rgb[...,2]-=np.maximum(signed,0)
    return Image.fromarray(np.rint(rgb*255).astype(np.uint8)),limit


def _save_visualization(path,original,references,reference_experts,output,target,psnr,ssim,attentions,spatial_size,weights):
    count=len(references)
    tile_height,tile_width=original.shape[-2:]
    gap=12;margin=12;header=54;footer=58
    # Keep labels readable even for small training/test images.
    column_width=max(tile_width,240)
    row_height=header+tile_height+gap
    canvas_width=2*margin+3*column_width+2*gap
    canvas_height=2*margin+(count+3)*row_height+2*footer
    annotated=Image.new("RGB",(canvas_width,canvas_height),"white")
    draw=ImageDraw.Draw(annotated)

    def left(column):return margin+column*(column_width+gap)
    def paste(tile,column,top):
        annotated.paste(tile,(left(column)+(column_width-tile.width)//2,top+header))
    def plain(tile):
        pixels=tile.detach().float().cpu().permute(1,2,0).numpy().clip(0,1)
        return Image.fromarray(np.rint(pixels*255).astype(np.uint8))

    top=margin
    draw.text((margin,top),f"PSNR: {psnr:.3f}  SSIM: {ssim:.4f}",fill="black")
    for column,(tile,label) in enumerate(((original,"Original"),(output,"Output"),(target,"Ground truth"))):
        draw.text((left(column),top+28),label,fill="black")
        paste(plain(tile),column,top)

    for j in range(count):
        top=margin+(j+1)*row_height
        attention=attentions[j]
        groups=[] if attention is None else _attention_groups(attention,spatial_size)
        message="No transfer attention (CNN fusion)" if attention is None else f"{len(groups)} attention groups (matching colors / IDs)"
        draw.text((margin,top),f"Ref {j+1} | selected: {float(weights[j].mean()):.2%} | {message}",fill="black")
        tiles=((original,"Original attention",0),
               (references[j],f"Reference {j+1} original attention",1),
               (reference_experts[j],f"Reference {j+1} expert attention",1))
        for column,(tile,label,side) in enumerate(tiles):
            draw.text((left(column),top+28),label,fill="black")
            paste(_mark_attention_regions(tile,groups,side,spatial_size),column,top)

    predicted_changes=_pixel_change_maps(original,output)
    target_changes=_pixel_change_maps(original,target)
    limits=[max(float(np.abs(predicted).max()),float(np.abs(actual).max()))
            for predicted,actual in zip(predicted_changes,target_changes)]
    labels=("Brightness (RGB luma)","Saturation (HSV S)","Temperature (warmth proxy)")
    for row,(name,changes) in enumerate((("Output",predicted_changes),("Ground truth",target_changes))):
        top=margin+(count+1)*row_height+row*(row_height+footer)
        draw.text((margin,top),f"{name} - original | blue = decrease, white = zero, red = increase",fill="black")
        for column,(label,change) in enumerate(zip(labels,changes)):
            draw.text((left(column),top+28),label,fill="black")
            heatmap,limit=_change_heatmap(change,limits[column])
            paste(heatmap,column,top)
            bar_top=top+header+tile_height+8
            ramp=np.tile(np.linspace(-1,1,column_width,dtype=np.float32),(12,1))
            bar,_=_change_heatmap(ramp)
            annotated.paste(bar,(left(column),bar_top))
            draw.text((left(column),bar_top+15),f"-{limit:.4f}",fill="black")
            draw.text((left(column)+column_width//2-3,bar_top+15),"0",fill="black")
            text=f"+{limit:.4f}"
            width=draw.textbbox((0,0),text)[2]
            draw.text((left(column)+column_width-width,bar_top+15),text,fill="black")
            note="Blue: cooler / Red: warmer" if column==2 else "Shared output / target scale"
            draw.text((left(column),bar_top+32),note,fill="black")
    annotated.save(path)


def _all_visual_attentions(model,original,references,reference_experts,weights):
    if model.ablation_cnn_fusion:return [None]*references.shape[1]
    current_tokens=model.tokens(model.resize(model.a(original)))
    attentions=[]
    for j in range(references.shape[1]):
        reference_edit=model.edit_features(references[:,j],reference_experts[:,j])
        _,attention=model.cross_attention(current_tokens,model.tokens(model.resize(reference_edit)))
        attentions.append((attention[0]*weights[0,j,:,None]).detach().cpu())
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
        attentions=_all_visual_attentions(model,o,ro,re,weights)
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
            attentions=_all_visual_attentions(model,o,ro,re,weights)
            psnr,ssim=metrics[-1];_save_visualization(visual_dir/f"{image_id}.png",o[0],ro[0],re[0],out[0],t[0],psnr,ssim,attentions,model.attention_spatial_size,weights[0].detach().cpu())
    result={"split":name,"psnr":float(np.mean([m[0] for m in metrics])),"ssim":float(np.mean([m[1] for m in metrics])),"lpips":None,"count":len(metrics)};save_json(Path(trainer.cfg.paths()["root"])/f"{name}_evaluation.json",result);return result
