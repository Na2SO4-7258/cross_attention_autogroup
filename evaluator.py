from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from utils import psnr_per_image,ssim_per_image,save_json

def _save_visualization(path,original,reference,reference_expert,output,target,psnr,ssim):
    tiles=[original,reference,reference_expert,output,target]
    panel=torch.cat([tile.cpu() for tile in tiles],2).permute(1,2,0).numpy()
    image=Image.fromarray((panel.clip(0,1)*255).astype(np.uint8))
    header_height=46;annotated=Image.new("RGB",(image.width,image.height+header_height),"white")
    annotated.paste(image,(0,header_height));draw=ImageDraw.Draw(annotated)
    draw.text((6,4),f"PSNR: {psnr:.3f}  SSIM: {ssim:.4f}",fill="black")
    tile_width=original.shape[2]
    for i,label in enumerate(["original","reference","reference expert","output","target"]):draw.text((i*tile_width+4,22),label,fill="black")
    annotated.save(path)

@torch.no_grad()
def save_visualizations(model,dataset,trainer,mode,name,max_images=None,split="val"):
    model.eval();visual_dir=Path(trainer.cfg.paths()["visuals"])/name;visual_dir.mkdir(parents=True,exist_ok=True)
    limit=len(dataset) if max_images is None else min(len(dataset),max_images)
    for index in range(limit):
        original,expert,image_id=dataset.load(index);o=torch.from_numpy(original[None]).to(trainer.device);t=torch.from_numpy(expert[None]).to(trainer.device)
        if mode=="self_reference":ro=o[:,None].repeat(1,trainer.cfg.num_reference,1,1,1);re=t[:,None].repeat(1,trainer.cfg.num_reference,1,1,1)
        else:ro,re=trainer._references([index],split,mode)
        out,_,_=model(o,t,ro,re,trainer.temperature);psnr=psnr_per_image(out,t).item();ssim=ssim_per_image(out,t).item()
        _save_visualization(visual_dir/f"{image_id}.png",o[0],ro[0,0],re[0,0],out[0],t[0],psnr,ssim)

@torch.no_grad()
def evaluate(model,dataset,trainer,mode,name):
    model.eval();metrics=[];visual_dir=Path(trainer.cfg.paths()["visuals"])/name;visual_dir.mkdir(parents=True,exist_ok=True)
    if mode=="dynamic_attention":
        # 测试样本只作为查询；参考候选始终来自完整训练集。
        if name=="test":trainer.update_grouping(dataset,name,0,trainer.train_set,"train")
        else:trainer.update_grouping(dataset,name,0)
    for index in range(len(dataset)):
        original,expert,image_id=dataset.load(index);o=torch.from_numpy(original[None]).to(trainer.device);t=torch.from_numpy(expert[None]).to(trainer.device)
        if mode=="self_reference":
            ro=o[:,None].repeat(1,trainer.cfg.num_reference,1,1,1);re=t[:,None].repeat(1,trainer.cfg.num_reference,1,1,1)
        else:ro,re=trainer._references([index],name,mode)
        out,_,_=model(o,t,ro,re,trainer.temperature);metrics.append((psnr_per_image(out,t).item(),ssim_per_image(out,t).item()))
        if index<trainer.cfg.max_visualizations:
            psnr,ssim=metrics[-1];_save_visualization(visual_dir/f"{image_id}.png",o[0],ro[0,0],re[0,0],out[0],t[0],psnr,ssim)
    result={"split":name,"mode":mode,"psnr":float(np.mean([m[0] for m in metrics])),"ssim":float(np.mean([m[1] for m in metrics])),"lpips":None,"count":len(metrics)};save_json(Path(trainer.cfg.paths()["root"])/f"{name}_evaluation.json",result);return result
