from pathlib import Path
import numpy as np
import torch
from PIL import Image
from utils import psnr_per_image,ssim_per_image,save_json

@torch.no_grad()
def evaluate(model,dataset,trainer,mode,name):
    model.eval();metrics=[];visual_dir=Path(trainer.cfg.paths()["visuals"])/name;visual_dir.mkdir(parents=True,exist_ok=True)
    for index in range(len(dataset)):
        original,expert,image_id=dataset.load(index);o=torch.from_numpy(original[None]).to(trainer.device);t=torch.from_numpy(expert[None]).to(trainer.device)
        if mode=="self_reference":
            ro=o[:,None].repeat(1,trainer.cfg.num_reference,1,1,1);re=t[:,None].repeat(1,trainer.cfg.num_reference,1,1,1)
        else:ro,re=trainer._references([index],"eval",mode)
        out,_,_=model(o,ro,re,trainer.temperature);metrics.append((psnr_per_image(out,t).item(),ssim_per_image(out,t).item()))
        if index<trainer.cfg.max_visualizations:
            reference=ro[0,0].cpu();panel=torch.cat([o[0].cpu(),reference,out[0].cpu(),t[0].cpu()],2).permute(1,2,0).numpy();Image.fromarray((panel.clip(0,1)*255).astype(np.uint8)).save(visual_dir/f"{image_id}.png")
    result={"split":name,"mode":mode,"psnr":float(np.mean([m[0] for m in metrics])),"ssim":float(np.mean([m[1] for m in metrics])),"lpips":None,"count":len(metrics)};save_json(Path(trainer.cfg.paths()["root"])/f"{name}_evaluation.json",result);return result
