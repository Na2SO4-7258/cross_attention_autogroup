import json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def ensure_dirs(config):
    for path in config.paths().values(): Path(path).mkdir(parents=True,exist_ok=True)

def save_json(path,payload):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as handle: json.dump(payload,handle,indent=2)

def ssim_per_image(x,y,window=7):
    c1=0.01**2;c2=0.03**2
    mu_x=F.avg_pool2d(x,window,1,window//2);mu_y=F.avg_pool2d(y,window,1,window//2)
    sx=F.avg_pool2d(x*x,window,1,window//2)-mu_x*mu_x
    sy=F.avg_pool2d(y*y,window,1,window//2)-mu_y*mu_y
    sxy=F.avg_pool2d(x*y,window,1,window//2)-mu_x*mu_y
    score=((2*mu_x*mu_y+c1)*(2*sxy+c2))/((mu_x*mu_x+mu_y*mu_y+c1)*(sx+sy+c2))
    return score.mean((1,2,3))

def psnr_per_image(x,y):
    mse=(x-y).pow(2).mean((1,2,3)).clamp_min(1e-10)
    return -10*torch.log10(mse)

def top_reference_payload(ids,matrix,k):
    result={}
    for i,image_id in enumerate(ids):
        order=np.argsort(-matrix[i])
        result[image_id]=[[ids[j],float(matrix[i,j])] for j in order if j!=i][:k]
    return result

def save_training_curve(path,history):
    if not history:return
    width,height,pad=760,300,36;values=[r["train"]["loss"] for r in history]+[r["val"]["loss"] for r in history];lo,hi=min(values),max(values)
    if hi==lo:hi=lo+1
    def points(key):
        values=[]
        for i,row in enumerate(history):
            x=pad+i*(width-2*pad)/max(1,len(history)-1);loss=row[key]["loss"];y=height-pad-(loss-lo)*(height-2*pad)/(hi-lo);values.append(f"{x:.1f},{y:.1f}")
        return " ".join(values)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/><polyline fill="none" stroke="#1565c0" stroke-width="2" points="{points("train")}"/><polyline fill="none" stroke="#c62828" stroke-width="2" points="{points("val")}"/><text x="50" y="20" font-family="sans-serif">loss: blue=train, red=validation</text></svg>'
    Path(path).write_text(svg,encoding="utf-8")

def grouping_stability(previous,current):
    if not previous:return 0.0
    overlaps=[]
    for key,entries in current.items():
        before={v[0] for v in previous.get(key,[])};after={v[0] for v in entries}
        if after:overlaps.append(len(before&after)/len(after))
    return float(np.mean(overlaps)) if overlaps else 0.0
