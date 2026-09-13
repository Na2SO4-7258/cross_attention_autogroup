import re
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from torch.utils.data import Dataset

EXT={".jpg",".jpeg",".png",".tif",".tiff",".dng"}
EXPERT_RE=re.compile(r"(?:expert|tiff16)[_\- ]*([abcde])(?:\b|_)",re.I)

def _key(path):
    value=path.stem.lower()
    if re.match(r"(?:original|input|expert|tiff16)[_\- ]*[abcde]?",value,re.I):
        return re.sub(r"[^a-z0-9]","",path.parent.name.lower())
    value=re.sub(r"(?:expert|tiff16)[_\- ]*[abcde].*$","",value)
    value=re.sub(r"[_\- ](?:a|b|c|d|e)$","",value)
    return re.sub(r"[^a-z0-9]","",value)

def _read(path,size):
    try:
        with Image.open(path) as image:
            image=ImageOps.exif_transpose(image).convert("RGB").resize((size,size),Image.Resampling.LANCZOS)
            return np.asarray(image,dtype=np.float32).transpose(2,0,1)/255.0
    except Exception as error:
        if path.suffix.lower()==".dng":
            try:
                import rawpy
                image=rawpy.imread(str(path)).postprocess(use_camera_wb=True,output_bps=8)
                return np.asarray(Image.fromarray(image).resize((size,size),Image.Resampling.LANCZOS),dtype=np.float32).transpose(2,0,1)/255.0
            except ImportError:pass
        raise RuntimeError(f"Cannot decode {path}. Install rawpy for DNG files.") from error

def discover_records(data_dir):
    groups={}
    for path in Path(data_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXT:continue
        text=str(path).lower();match=EXPERT_RE.search(text)
        label=match.group(1).upper() if match else None
        key=_key(path)
        bucket=groups.setdefault(key,{"id":key,"original":None,"experts":{}})
        if label:bucket["experts"][label]=str(path)
        elif any(word in text for word in ("original","input","dng")):bucket["original"]=str(path)
    records=[]
    for value in groups.values():
        if value["original"] and value["experts"]:records.append(value)
    if not records:raise RuntimeError(f"No matched Original/Expert pairs found in {data_dir}. Expected paths containing original/input and expert_A...expert_E (or tiff16_a...tiff16_e).")
    return sorted(records,key=lambda item:item["id"])

def make_splits(records,output_dir,seed,ratios):
    split_path=Path(output_dir)/"splits.json"
    ids=[r["id"] for r in records]
    if split_path.exists():
        import json
        saved=json.loads(split_path.read_text(encoding="utf-8"))
        if set().union(*map(set,saved.values()))==set(ids):return saved
    rng=np.random.default_rng(seed);order=rng.permutation(ids).tolist();n=len(order);n_train=int(n*ratios[0]);n_val=int(n*ratios[1])
    result={"train":order[:n_train],"val":order[n_train:n_train+n_val],"test":order[n_train+n_val:]}
    import json
    split_path.parent.mkdir(parents=True,exist_ok=True);split_path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

class FiveKDataset(Dataset):
    def __init__(self,records,ids,image_size,seed=0):
        lookup={r["id"]:r for r in records};self.records=[lookup[x] for x in ids];self.image_size=image_size;self.seed=seed;self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return len(self.records)
    def load(self,index,expert=None):
        record=self.records[index];labels=sorted(record["experts"]);label=expert or labels[(index+self.epoch+self.seed)%len(labels)]
        return _read(Path(record["original"]),self.image_size),_read(Path(record["experts"][label]),self.image_size),record["id"]
    def __getitem__(self,index):
        original,expert,image_id=self.load(index)
        return {"original":original,"expert":expert,"index":index,"id":image_id}
