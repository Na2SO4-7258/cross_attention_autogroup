import json,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from torch.utils.data import Dataset

EXT={".jpg",".jpeg",".png",".tif",".tiff",".dng"}
EXPERT_RE=re.compile(r"(?:expert|tiff16)[_\- ]*([abcde])(?:\b|_)",re.I)

def _key(path):
    value=path.stem.lower()
    if re.match(r"(?:original|input|expert|tiff16)[_\- ]*[abcde]?",value,re.I):return re.sub(r"[^a-z0-9]","",path.parent.name.lower())
    value=re.sub(r"(?:expert|tiff16)[_\- ]*[abcde].*$","",value);value=re.sub(r"[_\- ](?:a|b|c|d|e)$","",value)
    return re.sub(r"[^a-z0-9]","",value)

def _read(path,size):
    try:
        with Image.open(path) as image:return np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((size,size),Image.Resampling.LANCZOS),dtype=np.float32).transpose(2,0,1)/255
    except Exception as error:
        if path.suffix.lower()==".dng":
            try:
                import rawpy
                image=rawpy.imread(str(path)).postprocess(use_camera_wb=True,output_bps=8);return np.asarray(Image.fromarray(image).resize((size,size),Image.Resampling.LANCZOS),dtype=np.float32).transpose(2,0,1)/255
            except ImportError:pass
        raise RuntimeError(f"Cannot decode {path}. Install rawpy for DNG files.") from error

def discover_records(data_dir):
    groups={}
    for path in Path(data_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXT:continue
        text=str(path).lower();match=EXPERT_RE.search(text);label=match.group(1).upper() if match else None;key=_key(path);bucket=groups.setdefault(key,{"original":None,"experts":{}})
        if label:bucket["experts"][label]=str(path)
        elif any(word in text for word in ("original","input","dng")):bucket["original"]=str(path)
    records=[{"id":f"{image_id}_{label}","original":value["original"],"expert":expert} for image_id,value in groups.items() if value["original"] for label,expert in value["experts"].items()]
    if not records:raise RuntimeError(f"No matched Original/Expert pairs found in {data_dir}.")
    return sorted(records,key=lambda item:item["id"])

def make_splits(records,output_dir,seed,ratios,train_range=None,val_range=None):
    split_path=Path(output_dir)/"splits.json";ids=[r["id"] for r in records]
    if (train_range is None)!=(val_range is None):
        raise ValueError("train_range and val_range must be specified together")
    if train_range is not None:
        # 范围以“原图编号”计数；实际划分仍由完整的 (original, expert) 组对组成。
        source_ids=[]
        for record in records:
            source_id=record["id"].rsplit("_",1)[0]
            if not source_ids or source_ids[-1]!=source_id:source_ids.append(source_id)
        def selected(sample_range,name):
            start,end=sample_range
            if start is None or end is None or start<1 or end<start or end>len(source_ids):
                raise ValueError(f"Invalid {name} original-image range {sample_range}; valid 1-based range is 1..{len(source_ids)}")
            return set(source_ids[start-1:end])
        train_sources=selected(train_range,"training")
        val_sources=selected(val_range,"validation")
        train_ids=[r["id"] for r in records if r["id"].rsplit("_",1)[0] in train_sources]
        val_ids=[r["id"] for r in records if r["id"].rsplit("_",1)[0] in val_sources]
        if set(train_ids)&set(val_ids):
            raise ValueError("Training and validation sample ranges must not overlap")
        # 未指定的原图及其全部专家组对保留作测试集。
        result={"train":train_ids,"val":val_ids,"test":[x for x in ids if x not in set(train_ids)|set(val_ids)]}
        split_path.parent.mkdir(parents=True,exist_ok=True);split_path.write_text(json.dumps(result,indent=2),encoding="utf-8");return result
    if split_path.exists():
        saved=json.loads(split_path.read_text(encoding="utf-8"))
        if set().union(*map(set,saved.values()))==set(ids):return saved
    rng=np.random.default_rng(seed);order=rng.permutation(ids).tolist();n=len(order);n_train=int(n*ratios[0]);n_val=int(n*ratios[1]);result={"train":order[:n_train],"val":order[n_train:n_train+n_val],"test":order[n_train+n_val:]}
    split_path.parent.mkdir(parents=True,exist_ok=True);split_path.write_text(json.dumps(result,indent=2),encoding="utf-8");return result

class FiveKDataset(Dataset):
    def __init__(self,records,ids,image_size):
        lookup={r["id"]:r for r in records};self.records=[lookup[x] for x in ids];self.image_size=image_size
    def __len__(self):return len(self.records)
    def load(self,index):
        record=self.records[index];return _read(Path(record["original"]),self.image_size),_read(Path(record["expert"]),self.image_size),record["id"]
    def __getitem__(self,index):
        original,expert,image_id=self.load(index);return {"original":original,"expert":expert,"index":index,"id":image_id}
