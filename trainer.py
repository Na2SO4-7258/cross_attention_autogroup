import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ssim_per_image,psnr_per_image,save_json,top_reference_payload,grouping_stability

class Trainer:
    def __init__(self,model,config,train_set,val_set,test_set=None):
        self.model=model.to(config.device);self.cfg=config;self.train_set=train_set;self.val_set=val_set;self.test_set=test_set;self.device=torch.device(config.device);self.optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay);self.references={"train":{},"val":{},"test":{}};self.last_top={"train":{},"val":{},"test":{}};self.temperature=config.temperature
        self.log=logging.getLogger("fivek")
    def loader(self,dataset,shuffle):return DataLoader(dataset,batch_size=self.cfg.batch_size,shuffle=shuffle,num_workers=self.cfg.num_workers,pin_memory=self.device.type=="cuda")
    def _dataset(self,split):
        if split=="train":return self.train_set
        if split=="val":return self.val_set
        if split=="test" and self.test_set is not None:return self.test_set
        raise ValueError(f"Unknown or unavailable split: {split}")
    def _candidate_indices(self,index,split,mode):
        dataset=self._dataset(split)
        if mode=="self_reference":return [index]
        if mode=="dynamic_attention":
            choices=self.references.get(split,{}).get(index)
            if choices is None:raise RuntimeError(f"No dynamic references prepared for {split} sample {index}")
            return choices
        candidates=[x for x in range(len(dataset)) if x!=index]
        if not candidates:raise RuntimeError(f"{split} needs at least two (original, expert) pairs for a non-self reference")
        rng=np.random.default_rng(self.cfg.seed+index+len(dataset)*17)
        if mode=="fixed_reference":return candidates[:self.cfg.num_reference]
        return rng.choice(candidates,size=min(self.cfg.num_reference,len(candidates)),replace=False).tolist()
    def _references(self,indices,split,mode):
        source=self._dataset(split)
        refs_o=[];refs_e=[]
        for index in indices:
            if mode=="self_reference" and split!="train": choices=[int(index)]
            else: choices=self._candidate_indices(int(index),split,mode)
            rows=[source.load(i)[:2] for i in choices]
            while len(rows)<self.cfg.num_reference:rows.append(rows[-1])
            refs_o.append(np.stack([x[0] for x in rows]));refs_e.append(np.stack([x[1] for x in rows]))
        return torch.from_numpy(np.stack(refs_o)).to(self.device),torch.from_numpy(np.stack(refs_e)).to(self.device)
    def run_epoch(self,dataset,split,epoch,mode,train):
        self.model.train(train);values=[];psnrs=[];ssims=[]
        for batch in tqdm(self.loader(dataset,train),desc=f"{split} {epoch}",leave=False):
            original=batch["original"].to(self.device);target=batch["expert"].to(self.device);refs_o,refs_e=self._references(batch["index"].tolist(),split,mode)
            with torch.set_grad_enabled(train):
                output,_,_=self.model(original,target,refs_o,refs_e,self.temperature);l1=F.l1_loss(output,target);ssim=ssim_per_image(output,target).mean();loss=l1+self.cfg.lambda_ssim*(1-ssim)+self.cfg.residual_weight*(output-original).abs().mean()
                if train:self.optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);self.optimizer.step()
            values.append(loss.item());psnrs.extend(psnr_per_image(output.detach(),target).cpu().tolist());ssims.extend(ssim_per_image(output.detach(),target).cpu().tolist())
        return {"loss":float(np.mean(values)),"psnr":float(np.mean(psnrs)),"ssim":float(np.mean(ssims))}
    @torch.no_grad()
    def update_grouping(self,dataset,split,epoch):
        """Choose references for every pair in one split, never from a truncated pool."""
        if len(dataset)<2:raise RuntimeError(f"{split} needs at least two (original, expert) pairs for dynamic reference selection")
        self.model.eval();indices=list(range(len(dataset)));ids=[dataset.records[i]["id"] for i in indices];n=len(indices);matrix=np.full((n,n),-np.inf,dtype=np.float32)
        for i,idx in enumerate(tqdm(indices,desc=f"{split} similarity",leave=False)):
            current_original,current_expert=dataset.load(idx)[:2]
            current_original=torch.from_numpy(current_original[None]).to(self.device)
            current_expert=torch.from_numpy(current_expert[None]).to(self.device)
            for start in range(0,n,self.cfg.batch_size):
                subset=indices[start:start+self.cfg.batch_size];rows=[dataset.load(j)[:2] for j in subset]
                ro=torch.from_numpy(np.stack([r[0] for r in rows])).to(self.device);re=torch.from_numpy(np.stack([r[1] for r in rows])).to(self.device)
                current_o=current_original.expand(len(subset),-1,-1,-1)
                current_e=current_expert.expand(len(subset),-1,-1,-1)
                scores=self.model.retouch_similarity_pair(current_o,current_e,ro,re)
                matrix[i,start:start+len(subset)]=scores.cpu().numpy()
            matrix[i,i]=-np.inf
        payload=top_reference_payload(ids,matrix,self.cfg.num_reference);previous=self.last_top[split];self.last_top[split]=payload
        lookup={image_id:index for index,image_id in enumerate(ids)};self.references[split]={}
        for source_idx,record in enumerate(dataset.records):
            self.references[split][source_idx]=[lookup[v[0]] for v in payload[record["id"]]]
        np.save(Path(self.cfg.paths()["similarity"])/f"{split}_similarity_epoch_{epoch:03d}.npy",matrix);save_json(Path(self.cfg.paths()["similarity"])/f"{split}_top_reference_epoch_{epoch:03d}.json",payload)
        stability=grouping_stability(previous,payload);info={"average_similarity":float(matrix[np.isfinite(matrix)].mean()),"grouping_stability":stability,"example":next(iter(payload.items()))};save_json(Path(self.cfg.paths()["similarity"])/f"{split}_grouping_epoch_{epoch:03d}.json",info);return info
    def checkpoint(self,epoch,best,tag):
        state={"model_state":self.model.state_dict(),"optimizer_state":self.optimizer.state_dict(),"epoch":epoch,"best_psnr":best,"temperature":self.temperature,"references":self.references,"config":self.cfg.to_dict()};torch.save(state,Path(self.cfg.paths()["checkpoints"])/tag)
