import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ssim_per_image,psnr_per_image,save_json,top_reference_payload,grouping_stability

class Trainer:
    def __init__(self,model,config,train_set,val_set):
        self.model=model.to(config.device);self.cfg=config;self.train_set=train_set;self.val_set=val_set;self.device=torch.device(config.device);self.optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay);self.pool=list(range(min(len(train_set),getattr(config,"reference_pool_size",512))));self.references={};self.last_top={};self.temperature=config.temperature
        self.log=logging.getLogger("fivek")
    def loader(self,dataset,shuffle):return DataLoader(dataset,batch_size=self.cfg.batch_size,shuffle=shuffle,num_workers=self.cfg.num_workers,pin_memory=self.device.type=="cuda")
    def _candidate_indices(self,index,split,mode):
        dataset=self.train_set if split=="train" else self.val_set
        if mode=="self_reference":return [index]
        if split=="train" and index in self.references:return self.references[index]
        candidates=self.pool if split=="train" else list(range(len(self.train_set)))
        candidates=[x for x in candidates if not(split=="train" and x==index)]
        rng=np.random.default_rng(self.cfg.seed+index+len(dataset)*17)
        if mode=="fixed_reference":return candidates[:self.cfg.num_reference]
        return rng.choice(candidates,size=min(self.cfg.num_reference,len(candidates)),replace=False).tolist()
    def _references(self,indices,split,mode):
        source=self.val_set if mode=="self_reference" and split!="train" else self.train_set
        refs_o=[];refs_e=[]
        for index in indices:
            if mode=="self_reference" and split!="train": choices=[int(index)]
            else: choices=self._candidate_indices(int(index),split,mode)
            rows=[source.load(i)[:2] for i in choices]
            while len(rows)<self.cfg.num_reference:rows.append(rows[-1])
            refs_o.append(np.stack([x[0] for x in rows]));refs_e.append(np.stack([x[1] for x in rows]))
        return torch.from_numpy(np.stack(refs_o)).to(self.device),torch.from_numpy(np.stack(refs_e)).to(self.device)
    def run_epoch(self,dataset,split,epoch,mode,train):
        dataset.set_epoch(epoch);self.model.train(train);values=[];psnrs=[];ssims=[]
        for batch in tqdm(self.loader(dataset,train),desc=f"{split} {epoch}",leave=False):
            original=batch["original"].to(self.device);target=batch["expert"].to(self.device);refs_o,refs_e=self._references(batch["index"].tolist(),split,mode)
            with torch.set_grad_enabled(train):
                output,_,_=self.model(original,refs_o,refs_e,self.temperature);l1=F.l1_loss(output,target);ssim=ssim_per_image(output,target).mean();loss=l1+self.cfg.lambda_ssim*(1-ssim)+self.cfg.residual_weight*(output-original).abs().mean()
                if train:self.optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);self.optimizer.step()
            values.append(loss.item());psnrs.extend(psnr_per_image(output.detach(),target).cpu().tolist());ssims.extend(ssim_per_image(output.detach(),target).cpu().tolist())
        return {"loss":float(np.mean(values)),"psnr":float(np.mean(psnrs)),"ssim":float(np.mean(ssims))}
    @torch.no_grad()
    def update_grouping(self,epoch):
        self.model.eval();pool=self.pool;ids=[self.train_set.records[i]["id"] for i in pool];n=len(pool);matrix=np.full((n,n),-np.inf,dtype=np.float32)
        for i,idx in enumerate(tqdm(pool,desc="similarity",leave=False)):
            current=torch.from_numpy(self.train_set.load(idx)[0][None]).to(self.device)
            for start in range(0,n,self.cfg.batch_size):
                subset=pool[start:start+self.cfg.batch_size];rows=[self.train_set.load(j)[:2] for j in subset]
                ro=torch.from_numpy(np.stack([r[0] for r in rows])).to(self.device);re=torch.from_numpy(np.stack([r[1] for r in rows])).to(self.device)
                current_rep=current.expand(len(subset),-1,-1,-1);_,_,scores=self.model.pair(current_rep,ro,re);matrix[i,start:start+len(subset)]=scores.cpu().numpy()
            matrix[i,i]=-np.inf
        payload=top_reference_payload(ids,matrix,self.cfg.num_reference);previous=self.last_top;self.last_top=payload
        lookup={image_id:index for index,image_id in enumerate(ids)};self.references={}
        for source_idx,record in enumerate(self.train_set.records):
            if record["id"] in lookup:self.references[source_idx]=[pool[lookup[v[0]]] for v in payload[record["id"]]]
        np.save(Path(self.cfg.paths()["similarity"])/f"similarity_epoch_{epoch:03d}.npy",matrix);save_json(Path(self.cfg.paths()["similarity"])/f"top_reference_epoch_{epoch:03d}.json",payload)
        stability=grouping_stability(previous,payload);info={"average_similarity":float(matrix[np.isfinite(matrix)].mean()),"grouping_stability":stability,"example":next(iter(payload.items()))};save_json(Path(self.cfg.paths()["similarity"])/f"grouping_epoch_{epoch:03d}.json",info);return info
    def checkpoint(self,epoch,best,tag):
        state={"model_state":self.model.state_dict(),"optimizer_state":self.optimizer.state_dict(),"epoch":epoch,"best_psnr":best,"temperature":self.temperature,"references":self.references,"config":self.cfg.to_dict()};torch.save(state,Path(self.cfg.paths()["checkpoints"])/tag)
