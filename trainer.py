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
        self.model=model.to(config.device);self.cfg=config;self.train_set=train_set;self.val_set=val_set;self.test_set=test_set;self.device=torch.device(config.device);self.optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay);self.references={"train":{},"val":{},"test":{}};self.reference_sources={"train":"train","val":"val","test":"test"};self.last_top={"train":{},"val":{},"test":{}};self.temperature=config.temperature
        self.log=logging.getLogger("fivek")
    def loader(self,dataset,shuffle):return DataLoader(dataset,batch_size=self.cfg.batch_size,shuffle=shuffle,num_workers=self.cfg.num_workers,pin_memory=self.device.type=="cuda")
    def _dataset(self,split):
        if split=="train":return self.train_set
        if split=="val":return self.val_set
        if split=="test" and self.test_set is not None:return self.test_set
        raise ValueError(f"Unknown or unavailable split: {split}")
    @staticmethod
    def _source_id(dataset,index):return dataset.records[index]["id"].rsplit("_",1)[0]
    def _candidate_indices(self,index,split):
        choices=self.references.get(split,{}).get(index)
        if choices is None:raise RuntimeError(f"No dynamic references prepared for {split} sample {index}")
        return choices
    def _references(self,indices,split):
        source=self._dataset(self.reference_sources[split])
        refs_o=[];refs_e=[]
        for index in indices:
            choices=self._candidate_indices(int(index),split)
            rows=[source.load(i)[:2] for i in choices]
            while len(rows)<self.cfg.num_reference:rows.append(rows[-1])
            refs_o.append(np.stack([x[0] for x in rows]));refs_e.append(np.stack([x[1] for x in rows]))
        return torch.from_numpy(np.stack(refs_o)).to(self.device),torch.from_numpy(np.stack(refs_e)).to(self.device)
    def run_epoch(self,dataset,split,epoch,train):
        self.model.train(train);values=[];psnrs=[];ssims=[]
        for batch in tqdm(self.loader(dataset,train),desc=f"{split} {epoch}",leave=False):
            original=batch["original"].to(self.device);target=batch["expert"].to(self.device);refs_o,refs_e=self._references(batch["index"].tolist(),split)
            with torch.set_grad_enabled(train):
                output,_,_=self.model(original,target,refs_o,refs_e,self.temperature);l1=F.l1_loss(output,target);ssim=ssim_per_image(output,target).mean();loss=l1+self.cfg.lambda_ssim*(1-ssim)+self.cfg.residual_weight*(output-original).abs().mean()
                if train:self.optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);self.optimizer.step()
            values.append(loss.item());psnrs.extend(psnr_per_image(output.detach(),target).cpu().tolist());ssims.extend(ssim_per_image(output.detach(),target).cpu().tolist())
        return {"loss":float(np.mean(values)),"psnr":float(np.mean(psnrs)),"ssim":float(np.mean(ssims))}
    @torch.no_grad()
    def update_grouping(self,dataset,split,epoch,candidate_dataset=None,candidate_split=None,save_results=True):
        """Choose references for every pair; candidates may come from another split."""
        candidate_dataset=dataset if candidate_dataset is None else candidate_dataset;candidate_split=split if candidate_split is None else candidate_split
        if not len(candidate_dataset):raise RuntimeError(f"{candidate_split} has no reference pairs")
        if self.cfg.num_reference<1:raise ValueError("num_reference must be positive")
        if self.cfg.ablation_random_reference:
            payload={};references={}
            for index,record in enumerate(dataset.records):
                candidates=[j for j in range(len(candidate_dataset)) if self._source_id(candidate_dataset,j)!=self._source_id(dataset,index)]
                if not candidates:raise RuntimeError(f"No eligible references for {split} sample {record['id']}")
                choices=np.random.choice(candidates,size=min(self.cfg.num_reference,len(candidates)),replace=False).tolist();references[index]=choices;payload[record["id"]]=[[candidate_dataset.records[j]["id"],None] for j in choices]
            previous=self.last_top[split];self.last_top[split]=payload;self.references[split]=references;self.reference_sources[split]=candidate_split
            info={"selection":"random","average_distance":None,"grouping_stability":grouping_stability(previous,payload),"example":next(iter(payload.items()),None)}
            if save_results:
                folder=Path(self.cfg.paths()["similarity"]);save_json(folder/f"{split}_top_reference_epoch_{epoch:03d}.json",payload);save_json(folder/f"{split}_grouping_epoch_{epoch:03d}.json",info)
            return info
        self.model.eval();indices=list(range(len(dataset)));candidate_indices=list(range(len(candidate_dataset)));ids=[dataset.records[i]["id"] for i in indices];candidate_ids=[candidate_dataset.records[i]["id"] for i in candidate_indices]
        def embeddings(source,source_indices,label):
            vectors=[]
            for start in tqdm(range(0,len(source_indices),self.cfg.batch_size),desc=f"{label} embedding",leave=False):
                subset=source_indices[start:start+self.cfg.batch_size];rows=[source.load(j)[:2] for j in subset]
                original=torch.from_numpy(np.stack([row[0] for row in rows])).to(self.device);expert=torch.from_numpy(np.stack([row[1] for row in rows])).to(self.device)
                vectors.append(self.model.grouping_embedding(original,expert))
            return torch.cat(vectors).cpu().numpy()
        query_vectors=embeddings(dataset,indices,split)
        candidate_vectors=query_vectors if dataset is candidate_dataset else embeddings(candidate_dataset,candidate_indices,f"{candidate_split} reference")
        # 每个组对均由原有 9 通道编码器的逐通道全局平均池化向量表示；欧氏距离越小，参考越相近。
        matrix=np.linalg.norm(query_vectors[:,None,:]-candidate_vectors[None,:,:],axis=2).astype(np.float32)
        for i,idx in enumerate(indices):
            current_source=self._source_id(dataset,idx)
            for candidate_pos,candidate_idx in enumerate(candidate_indices):
                if self._source_id(candidate_dataset,candidate_idx)==current_source:matrix[i,candidate_pos]=np.inf
        payload={image_id:[[candidate_ids[j],float(matrix[i,j])] for j in np.argsort(matrix[i]) if np.isfinite(matrix[i,j])][:self.cfg.num_reference] for i,image_id in enumerate(ids)};previous=self.last_top[split];self.last_top[split]=payload
        lookup={image_id:index for index,image_id in enumerate(candidate_ids)};self.references[split]={};self.reference_sources[split]=candidate_split
        for source_idx,record in enumerate(dataset.records):
            self.references[split][source_idx]=[lookup[v[0]] for v in payload[record["id"]]]
        if save_results:
            np.save(Path(self.cfg.paths()["similarity"])/f"{split}_similarity_epoch_{epoch:03d}.npy",matrix);save_json(Path(self.cfg.paths()["similarity"])/f"{split}_top_reference_epoch_{epoch:03d}.json",payload)
        stability=grouping_stability(previous,payload);info={"average_distance":float(matrix[np.isfinite(matrix)].mean()),"grouping_stability":stability,"example":next(iter(payload.items()))}
        if save_results:save_json(Path(self.cfg.paths()["similarity"])/f"{split}_grouping_epoch_{epoch:03d}.json",info)
        return info
    def checkpoint(self,epoch,best,tag):
        state={"model_state":self.model.state_dict(),"optimizer_state":self.optimizer.state_dict(),"epoch":epoch,"best_psnr":best,"temperature":self.temperature,"references":self.references,"reference_sources":self.reference_sources,"config":self.cfg.to_dict()};torch.save(state,Path(self.cfg.paths()["checkpoints"])/tag)
