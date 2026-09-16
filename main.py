import argparse,logging
from pathlib import Path
import torch
from config import Config
from utils import seed_everything,ensure_dirs,save_json,save_training_curve
from download_fivek import ensure_fivek
from dataset import discover_records,make_splits,FiveKDataset
from models import LatentRetouchModel
from trainer import Trainer
from evaluator import evaluate,save_visualizations

def arguments():
    parser=argparse.ArgumentParser(description="Latent reference learning for MIT-Adobe FiveK retouching")
    parser.add_argument("--data-dir");parser.add_argument("--output-dir");parser.add_argument("--download-url");parser.add_argument("--epochs",type=int);parser.add_argument("--batch-size",type=int);parser.add_argument("--image-size",type=int);parser.add_argument("--num-reference",type=int);parser.add_argument("--num-workers",type=int);parser.add_argument("--group-update-interval",type=int);parser.add_argument("--attention-dim",type=int);parser.add_argument("--edit-dim",type=int);parser.add_argument("--cross-attention-heads",type=int);parser.add_argument("--attention-spatial-size",type=int);parser.add_argument("--train-start",type=int);parser.add_argument("--train-end",type=int);parser.add_argument("--val-start",type=int);parser.add_argument("--val-end",type=int);parser.add_argument("--resume");parser.add_argument("--evaluate-only",action="store_true");parser.add_argument("--visualize-val-each-epoch",action="store_true",default=None)
    return parser.parse_args()

def main():
    args=arguments();cfg=Config()
    for key,value in vars(args).items():
        if value is not None and hasattr(cfg,key.replace("epochs","num_epochs")):setattr(cfg,key.replace("epochs","num_epochs"),value)
    ensure_dirs(cfg);logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(Path(cfg.paths()["logs"])/"train.log",encoding="utf-8")]);log=logging.getLogger("fivek");seed_everything(cfg.seed)
    ensure_fivek(cfg.data_dir,cfg.download_url);records=discover_records(cfg.data_dir);splits=make_splits(records,cfg.output_dir,cfg.seed,(cfg.train_ratio,cfg.val_ratio),(cfg.train_start,cfg.train_end) if cfg.train_start is not None or cfg.train_end is not None else None,(cfg.val_start,cfg.val_end) if cfg.val_start is not None or cfg.val_end is not None else None);train=FiveKDataset(records,splits["train"],cfg.image_size);val=FiveKDataset(records,splits["val"],cfg.image_size);test=FiveKDataset(records,splits["test"],cfg.image_size)
    if not train or not val or not test:raise RuntimeError("FiveK split is too small; at least three matched image records are required.")
    model=LatentRetouchModel(cfg.attention_dim,cfg.edit_dim,cfg.cross_attention_heads,cfg.attention_spatial_size,cfg.encoder_base_dim,cfg.encoder_mid_dim,cfg.enhancer_base_dim,cfg.enhancer_mid_dim);trainer=Trainer(model,cfg,train,val,test);start=1;best=-float("inf")
    if args.resume:
        state=torch.load(args.resume,map_location=cfg.device);model.load_state_dict(state["model_state"]);trainer.optimizer.load_state_dict(state["optimizer_state"]);loaded_refs=state.get("references",{});trainer.references=loaded_refs if all(key in loaded_refs for key in ("train","val","test")) else {"train":loaded_refs,"val":{},"test":{}};trainer.reference_sources=state.get("reference_sources",trainer.reference_sources);trainer.temperature=state.get("temperature",cfg.temperature);start=state["epoch"]+1;best=state.get("best_psnr",best)
    else:
        best_path=Path(cfg.paths()["checkpoints"])/"best_psnr.pt"
        if best_path.exists():
            state=torch.load(best_path,map_location=cfg.device);model.load_state_dict(state["model_state"]);best=state.get("best_psnr",best);logging.getLogger("fivek").info("loaded best-PSNR model: %s (PSNR=%.3f)",best_path,best)
    if args.evaluate_only:
        result=evaluate(model,test,trainer,"test");log.info(result);return
    history=[]
    for epoch in range(start,cfg.num_epochs+1):
        trainer.temperature=max(cfg.min_temperature,cfg.temperature*(1-(epoch-1)/max(cfg.num_epochs,1)))
        row={"epoch":epoch,"temperature":trainer.temperature}
        if epoch==1 or epoch%cfg.group_update_interval==0:row["train_grouping"]=trainer.update_grouping(train,"train",epoch)
        train_metrics=trainer.run_epoch(train,"train",epoch,True)
        row["val_grouping"]=trainer.update_grouping(val,"val",epoch)
        val_metrics=trainer.run_epoch(val,"val",epoch,False);row.update({"train":train_metrics,"val":val_metrics});
        if cfg.visualize_val_each_epoch:save_visualizations(model,val,trainer,f"val_epoch_{epoch:03d}")
        trainer.checkpoint(epoch,best,f"epoch_{epoch:03d}.pt")
        if val_metrics["psnr"]>best:best=val_metrics["psnr"];trainer.checkpoint(epoch,best,"best_psnr.pt")
        trainer.checkpoint(epoch,best,"latest.pt");history.append(row);save_json(Path(cfg.paths()["logs"])/"history.json",history);save_training_curve(Path(cfg.paths()["logs"])/"training_loss.svg",history);log.info("epoch=%d loss=%.5f val_loss=%.5f psnr=%.3f ssim=%.4f lr=%.2e temp=%.3f",epoch,train_metrics["loss"],val_metrics["loss"],val_metrics["psnr"],val_metrics["ssim"],trainer.optimizer.param_groups[0]["lr"],trainer.temperature)
    state=torch.load(Path(cfg.paths()["checkpoints"])/"best_psnr.pt",map_location=cfg.device);model.load_state_dict(state["model_state"]);result=evaluate(model,test,trainer,"test");log.info("test=%s",result)

if __name__=="__main__":main()
