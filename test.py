"""Validate a checkpoint using references from validation and reference groups."""

from pathlib import Path

# 常用参数：修改后直接运行 python test.py；命令行参数可覆盖这里的值。
# 组编号从 1 开始，包含起止组，每组包含该原图的全部专家图对。
VAL_START = 250  # 验证从第几组开始
VAL_END = 260    # 验证到第几组结束
REF_START = 1    # 参考图区间从第几组开始
REF_END = 500    # 参考图区间到第几组结束
# 参考候选池 = 验证区间 ∪ 参考区间；排除与当前原图同源的图对。
CHECKPOINT = "outputs/checkpoints/best_psnr.pt"
OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")  # 图片保存到此目录下的 visuals/test/

import argparse
import logging

import torch

from config import Config
from dataset import FiveKDataset, discover_records
from evaluator import save_visualizations
from models import LatentRetouchModel
from trainer import Trainer
from utils import seed_everything


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--resume", default=CHECKPOINT)
    parser.add_argument("--val-start", type=int, default=VAL_START)
    parser.add_argument("--val-end", type=int, default=VAL_END)
    parser.add_argument("--ref-start", "--reference-start", type=int, default=REF_START)
    parser.add_argument("--ref-end", "--reference-end", type=int, default=REF_END)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--device")
    for name in ("batch-size", "num-workers", "image-size", "num-reference"):
        parser.add_argument(f"--{name}", type=int)
    return parser.parse_args()


def select_ranges(records, val_range, ref_range):
    """Use the same stable, 1-based original-image numbering as make_splits."""
    source_ids = list(dict.fromkeys(record["id"].rsplit("_", 1)[0] for record in records))

    def selected(bounds, name):
        start, end = bounds
        if not 1 <= start <= end <= len(source_ids):
            raise ValueError(f"Invalid {name} range {bounds}; valid group range is 1..{len(source_ids)}")
        return set(source_ids[start - 1:end])

    validation = selected(val_range, "validation")
    candidates = validation | selected(ref_range, "reference")
    if len(candidates) < 2:
        raise ValueError("At least two original-image groups are needed to exclude self references")
    val_ids = [r["id"] for r in records if r["id"].rsplit("_", 1)[0] in validation]
    ref_ids = [r["id"] for r in records if r["id"].rsplit("_", 1)[0] in candidates]
    return val_ids, ref_ids


def main():
    args = arguments()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = Config()
    for key, value in state.get("config", {}).items():
        if hasattr(cfg, key) and key != "device":
            setattr(cfg, key, value)
    for key in ("data_dir", "output_dir", "device", "batch_size", "num_workers",
                "image_size", "num_reference"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    for key in ("batch_size", "image_size", "num_reference"):
        if getattr(cfg, key) < 1:
            raise ValueError(f"{key} must be positive")
    if cfg.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    records = discover_records(cfg.data_dir)
    val_ids, ref_ids = select_ranges(records, (args.val_start, args.val_end), (args.ref_start, args.ref_end))
    val = FiveKDataset(records, val_ids, cfg.image_size)
    references = FiveKDataset(records, ref_ids, cfg.image_size)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler()])
    log = logging.getLogger("fivek")
    seed_everything(cfg.seed)
    model = LatentRetouchModel(
        cfg.attention_dim, cfg.edit_dim, cfg.cross_attention_heads, cfg.attention_spatial_size,
        cfg.encoder_base_dim, cfg.encoder_mid_dim, cfg.enhancer_base_dim, cfg.enhancer_mid_dim,
        ablation_uniform_weight=cfg.ablation_uniform_weight, ablation_cnn_fusion=cfg.ablation_cnn_fusion)
    model.load_state_dict(state["model_state"])
    trainer = Trainer(model, cfg, references, val)
    trainer.temperature = state.get("temperature", cfg.temperature)
    epoch = state.get("epoch", 0)
    log.info("Loaded %s; validation=%d pairs; reference pool=%d pairs; temperature=%g",
             args.checkpoint, len(val), len(references), trainer.temperature)
    trainer.update_grouping(val, "val", epoch, references, "train", save_results=False)
    metrics = trainer.run_epoch(val, "val", epoch, False)
    log.info("Final average: loss=%.6f PSNR=%.3f SSIM=%.4f count=%d",
             metrics["loss"], metrics["psnr"], metrics["ssim"], len(val))
    save_visualizations(model, val, trainer, "test", max_images=len(val), split="val")
    log.info("Results saved to %s", (Path(cfg.paths()["visuals"]) / "test").resolve())


if __name__ == "__main__":
    main()
