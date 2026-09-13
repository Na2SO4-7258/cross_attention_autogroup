# Latent FiveK Retouching

End-to-end PyTorch implementation of spatial cross-attention reference selection and edit transfer for MIT-Adobe FiveK. It does not use expert identity as a grouping label, KMeans, diffusion, or reinforcement learning. The downloader retrieves public FiveK manifests and source images, then converts them to compact RGB PNG files in `0001` through `5000` directories. Set `DOWNLOAD_MAX_SIDE` (edge size) and `DOWNLOAD_UP_TO` (last image number) at the top of `download_fivek.py`; they default to `500` and `5000`.

```powershell
python -m pip install -r requirements.txt
python main.py --data-dir data/fivek --epochs 30 --mode dynamic_attention
```

The loader discovers common `original`/`input` and `expert_A`...`expert_E`/`tiff16_a`...`tiff16_e` layouts, preserves image-level splits, decodes TIFF/JPEG/PNG and uses `rawpy` for DNG. Outputs include checkpoints, similarity matrices, top-reference JSON, grouping stability, training history, metrics, and original/reference/output/expert panels.

Modes: `self_reference` (sanity check only), `random_reference`, `fixed_reference`, and `dynamic_attention`.
