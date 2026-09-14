# Latent FiveK Retouching

End-to-end PyTorch implementation of spatial cross-attention reference selection and edit transfer for MIT-Adobe FiveK. It does not use expert identity as a grouping label, KMeans, diffusion, or reinforcement learning. The downloader retrieves public FiveK manifests and source images, then converts them to compact RGB PNG files in `0001` through `5000` directories. Set `DOWNLOAD_MAX_SIDE` (edge size) and `DOWNLOAD_UP_TO` (last image number) at the top of `download_fivek.py`; they default to `500` and `5000`.

```powershell
python -m pip install -r requirements.txt
python main.py --data-dir data/fivek --epochs 30 --mode dynamic_attention
```

The loader discovers common `original`/`input` and `expert_A`...`expert_E`/`tiff16_a`...`tiff16_e` layouts, decodes TIFF/JPEG/PNG and uses `rawpy` for DNG. The atomic sample is an `(original, expert)` pair: each pair independently selects a `(reference original, reference expert)` pair from a different original image. In `dynamic_attention` mode, at the start of every epoch, cross-attention scores every pair against valid pairs in the same split, then training starts. Validation and test selection are likewise made only within their own split; there is no truncated reference pool or random fallback.

To select explicit original-image-number ranges (1-based, inclusive, in the stable `discover_records` order), use. Every expert pair for each selected original image is included:

```powershell
python main.py --train-start 1 --train-end 16000 --val-start 16001 --val-end 18000
```

All remaining pairs become the test split. The saved preview layout is: original, reference original, reference expert, output, target.

Modes: `self_reference` (sanity check only), `random_reference`, `fixed_reference`, and `dynamic_attention`.
