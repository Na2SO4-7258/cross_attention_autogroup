"""MIT-Adobe FiveK retrieval and local compact-RGB preparation."""
DOWNLOAD_MAX_SIDE=500
DOWNLOAD_UP_TO=500
DOWNLOAD_WORKERS=8
MANIFEST_URLS=(
    "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/resolve/main/training.json",
    "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/resolve/main/validation.json",
    "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/resolve/main/testing.json",
)
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from io import BytesIO
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import quote,urlsplit,urlunsplit
from PIL import Image,ImageOps
from tqdm import tqdm

IMAGE_EXT={".jpg",".jpeg",".png",".tif",".tiff",".dng"}

def _safe_url(url):
    parsed=urlsplit(url)
    return urlunsplit((parsed.scheme,parsed.netloc,quote(parsed.path,safe="/%:@"),quote(parsed.query,safe="=&%:/?"),parsed.fragment))
def _response(url):return urlopen(Request(_safe_url(url),headers={"User-Agent":"FiveK-research-loader/2.0"}),timeout=90)
def _bytes(url):
    with _response(url) as response:return response.read()
def _resize(image):
    image=ImageOps.exif_transpose(image).convert("RGB");scale=min(1.0,DOWNLOAD_MAX_SIDE/max(image.size))
    if scale<1:image=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
    return image
def _dng_to_image(payload):
    try:import rawpy
    except ImportError as error:raise RuntimeError("rawpy is required to convert FiveK DNG originals. Run: python -m pip install rawpy") from error
    handle=tempfile.NamedTemporaryFile(suffix=".dng",delete=False);temp=Path(handle.name);handle.close()
    try:
        temp.write_bytes(payload)
        with rawpy.imread(str(temp)) as raw:return Image.fromarray(raw.postprocess(use_camera_wb=True,output_bps=8))
    finally:
        if temp.exists():temp.unlink()
def _prepare_one(url,destination):
    if destination.exists():return
    payload=_bytes(url);image=_dng_to_image(payload) if url.lower().endswith(".dng") else Image.open(BytesIO(payload));output=_resize(image);destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix(".tmp.png");output.save(temporary,"PNG");temporary.replace(destination)
def _manifests():
    result={}
    for url in MANIFEST_URLS:result.update(json.loads(_bytes(url).decode("utf-8")))
    return result
def _jobs(root):
    for image_id,metadata in _manifests().items():
        numeric_id=int(metadata.get("id",str(image_id).lstrip("a")[:4]))
        if numeric_id>DOWNLOAD_UP_TO:continue
        urls=metadata["urls"];folder=root/f"{numeric_id:04d}";yield urls["dng"],folder/"original.png"
        for expert,url in urls["tiff16"].items():yield url,folder/f"expert_{expert.upper()}.png"
def ensure_fivek(data_dir,url=""):
    root=Path(data_dir)
    marker=root/f".fivek_complete_{DOWNLOAD_UP_TO:04d}"
    if marker.exists():return root
    if url:raise RuntimeError("Archive URLs are no longer used because the official FiveK archive endpoint is not stable. Remove --download-url to use the supported public manifest downloader.")
    root.mkdir(parents=True,exist_ok=True);jobs=list(_jobs(root));failures=[]
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures={pool.submit(_prepare_one,source,target):(source,target) for source,target in jobs}
        for future in tqdm(as_completed(futures),total=len(futures),desc=f"Downloading FiveK 0001-{DOWNLOAD_UP_TO:04d}",unit="file"):
            source,target=futures[future]
            try:future.result()
            except Exception as error:failures.append(f"{target.name}: {error}")
    if failures:raise RuntimeError(f"FiveK download/preparation failed for {len(failures)} files. First failure: {failures[0]}")
    marker.write_text("prepared",encoding="utf-8")
    return root
