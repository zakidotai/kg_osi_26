"""Download official checkpoints/preprocessing assets into the ignored model cache."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "data/local/embedding_models"
MAT2VEC_REVISION = "7ca8961c09ad9995694a41edfd80da6c4ac7e307"
MATSCIBERT_CODE_REVISION = "564750d9a8be0c058ee20f41def9c3de3718e079"
MATSCIBERT_REVISION = "ced9d8f5f208712c4a90f98a246fe32155b29995"
MATSCIBERT_ID = "m3rg-iitd/matscibert"

MAT2VEC_RAW = f"https://raw.githubusercontent.com/materialsintelligence/mat2vec/{MAT2VEC_REVISION}"
MATSCIBERT_RAW = f"https://raw.githubusercontent.com/M3RG-IITD/MatSciBERT/{MATSCIBERT_CODE_REVISION}"
ASSETS = {
    **{
        f"mat2vec/{name}": f"https://storage.googleapis.com/mat2vec/{name}"
        for name in (
            "pretrained_embeddings", "pretrained_embeddings.wv.vectors.npy",
            "pretrained_embeddings.trainables.syn1neg.npy",
        )
    },
    "mat2vec/process.py": f"{MAT2VEC_RAW}/mat2vec/processing/process.py",
    "mat2vec/phraser.pkl": f"{MAT2VEC_RAW}/mat2vec/processing/models/phraser.pkl",
    "mat2vec/LICENSE": f"{MAT2VEC_RAW}/LICENSE",
    "mat2vec/punkt_chem-1.0.pickle": "http://data.chemdataextractor.org/models/punkt_chem-1.0.pickle",
    "mat2vec/clusters_chem1500-1.0.pickle": "http://data.chemdataextractor.org/models/clusters_chem1500-1.0.pickle",
    "matscibert/normalize_text.py": f"{MATSCIBERT_RAW}/normalize_text.py",
    "matscibert/vocab_mappings.txt": f"{MATSCIBERT_RAW}/vocab_mappings.txt",
    "matscibert/LICENSE": f"{MATSCIBERT_RAW}/LICENSE",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_models(model_dir=DEFAULT_MODEL_DIR):
    """Explicit setup step; inference uses only the downloaded local assets."""
    import requests
    from huggingface_hub import snapshot_download

    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous_files = previous.get("files", {})
    lock = Lock()

    def download(item):
        relative, url = item
        target = model_dir / relative
        previous_entry = previous_files.get(relative, {})
        if target.is_file() and previous_entry.get("source") == url and previous_entry.get("sha256") == sha256_file(target):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        print(f"Downloading {relative}", flush=True)
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    handle.write(block)
        partial.replace(target)
        receipt = {"source": url, "sha256": sha256_file(target), "bytes": target.stat().st_size}
        with lock:
            previous_files[relative] = receipt
            # Retain successful files if another download is interrupted.
            receipt_path = manifest_path.with_suffix(".partial")
            receipt_path.write_text(json.dumps({"files": previous_files}, indent=2) + "\n")
            receipt_path.replace(manifest_path)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, ASSETS.items()))
    print("Downloading MatSciBERT checkpoint", flush=True)
    snapshot_download(
        MATSCIBERT_ID,
        revision=MATSCIBERT_REVISION,
        local_dir=model_dir / "matscibert",
        cache_dir=model_dir / "hf_cache",
        allow_patterns=[
            "config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "model.safetensors",
        ],
    )
    files = {}
    for directory in ("mat2vec", "matscibert"):
        for path in sorted((model_dir / directory).iterdir()):
            if path.is_file() and not path.name.endswith(".partial"):
                relative = str(path.relative_to(model_dir))
                files[relative] = {
                    "sha256": sha256_file(path), "bytes": path.stat().st_size,
                    "source": ASSETS.get(relative, f"{MATSCIBERT_ID}@{MATSCIBERT_REVISION}"),
                }
    manifest = {
        "mat2vec_revision": MAT2VEC_REVISION,
        "matscibert_code_revision": MATSCIBERT_CODE_REVISION,
        "matscibert_checkpoint": f"{MATSCIBERT_ID}@{MATSCIBERT_REVISION}",
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Models ready: {model_dir}", flush=True)
