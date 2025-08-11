import os, zipfile
from pathlib import Path

def ensure_kaggle_api():
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            os.chmod(kaggle_json, 0o600)
        except Exception:
            pass
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return False

def main():
    if not ensure_kaggle_api():
        print("Kaggle API credentials not found.")
        return

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    dataset = "paramaggarwal/fashion-product-images-dataset"
    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset} to {out_dir} ...")
    api.dataset_download_files(dataset, path=str(out_dir), unzip=False, quiet=False)

    for p in out_dir.glob("*.zip"):
        print(f"Unzipping {p.name} ...")
        with zipfile.ZipFile(p, 'r') as zf:
            zf.extractall(out_dir)
        p.unlink(missing_ok=True)

    for p in list(out_dir.rglob("*.zip")):
        print(f"Unzipping nested archive {p.relative_to(out_dir)} ...")
        with zipfile.ZipFile(p, 'r') as zf:
            zf.extractall(p.parent)
        p.unlink(missing_ok=True)

    print("Done.")

if __name__ == "__main__":
    main()