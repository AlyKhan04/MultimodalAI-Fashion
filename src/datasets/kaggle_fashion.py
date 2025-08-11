from pathlib import Path
import pandas as pd

def load_styles(styles_path: str):
    p = Path(styles_path)
    if not p.exists():
        raise FileNotFoundError(f"{styles_path} not found.")
    return pd.read_csv(p)