import os
import random
import shutil
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent   # project folder
SRC = BASE_DIR / "data" / "raw"
DEST = BASE_DIR / "data"

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

classes = os.listdir(SRC)

for cls in classes:
    files = os.listdir(os.path.join(SRC, cls))
    random.shuffle(files)

    n = len(files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, split_files in splits.items():
        out_dir = os.path.join(DEST, split, cls)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for f in split_files:
            src_path = os.path.join(SRC, cls, f)
            dst_path = os.path.join(out_dir, f)
            shutil.copy2(src_path, dst_path)

print("Dataset split complete.")