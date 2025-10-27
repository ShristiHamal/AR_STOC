import numpy as np
from pathlib import Path
from PIL import Image

def save_npy_as_images(image_npy_path, mask_npy_path, out_dir):
    images = np.load(image_npy_path)
    masks = np.load(mask_npy_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for i in range(len(images)):
        Image.fromarray(images[i]).save(out_dir / f"img_{i}.png")
        Image.fromarray(masks[i]).save(out_dir / f"mask_{i}.png")
