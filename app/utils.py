# app/utils.py
import json
import numpy as np
from PIL import Image

def read_img(path):
    """Load image as RGB."""
    return Image.open(path).convert("RGB")

def load_openpose_json(jpath):
    """Load COCO-format openpose keypoints from JSON."""
    with open(jpath, 'r') as f:
        data = json.load(f)
    if 'people' in data and len(data['people']) > 0:
        kp = data['people'][0].get('pose_keypoints_2d', [])
        kp = np.array(kp).reshape(-1,3)[:,:2]  # x,y only
        return kp
    return None

def torso_bbox_from_keypoints(kp):
    """Compute torso bbox from shoulders and hips."""
    if kp is None:
        return None
    idxs = [5,6,11,12]  # COCO left_shoulder, right_shoulder, left_hip, right_hip
    pts = kp[idxs]
    pts = pts[~np.isnan(pts).any(axis=1)]
    if pts.shape[0] == 0:
        return None
    x_min, y_min = int(np.min(pts[:,0])), int(np.min(pts[:,1]))
    x_max, y_max = int(np.max(pts[:,0])), int(np.max(pts[:,1]))
    w, h = x_max-x_min, y_max-y_min
    pad_w, pad_h = int(w*0.4), int(h*0.6)
    x1, x2 = max(0, x_min-pad_w), x_max+pad_w
    y1, y2 = max(0, y_min-pad_h), y_max+pad_h
    return (x1, y1, x2, y2)

def alpha_blend(base_img, cloth_img, cloth_mask, bbox):
    """Blend cloth onto person image using mask and bbox."""
    x1,y1,x2,y2 = bbox
    base = base_img.copy()
    w, h = x2-x1, y2-y1
    if w<=0 or h<=0:
        return base
    cloth_resized = cloth_img.resize((w,h), Image.BILINEAR)
    mask_resized = cloth_mask.resize((w,h), Image.NEAREST)
    base_crop = base.crop((x1,y1,x2,y2)).convert("RGBA")
    cloth_rgba = Image.new("RGBA", (w,h))
    cloth_rgba.paste(cloth_resized.convert("RGBA"), (0,0), mask_resized.convert("L"))
    composed = Image.alpha_composite(base_crop, cloth_rgba)
    base.paste(composed.convert("RGB"), (x1,y1))
    return base
