import albumentations as A
import cv2, numpy as np

_common = {
    "Resize": lambda p: A.Resize(**p),
    "RandomBrightnessContrast": lambda p: A.RandomBrightnessContrast(**p),
    "GaussianBlur": lambda p: A.GaussianBlur(**p),
}

def apply_common(img: np.ndarray, cfg: list):
    transforms = [_common[name](params) for name, params in ((k, v) for d in cfg for k, v in d.items())]
    pipe = A.Compose(transforms)
    return pipe(image=img)["image"]
