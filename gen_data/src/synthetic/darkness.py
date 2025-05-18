import cv2, numpy as np, random

def apply(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), int(0.9*max(h,w)), 255, -1)
    mask = cv2.GaussianBlur(mask, (151,151), 0)
    factor = random.uniform(0.4, 0.6)
    dark = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    dark[:,:,2] = (dark[:,:,2].astype(np.float32)*factor).astype(np.uint8)
    dark = cv2.cvtColor(dark, cv2.COLOR_HSV2RGB)
    return np.where(mask[...,None]==255, dark, img)