import cv2, numpy as np, random

def apply(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    center = (random.randint(0, w), random.randint(0, h//2))
    max_rad = int(0.6 * max(h, w))
    for r in range(max_rad, 0, -max_rad//10):
        alpha = max(0, 1 - r/max_rad)
        cv2.circle(overlay, center, r, (255,255,255), -1)
        img = cv2.addWeighted(img, 1, overlay, alpha*0.1, 0)
    return img