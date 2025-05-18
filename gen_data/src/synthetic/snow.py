import cv2, numpy as np, random

def apply(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    snow = img.copy()
    n_flakes = random.randint(300, 500)
    for _ in range(n_flakes):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        r = random.randint(1, 3)
        cv2.circle(snow, (x, y), r, (255,255,255), -1)
    snow = cv2.GaussianBlur(snow, (5,5), 0)
    alpha = random.uniform(0.15, 0.3)
    return cv2.addWeighted(img, 1-alpha, snow, alpha, 0)