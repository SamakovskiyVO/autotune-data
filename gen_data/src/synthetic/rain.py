"""Adds rain streaks & droplets."""
import cv2, numpy as np, random

def _make_streaks(h, w):
    rain_layer = np.zeros((h, w, 3), np.uint8)
    n_drops = random.randint(600, 1000)
    for _ in range(n_drops):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        length = random.randint(15, 25)
        thickness = random.randint(1, 2)
        cv2.line(rain_layer, (x, y), (x, y+length), (255,255,255), thickness)
    rain_layer = cv2.blur(rain_layer, (3,3))
    return rain_layer

def apply(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    rain = _make_streaks(h, w)
    intensity = random.uniform(0.2, 0.4)
    blended = cv2.addWeighted(img, 1-intensity, rain, intensity, 0)
    return blended