import cv2, numpy as np, random
def apply(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    alpha = random.uniform(0.05, 0.2)
    fog = np.full((h, w, 3), 255, np.uint8)
    blur_k = random.choice([91, 101, 111])
    fog = cv2.GaussianBlur(fog, (blur_k, blur_k), 0)
    return cv2.addWeighted(img, 1-alpha, fog, alpha, 0)
