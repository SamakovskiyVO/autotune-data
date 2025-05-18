import cv2, numpy as np, random, pathlib
ASSETS = list((pathlib.Path(__file__).parent.parent/"assets"/"dirt_masks").glob("*.png"))

def _random_mask(h, w):
    if ASSETS:
        mask = cv2.imread(str(random.choice(ASSETS)), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (w, h))
    else:
        # fallback: generate blobs procedurally
        mask = np.zeros((h,w), np.uint8)
        for _ in range(random.randint(3,6)):
            x,y = random.randint(0,w-1), random.randint(0,h-1)
            r = random.randint(60,120)
            cv2.circle(mask, (x,y), r, 255, -1)
        mask = cv2.GaussianBlur(mask, (151,151), 0)
    return mask.astype(np.float32)/255.

def apply(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = _random_mask(h, w)
    dirt_color = np.full_like(img, (50,40,30))
    alpha = mask[...,None]*random.uniform(0.5,0.9)
    return (img*(1-alpha) + dirt_color*alpha).astype(np.uint8)