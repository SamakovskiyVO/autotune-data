"""
Генерация изображений из текстовых промптов:
* Stable-Diffusion XL – локально;
* Yandex Diffusion API – облачно.
"""
import pathlib, itertools, random, requests, base64, io
from typing import List
from loguru import logger

# --- Stable Diffusion XL (offline) ---
def run_sdxl(prompts: List[str], out_dir: pathlib.Path,
             batch_size: int = 4, num_steps: int = 25,
             negative_prompt: str = ""):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype="auto",
    ).to("cuda")

    for i, chunk in enumerate(chunks(prompts, batch_size)):
        logger.info(f"SDXL batch {i}")
        images = pipe(chunk,
                      negative_prompt=[negative_prompt]*len(chunk),
                      num_inference_steps=num_steps).images
        for p, img in zip(chunk, images):
            name = (out_dir/f"sdxl_{hash(p)%10**8}.jpg")
            img.save(name)

# --- Yandex Cloud Diffusion ---
def run_yandex(prompts: List[str], out_dir: pathlib.Path,
               api_key: str, folder_id: str,
               batch_size: int = 4, num_steps: int = 25,
               negative_prompt: str = ""):
    url = "https://vision.api.cloud.yandex.net/vision/v1/im.generate"
    headers = {"Authorization": f"Api-Key {api_key}"}

    for i, chunk in enumerate(chunks(prompts, batch_size)):
        logger.info(f"Yandex batch {i}")
        body = {
            "folderId": folder_id,
            "generationOptions": {
                "steps": num_steps,
                "negativePrompt": negative_prompt,
            },
            "requests": [{"text": p} for p in chunk]
        }
        resp = requests.post(url, headers=headers, json=body, timeout=300)
        resp.raise_for_status()
        for j, reply in enumerate(resp.json()["responses"]):
            img_bytes = base64.b64decode(reply["image"]["content"])
            (out_dir / f"ya_{hash(chunk[j])%10**8}.jpg").write_bytes(img_bytes)

# --- helpers ---
def chunks(seq, n):
    it = iter(seq)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch
