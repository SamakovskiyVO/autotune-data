import cv2, numpy as np, random, pathlib, itertools, shutil
from collections import defaultdict
import yaml
from . import utils, augmentations as aug, diffusion
from .synthetic import fog, rain, snow, glare, darkness, dirt  # noqa

RAW_ROOT = pathlib.Path("raw")
SYNTH_MODULES = {
    "fog": fog, "rain": rain, "snow": snow,
    "glare": glare, "darkness": darkness, "dirt": dirt,
}

def build_dataset(cfg: dict, log):
    utils.fix_seed(cfg["seed"])
    out_dir = pathlib.Path(cfg["output_dir"]).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # --- офлайн диффузия ---
    if "diffusion" in cfg:
        _run_diffusion(cfg, out_dir, log)

    summary = defaultdict(int)
    for cls, opts in cfg["classes"].items():
        target = opts["target"]
        synth_list = opts.get("synthetic", [])
        src_paths = list((RAW_ROOT/cls).glob("**/*.*"))
        log.info(f"{cls}: {len(src_paths)} raw → {target}")
        cycle = itertools.cycle(src_paths)
        cls_dir = out_dir/cls; cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(target):
            img_p = next(cycle)
            img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
            img = aug.apply_common(img, cfg["augmentations"]["common"])
            for eff in synth_list:
                img = SYNTH_MODULES[eff].apply(img)
            cv2.imwrite(str(cls_dir/f"{cls}_{i:05}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            summary[cls] += 1
    log.info("Done: " + ", ".join(f"{k}={v}" for k, v in summary.items()))

def _run_diffusion(cfg, out_dir, log):
    diff_cfg = cfg["diffusion"]
    prompts = []
    for cls, opts in cfg["classes"].items():
        p_list = opts.get("diffusion_prompts", [])
        prompts.extend(p_list)
    if not prompts:
        log.warning("No diffusion prompts specified"); return
    log.info(f"Generating {len(prompts)} images via diffusion")

    engine = diff_cfg["engine"]
    batch_size = diff_cfg["batch_size"]
    num_steps = diff_cfg["num_steps"]
    neg = diff_cfg["negative_prompt"]

    diff_out = out_dir / "_diffusion"
    diff_out.mkdir(exist_ok=True)

    if engine == "sdxl":
        diffusion.run_sdxl(prompts, diff_out, batch_size, num_steps, neg)
    elif engine == "yandex":
        diffusion.run_yandex(prompts, diff_out,
                             diff_cfg["yandex"]["api_key"],
                             diff_cfg["yandex"]["folder_id"],
                             batch_size, num_steps, neg)
    else:
        log.error(f"Unknown diffusion engine {engine}")
