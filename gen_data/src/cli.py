import argparse, pathlib, yaml
from .generate_dataset import build_dataset
from .utils import make_logger
from rich.console import Console
from rich.table import Table
import random, cv2
import matplotlib.pyplot as plt

console = Console()

def _parse():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s_full = sub.add_parser("full")
    s_full.add_argument("--config", required=True, type=pathlib.Path)
    s_sample = sub.add_parser("sample")
    s_sample.add_argument("--n", type=int, default=8)
    s_sample.add_argument("--folder", default="output")
    return p.parse_args()

def preview_grid(folder: str, n: int):
    paths = list(pathlib.Path(folder).rglob("*.jpg"))
    random.shuffle(paths); paths = paths[:n]
    imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
    cols = 4
    plt.figure(figsize=(12, 3*(n//cols+1)))
    for i, im in enumerate(imgs, 1):
        plt.subplot(n//cols+1, cols, i); plt.axis("off"); plt.imshow(im)
        plt.title(paths[i-1].parent.name)
    plt.show()

if __name__ == "__main__":
    args = _parse()
    log = make_logger()
    if args.cmd == "full":
        cfg = yaml.safe_load(args.config.read_text())
        build_dataset(cfg, log)
    elif args.cmd == "sample":
        preview_grid(args.folder, args.n)
