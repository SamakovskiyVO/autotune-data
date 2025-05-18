from src.generate_dataset import build_dataset
from src.utils import make_logger
DEF_CFG = {
    "seed": 1,
    "output_dir": "tmp_out",
    "classes": {c: {"target": 3} for c in
                ["good", "rain", "fog", "snow", "glare", "dark", "dirt"]},
    "augmentations": {"common": []},
}
def test_build():
    build_dataset(DEF_CFG, make_logger())
