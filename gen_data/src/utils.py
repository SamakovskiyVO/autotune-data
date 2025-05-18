from loguru import logger
import random, numpy as np, torch

def make_logger():
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))
    return logger

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
