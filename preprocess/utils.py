import torch
import random
import numpy as np
from transformers import set_seed
from deepspeed.accelerator import get_accelerator

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)