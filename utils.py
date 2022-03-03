import logging
import os
import torch
from time import time
import json

def get_logger(name, out_file_name=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    if out_file_name:
        file_handler = logging.FileHandler(out_file_name, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def output_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

class Timer:
    def __init__(self, logger):
        self.logger = logger
        self.t = time()
    
    def log(self):
        now = time()
        self.logger.info(f"elapsed time:{now - self.t} s")
        self.t = now

def try_create_dir(path):
    os.system(f"mkdir -p {path}")

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        idxs_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[idxs_to_remove] = filter_value
    if top_p > 0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cummulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_idx_to_remove = cummulative_probs > top_p
        sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
        sorted_idx_to_remove[..., 0] = 0

        idxs_to_remove = sorted_idx[sorted_idx_to_remove]
        logits[idxs_to_remove] = filter_value
    idxs_to_remove = logits < threshold
    logits[idxs_to_remove] = filter_value
    # print(logits.size())
    return logits