import logging

import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


logger = logging.getLogger("ova")


def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
        logger.info(f"CUDA available. Using {device}")
    else:
        device = "cpu"
        logger.warning("CUDA not available. Falling back to CPU (this will be slower)")
    return device