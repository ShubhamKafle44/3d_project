import logging
import numpy as np
from PIL import Image
import config

logger = logging.getLogger(__name__)

def load_background() -> np.ndarray:
    try:
        img = Image.open(config.BACKGROUND_PATH).resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE)
        )
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]          # drop alpha channel if present
        return arr.astype(np.float32) / 255.0
    except FileNotFoundError:
        logger.warning("'%s' not found — using grey fallback.", config.BACKGROUND_PATH)
        return np.full(
            (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), 0.2, dtype=np.float32
        )
