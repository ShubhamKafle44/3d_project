import logging
import numpy as np
from PIL import Image
from config import BACKGROUND_PATH, IMAGE_SIZE

logger = logging.getLogger(__name__)

def load_background() -> np.ndarray:
    try:
        img = Image.open(BACKGROUND_PATH).resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(img)

        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        return arr / 255.0

    except FileNotFoundError:
        logger.warning("'%s' not found — using grey fallback.", BACKGROUND_PATH)
        return np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 0.2, dtype=np.float32)