import logging

import numpy as np
from PIL import Image

from config import BACKGROUND_PATH, IMAGE_SIZE

# Setting up a logger so we can see what's happening if things go sideways
logger = logging.getLogger(__name__)


def load_background() -> np.ndarray:
    """Load background image, resize to IMAGE_SIZE, and normalise to [0, 1].

    Falls back to a solid dark-grey canvas when the file is missing.
    """
    try:
        # Open the image from the config path and force it into our required dimensions
        img = Image.open(BACKGROUND_PATH).resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert the pixel data into a math-friendly numpy array
        arr = np.array(img)
        
        # If the image has an Alpha (transparency) channel, we chop it off 
        # because we only need the standard RGB colors (Red, Green, Blue)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
            
        # Scale the pixel values from 0-255 down to a 0.0-1.0 range
        # This makes the math much easier for most image processing algorithms
        return arr / 255.0
        
    except FileNotFoundError:
        # If the file is missing, we don't want to crash the whole program
        # We'll log a warning and just generate a boring dark-grey square instead
        logger.warning("'%s' not found — using grey fallback.", BACKGROUND_PATH)
        
        # Creating a placeholder array filled with 0.2 (dark grey) for R, G, and B
        return np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 0.2, dtype=np.float32)