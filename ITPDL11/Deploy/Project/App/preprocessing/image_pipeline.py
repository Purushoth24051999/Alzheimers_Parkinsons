"""
Image Pipeline Utilities
========================
Helper functions for loading, transforming, and encoding MRI images
at the boundary between the filesystem, the noise-removal stage,
and the Keras LeNet model.
"""

import base64
import io

import numpy as np
from PIL import Image, ImageOps


def load_image_for_preview(image_path: str) -> np.ndarray:
    """
    Load an image from disk as a uint8 RGB numpy array.

    Preserves the original resolution so that the preview window shows
    the image at its natural size. Resizing for the model happens later
    in prepare_for_model().

    Args:
        image_path: Absolute or relative path to a .jpg / .png file.

    Returns:
        uint8 numpy array of shape (H, W, 3) in RGB colour order.
    """
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img)


def prepare_for_model(image_array: np.ndarray) -> np.ndarray:
    """
    Resize and normalise an image for the LeNet Keras model.

    The model was trained on 224×224 RGB images normalised to [-1, 1].
    This function replicates the exact same preprocessing used in the
    original Deploy_8 view so that the model receives identical input
    regardless of which noise filter was applied.

    Args:
        image_array: uint8 RGB numpy array of any size (H, W, 3).

    Returns:
        float32 numpy array of shape (1, 224, 224, 3) with values in [-1, 1],
        ready to pass to keras_model.predict().
    """
    # Convert back to PIL for high-quality resampling
    img = Image.fromarray(image_array.astype(np.uint8))

    # Fit (crop + resize) to 224×224 — matches ImageOps.fit used in the original view
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.LANCZOS)

    img_array = np.asarray(img).astype(np.float32)

    # Normalise to [-1, 1] — identical to the original pipeline
    normalised = (img_array / 127.0) - 1.0

    # Add batch dimension
    return np.expand_dims(normalised, axis=0)


def image_to_base64(image_array: np.ndarray) -> str:
    """
    Encode a numpy image array as a base64 PNG data-URI string.

    Used to embed both the original and denoised images directly into
    the preview HTML/JSON response without writing temporary files to disk.

    Args:
        image_array: uint8 numpy array of shape (H, W, 3).

    Returns:
        A string like "data:image/png;base64,iVBORw0..." suitable for use
        as an <img src="..."> value in HTML or a JSON response field.
    """
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
