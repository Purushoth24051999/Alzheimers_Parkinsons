"""
MRI Noise Removal Filters
=========================
This module provides noise removal functions for brain MRI images.

MRI images suffer from several types of noise:
- Gaussian noise   : thermal/electronic noise from scanner hardware
- Salt-and-pepper  : random bright/dark pixel spikes from motion or signal dropout
- Rician noise     : dominant in magnitude MRI images (combination of Gaussian in real/imaginary channels)
- Speckle noise    : interference patterns common in gradient-echo sequences

Each filter below targets one or more of these noise types.
"""

import cv2
import numpy as np

# scikit-image is used for wavelet denoising; imported lazily to avoid hard crash
# if the library is not installed.
def _skimage_wavelet():
    try:
        from skimage.restoration import denoise_wavelet, estimate_sigma
        return denoise_wavelet, estimate_sigma
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for wavelet denoising. "
            "Install it with: pip install scikit-image"
        ) from exc


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def apply_gaussian_filter(image_array: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to the image.

    Target noise: Gaussian / thermal noise
    How it works: Convolves the image with a Gaussian kernel, averaging
    neighbouring pixels weighted by distance. Effective for high-frequency
    Gaussian noise but slightly blurs edges.

    Args:
        image_array: uint8 RGB numpy array of shape (H, W, 3).

    Returns:
        Filtered uint8 RGB numpy array of the same shape.
    """
    # Kernel size 5×5, sigma=0 lets OpenCV compute sigma from kernel size.
    return cv2.GaussianBlur(image_array, ksize=(5, 5), sigmaX=0)


def apply_median_filter(image_array: np.ndarray) -> np.ndarray:
    """
    Apply median filter to the image.

    Target noise: Salt-and-pepper noise, random motion artefacts
    How it works: Replaces each pixel with the median value in its
    neighbourhood. Very effective at removing isolated outlier pixels
    (salt-and-pepper) while preserving edges better than a Gaussian blur.

    Args:
        image_array: uint8 RGB numpy array of shape (H, W, 3).

    Returns:
        Filtered uint8 RGB numpy array of the same shape.
    """
    # Kernel size must be an odd integer; 5 is a good balance for MRI.
    return cv2.medianBlur(image_array, ksize=5)


def apply_bilateral_filter(image_array: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filter to the image.

    Target noise: Rician noise (edge-preserving smoothing)
    How it works: Smooths pixels based on both spatial proximity AND
    intensity similarity. Only neighbouring pixels with similar brightness
    are averaged, so edges and tissue boundaries are preserved. Well-suited
    to MRI where anatomical boundaries are clinically important.

    Args:
        image_array: uint8 RGB numpy array of shape (H, W, 3).

    Returns:
        Filtered uint8 RGB numpy array of the same shape.
    """
    # d=9 : diameter of pixel neighbourhood
    # sigmaColor=75 : colour space filter strength
    # sigmaSpace=75 : coordinate space filter strength
    return cv2.bilateralFilter(image_array, d=9, sigmaColor=75, sigmaSpace=75)


def apply_nlm_denoising(image_array: np.ndarray) -> np.ndarray:
    """
    Apply Non-Local Means (NLM) denoising to the image.

    Target noise: Speckle noise, Rician noise
    How it works: For each pixel, searches for similar patches across the
    entire image (non-local) and averages them. This exploits the repetitive
    texture structure of brain MRI images (grey matter, white matter regions)
    to produce very clean results without blurring fine details.

    Args:
        image_array: uint8 RGB numpy array of shape (H, W, 3).

    Returns:
        Filtered uint8 RGB numpy array of the same shape.
    """
    # h=10          : filter strength for luminance channel
    # hColor=10     : filter strength for colour channels
    # templateWindowSize=7 : size of patch used for comparison
    # searchWindowSize=21  : size of window searched for similar patches
    return cv2.fastNlMeansDenoisingColored(
        image_array,
        None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def apply_anisotropic_diffusion(image_array: np.ndarray) -> np.ndarray:
    """
    Apply Perona-Malik anisotropic diffusion to the image.

    Target noise: Speckle noise, intra-region noise
    How it works: Iteratively smooths homogeneous regions (low gradient)
    while deliberately NOT smoothing across strong edges (high gradient).
    This is achieved through a diffusion coefficient that decreases with
    gradient magnitude, preserving tissue boundaries in MRI scans.

    Implemented purely in NumPy — no extra library required.

    Args:
        image_array: uint8 RGB numpy array of shape (H, W, 3).

    Returns:
        Filtered uint8 RGB numpy array of the same shape.
    """
    # Work in float for numerical stability
    img = image_array.astype(np.float32)

    num_iter = 15       # Number of diffusion iterations
    kappa = 30.0        # Edge-stopping threshold (lower = more edge-sensitive)
    gamma = 0.1         # Diffusion rate (must be ≤ 0.25 for stability)

    for _ in range(num_iter):
        # Compute gradients in 4 directions (North, South, East, West)
        delta_n = np.roll(img, -1, axis=0) - img
        delta_s = np.roll(img,  1, axis=0) - img
        delta_e = np.roll(img, -1, axis=1) - img
        delta_w = np.roll(img,  1, axis=1) - img

        # Perona-Malik conduction coefficient (exponential variant)
        # c → 1 when gradient is small (smooth interior), c → 0 at edges
        c_n = np.exp(-(delta_n / kappa) ** 2)
        c_s = np.exp(-(delta_s / kappa) ** 2)
        c_e = np.exp(-(delta_e / kappa) ** 2)
        c_w = np.exp(-(delta_w / kappa) ** 2)

        img = img + gamma * (
            c_n * delta_n + c_s * delta_s +
            c_e * delta_e + c_w * delta_w
        )

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_wavelet_denoising(image_array: np.ndarray) -> np.ndarray:
    """
    Apply wavelet-based denoising to the image.

    Target noise: General MRI noise (Rician + Gaussian components)
    How it works: Decomposes the image into wavelet coefficients across
    multiple frequency sub-bands. Noise (mostly in high-frequency bands)
    is suppressed by soft-thresholding the coefficients. The image is then
    reconstructed from the thresholded coefficients, yielding a clean image
    that preserves structural details at all scales — ideal for MRI.

    Requires: scikit-image  (pip install scikit-image)

    Args:
        image_array: uint8 RGB numpy array of shape (H, W, 3).

    Returns:
        Filtered uint8 RGB numpy array of the same shape.
    """
    denoise_wavelet, estimate_sigma = _skimage_wavelet()

    # Normalise to [0, 1] for skimage
    img_float = image_array.astype(np.float32) / 255.0

    # Estimate per-channel noise sigma from the image itself
    sigma_est = estimate_sigma(img_float, channel_axis=-1, average_sigmas=False)

    # Apply wavelet denoising; 'BayesShrink' adapts threshold per sub-band
    denoised = denoise_wavelet(
        img_float,
        method="BayesShrink",
        mode="soft",
        sigma=sigma_est,
        channel_axis=-1,
        convert2ycbcr=True,   # process luminance separately (better for colour MRI)
    )

    return (np.clip(denoised, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Maps UI dropdown values to filter functions
_FILTER_MAP = {
    "gaussian":    apply_gaussian_filter,
    "median":      apply_median_filter,
    "bilateral":   apply_bilateral_filter,
    "nlm":         apply_nlm_denoising,
    "anisotropic": apply_anisotropic_diffusion,
    "wavelet":     apply_wavelet_denoising,
}


def apply_selected_filter(image_array: np.ndarray, filter_type: str) -> np.ndarray:
    """
    Dispatch to the appropriate noise removal function.

    Args:
        image_array: uint8 RGB numpy array.
        filter_type: One of "skip", "gaussian", "median", "bilateral",
                     "nlm", "anisotropic", "wavelet".

    Returns:
        Filtered (or unchanged if skip) uint8 RGB numpy array.

    Raises:
        ValueError: If an unknown filter_type is provided.
    """
    if filter_type == "skip":
        return image_array

    fn = _FILTER_MAP.get(filter_type)
    if fn is None:
        raise ValueError(
            f"Unknown filter type: '{filter_type}'. "
            f"Valid options: skip, {', '.join(_FILTER_MAP.keys())}"
        )
    return fn(image_array)
