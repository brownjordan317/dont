from pathlib import Path
from typing import Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def _resize_image(img: np.ndarray, resize_factor: float) -> np.ndarray:
    if abs(resize_factor - 1.0) < 1e-9:
        return img

    if cv2 is not None:
        interpolation = cv2.INTER_LINEAR if resize_factor > 1.0 else cv2.INTER_AREA
        return cv2.resize(
            img,
            dsize=None,
            fx=resize_factor,
            fy=resize_factor,
            interpolation=interpolation,
        )

    # Lightweight nearest-neighbor fallback for API callers that pass an image
    # array directly in environments where OpenCV is not installed.
    new_h = max(1, int(round(img.shape[0] * resize_factor)))
    new_w = max(1, int(round(img.shape[1] * resize_factor)))
    row_idx = np.clip(
        np.round(np.linspace(0, img.shape[0] - 1, new_h)).astype(np.int32),
        0,
        img.shape[0] - 1,
    )
    col_idx = np.clip(
        np.round(np.linspace(0, img.shape[1] - 1, new_w)).astype(np.int32),
        0,
        img.shape[1] - 1,
    )
    return img[row_idx][:, col_idx]


def _ensure_grayscale_uint8(image_source: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image_source, (str, Path)):
        image_path = Path(image_source)
        if cv2 is not None:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not read heatmap image: {image_path}")
            return img

        try:
            import matplotlib.image as mpimg
        except ImportError as exc:
            raise ImportError(
                "opencv-python or matplotlib is required to load heatmap images from a file path."
            ) from exc

        img = np.asarray(mpimg.imread(str(image_path)))
        if img.ndim == 3:
            img = np.rint(np.mean(img[:, :, :3], axis=2))
        if np.issubdtype(img.dtype, np.floating):
            img_max = float(np.max(img)) if img.size else 0.0
            if img_max <= 1.0:
                img = img * 255.0
        return img.astype(np.uint8)

    img = np.asarray(image_source)
    if img.ndim == 3:
        # Accept RGB/RGBA arrays from API callers without forcing them to preconvert.
        channels = img.shape[2]
        if channels == 4 and cv2 is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif channels >= 3 and cv2 is not None:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif channels >= 3:
            img = np.rint(np.mean(img[:, :, :3], axis=2))
        else:
            raise ValueError("Heatmap image array must have 3 or 4 channels.")

    if img.ndim != 2:
        raise ValueError("Heatmap image must be a grayscale or RGB/RGBA array.")

    if np.issubdtype(img.dtype, np.floating):
        img_max = float(np.max(img)) if img.size else 0.0
        img = np.clip(img, 0.0, 1.0 if img_max <= 1.0 else 255.0)
        if img_max <= 1.0:
            img = img * 255.0

    return img.astype(np.uint8)


def make_heatmap_from_source(
    image_source: Union[str, Path, np.ndarray],
    source_image_ppm: float,
    resolution: float,
) -> np.ndarray:
    img = _ensure_grayscale_uint8(image_source)

    target_ppm = 1.0 / resolution
    resize_factor = target_ppm / source_image_ppm
    img = _resize_image(img, resize_factor)
    return img.astype(np.float32) / 255.0
