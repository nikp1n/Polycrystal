from pathlib import Path
import hashlib
import shutil

import numpy as np
import tifffile
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path):
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError('Поддерживаются PNG, JPEG, BMP и TIFF.')
    if path.suffix.lower() in {'.tif', '.tiff'}:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != 1 or len(tif.series) != 1:
                raise ValueError('Многостраничный TIFF нужно разделить на отдельные двумерные кадры.')
            series = tif.series[0]
            if any(axis in series.axes for axis in 'ZTC') and series.axes not in ('YXS', 'SYX'):
                raise ValueError('Выберите отдельный двумерный кадр, а не стек TIFF.')
            array = series.asarray()
            if series.axes == 'SYX':
                array = np.moveaxis(array, 0, -1)
    else:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ('L', 'I', 'F', 'I;16', 'I;16L', 'I;16B', 'RGB', 'RGBA'):
                img = img.convert('RGB')
            array = np.asarray(img).copy()
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] not in (3, 4)):
        raise ValueError('Нужен один серый или RGB/RGBA кадр, без стека страниц.')
    if min(array.shape[:2]) < 16:
        raise ValueError('Минимальный размер изображения — 16 × 16 пикселей.')
    if not np.isfinite(array).all():
        raise ValueError('Изображение содержит NaN или бесконечные значения.')
    return np.ascontiguousarray(array)


def import_original(path):
    """Копирует исходный файл без перекодирования; совпадения имён не затираются."""
    path = Path(path).resolve()
    directory = ROOT / 'originals'
    directory.mkdir(exist_ok=True)
    if path.parent == directory.resolve():
        return path
    target = directory / path.name
    if target.exists():
        if file_hash(path) == file_hash(target):
            return target
        target = directory / f'{path.stem}_{file_hash(path)[:12]}{path.suffix.lower()}'
        if target.exists() and file_hash(target) != file_hash(path):
            raise ValueError('Конфликт имён исходных файлов.')
    if not target.exists():
        shutil.copy2(path, target)
    return target


def display_u8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    values = array.astype(np.float64)
    low, high = np.percentile(values, (0.2, 99.8))
    if high <= low:
        low, high = values.min(), values.max()
    return np.ascontiguousarray(np.clip((values-low) * 255 / max(high-low, 1e-12), 0, 255).astype(np.uint8))


def gray_u8(array):
    import cv2
    arr = display_u8(array)
    return arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY if arr.shape[-1] == 4 else cv2.COLOR_RGB2GRAY)


def rgb_u8(array):
    arr = display_u8(array)
    return np.repeat(arr[..., None], 3, axis=2) if arr.ndim == 2 else arr[..., :3].copy()
