"""Единая сегментация: Watershed формирует области и подсказки для MobileSAM."""
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation

from .image_loader import ROOT, gray_u8, rgb_u8

MODEL_SHA256 = '6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f'
MODEL_REVISION = 'f706ad9c4eb7f219c00d9050e46328518ffb65d2'


@dataclass
class Settings:
    crop_height: int = 0  # 0: консервативное автоопределение, -1: весь кадр
    min_area: int = 20
    smoothing: float = 1.2
    marker_depth: float = 8.0
    pred_iou: float = 0.72
    stability: float = 0.88
    adaptive_detail: bool = True
    refine_boundaries: bool = True
    exclude_border: bool = True
    scale: float = 0.0
    unit: str = 'µm'


def working_height(gray, requested=0):
    height, width = gray.shape
    if requested == -1:
        return height
    if requested > 0:
        if not 16 <= requested <= height:
            raise ValueError('Высота рабочей области должна быть от 16 до высоты кадра.')
        return requested
    # SEM-полоса: резкое падение яркости и несколько почти чёрных строк.
    for y in range(int(height * .7), height - 4):
        band = gray[y:y+4]
        if np.mean(band < 45) > .86 and np.mean(gray[max(0,y-8):y]) > np.mean(band) + 20:
            if np.mean(gray[y:] < 55) > .55:
                return y
    return height


def prepare(image, settings):
    gray = gray_u8(image)
    height = working_height(gray, settings.crop_height)
    work = gray[:height]
    if float(np.std(work)) < .7:
        raise ValueError('Рабочая область почти однородна: достоверные границы не определяются.')
    smooth = cv2.GaussianBlur(work, (0, 0), max(.1, settings.smoothing))
    enhanced = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(smooth)
    edge = filters.scharr(enhanced.astype(np.float32) / 255)
    edge /= max(float(np.percentile(edge, 99.5)), 1e-6)
    preprocessed = np.zeros(gray.shape, np.uint8)
    edges = np.zeros(gray.shape, np.uint8)
    preprocessed[:height] = enhanced
    edges[:height] = np.clip(edge * 255, 0, 255).astype(np.uint8)
    return gray, enhanced, preprocessed, edges, height


def relabel(labels):
    ids, inverse = np.unique(labels, return_inverse=True)
    lookup = np.arange(len(ids), dtype=np.int32) + (0 if ids[0] == 0 else 1)
    return lookup[inverse].reshape(labels.shape)


def postprocess(labels, min_area):
    # Не расширяем маски и не удаляем зёрна по эксцентриситету/среднему размеру.
    output = np.zeros(labels.shape, np.int32)
    index = 0
    for label_id, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        components, count = ndi.label(labels[box] == label_id)
        sizes = np.bincount(components.ravel())
        view = output[box]
        for component in range(1, count + 1):
            if sizes[component] >= min_area:
                index += 1
                view[components == component] = index
    return output


@lru_cache(maxsize=1)
def load_model():
    checkpoint = ROOT / 'models' / 'mobile_sam.pt'
    if not checkpoint.exists():
        raise ValueError('Нет models/mobile_sam.pt. Выполните python setup_model.py.')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError('Контрольная сумма MobileSAM не совпадает. Выполните python setup_model.py.')
    try:
        import torch
        from mobile_sam import sam_model_registry
    except ImportError as exc:
        raise ValueError('Установите зависимости из requirements.txt, включая MobileSAM и PyTorch.') from exc
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = sam_model_registry['vit_t']()  # Без небезопасного torch.load внутри upstream.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    return model, device


def watershed_points(proposals):
    """По одному внутреннему маркеру на связную область первого этапа."""
    points = []
    for idx, box in enumerate(ndi.find_objects(proposals), 1):
        if box is None:
            continue
        mask = proposals[box] == idx
        distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        points.append((float(x+box[1].start)+.5, float(y+box[0].start)+.5))
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def sam_labels(image, height, proposals, settings, progress):
    model, device = load_model()
    from mobile_sam import SamAutomaticMaskGenerator
    from mobile_sam.utils.amg import rle_to_mask
    rgb = rgb_u8(image)[:height]
    width = rgb.shape[1]
    points = watershed_points(proposals)
    if not len(points):
        raise ValueError('Watershed не подготовил областей для MobileSAM. Уменьшите минимальную площадь или глубину маркеров.')
    crops = [(0, 0, width, height)]
    detailed = bool(settings.adaptive_detail and len(points) > 250)
    if detailed:
        crop_w, crop_h = int(np.ceil(width*.68)), int(np.ceil(height*.68))
        crops += [(x, y, x+crop_w, y+crop_h)
                  for y in (0, height-crop_h) for x in (0, width-crop_w)]
    candidates = []
    for crop_index, (x0, y0, x1, y1) in enumerate(crops):
        chosen = points[(points[:, 0] >= x0) & (points[:, 0] < x1)
                        & (points[:, 1] >= y0) & (points[:, 1] < y1)]
        if not len(chosen):
            continue
        prompt_grid = (chosen - [x0, y0]) / [x1-x0, y1-y0]
        progress(f'Этап 2/2 · MobileSAM: уточнение областей Watershed ({crop_index+1}/{len(crops)})…')
        generator = SamAutomaticMaskGenerator(
            model, points_per_side=None, point_grids=[prompt_grid.astype(np.float32)],
            points_per_batch=16, pred_iou_thresh=settings.pred_iou,
            stability_score_thresh=settings.stability, crop_n_layers=0,
            min_mask_region_area=0, output_mode='uncompressed_rle',
        )
        for item in generator.generate(rgb[y0:y1, x0:x1]):
            item['offset'] = (x0, y0, x1, y1)
            candidates.append(item)
    candidates.sort(key=lambda item: item['area'])
    labels = np.zeros((height, width), np.int32)
    confidence = np.zeros(labels.shape, np.float32)
    accepted, accepted_areas = [], [0]
    for item in candidates:
        x0, y0, x1, y1 = item['offset']
        mask = rle_to_mask(item['segmentation'])
        # Обрезанные внутренним краем фрагмента маски не считаем целыми зёрнами.
        if ((x0 > 0 and mask[:, :2].any()) or (x1 < width and mask[:, -2:].any())
                or (y0 > 0 and mask[:2].any()) or (y1 < height and mask[-2:].any())):
            continue
        area = int(mask.sum())
        if not area:
            continue
        view = labels[y0:y1, x0:x1]
        overlaps = np.bincount(view[mask], minlength=len(accepted_areas))
        contained = sum(overlaps[idx] >= .85*accepted_areas[idx] for idx in range(1, len(accepted_areas)))
        if contained >= 2 or np.count_nonzero(mask & (view > 0))/area > .5:
            continue
        available = mask & (view == 0)
        label_id = len(accepted)+1
        view[available] = label_id
        confidence[y0:y1, x0:x1][available] = float(item['predicted_iou'])
        accepted_areas.append(int(available.sum()))
        accepted.append({'id': label_id, 'predicted_iou': float(item['predicted_iou']),
                         'stability_score': float(item['stability_score'])})
    if not accepted:
        raise ValueError('MobileSAM не подтвердила маски по областям Watershed. Уменьшите пороги SAM или проверьте рабочую область.')
    return labels, confidence, {
        'device': device, 'candidate_masks': len(candidates), 'watershed_prompt_count': len(points),
        'prompt_source': 'Watershed regions', 'watershed_prompt_points_xy': points.tolist(),
        'sam_crop_layers': int(detailed), 'mask_scores': accepted,
        'model_sha256': MODEL_SHA256, 'model_revision': MODEL_REVISION,
    }


def watershed_labels(enhanced, settings):
    # Локальные минимумы рельефа: глубина задаётся в уровнях яркости,
    # а не долей максимального радиуса самого крупного зерна.
    relief = cv2.GaussianBlur(enhanced, (0, 0), 1.0).astype(np.float32)
    markers, count = ndi.label(morphology.h_maxima(relief, settings.marker_depth))
    if count == 0:
        raise ValueError('Нет маркеров. Уменьшите глубину маркеров.')
    labels = segmentation.watershed(-relief, markers, watershed_line=False)
    return labels.astype(np.int32), {'markers': int(count)}


def complete_from_proposals(raw, proposals, gray, height):
    """Уточняет неполные SAM-маски в пределах уже рассчитанных областей Watershed.

    Повторного запуска Watershed после MobileSAM нет. Новые объекты без
    подтверждения SAM не создаются; границы первого этапа ограничивают заполнение.
    """
    output = raw.copy()
    unsupported = 0
    surface = cv2.GaussianBlur(gray[:height], (0, 0), .8)
    foreground = surface >= min(max(float(np.percentile(surface, 1)), 1.0), 35.0)
    for idx, box in enumerate(ndi.find_objects(proposals), 1):
        if box is None:
            continue
        region = proposals[box] == idx
        current = raw[box]
        supported = np.where(region, current, 0)
        ids, counts = np.unique(supported[supported > 0], return_counts=True)
        if not len(ids) or counts.sum() < max(3, .1*int(region.sum())):
            unsupported += 1
            continue
        missing = region & (current == 0) & foreground[box]
        if not missing.any():
            continue
        if len(ids) == 1:
            output[box][missing] = ids[0]
        else:
            _, nearest = ndi.distance_transform_edt(supported == 0, return_indices=True)
            output[box][missing] = supported[nearest[0], nearest[1]][missing]
    return output, unsupported


def run_segmentation(image, settings, progress=lambda message: None):
    progress('Подготовка рабочей области и карты границ…')
    gray, enhanced, preprocessed, edges, height = prepare(image, settings)
    progress('Этап 1/2 · Watershed: выделение предварительных областей зёрен…')
    watershed_raw, watershed_info = watershed_labels(enhanced, settings)
    proposals = postprocess(watershed_raw, settings.min_area)
    if not np.any(proposals):
        raise ValueError('Все области Watershed меньше минимальной площади. Уменьшите её.')
    labels, confidence, info = sam_labels(image, height, proposals, settings, progress)
    raw = np.zeros(gray.shape, np.int32)
    raw[:height] = labels
    full_watershed = np.zeros(gray.shape, np.int32)
    full_watershed[:height] = watershed_raw
    progress('Постобработка масок MobileSAM и измерение зёрен…')
    cleaned = postprocess(raw, settings.min_area)
    unsupported = 0
    if settings.refine_boundaries:
        refined_work, unsupported = complete_from_proposals(cleaned[:height], proposals, gray, height)
        cleaned[:height] = refined_work
    final = postprocess(cleaned, settings.min_area)
    if not np.any(final):
        raise ValueError('Итоговая карта пуста. Проверьте минимальную площадь и пороги SAM.')
    info.update({
        'method': 'Watershed → MobileSAM', 'pipeline': ['Watershed', 'MobileSAM'],
        'watershed_markers': watershed_info['markers'], 'watershed_regions': int(proposals.max()),
        'unsupported_watershed_regions': unsupported, 'working_height': height,
        'settings': asdict(settings), 'raw_count': int(raw.max()), 'final_count': int(final.max()),
        'metrics': None, 'manually_edited': False,
        'postprocessing': 'SAM masks completed within prior Watershed regions' if settings.refine_boundaries else 'connected components',
    })
    full_confidence = np.zeros(gray.shape, np.float32)
    full_confidence[:height] = confidence
    return {'watershed': full_watershed, 'raw': raw, 'labels': final,
            'preprocessed': preprocessed, 'edges': edges, 'confidence': full_confidence,
            'info': info, 'classes': {}}


def labels_to_color(labels):
    ids = np.arange(int(labels.max()) + 1, dtype=np.uint32)
    colors = np.column_stack((55 + ids*67 % 185, 55 + ids*109 % 185, 55 + ids*149 % 185)).astype(np.uint8)
    colors[0] = 0
    return colors[labels]


def make_overlay(original, labels, selected=()):
    base = rgb_u8(original)
    boundaries = segmentation.find_boundaries(labels, mode='inner') & (labels > 0)
    base[boundaries] = (41, 238, 179)
    if selected:
        chosen = np.isin(labels, list(selected))
        base[chosen] = (.5 * base[chosen] + .5 * np.array([255, 191, 55])).astype(np.uint8)
    return base
