"""Геометрия зёрен и метрики относительно независимой эталонной разметки."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from .image_loader import gray_u8

CLASSES = {1: 'Зерно', 2: 'Дефект', 3: 'Исключить'}


def border_ids(labels, height, margin=2):
    """Учитывает тонкую нулевую рамку, оставленную SAM/водоразделом."""
    work = labels[:height]
    if work.size == 0:
        return set()
    strip = max(1, int(margin)+1)
    ids = np.unique(np.concatenate((work[:strip].ravel(), work[-strip:].ravel(),
                                   work[:, :strip].ravel(), work[:, -strip:].ravel())))
    return set(map(int, ids[ids > 0]))


def analysis_labels(result):
    labels = result['labels']
    excluded = {idx for idx, cls in result['classes'].items() if cls != 1}
    if result['info']['settings']['exclude_border']:
        excluded |= border_ids(labels, result['info']['working_height'])
    output = labels.copy()
    if excluded:
        output[np.isin(output, list(excluded))] = 0
    return output


def measure(image, result):
    labels, info = result['labels'], result['info']
    settings = info['settings']
    height = info['working_height']
    scale = float(settings['scale'])
    factor = scale if scale > 0 else 1.0
    unit = settings['unit'] if scale > 0 else 'px'
    gray = gray_u8(image)
    edge_ids = border_ids(labels, height)
    rows = []
    for prop in regionprops(labels):
        y0, x0, y1, x1 = prop.bbox
        border = int(prop.label) in edge_ids
        cls = result['classes'].get(prop.label, 1)
        boundary = find_boundaries(np.pad(prop.image, 1), mode='inner')[1:-1, 1:-1] & prop.image
        intensities = gray[y0:y1, x0:x1][boundary].astype(float)
        cv = float(intensities.std() / max(intensities.mean(), 1e-12)) if intensities.size else 0.0
        values = result['edges'][y0:y1, x0:x1][boundary]
        rows.append({
            'id': int(prop.label), 'class': CLASSES.get(cls, 'Зерно'),
            'included': bool(cls == 1 and not (border and settings['exclude_border'])),
            'touches_border': bool(border), 'area_px2': int(prop.area),
            'centroid_x_px': float(prop.centroid[1]), 'centroid_y_px': float(prop.centroid[0]),
            'perimeter_px': float(prop.perimeter_crofton),
            'major_axis_px': float(prop.axis_major_length), 'minor_axis_px': float(prop.axis_minor_length),
            'feret_max_px': float(prop.feret_diameter_max), 'eccentricity': float(prop.eccentricity),
            'equivalent_diameter_px': float(prop.equivalent_diameter_area),
            'boundary_intensity_cv': cv, 'boundary_uniformity': 1.0/(1+cv),
            'boundary_edge_strength': float(np.mean(values)/255) if values.size else 0.0,
            'unit': unit, 'area': float(prop.area)*factor**2,
            'perimeter': float(prop.perimeter_crofton)*factor,
            'major_axis': float(prop.axis_major_length)*factor, 'minor_axis': float(prop.axis_minor_length)*factor,
            'feret_max': float(prop.feret_diameter_max)*factor,
            'centroid_x': float(prop.centroid[1])*factor, 'centroid_y': float(prop.centroid[0])*factor,
        })
    return rows


def quality_metrics(prediction, reference, height=None):
    """Нулевые ID — фон, положительные ID — отдельные объекты; ID могут различаться."""
    if prediction.shape != reference.shape or reference.ndim != 2:
        raise ValueError('Эталон должен иметь размер исходного кадра и одну целочисленную метку на зерно.')
    if reference.dtype.kind not in 'ui' or np.any(reference < 0):
        raise ValueError('Эталон: целочисленный TIFF/NPY, 0 — фон, 1…N — разные зёрна.')
    height = height or prediction.shape[0]
    pred, ref = prediction[:height], reference[:height]
    p, g = pred > 0, ref > 0
    intersection, union = np.count_nonzero(p & g), np.count_nonzero(p | g)
    dice = 2*intersection/max(np.count_nonzero(p)+np.count_nonzero(g), 1)
    _, pi = np.unique(pred, return_inverse=True)
    _, gi = np.unique(ref, return_inverse=True)
    # Вставляем нулевую строку/столбец даже при отсутствии фона.
    pids, gids = np.unique(pred), np.unique(ref)
    if len(pids)*len(gids) > 20_000_000:
        raise ValueError('Слишком много меток: проверьте, что эталон содержит ID объектов, а не яркость.')
    table = np.bincount(pi*len(gids)+gi, minlength=len(pids)*len(gids)).reshape(len(pids), len(gids))
    valid_p, valid_g = pids > 0, gids > 0
    inter = table[np.ix_(valid_p, valid_g)]
    pa, ga = table.sum(axis=1)[valid_p], table.sum(axis=0)[valid_g]
    iou = inter / np.maximum(pa[:, None]+ga[None, :]-inter, 1)
    if iou.size:
        # Сначала максимизируем число соответствий IoU >= .5, затем их IoU.
        a, b = linear_sum_assignment(-((iou >= .5).astype(float)*1000 + iou))
        matched = iou[a, b]
        matched = matched[matched >= .5]
    else:
        matched = np.array([])
    tp, npred, nref = len(matched), len(pa), len(ga)
    pb, gb = find_boundaries(pred), find_boundaries(ref)
    tolerance = 2
    bprecision = np.count_nonzero(pb & ndi.binary_dilation(gb, iterations=tolerance))/max(np.count_nonzero(pb), 1)
    brecall = np.count_nonzero(gb & ndi.binary_dilation(pb, iterations=tolerance))/max(np.count_nonzero(gb), 1)
    return {
        'foreground_iou': float(intersection/max(union, 1)), 'foreground_dice': float(dice),
        'instance_precision_iou50': tp/max(npred, 1), 'instance_recall_iou50': tp/max(nref, 1),
        'instance_f1_iou50': 2*tp/max(npred+nref, 1),
        'matched_mean_iou': float(matched.mean()) if tp else 0.0,
        'panoptic_quality_iou50': float(matched.sum()/max(.5*(npred+nref), 1)),
        'boundary_f1_2px': 2*bprecision*brecall/max(bprecision+brecall, 1e-12),
        'true_positive': tp, 'false_positive': npred-tp, 'false_negative': nref-tp,
        'reference_objects': nref, 'predicted_objects': npred,
        'evaluation_height': height, 'scope': 'all instances in working area, before border exclusion',
    }
