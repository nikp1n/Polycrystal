import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def merge(labels, ids):
    ids = sorted(set(ids) - {0})
    if len(ids) < 2:
        raise ValueError('Выберите минимум два соседних объекта (Ctrl + щелчок).')
    chosen = np.isin(labels, ids)
    # Закрывает только тонкую фоновую линию между выбранными масками.
    near_min = ndi.minimum_filter(np.where(chosen, labels, np.iinfo(np.int32).max), size=5)
    near_max = ndi.maximum_filter(np.where(chosen, labels, 0), size=5)
    bridge = (ndi.binary_closing(chosen, structure=np.ones((3, 3)))
              & (labels == 0) & (near_min < near_max))
    connected = chosen | bridge
    if ndi.label(connected)[1] != 1:
        raise ValueError('Выберите соседние зёрна. Для соединения через большой промежуток используйте кисть.')
    output = labels.copy()
    output[connected] = ids[0]
    return output, ids[0]


def split(labels, label_id, points):
    mask = labels == label_id
    line = np.zeros(labels.shape, np.uint8)
    coords = np.rint(points).astype(np.int32)
    if len(coords) < 2:
        raise ValueError('Проведите линию через зерно от одного края до другого.')
    cv2.polylines(line, [coords], False, 1, thickness=2)
    components, count = ndi.label(mask & ~line.astype(bool))
    sizes = np.bincount(components.ravel())
    valid = np.flatnonzero(sizes[1:] >= 3) + 1
    if len(valid) < 2:
        raise ValueError('Линия не разделила зерно. Продлите её за обе границы объекта.')
    markers = np.zeros(labels.shape, np.int32)
    for i, idx in enumerate(valid, 1):
        markers[components == idx] = i
    parts = watershed(np.zeros(labels.shape), markers, mask=mask)
    output = labels.copy()
    output[mask] = 0
    for i in range(1, len(valid)+1):
        output[parts == i] = label_id if i == 1 else int(labels.max())+i-1
    return output


def polygon(labels, points, height, label_id=0):
    if len(points) < 3:
        raise ValueError('Обведите замкнутый контур минимум из трёх точек.')
    mask = np.zeros(labels.shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    mask[height:] = 0
    if np.count_nonzero(mask) < 3:
        raise ValueError('Контур слишком мал или лежит вне рабочей области.')
    output = labels.copy()
    if label_id:
        output[labels == label_id] = 0
    else:
        label_id = int(labels.max())+1
    output[mask > 0] = label_id
    if not np.any(output == label_id):
        raise ValueError('Контур полностью перекрыт другими объектами.')
    if ndi.label(output == label_id)[1] != 1:
        raise ValueError('Контур образует несколько несвязных частей из-за соседних масок. Измените контур или сначала удалите мешающий объект.')
    return output, label_id


def paint(labels, points, height, width, label_id=0, erase=False):
    if not points:
        raise ValueError('Проведите кистью по изображению.')
    coords = np.rint(points).astype(np.int32)
    mask = np.zeros(labels.shape, np.uint8)
    thickness = max(1, int(width))
    cv2.polylines(mask, [coords], False, 1, thickness=thickness)
    for point in (coords[0], coords[-1]):
        cv2.circle(mask, tuple(point), max(1, thickness//2), 1, -1)
    mask[height:] = 0
    output = labels.copy()
    target = 0 if erase else (label_id or int(labels.max())+1)
    output[mask > 0] = target
    return output, target, mask > 0


def normalize_components(labels, classes):
    """После правки каждая связная часть остаётся самостоятельным экземпляром."""
    output = labels.copy()
    classes = dict(classes)
    next_id = int(labels.max())
    for idx, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        local, count = ndi.label(labels[box] == idx)
        if count <= 1:
            continue
        sizes = np.bincount(local.ravel())
        largest = int(np.argmax(sizes[1:])+1)
        view = output[box]
        for component in range(1, count+1):
            if component != largest:
                next_id += 1
                view[local == component] = next_id
                classes[next_id] = classes.get(idx, 1)
    present = set(map(int, np.unique(output)))
    classes = {idx: cls for idx, cls in classes.items() if idx in present and idx > 0}
    return output, classes


def restore_region(labels, deleted, x, y, height, target=0):
    if not (0 <= y < height and 0 <= x < labels.shape[1]) or labels[y, x] > 0:
        raise ValueError('Щёлкните внутри удалённой или пустой области.')
    if deleted[y, x] > 0:
        region = (deleted == deleted[y, x]) & (labels == 0)
    else:
        region = labels == 0
        region[height:] = False
    parts, _ = ndi.label(region)
    component = parts == parts[y, x]
    if deleted[y, x] == 0 and (component[0].any() or component[height-1].any()
                              or component[:, 0].any() or component[:, -1].any()):
        raise ValueError('Эта область соединена с общим фоном. Обведите зерно контуром или закрасьте кистью.')
    output = labels.copy()
    new_id = int(labels.max())+1
    output[component] = new_id
    if target:
        output, new_id = merge(output, [target, new_id])
    return output, new_id
