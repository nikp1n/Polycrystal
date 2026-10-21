from datetime import datetime
from pathlib import Path
import csv
import json
import uuid

import numpy as np
import tifffile
from PIL import Image

from .analysis import measure, analysis_labels
from .image_loader import ROOT, file_hash, display_u8
from .segmentation import labels_to_color, make_overlay


def save_result(source, original, result):
    directory = ROOT / 'results' / (Path(source).stem + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    rows = measure(original, result)
    analytical = analysis_labels(result)
    info = dict(result['info'])
    info.update({'source_name': Path(source).name, 'source_sha256': file_hash(source),
                 'source_dtype': str(original.dtype), 'source_shape': list(original.shape),
                 'classes': {str(k): v for k, v in result['classes'].items()},
                 'exported_at': datetime.now().astimezone().isoformat(),
                 'included_grains': sum(row['included'] for row in rows),
                 'excluded_objects': sum(not row['included'] for row in rows),
                 'border_margin_px': 2,
                 'label_convention': '0 background; positive integers instance IDs',
                 'confidence_note': 'SAM predicted IoU, not measured accuracy or pixel probability'})
    stages = {'01_original': display_u8(original), '02_preprocessed': result['preprocessed'],
              '03_edge_map': result['edges'], '04_raw_segmentation': labels_to_color(result['raw']),
              '05_postprocessed_segmentation': labels_to_color(analytical),
              '06_overlay': make_overlay(original, analytical)}
    for name, array in stages.items():
        Image.fromarray(array).save(directory / f'{name}.png')
    if result.get('watershed') is not None:
        Image.fromarray(labels_to_color(result['watershed'])).save(directory / '03a_watershed.png')
        tifffile.imwrite(directory / 'watershed_labels.tif', result['watershed'].astype(np.int32))
    tifffile.imwrite(directory / 'raw_labels.tif', result['raw'].astype(np.int32))
    tifffile.imwrite(directory / 'labels.tif', result['labels'].astype(np.int32))
    tifffile.imwrite(directory / 'analysis_labels.tif', analytical.astype(np.int32))
    class_map = np.zeros(result['labels'].shape, np.uint8)
    for row in rows:
        class_map[result['labels'] == row['id']] = result['classes'].get(row['id'], 1)
    tifffile.imwrite(directory / 'classes.tif', class_map)
    if result['confidence'] is not None:
        np.save(directory / 'model_confidence.npy', result['confidence'])
    if rows:
        with open(directory / 'grains.csv', 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=';')
            writer.writeheader()
            writer.writerows(row for row in rows if row['included'])
    else:
        (directory / 'grains.csv').write_text('id;class;included\n', encoding='utf-8-sig')
    (directory / 'summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Только числовые массивы, без pickle; исходный файл проверяется по SHA-256.
    arrays = {k: result[k] for k in ('raw', 'labels', 'preprocessed', 'edges')}
    if result.get('watershed') is not None:
        arrays['watershed'] = result['watershed']
    arrays['deleted_labels'] = result.get('deleted_labels', np.zeros(result['labels'].shape, np.int32))
    if result['confidence'] is not None:
        arrays['confidence'] = result['confidence']
    np.savez_compressed(directory / 'session.npz', **arrays)
    return directory


def load_result(session_path, source, original):
    path = Path(session_path)
    info = json.loads(path.with_name('summary.json').read_text(encoding='utf-8'))
    if info['source_sha256'] != file_hash(source):
        raise ValueError('Сессия относится к другому исходному файлу. Откройте соответствующее фото.')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in ('raw', 'labels', 'preprocessed', 'edges')}
        arrays['confidence'] = archive['confidence'].copy() if 'confidence' in archive else None
        arrays['watershed'] = archive['watershed'].copy() if 'watershed' in archive else None
        arrays['deleted_labels'] = archive['deleted_labels'].copy() if 'deleted_labels' in archive else np.zeros(original.shape[:2], np.int32)
    for key in ('raw', 'labels', 'preprocessed', 'edges'):
        if arrays[key].shape != original.shape[:2]:
            raise ValueError('Повреждённая сессия: неверный размер массива.')
    if arrays['deleted_labels'].shape != original.shape[:2] or arrays['deleted_labels'].dtype.kind not in 'ui' or arrays['deleted_labels'].min() < 0:
        raise ValueError('Повреждённая карта удалённых областей.')
    for key in ('raw', 'labels'):
        if arrays[key].dtype.kind not in 'ui' or arrays[key].min() < 0:
            raise ValueError('Повреждённая карта меток.')
    if not 16 <= int(info['working_height']) <= original.shape[0]:
        raise ValueError('Повреждённая сессия: неверная рабочая область.')
    if np.any(arrays['labels'][info['working_height']:] > 0):
        raise ValueError('Повреждённая сессия: метки вне рабочей области.')
    if arrays['confidence'] is not None and arrays['confidence'].shape != original.shape[:2]:
        raise ValueError('Повреждённая карта уверенности.')
    if arrays['watershed'] is not None and (arrays['watershed'].shape != original.shape[:2] or arrays['watershed'].dtype.kind not in 'ui' or arrays['watershed'].min() < 0):
        raise ValueError('Повреждённая карта областей Watershed.')
    arrays['classes'] = {int(k): int(v) for k, v in info.get('classes', {}).items()}
    arrays['info'] = info
    return arrays
