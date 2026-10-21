"""Загрузка официальных весов с проверкой SHA-256; сетевой доступ только по команде."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent
EXPECTED = '6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f'
URL = 'https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt'


def main():
    target = ROOT / 'models' / 'mobile_sam.pt'
    target.parent.mkdir(exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED:
        print('MobileSAM готова. Контрольная сумма совпадает.')
        return
    temporary = target.with_suffix('.download')
    try:
        print('Загрузка официальной модели MobileSAM (около 39 МБ)…')
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024*1024):
                stream.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED:
            raise ValueError('Неверная контрольная сумма. Файл не установлен.')
        temporary.replace(target)
        print('MobileSAM готова.')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
