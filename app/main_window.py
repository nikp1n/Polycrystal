from pathlib import Path
import time

import numpy as np
import tifffile
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut, QFontDatabase, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QListWidgetItem, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QProgressBar,
    QScrollArea, QInputDialog, QAbstractItemView, QMenu, QLineEdit,
)

from core.image_loader import ROOT, SUPPORTED_EXTENSIONS, load_image, import_original
from core.segmentation import Settings, run_segmentation, labels_to_color, make_overlay
from core.analysis import measure, quality_metrics, CLASSES, analysis_labels
from core.storage import save_result, load_result
from core import editing
from .image_viewer import ImageViewer
from .file_drop import FileDropMixin, DropSurface, DropFileList
from .theme import APP_STYLE, apply_light_theme

STAGES = ['Исходный снимок', 'Предобработка', 'Карта границ', 'Маски MobileSAM', 'Итоговые маски', 'Границы на снимке', 'Области Watershed']
NOTES = [
    'Исходный кадр. Оригинал и его разрядность сохраняются в originals без изменения.',
    'Рабочая область: сглаживание и локальный контраст CLAHE. Служебная полоса исключена.',
    'Градиент Scharr: светлые линии означают выраженные перепады яркости; это не вероятность модели.',
    'Маски до постобработки и ручных исправлений. Для SAM перекрытия разрешены по вложенности.',
    'Маски после фильтра площади, адаптивного уточнения границ при его применении и ручной правки. Чёрный — фон или пропуск.',
    'Границы на оригинале. Щелчок — выбрать зерно; Ctrl + щелчок — добавить к выбору.',
    'Первый этап: предварительные области Watershed. Их внутренние маркеры передаются MobileSAM для уточнения.',
]


class ProcessingWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    batch_item = Signal(str, str)

    def __init__(self, image, settings, paths=None):
        super().__init__()
        self.image, self.settings, self.paths = image, settings, paths

    def run(self):
        if self.paths is not None:
            failures = []
            for index, path in enumerate(self.paths, 1):
                if self.isInterruptionRequested():
                    break
                self.progress.emit(f'Пакет: {index}/{len(self.paths)} · {path.name}')
                try:
                    image = load_image(path)
                    started = time.perf_counter()
                    result = run_segmentation(image, self.settings, self.progress.emit)
                    result['info']['processing_seconds'] = round(time.perf_counter()-started, 3)
                    destination = save_result(path, image, result)
                    self.batch_item.emit(str(path), str(destination))
                except Exception as exc:
                    failures.append(f'{path.name}: {exc}')
            self.completed.emit({'batch': True, 'failures': failures, 'cancelled': self.isInterruptionRequested()})
            return
        try:
            started = time.perf_counter()
            result = run_segmentation(self.image, self.settings, self.progress.emit)
            result['info']['processing_seconds'] = round(time.perf_counter()-started, 3)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(FileDropMixin, QMainWindow):
    files_dropped = Signal(list)
    def __init__(self):
        super().__init__()
        apply_light_theme()
        # Явная загрузка также обеспечивает читаемый offscreen-просмотр Qt.
        font_path = Path('C:/Windows/Fonts/segoeui.ttf')
        if font_path.exists():
            QFontDatabase.addApplicationFont(str(font_path))
        self.setFont(QFont('Segoe UI', 10))
        self.setWindowTitle('PolyCrystal Analyzer · анализ SEM-снимков')
        self.resize(1440, 960)
        self.setMinimumSize(1100, 740)
        self.setAcceptDrops(True)
        self.setStyleSheet(APP_STYLE)
        self.path = None
        self.original = None
        self.result = None
        self.selected = set()
        self.history = []
        self.last_result_dir = None
        self.dirty = False
        self.worker = None
        self.pending_result = None
        self.busy = False
        self.build_ui()
        self.files_dropped.connect(self.import_paths, Qt.ConnectionType.QueuedConnection)
        for field in self.findChildren(QLineEdit):
            field.setAcceptDrops(False)
        for path in sorted((ROOT / 'originals').iterdir()):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.add_item(path)
        self.update_controls()
        QShortcut(QKeySequence.StandardKey.Open, self, activated=self.open_files)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence('Delete'), self, activated=self.delete_selected)

    def button(self, text, slot, primary=False):
        button = QPushButton(text)
        button.clicked.connect(slot)
        if primary:
            button.setObjectName('Primary')
        return button

    def build_ui(self):
        root = DropSurface()
        root.setObjectName('MainSurface')
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 10)
        outer.setSpacing(12)
        heading = QHBoxLayout()
        title = QLabel('PolyCrystal Analyzer')
        title.setObjectName('Title')
        heading.addWidget(title)
        heading.addStretch()
        self.open_button = self.button('Открыть снимки', self.open_files)
        self.results_button = self.button('Папка результатов', self.open_results)
        self.results_button.setToolTip('Открыть последнюю сохранённую обработку или общую папку results')
        self.save_button = self.button('Сохранить', self.save)
        for button in (self.open_button, self.results_button, self.save_button):
            heading.addWidget(button)
        outer.addLayout(heading)

        file_menu = self.menuBar().addMenu('Файл')
        file_menu.addAction('Открыть снимки…', self.open_files)
        self.restore_button = file_menu.addAction('Открыть сохранённую обработку…', self.restore)
        file_menu.addAction('Папка исходных снимков', lambda: self.open_dir(ROOT/'originals'))
        file_menu.addAction('Все результаты', lambda: self.open_dir(ROOT/'results'))
        self.menuBar().addAction('Руководство', lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT/'README.md'))))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(285)
        scroll.setMaximumWidth(320)
        sidebar = QWidget()
        sidebar.setObjectName('Sidebar')
        left = QVBoxLayout(sidebar)
        left.setContentsMargins(13, 13, 13, 13)
        left.setSpacing(10)
        label = QLabel('Снимки')
        label.setObjectName('CardTitle')
        left.addWidget(label)
        self.files = DropFileList()
        self.files.setMinimumHeight(155)
        self.files.setMaximumHeight(220)
        self.files.setAcceptDrops(True)
        self.files.viewport().setAcceptDrops(True)
        self.files.currentItemChanged.connect(self.choose_file)
        left.addWidget(self.files)
        self.drop_hint = QLabel('Перетащите фото в окно\nPNG, JPEG, BMP или TIFF')
        self.drop_hint.setObjectName('DropHint')
        self.drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.drop_hint)
        self.image_info = QLabel('Выберите снимок из списка')
        self.image_info.setObjectName('Hint')
        self.image_info.setWordWrap(True)
        left.addWidget(self.image_info)
        method_label = QLabel('Watershed → MobileSAM')
        method_label.setObjectName('CardTitle')
        left.addWidget(method_label)
        self.run_button = self.button('Сегментировать', self.run, True)
        left.addWidget(self.run_button)
        self.cancel_button = self.button('Остановить пакет', self.cancel_batch)
        left.addWidget(self.cancel_button)
        self.border = QCheckBox('Исключить краевые из анализа')
        self.border.setChecked(True)
        self.border.setToolTip('Убрать краевые объекты из таблицы, итогового изображения и аналитического экспорта. Исходные маски остаются доступными.')
        left.addWidget(self.border)
        self.advanced_toggle = self.button('Показать настройки', self.toggle_advanced)
        self.advanced_toggle.setCheckable(True)
        left.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        settings_tabs = QTabWidget()
        advanced_layout.addWidget(settings_tabs)
        algorithm = QWidget()
        form = QFormLayout(algorithm)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.crop_mode = QComboBox()
        self.crop_mode.addItems(['Авто: убрать SEM-полосу', 'Весь кадр', 'Высота вручную'])
        form.addRow('Рабочая область', self.crop_mode)
        self.crop_height = QSpinBox()
        self.crop_height.setRange(16, 100000)
        self.crop_height.setValue(270)
        self.crop_height.setSuffix(' px')
        form.addRow('Высота', self.crop_height)
        self.min_area = QSpinBox()
        self.min_area.setRange(1, 1000000)
        self.min_area.setValue(20)
        self.min_area.setSuffix(' px²')
        form.addRow('Минимальная площадь', self.min_area)
        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(.1, 5)
        self.smoothing.setSingleStep(.1)
        self.smoothing.setValue(1.2)
        form.addRow('Сглаживание σ', self.smoothing)
        self.marker_depth = QDoubleSpinBox()
        self.marker_depth.setRange(1, 100)
        self.marker_depth.setValue(8)
        form.addRow('Глубина маркеров', self.marker_depth)
        self.detail = QCheckBox('Детализация мелких зёрен')
        self.detail.setChecked(True)
        form.addRow(self.detail)
        self.refine = QCheckBox('Дополнить маски по Watershed')
        self.refine.setChecked(True)
        form.addRow(self.refine)
        self.iou = QDoubleSpinBox()
        self.iou.setRange(.1, .99)
        self.iou.setSingleStep(.02)
        self.iou.setValue(.72)
        form.addRow('Порог predicted IoU', self.iou)
        self.stability = QDoubleSpinBox()
        self.stability.setRange(.1, 1)
        self.stability.setSingleStep(.02)
        self.stability.setValue(.88)
        form.addRow('Стабильность SAM', self.stability)
        self.batch_button = self.button('Обработать весь список', self.batch)
        form.addRow(self.batch_button)
        settings_tabs.addTab(algorithm, 'Алгоритм')
        calibration = QWidget()
        cf = QFormLayout(calibration)
        cf.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.scale = QDoubleSpinBox()
        self.scale.setDecimals(8)
        self.scale.setRange(0, 1e6)
        self.scale.setSingleStep(.001)
        self.unit = QComboBox()
        self.unit.addItems(['µm', 'nm'])
        cf.addRow('Единиц на пиксель (0 = px)', self.scale)
        cf.addRow('Единицы', self.unit)
        self.calibrate_button = self.button('Измерить линейку на снимке', self.start_calibration)
        cf.addRow(self.calibrate_button)
        settings_tabs.addTab(calibration, 'Масштаб')
        self.advanced.hide()
        left.addWidget(self.advanced)
        self.status = QLabel('Откройте снимок → сегментируйте → проверьте результат')
        self.status.setObjectName('Hint')
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        left.addStretch()
        scroll.setWidget(sidebar)
        splitter.addWidget(scroll)

        center = QWidget()
        main = QVBoxLayout(center)
        main.setContentsMargins(4, 0, 0, 0)
        main.setSpacing(8)
        toolbar = QHBoxLayout()
        self.stage = QComboBox()
        self.stage.addItems(STAGES)
        self.stage.setMinimumWidth(205)
        self.stage.currentIndexChanged.connect(self.render)
        toolbar.addWidget(self.stage)
        self.fit_button = self.button('Вписать', lambda: self.viewer.fit_image())
        toolbar.addWidget(self.fit_button)
        toolbar.addStretch()
        self.edit_toggle = QCheckBox('Редактировать зёрна')
        self.edit_toggle.toggled.connect(self.toggle_editor)
        toolbar.addWidget(self.edit_toggle)
        main.addLayout(toolbar)
        self.activity = QLabel('Загрузите снимок, чтобы начать')
        self.activity.setObjectName('CardTitle')
        self.activity.setWordWrap(True)
        main.addWidget(self.activity)
        self.viewer = ImageViewer()
        self.viewer.clicked.connect(self.select)
        self.viewer.stroke.connect(self.stroke)
        self.viewer.calibrated.connect(self.finish_calibration)
        main.addWidget(self.viewer, 1)
        self.editor_panel = QWidget()
        self.editor_panel.setObjectName('EditorPanel')
        el = QVBoxLayout(self.editor_panel)
        el.setContentsMargins(9, 8, 9, 8)
        erow = QHBoxLayout()
        self.tool = QComboBox()
        self.tool.addItems(['Выбрать зерно', 'Добавить зерно по контуру', 'Изменить контур', 'Кисть', 'Ластик', 'Заполнить удалённую область', 'Разделить зерно'])
        self.tool.currentIndexChanged.connect(self.change_tool)
        erow.addWidget(self.tool, 1)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 100)
        self.brush_size.setValue(9)
        self.brush_size.setSuffix(' px')
        self.brush_size.setToolTip('Диаметр кисти или ластика в пикселях исходного кадра')
        self.brush_size.valueChanged.connect(lambda size: setattr(self.viewer, 'brush_width', size))
        erow.addWidget(self.brush_size)
        self.merge_button = self.button('Объединить', self.merge_selected)
        self.delete_button = self.button('Удалить', self.delete_selected)
        self.undo_button = self.button('Отменить', self.undo)
        for button in (self.merge_button, self.delete_button, self.undo_button):
            erow.addWidget(button)
        el.addLayout(erow)
        second = QHBoxLayout()
        self.selection_text = QLabel('Выбор: —')
        second.addWidget(self.selection_text)
        self.clear_button = self.button('Снять выбор', self.clear_selection)
        second.addWidget(self.clear_button)
        second.addStretch()
        self.class_combo = QComboBox()
        self.class_combo.addItems(list(CLASSES.values()))
        self.class_combo.hide()
        self.class_button = QPushButton('Класс объекта')
        class_menu = QMenu(self.class_button)
        for cls, name in CLASSES.items():
            class_menu.addAction(name, lambda checked=False, value=cls: self.set_class(value))
        self.class_button.setMenu(class_menu)
        second.addWidget(self.class_button)
        el.addLayout(second)
        self.editor_panel.hide()
        main.addWidget(self.editor_panel)
        self.note = QLabel('')
        self.note.setWordWrap(True)
        self.note.setObjectName('Hint')
        main.addWidget(self.note)
        self.tabs = QTabWidget()
        self.tabs.setMaximumHeight(225)
        table_page = QWidget()
        tl = QVBoxLayout(table_page)
        tl.setContentsMargins(6, 6, 6, 6)
        self.show_excluded = QCheckBox('Показать исключённые объекты')
        self.show_excluded.toggled.connect(self.refresh_result)
        tl.addWidget(self.show_excluded)
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(['ID', 'Класс', 'В анализе', 'Площадь', 'Периметр', 'Фере', 'Центр X', 'Центр Y', 'Эксцентр.'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellClicked.connect(self.table_select)
        tl.addWidget(self.table)
        self.tabs.addTab(table_page, 'Зёрна и измерения')
        validation = QWidget()
        vl = QVBoxLayout(validation)
        self.metrics_text = QLabel('Для расчёта качества загрузите независимую эталонную разметку.')
        self.metrics_text.setWordWrap(True)
        self.metrics_button = self.button('Сравнить с эталоном (TIFF / NPY)', self.evaluate)
        vl.addWidget(self.metrics_text)
        vl.addWidget(self.metrics_button)
        vl.addStretch()
        self.tabs.addTab(validation, 'Оценка качества')
        main.addWidget(self.tabs)
        splitter.addWidget(center)
        splitter.setStretchFactor(1, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        outer.addWidget(self.progress)
        self.statusBar().showMessage('Ctrl+O — открыть   ·   Ctrl+S — сохранить   ·   Ctrl+Z — отменить правку')
        self.crop_mode.currentIndexChanged.connect(self.update_controls)
        self.scale.valueChanged.connect(self.measurement_settings_changed)
        self.unit.currentTextChanged.connect(self.measurement_settings_changed)
        self.border.toggled.connect(self.measurement_settings_changed)

    def toggle_advanced(self):
        visible = self.advanced_toggle.isChecked()
        self.advanced.setVisible(visible)
        self.advanced_toggle.setText('Скрыть настройки' if visible else 'Показать настройки')

    def toggle_editor(self, enabled):
        self.editor_panel.setVisible(enabled)
        if enabled:
            self.stage.setCurrentIndex(5)
        else:
            self.tool.setCurrentIndex(0)
            self.viewer.set_mode('select')
        self.render()

    def clear_selection(self):
        self.selected = set()
        self.update_controls()
        self.render()

    def set_class(self, cls):
        self.class_combo.setCurrentIndex(cls-1)
        self.assign_class()

    def open_results(self):
        directory = self.last_result_dir if self.last_result_dir and self.last_result_dir.exists() else ROOT/'results'
        self.open_dir(directory)

    def open_dir(self, path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def add_item(self, path):
        for index in range(self.files.count()):
            if self.files.item(index).data(Qt.ItemDataRole.UserRole) == str(path):
                return self.files.item(index)
        item = QListWidgetItem(path.name)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setToolTip(str(path))
        self.files.addItem(item)
        return item

    def open_files(self):
        if self.busy:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, 'Открыть SEM-снимки', str(ROOT / 'originals'), 'Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        self.import_paths(paths)

    def import_paths(self, paths):
        if self.busy:
            return
        expanded = []
        for value in paths:
            path = Path(value)
            if path.is_dir():
                expanded.extend(str(item) for item in sorted(path.iterdir())
                                if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS)
            else:
                expanded.append(str(path))
        first = None
        for path in dict.fromkeys(expanded):
            try:
                load_image(path)  # Повреждённые файлы не попадают в originals.
                copied = import_original(path)
                item = self.add_item(copied)
                first = first or item
            except Exception as exc:
                self.error(str(exc))
        if first is not None:
            self.files.setCurrentItem(first)

    def confirm_discard(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, 'Несохранённый результат', 'Сохранить результат и исправления перед продолжением?', QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def choose_file(self, item, previous):
        if item is None or self.busy:
            return
        if not self.confirm_discard():
            self.files.blockSignals(True)
            self.files.setCurrentItem(previous)
            self.files.blockSignals(False)
            return
        try:
            path = Path(item.data(Qt.ItemDataRole.UserRole))
            image = load_image(path)
            self.path, self.original = path, image
            self.result, self.selected, self.history = None, set(), []
            self.dirty = False
            self.edit_toggle.setChecked(False)
            self.viewer.cancel_drawing()
            self.table.setRowCount(0)
            self.metrics_text.setText('Эталон не загружен. Метрики качества не измерены.')
            self.scale.setValue(0)  # Масштаб другого снимка не переносится неявно.
            self.crop_height.setMaximum(image.shape[0])
            self.crop_height.setValue(image.shape[0])
            self.image_info.setText(f'{path.name}\n{image.shape[1]} × {image.shape[0]} px · {image.dtype}')
            self.stage.setCurrentIndex(0)
            self.tool.setCurrentIndex(0)
            self.viewer.set_mode('select')
            self.viewer.set_image(image, fit=True)
            self.status.setText('Проверьте рабочую область и запустите сегментацию.')
            self.activity.setText('Снимок загружен · результат сегментации ещё не получен')
            self.update_controls()
        except Exception as exc:
            self.files.blockSignals(True)
            self.files.setCurrentItem(previous)
            self.files.blockSignals(False)
            self.error(str(exc))

    def settings(self):
        crop = [0, -1, self.crop_height.value()][self.crop_mode.currentIndex()]
        return Settings(crop_height=crop,
                        min_area=self.min_area.value(), smoothing=self.smoothing.value(),
                        marker_depth=self.marker_depth.value(),
                        pred_iou=self.iou.value(), stability=self.stability.value(),
                        adaptive_detail=self.detail.isChecked(),
                        refine_boundaries=self.refine.isChecked(),
                        exclude_border=self.border.isChecked(), scale=self.scale.value(), unit=self.unit.currentText())

    def update_controls(self, *args):
        loaded = self.original is not None
        ready = self.result is not None
        idle = not self.busy
        self.open_button.setEnabled(idle)
        self.files.setEnabled(idle)
        self.run_button.setEnabled(loaded and idle)
        self.batch_button.setEnabled(self.files.count() > 0 and idle)
        self.cancel_button.setVisible(self.busy and self.worker is not None and self.worker.paths is not None)
        self.save_button.setEnabled(ready and idle)
        self.restore_button.setEnabled(loaded and idle)
        self.metrics_button.setEnabled(ready and idle)
        self.tool.setEnabled(ready and idle)
        self.edit_toggle.setEnabled(ready and idle)
        self.show_excluded.setEnabled(ready and idle)
        self.brush_size.setEnabled(ready and idle)
        self.brush_size.setVisible(self.tool.currentIndex() in (3, 4))
        self.stage.setEnabled(ready)
        self.merge_button.setEnabled(ready and idle and len(self.selected) > 1)
        self.delete_button.setEnabled(ready and idle and bool(self.selected))
        self.class_button.setEnabled(ready and idle and bool(self.selected))
        self.undo_button.setEnabled(bool(self.history) and idle)
        self.calibrate_button.setEnabled(loaded and idle)
        for control in (self.crop_mode, self.min_area, self.smoothing, self.scale, self.unit, self.border):
            control.setEnabled(idle)
        self.crop_height.setEnabled(idle and self.crop_mode.currentIndex() == 2)
        self.marker_depth.setEnabled(idle)
        for control in (self.iou, self.stability, self.detail, self.refine):
            control.setEnabled(idle)
        self.selection_text.setText('Выбор: ' + (', '.join(map(str, sorted(self.selected))) or '—'))

    def start_worker(self, paths=None):
        self.busy = True
        self.pending_result = None
        self.worker = ProcessingWorker(self.original, self.settings(), paths)
        self.worker.progress.connect(self.show_progress)
        self.worker.completed.connect(self.received_result)
        self.worker.failed.connect(self.processing_failed)
        self.worker.batch_item.connect(self.batch_saved)
        self.worker.finished.connect(self.worker_finished)
        self.progress.show()
        self.update_controls()
        self.worker.start()

    def run(self):
        if self.original is None or self.busy or not self.confirm_discard():
            return
        self.start_worker()

    def batch(self):
        if self.busy:
            return
        # Масштаб/высота задаются для каждого кадра отдельно: пакет всегда автообрезка и px.
        settings = self.settings()
        if settings.scale > 0 or settings.crop_height != 0:
            self.error('Для пакета выберите автоматическую рабочую область и масштаб 0 (пиксели). Физическую калибровку выполняйте отдельно для каждого снимка.')
            return
        self.start_worker([Path(self.files.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.files.count())])

    def cancel_batch(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.status.setText('Остановка после завершения текущего снимка…')

    def received_result(self, result):
        self.pending_result = result

    def show_progress(self, message):
        self.status.setText(message)
        self.activity.setText(message)

    def processing_failed(self, message):
        self.status.setText('Обработка не выполнена. Предыдущий результат сохранён в окне.')
        self.activity.setText('Обработка не выполнена · проверьте сообщение об ошибке')
        self.error(message)

    def worker_finished(self):
        worker, self.worker = self.worker, None
        self.busy = False
        self.progress.hide()
        if self.pending_result is not None:
            result = self.pending_result
            if result.get('batch'):
                message = 'Пакет остановлен.' if result['cancelled'] else 'Пакетная обработка завершена.'
                self.status.setText(message + ' Результаты находятся в results.')
                self.activity.setText(message + ' Результаты находятся в results.')
                if result['failures']:
                    self.error('\n'.join(result['failures']))
            else:
                self.result = result
                self.viewer.cancel_drawing()
                self.tool.setCurrentIndex(0)
                self.selected, self.history = set(), []
                self.dirty = True
                self.stage.setCurrentIndex(5)
                self.refresh_result()
        self.pending_result = None
        worker.deleteLater()
        self.update_controls()

    def render(self, *args):
        if self.original is None:
            return
        index = self.stage.currentIndex()
        self.note.setText(NOTES[index])
        if index == 0 or self.result is None:
            image = self.original
        elif index == 1:
            image = self.result['preprocessed']
        elif index == 2:
            image = self.result['edges']
        elif index == 3:
            image = labels_to_color(self.result['raw'])
        elif index == 6:
            watershed = self.result.get('watershed')
            if watershed is None:
                self.note.setText('Области Watershed не сохранены в этой сессии. Запустите общую сегментацию, чтобы получить оба этапа.')
                image = self.original
            else:
                image = labels_to_color(watershed)
        else:
            labels = self.result['labels'] if self.show_excluded.isChecked() else analysis_labels(self.result)
            image = labels_to_color(labels) if index == 4 else make_overlay(self.original, labels, self.selected)
        self.viewer.set_image(image)
        if self.edit_toggle.isChecked():
            self.note.setText(self.tool_help())

    def refresh_result(self, *args):
        if self.result is None:
            return
        self.result.setdefault('deleted_labels', np.zeros(self.result['labels'].shape, np.int32))
        all_rows = measure(self.original, self.result)
        self.measurements = all_rows
        rows = all_rows if self.show_excluded.isChecked() else [row for row in all_rows if row['included']]
        self.table.setRowCount(len(rows))
        unit = all_rows[0]['unit'] if all_rows else 'px'
        self.table.setHorizontalHeaderLabels(['ID', 'Класс', 'В анализе', f'Площадь {unit}²', f'Периметр {unit}', f'Фере {unit}', f'Центр X {unit}', f'Центр Y {unit}', 'Эксцентр.'])
        for i, row in enumerate(rows):
            reason = 'да' if row['included'] else ('класс' if row['class'] != 'Зерно' else 'край')
            values = [row['id'], row['class'], reason, row['area'], row['perimeter'], row['feret_max'], row['centroid_x'], row['centroid_y'], row['eccentricity']]
            for j, value in enumerate(values):
                item = QTableWidgetItem(f'{value:.3f}' if isinstance(value, float) else str(value))
                if j == 0:
                    item.setData(Qt.ItemDataRole.UserRole, row['id'])
                if not row['included']:
                    from PySide6.QtGui import QColor
                    item.setForeground(QColor('#54657c'))
                    item.setBackground(QColor('#eef1f5'))
                self.table.setItem(i, j, item)
        included = sum(row['included'] for row in all_rows)
        excluded = len(all_rows)-included
        edge_count = sum(row['touches_border'] and not row['included'] and row['class']=='Зерно' for row in all_rows)
        self.status.setText(f'Найдено объектов: {len(all_rows)}\nВ анализе: {included}\nИсключено по краю: {edge_count}')
        self.activity.setText(f'В анализе {included} зёрен · исключено {excluded} · всего {len(all_rows)}')
        self.show_metrics()
        self.update_controls()
        self.render()

    def select(self, x, y, multi):
        if self.result is None or self.busy or self.stage.currentIndex() not in (4, 5):
            return
        labels = self.result['labels']
        if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
            return
        if self.viewer.mode == 'fill':
            try:
                target = next(iter(self.selected), 0) if len(self.selected) == 1 else 0
                output, label_id = editing.restore_region(labels, self.result['deleted_labels'], x, y, self.result['info']['working_height'], target)
                self.checkpoint()
                self.result['labels'] = output
                self.changed({label_id})
            except Exception as exc:
                self.error(str(exc))
            return
        visible = labels if self.show_excluded.isChecked() else analysis_labels(self.result)
        label_id = int(visible[y, x])
        if not multi:
            self.selected = {label_id} if label_id else set()
        elif label_id:
            self.selected.symmetric_difference_update({label_id})
        self.update_controls()
        self.render()

    def table_select(self, row, column):
        if self.result is not None and not self.busy:
            self.selected = {self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)}
            self.edit_toggle.setChecked(True)
            self.stage.setCurrentIndex(5)
            self.render()
            self.update_controls()

    def checkpoint(self):
        self.result.setdefault('deleted_labels', np.zeros(self.result['labels'].shape, np.int32))
        self.history.append((self.result['labels'].copy(), dict(self.result['classes']),
                             self.result['deleted_labels'].copy(), set(self.selected)))
        self.history = self.history[-20:]

    def changed(self, selection=None):
        labels, classes = editing.normalize_components(self.result['labels'], self.result['classes'])
        self.result['labels'], self.result['classes'] = labels, classes
        self.result['deleted_labels'][labels > 0] = 0
        self.result['info']['manually_edited'] = True
        self.result['info']['metrics'] = None
        self.result['info'].pop('reference_sha256', None)
        self.dirty = True
        self.selected = set(selection or ()) & set(map(int, np.unique(labels))) - {0}
        self.result['info']['final_count'] = len(np.unique(labels[labels > 0]))
        self.refresh_result()

    def undo(self):
        if self.history and not self.busy:
            labels, classes, deleted, selected = self.history.pop()
            self.result['labels'], self.result['classes'], self.result['deleted_labels'] = labels, classes, deleted
            self.changed(selected)

    def merge_selected(self):
        if self.result is None or self.busy:
            return
        try:
            labels, target = editing.merge(self.result['labels'], self.selected)
            self.checkpoint()
            cls = self.result['classes'].get(target, 1)
            for idx in self.selected:
                self.result['classes'].pop(idx, None)
            self.result['classes'][target] = cls
            self.result['labels'] = labels
            self.changed({target})
        except Exception as exc:
            self.error(str(exc))

    def delete_selected(self):
        if self.result is None or not self.selected or self.busy:
            return
        self.checkpoint()
        for idx in self.selected:
            mask = self.result['labels'] == idx
            self.result['deleted_labels'][mask] = int(self.result['deleted_labels'].max())+1
            self.result['labels'][mask] = 0
            self.result['classes'].pop(idx, None)
        self.changed()
        self.statusBar().showMessage('Область удалена. Верните её инструментом «Заполнить удалённую область» или добавьте новое зерно контуром / кистью.', 10000)

    def assign_class(self):
        if self.result is None or not self.selected or self.busy:
            return
        self.checkpoint()
        for idx in self.selected:
            self.result['classes'][idx] = self.class_combo.currentIndex()+1
        self.changed(self.selected)

    def tool_help(self):
        return [
            'Щелчок — выбрать; Ctrl + щелчок — выбрать несколько. Перетаскивание — перемещение снимка.',
            'Отметьте вершины контура и нажмите Enter / двойной щелчок. Можно обвести зерно удерживаемой кнопкой. Esc — отмена контура.',
            'Выберите одно зерно и обведите его новый контур. Enter / двойной щелчок — завершить.',
            'Кисть дополняет выбранное зерно. Чтобы рисовать новое, нажмите «Снять выбор». Размер указан в пикселях снимка.',
            'Ластик удаляет только закрашенные пиксели. Их можно вернуть заливкой удалённой области или кистью.',
            'Щёлкните удалённую область: без выбора — восстановить отдельное зерно; с выбранным соседом — присоединить к нему.',
            'Выберите одно зерно, затем проведите линию поперёк него от одного края за другой.',
        ][self.tool.currentIndex()]

    def change_tool(self, *args):
        mode = ['select', 'polygon', 'replace', 'paint', 'erase', 'fill', 'split'][self.tool.currentIndex()]
        self.viewer.set_mode(mode)
        self.brush_size.setVisible(mode in ('paint', 'erase'))
        if mode != 'select':
            self.stage.setCurrentIndex(5)
        self.note.setText(self.tool_help())

    def stroke(self, points):
        if self.result is None or self.busy:
            return
        mode = self.viewer.mode
        try:
            previous = self.result['labels']
            if mode in ('split', 'replace') and len(self.selected) != 1:
                raise ValueError('Сначала в режиме «Выбрать зерно» выберите одно зерно.')
            selected_id = next(iter(self.selected), 0) if len(self.selected) == 1 else 0
            height = self.result['info']['working_height']
            if mode == 'split':
                labels = editing.split(previous, selected_id, points)
                target = selected_id
            elif mode in ('paint', 'erase'):
                labels, target, mask = editing.paint(previous, points, height, self.brush_size.value(), selected_id, mode=='erase')
            elif mode in ('polygon', 'replace'):
                labels, target = editing.polygon(previous, points, height, selected_id if mode=='replace' else 0)
            else:
                return
            self.checkpoint()
            removed = (previous > 0) & (labels == 0)
            if removed.any():
                self.result['deleted_labels'][removed] = int(self.result['deleted_labels'].max())+1
            if mode == 'split':
                for idx in np.unique(labels[previous == selected_id]):
                    if idx > 0:
                        self.result['classes'][int(idx)] = self.result['classes'].get(selected_id, 1)
            self.result['labels'] = labels
            self.changed({target} if target else self.selected)
        except Exception as exc:
            self.error(str(exc))

    def start_calibration(self):
        if self.original is None or self.busy:
            return
        self.stage.setCurrentIndex(0)
        self.viewer.set_mode('calibrate')
        self.statusBar().showMessage('Проведите отрезок от одного конца масштабной линейки до другого на исходном кадре.')

    def finish_calibration(self, pixels):
        self.viewer.set_mode('select')
        self.tool.setCurrentIndex(0)
        if self.busy:
            return
        if pixels < 1:
            self.error('Масштабная линейка должна занимать хотя бы один пиксель.')
            return
        length, ok = QInputDialog.getDouble(self, 'Калибровка', f'Длина линейки в {self.unit.currentText()} ({pixels:.2f} px):', 1, .000001, 1e9, 6)
        if ok:
            self.scale.setValue(length/pixels)
            self.statusBar().showMessage(f'Масштаб: {length/pixels:.8f} {self.unit.currentText()}/px', 8000)

    def measurement_settings_changed(self, *args):
        if self.result is not None and not self.busy:
            settings = self.result['info']['settings']
            settings.update(scale=self.scale.value(), unit=self.unit.currentText(), exclude_border=self.border.isChecked())
            self.dirty = True
            self.refresh_result()

    def evaluate(self):
        if self.result is None or self.busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Эталонные ID отдельных зёрен', str(ROOT), 'Карта меток (*.tif *.tiff *.npy)')
        if not path:
            return
        try:
            from core.image_loader import file_hash
            reference = np.load(path, allow_pickle=False) if Path(path).suffix.lower() == '.npy' else tifffile.imread(path)
            metrics = quality_metrics(self.result['labels'], reference, self.result['info']['working_height'])
            self.result['info']['metrics'] = metrics
            self.result['info']['reference_sha256'] = file_hash(path)
            self.result['info']['reference_name'] = Path(path).name
            self.dirty = True
            self.show_metrics()
        except Exception as exc:
            self.error(str(exc))

    def show_metrics(self):
        if self.result is None:
            return
        metrics = self.result['info'].get('metrics')
        if metrics is None:
            self.metrics_text.setText('Эталон не загружен или маски изменены. Метрики качества не измерены.\nДля проверки загрузите независимую карту ID: 0 — фон, 1…N — отдельные зёрна.\nУверенность SAM не заменяет экспериментальную точность.')
        else:
            self.metrics_text.setText(f"Foreground IoU: {metrics['foreground_iou']:.3f} · Dice: {metrics['foreground_dice']:.3f}\nInstance Precision / Recall / F1 @ IoU≥0.5: {metrics['instance_precision_iou50']:.3f} / {metrics['instance_recall_iou50']:.3f} / {metrics['instance_f1_iou50']:.3f}\nPQ: {metrics['panoptic_quality_iou50']:.3f} · Boundary F1 (2 px): {metrics['boundary_f1_2px']:.3f}\nTP / FP / FN: {metrics['true_positive']} / {metrics['false_positive']} / {metrics['false_negative']} · Все объекты рабочей области, включая краевые.")

    def save(self):
        if self.result is None or self.busy:
            return False
        try:
            directory = save_result(self.path, self.original, self.result)
            self.dirty = False
            self.last_result_dir = directory
            self.statusBar().showMessage(f'Сохранено: {directory}', 15000)
            return True
        except Exception as exc:
            self.error(str(exc))
            return False

    def restore(self):
        if self.original is None or self.busy or not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Открыть сохранённую сессию', str(ROOT / 'results'), 'Сессия (session.npz)')
        if not path:
            return
        try:
            self.result = load_result(path, self.path, self.original)
            self.last_result_dir = Path(path).parent
            self.viewer.cancel_drawing()
            self.tool.setCurrentIndex(0)
            settings = self.result['info']['settings']
            controls = [self.scale, self.unit, self.border]
            for control in controls:
                control.blockSignals(True)
            self.scale.setValue(settings['scale'])
            self.unit.setCurrentText(settings['unit'])
            self.border.setChecked(settings['exclude_border'])
            for control in controls:
                control.blockSignals(False)
            crop = settings['crop_height']
            self.crop_mode.setCurrentIndex(0 if crop == 0 else 1 if crop == -1 else 2)
            if crop > 0:
                self.crop_height.setValue(crop)
            self.min_area.setValue(settings['min_area'])
            self.smoothing.setValue(settings['smoothing'])
            self.marker_depth.setValue(settings['marker_depth'])
            self.iou.setValue(settings['pred_iou'])
            self.stability.setValue(settings['stability'])
            self.detail.setChecked(settings.get('adaptive_detail', True))
            self.refine.setChecked(settings.get('refine_boundaries', True))
            self.selected, self.history, self.dirty = set(), [], False
            self.stage.setCurrentIndex(5)
            self.refresh_result()
        except Exception as exc:
            self.error(str(exc))

    def error(self, message):
        QMessageBox.warning(self, 'PolyCrystal Analyzer', message)

    def batch_saved(self, path, directory):
        self.last_result_dir = Path(directory)
        self.statusBar().showMessage(f'{Path(path).name}: результат сохранён')

    def closeEvent(self, event):
        if self.busy:
            self.error('Дождитесь текущей обработки. Пакет можно остановить после текущего снимка.')
            event.ignore()
        elif self.confirm_discard():
            event.accept()
        else:
            event.ignore()
