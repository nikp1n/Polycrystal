import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPainterPath, QFont
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from core.image_loader import display_u8
from .file_drop import FileDropMixin


class ImageViewer(FileDropMixin, QGraphicsView):
    clicked = Signal(int, int, bool)
    stroke = Signal(list)
    calibrated = Signal(float)

    def __init__(self):
        super().__init__()
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.item = QGraphicsPixmapItem()
        self.scene_.addItem(self.item)
        self.setBackgroundBrush(QColor('#edf1f6'))
        self.setStyleSheet('QGraphicsView {background: #edf1f6; border: 1px solid #cbd6e4; border-radius: 7px;}')
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMinimumSize(350, 220)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.mode = 'select'
        self.points = []
        self.path_item = None
        self.press = None
        self.has_image = False
        self.brush_width = 9
        self.auto_fit = True
        self.ignore_release = False
        self.empty = self.scene_.addText('Перетащите снимки сюда\nили нажмите «Открыть снимки»')
        self.empty.setFont(QFont('Segoe UI', 15))
        self.empty.setDefaultTextColor(QColor('#354f70'))

    def set_image(self, image, fit=False):
        array = display_u8(image)
        if array.ndim == 3 and array.shape[-1] == 4:
            array = np.ascontiguousarray(array[..., :3])
        h, w = array.shape[:2]
        fmt = QImage.Format.Format_Grayscale8 if array.ndim == 2 else QImage.Format.Format_RGB888
        qimage = QImage(array.data, w, h, array.strides[0], fmt).copy()
        changed = self.item.pixmap().size() != qimage.size()
        self.item.setPixmap(QPixmap.fromImage(qimage))
        self.scene_.setSceneRect(0, 0, w, h)
        self.empty.hide()
        self.has_image = True
        if changed or fit:
            self.fit_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.has_image and self.auto_fit:
            self.fit_image()
        elif not self.has_image:
            self.scene_.setSceneRect(0, 0, self.viewport().width(), self.viewport().height())
            rect = self.empty.boundingRect()
            self.empty.setPos(max(0, (self.viewport().width()-rect.width())/2), max(0, (self.viewport().height()-rect.height())/2))

    def fit_image(self):
        if self.has_image:
            self.auto_fit = True
            self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)

    def cancel_drawing(self):
        if self.path_item is not None:
            self.scene_.removeItem(self.path_item)
        self.path_item = None
        self.points = []

    def set_mode(self, mode):
        self.cancel_drawing()
        self.mode = mode
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if mode == 'select' else QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.OpenHandCursor if mode == 'select' else Qt.CursorShape.CrossCursor)

    def wheelEvent(self, event):
        if self.has_image:
            factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
            if .02 <= self.transform().m11()*factor <= 40:
                self.auto_fit = False
                self.scale(factor, factor)
        event.accept()

    def draw_preview(self, cursor=None):
        if not self.points:
            return
        path = QPainterPath()
        path.moveTo(*self.points[0])
        for point in self.points[1:]:
            path.lineTo(*point)
        if cursor is not None:
            path.lineTo(*cursor)
        color = QColor('#d52c43' if self.mode == 'erase' else '#0865bd')
        pen = QPen(color, self.brush_width if self.mode in ('paint', 'erase') else 0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if self.path_item is None:
            self.path_item = self.scene_.addPath(path, pen)
            self.path_item.setZValue(10)
        else:
            self.path_item.setPath(path)

    def finish_polygon(self):
        points = list(self.points)
        if len(points) >= 3:
            self.cancel_drawing()
            self.stroke.emit(points)

    def mousePressEvent(self, event):
        self.setFocus()
        self.press = event.position().toPoint()
        if self.has_image and event.button() == Qt.MouseButton.LeftButton and self.mode not in ('select', 'fill'):
            point = self.mapToScene(self.press)
            if self.mode not in ('polygon', 'replace'):
                self.cancel_drawing()
            self.points.append((point.x(), point.y()))
            self.draw_preview()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.points and self.mode not in ('select', 'fill'):
            point = self.mapToScene(event.position().toPoint())
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.points.append((point.x(), point.y()))
                self.draw_preview()
            elif self.mode in ('polygon', 'replace'):
                self.draw_preview((point.x(), point.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.mode in ('polygon', 'replace'):
            point = self.mapToScene(event.position().toPoint())
            endpoint = (point.x(), point.y())
            if not self.points or np.linalg.norm(np.asarray(self.points[-1])-endpoint) > .5:
                self.points.append(endpoint)
            self.finish_polygon()
            self.ignore_release = True
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        if self.ignore_release:
            self.ignore_release = False
            event.accept()
            return
        if self.points and self.mode not in ('select', 'fill'):
            point = self.mapToScene(event.position().toPoint())
            if self.mode in ('polygon', 'replace'):
                if self.press is not None and (self.press-event.position().toPoint()).manhattanLength() > 5:
                    self.points.append((point.x(), point.y()))
                    self.finish_polygon()
                event.accept()
                return
            points = self.points + [(point.x(), point.y())]
            self.cancel_drawing()
            if self.mode == 'calibrate':
                self.calibrated.emit(float(np.linalg.norm(np.asarray(points[-1])-points[0])))
            else:
                self.stroke.emit(points)
            event.accept()
            return
        if self.press is not None and (self.press-event.position().toPoint()).manhattanLength() < 5 and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            self.clicked.emit(int(np.floor(point.x())), int(np.floor(point.y())), bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier))
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.mode in ('polygon', 'replace'):
            self.finish_polygon()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_drawing()
            event.accept()
        else:
            super().keyPressEvent(event)
