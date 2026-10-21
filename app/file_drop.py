"""Приём файлов из Проводника: явные Qt drop-targets и отложенный импорт."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QListWidget


def local_paths(mime):
    if not mime.hasUrls():
        return []
    return list(dict.fromkeys(url.toLocalFile() for url in mime.urls()
                             if url.isLocalFile() and url.toLocalFile()))


class FileDropMixin:
    def accept_file_drag(self, event):
        window = self.window()
        if getattr(window, 'busy', False) or not local_paths(event.mimeData()):
            event.ignore()
            return False
        if not event.possibleActions() & Qt.DropAction.CopyAction:
            event.ignore()
            return False
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        return True

    def dragEnterEvent(self, event):
        self.accept_file_drag(event)

    def dragMoveEvent(self, event):
        self.accept_file_drag(event)

    def dropEvent(self, event):
        if self.accept_file_drag(event):
            # Скопировать URL до завершения Windows OLE Drop. Файловые операции,
            # диалоги и смена снимка выполняются queued-слотом после возврата из Drop.
            paths = local_paths(event.mimeData())
            self.window().files_dropped.emit(paths)


class DropSurface(FileDropMixin, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)


class DropFileList(FileDropMixin, QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(False)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
