import sys
sys.dont_write_bytecode = True


def main():
    from pathlib import Path
    from app.windows_launch import relaunch_on_desktop
    try:
        if relaunch_on_desktop(Path(__file__)):
            return
    except OSError as exc:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.critical(None, 'Запуск приложения',
                             f'Для переноса файлов из Проводника нужны обычные права.\n{exc}')
        sys.exit(1)
    from PySide6.QtWidgets import QApplication
    from app.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName('PolyCrystal Analyzer')
    app.setOrganizationName('PolyCrystal')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
