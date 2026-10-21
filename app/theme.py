from pathlib import Path
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication


def apply_light_theme():
    app = QApplication.instance()
    app.setStyle('Fusion')
    palette = QPalette()
    colors = {
        'Window': '#f4f6fa', 'WindowText': '#18263b', 'Base': '#ffffff',
        'AlternateBase': '#f3f6fa', 'Text': '#18263b', 'Button': '#ffffff',
        'ButtonText': '#18263b', 'Highlight': '#dceafe', 'HighlightedText': '#143c74',
        'ToolTipBase': '#ffffff', 'ToolTipText': '#18263b', 'PlaceholderText': '#526176',
        'Link': '#175fc3', 'Light': '#ffffff', 'Midlight': '#e9eef5',
        'Mid': '#c9d3e1', 'Dark': '#8898ad', 'Shadow': '#526176',
    }
    for role, color in colors.items():
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled):
            palette.setColor(group, getattr(QPalette.ColorRole, role), QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor('#65758c'))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor('#65758c'))
    app.setPalette(palette)


APP_STYLE = '''
QWidget {font-family: "Segoe UI"; font-size: 10pt; color: #18263b;}
QMainWindow, QWidget#MainSurface {background: #f4f6fa;}
QWidget#Sidebar, QWidget#EditorPanel {background: #ffffff; border: 1px solid #d6dfea; border-radius: 9px;}
QLabel {background: transparent; color: #18263b;}
QLabel#Title {font-size: 20pt; font-weight: 700; color: #172b49;}
QLabel#Subtitle, QLabel#Hint {color: #485c76;}
QLabel#CardTitle {font-size: 10pt; font-weight: 600; color: #29496d;}
QLabel#DropHint {background: #edf4ff; color: #254d80; border: 1px dashed #92aed2; border-radius: 6px; padding: 10px;}
QPushButton, QToolButton {background: #ffffff; color: #203550; border: 1px solid #b8c7d9; border-radius: 6px; padding: 7px 11px;}
QPushButton:hover, QToolButton:hover {background: #eaf2ff; border-color: #5087c6;}
QPushButton:checked, QToolButton:checked {background: #e4efff; color: #154b92; border-color: #5087c6;}
QPushButton:disabled, QToolButton:disabled {background: #f1f4f8; color: #65758c; border-color: #d4dce7;}
QPushButton#Primary {background: #2465b4; color: #ffffff; border-color: #2465b4; font-weight: 600;}
QPushButton#Primary:hover {background: #1b5195;}
QPushButton#Primary:disabled {background: #dbe5f2; color: #4d6480; border-color: #c6d4e5;}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {background: #ffffff; color: #18263b; selection-background-color: #dceafe; selection-color: #143c74; border: 1px solid #bac9dc; border-radius: 5px; padding: 5px; min-height: 22px;}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {background: #eff3f8; color: #65758c;}
QComboBox QAbstractItemView {background: #ffffff; color: #18263b; selection-background-color: #dceafe; selection-color: #143c74; border: 1px solid #aebed1;}
QListWidget, QTableWidget {background: #ffffff; color: #18263b; border: 1px solid #d6dfea; border-radius: 6px; alternate-background-color: #f3f6fa; selection-background-color: #dceafe; selection-color: #143c74;}
QListWidget::item {padding: 7px 9px; color: #18263b;}
QListWidget::item:selected {background: #dceafe; color: #143c74;}
QHeaderView::section {background: #eaf0f7; color: #203550; padding: 7px; border: 0; border-right: 1px solid #ffffff; font-weight: 600;}
QTabWidget::pane {border: 1px solid #d6dfea; border-radius: 6px; background: #ffffff;}
QTabBar::tab {background: #e8eef6; color: #324b68; padding: 8px 12px; margin-right: 2px; border: 0;}
QTabBar::tab:selected {background: #ffffff; color: #164f94; border-top: 2px solid #2465b4; padding-top: 6px;}
QGroupBox {background: #ffffff; border: 1px solid #d6dfea; border-radius: 7px; margin-top: 12px; padding: 14px 9px 8px; color: #203550;}
QGroupBox::title {subcontrol-origin: margin; left: 9px; padding: 0 4px; color: #203550;}
QCheckBox {color: #203550; background: transparent; spacing: 7px;}
QCheckBox::indicator {width: 16px; height: 16px; border: 1px solid #8da1ba; border-radius: 3px; background: #ffffff;}
QCheckBox::indicator:checked {background: #2465b4; border-color: #2465b4;}
QStatusBar, QMenuBar, QMenu {background: #ffffff; color: #203550;}
QMenu::item:selected {background: #dceafe; color: #143c74;}
QStatusBar {border-top: 1px solid #d6dfea;}
QProgressBar {background: #e3ebf6; border: 0; max-height: 6px;}
QProgressBar::chunk {background: #2465b4;}
QScrollArea {background: transparent; border: 0;}
QToolTip {background: #ffffff; color: #18263b; border: 1px solid #aebed1; padding: 7px;}
'''
APP_STYLE += 'QCheckBox::indicator:checked {image: url("' + Path(__file__).with_name('check.svg').as_posix() + '");}'
