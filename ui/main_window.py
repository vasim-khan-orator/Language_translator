from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from services.app_launcher import AppLauncher
from ui.taskbar import Taskbar


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._launcher = AppLauncher()

        self.setWindowTitle("AR Shell")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowOpacity(0.9)

        self._root = QWidget()
        self._root.setObjectName("root")
        self._root.setStyleSheet(
            "#root { background: transparent; }"
        )
        self.setCentralWidget(self._root)

        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self._root)
        layout.setContentsMargins(24, 24, 24, 32)
        layout.setSpacing(16)

        self._content_area = QFrame()
        self._content_area.setObjectName("contentArea")
        self._content_area.setStyleSheet(
            "#contentArea {"
            "  background: rgba(10, 16, 28, 120);"
            "  border: 1px solid rgba(90, 200, 255, 120);"
            "  border-radius: 20px;"
            "}"
        )

        content_layout = QVBoxLayout(self._content_area)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(12)

        header = QLabel("AR Shell Workspace")
        header.setStyleSheet(
            "color: rgb(190, 235, 255);"
            "font-size: 22px;"
            "font-weight: 600;"
        )

        hint = QLabel(
            "Embed modules here: translator, face attendance, detection, voice assistant."
        )
        hint.setStyleSheet(
            "color: rgba(190, 235, 255, 160);"
            "font-size: 14px;"
        )

        self._embed_container = QFrame()
        self._embed_container.setObjectName("embedContainer")
        self._embed_container.setStyleSheet(
            "#embedContainer {"
            "  background: rgba(0, 0, 0, 90);"
            "  border: 1px dashed rgba(120, 220, 255, 120);"
            "  border-radius: 16px;"
            "  min-height: 300px;"
            "}"
        )

        embed_layout = QVBoxLayout(self._embed_container)
        embed_layout.setContentsMargins(16, 16, 16, 16)
        embed_layout.addWidget(
            QLabel("Embedded app container (future CEF/Chromium).")
        )

        content_layout.addWidget(header)
        content_layout.addWidget(hint)
        content_layout.addWidget(self._embed_container, 1)

        self._taskbar = Taskbar()
        self._taskbar.home_clicked.connect(self._handle_home)
        self._taskbar.browser_clicked.connect(self._handle_browser)
        self._taskbar.settings_clicked.connect(self._open_settings)
        self._taskbar.exit_clicked.connect(self.close)

        layout.addWidget(self._content_area, 1)
        layout.addWidget(self._taskbar, 0, Qt.AlignmentFlag.AlignHCenter)

    def _handle_home(self) -> None:
        self._launcher.log_action("home")

    def _handle_browser(self) -> None:
        self._launcher.launch_chrome()

    def _open_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Shell Opacity")
        dialog.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        dialog.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        dialog.setStyleSheet(
            "QDialog {"
            "  background: rgba(8, 12, 20, 220);"
            "  border: 1px solid rgba(90, 200, 255, 160);"
            "  border-radius: 16px;"
            "}"
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Opacity")
        title.setStyleSheet(
            "color: rgb(200, 240, 255);"
            "font-size: 16px;"
            "font-weight: 600;"
        )

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(30, 100)
        slider.setValue(int(self.windowOpacity() * 100))
        slider.valueChanged.connect(self._apply_opacity)

        layout.addWidget(title)
        layout.addWidget(slider)
        dialog.setFixedWidth(300)
        dialog.exec()

    def _apply_opacity(self, value: int) -> None:
        self.setWindowOpacity(value / 100.0)

    def set_embedded_widget(self, widget: QWidget) -> None:
        widget.setParent(self._embed_container)
        embed_layout = self._embed_container.layout()
        if embed_layout is not None:
            embed_layout.addWidget(widget)
