from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QToolButton


class Taskbar(QFrame):
    home_clicked = pyqtSignal()
    browser_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    exit_clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("taskbar")
        self.setFixedHeight(64)
        self.setStyleSheet(
            "#taskbar {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "    stop:0 rgba(10, 16, 28, 210), stop:1 rgba(22, 32, 50, 210));"
            "  border: 1px solid rgba(90, 200, 255, 170);"
            "  border-radius: 28px;"
            "}"
            "QToolButton {"
            "  color: rgb(200, 240, 255);"
            "  background: transparent;"
            "  font-size: 14px;"
            "  padding: 6px 16px;"
            "}"
            "QToolButton:hover {"
            "  background: rgba(90, 200, 255, 40);"
            "  border-radius: 16px;"
            "}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(12)

        self._home_button = self._build_button("Home")
        self._browser_button = self._build_button("Browser")
        self._settings_button = self._build_button("Settings")
        self._exit_button = self._build_button("Exit")

        self._home_button.clicked.connect(self.home_clicked.emit)
        self._browser_button.clicked.connect(self.browser_clicked.emit)
        self._settings_button.clicked.connect(self.settings_clicked.emit)
        self._exit_button.clicked.connect(self.exit_clicked.emit)

        layout.addWidget(self._home_button)
        layout.addWidget(self._browser_button)
        layout.addWidget(self._settings_button)
        layout.addWidget(self._exit_button)

    def _build_button(self, label: str) -> QToolButton:
        button = QToolButton(self)
        button.setText(label)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button
