from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Iterable


class AppLauncher:
    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(message)s",
        )
        self._logger = logging.getLogger("ar_shell.launcher")

    def log_action(self, action: str) -> None:
        self._logger.info("Action requested: %s", action)

    def launch_chrome(self) -> None:
        commands = [
            "google-chrome",
            "google-chrome-stable",
            "chromium-browser",
            "chromium",
        ]
        command = self._find_command(commands)
        if command is None:
            self._logger.warning("Chrome not found in PATH")
            return

        try:
            subprocess.Popen([command])
            self._logger.info("Launched %s", command)
        except OSError as exc:
            self._logger.error("Failed to launch %s: %s", command, exc)

    @staticmethod
    def _find_command(candidates: Iterable[str]) -> str | None:
        for candidate in candidates:
            if shutil.which(candidate):
                return candidate
        return None
